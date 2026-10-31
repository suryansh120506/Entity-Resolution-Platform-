"""`resolution` stage — the confidence gate, Tier 1 of automated triage.

Routes each scored cluster using the tenant's own thresholds (FIG 2.1,
migration 0010), with no human in the default path:

    confidence >= auto_merge AND dense       -> AUTO_MERGED        (Tier 1)
    confidence <  api_floor (0.80)           -> REJECTED: separate (Tier 1)
    in between, or a suspected chain         -> PENDING_API        (Tier 2)
    unscored, or steward rules contradict    -> REJECTED: separate (Tier 1)

PENDING_API now means "waiting for the automated triage worker"
(`enrichment.triage_job`: web search + an LLM YES/NO per pair). It needs an
LLM; when none is configured the middle band goes to PENDING_REVIEW for a
steward, as before, so a missing setting never silently separates it.

Doubt resolves to *separate*: a wrong merge hides a real entity inside
another, a missed duplicate is one extra row. Each machine decision records
who made it (`resolution_source`) and on what evidence (`auto_decision`).

**Chain guard.** A cluster's confidence is its weakest spanning-tree edge,
which cannot see chaining: A~B~C~D at 0.97 each scores 0.97 while A and D
were never linked. A cluster of three or more whose edge density is below
`auto_merge_min_density` is never auto-merged, however high its score; it
goes to Tier 2, which checks every member against one anchor record.

Reading thresholds per tenant rather than from `app.core.constants` is the
point of them being columns: the constants are provisioning defaults.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select

from app.core.config import get_settings
from app.db.session import tenant_sync_session
from app.enrichment.llm_config import llm_available, load_settings_row
from app.ml_pipeline.survivorship import select_master_record
from app.models.cluster import Cluster, ClusterMember
from app.models.enums import ClusterStatus, JobStatus
from app.models.school_record import SchoolRecord
from app.models.tenant import Tenant
from app.queue.core import enqueue, stage_dedupe_key
from app.queue.registry import task
from app.workers.job_state import StaleStageError, advance_status, record_progress

logger = logging.getLogger(__name__)

STAGE = "resolution"

#: Statuses representing a decision already taken — never re-routed.
_PROTECTED_STATUSES = (ClusterStatus.RESOLVED.value, ClusterStatus.REJECTED.value)

TRIAGE_TASK = "enrichment.triage_job"
GOLDEN_TASK = "enrichment.enrich_golden"
#: Triage gates a job's completion; golden enrichment only adds to a
#: finished job. On the shared one-slot enrichment queue, triage goes first.
TRIAGE_PRIORITY = 5

#: Every band `route` can return.
BANDS = (
    "auto_merged",
    "below_floor",
    "unscored",
    "constraint_conflict",
    "triage",
    "chain_triage",
    "pending_review",
    "chain_suspect",
)


@task("resolution.resolve_by_confidence")
def resolve_by_confidence(job_id: str, tenant_id: str) -> dict[str, Any]:
    """Route scored clusters into bands. Idempotent: routing is a pure
    function of score, density and the tenant's thresholds, and clusters
    already decided (by a steward or by Tier 1/2) are left untouched."""
    settings = get_settings()
    min_density = settings.auto_merge_min_density

    with tenant_sync_session(tenant_id) as session:
        try:
            tenant = session.execute(select(Tenant).where(Tenant.id == tenant_id)).scalar_one()
            auto_merge: Decimal = tenant.confidence_auto_merge
            floor: Decimal = tenant.confidence_api_floor
            # Tier 2 needs an LLM: the tenant's own key or the server's.
            triage = settings.auto_triage_enabled and llm_available(
                load_settings_row(session, tenant_id), settings
            )

            clusters = (
                session.execute(
                    select(Cluster).where(
                        Cluster.job_id == job_id,
                        Cluster.status.notin_(_PROTECTED_STATUSES),
                    )
                )
                .scalars()
                .all()
            )
            members = load_cluster_members(session, [c.id for c in clusters])

            bands = dict.fromkeys(BANDS, 0)
            for cluster in clusters:
                band = route(
                    cluster,
                    members.get(cluster.id, []),
                    auto_merge,
                    min_density,
                    floor=floor,
                    triage=triage,
                )
                bands[band] += 1

            session.flush()
            record_progress(
                session,
                job_id,
                STAGE,
                done=len(clusters),
                total=len(clusters),
                details={
                    **bands,
                    "auto_merge_min_density": min_density,
                    "auto_merge": str(auto_merge),
                    "separate_below": str(floor),
                    "automated_triage": triage,
                },
            )

            statuses = statuses_after_gate(session, job_id)
            for status in statuses:
                advance_status(session, job_id, status)
            enqueue_followups(session, tenant_id, job_id, statuses)

            logger.info("Job %s routed: %s", job_id, bands)
            return {"job_id": job_id, **bands}

        except StaleStageError as stale:
            logger.info("resolve_by_confidence skipped for job %s: %s", job_id, stale)
            return {"job_id": job_id, "skipped": str(stale)}


def statuses_after_gate(session: Any, job_id: str) -> tuple[JobStatus, ...]:
    """Where the job goes once routing is done.

    Counted from the table, not from this pass's bands: a re-run routes only
    the undecided clusters, and a job whose earlier clusters still await a
    steward must not be completed because this pass found nothing new.

    FIG 2.2 order matters: AWAITING_REVIEW precedes RESOLVING, so a job that
    needs a human must not pass through RESOLVING first (that would make
    the move into review a backwards transition). A job whose open clusters
    are all waiting for the triage worker is RESOLVING: no one is needed.
    """
    if count_clusters(session, job_id, ClusterStatus.PENDING_REVIEW):
        return (JobStatus.AWAITING_REVIEW,)
    if count_clusters(session, job_id, ClusterStatus.PENDING_API):
        return (JobStatus.RESOLVING,)
    return (JobStatus.RESOLVING, JobStatus.COMPLETED)


def enqueue_followups(
    session: Any, tenant_id: str, job_id: str, statuses: tuple[JobStatus, ...]
) -> None:
    """Tier 2 for clusters waiting on it; Tier 3 once the job is complete.

    Enqueued in the gate's own transaction (the queue is a table), so the
    routing and the work it implies commit or roll back together.
    """
    if count_clusters(session, job_id, ClusterStatus.PENDING_API):
        enqueue_stage(session, tenant_id, job_id, TRIAGE_TASK, priority=TRIAGE_PRIORITY)
    if statuses and statuses[-1] is JobStatus.COMPLETED:
        enqueue_stage(session, tenant_id, job_id, GOLDEN_TASK)


def enqueue_stage(
    session: Any, tenant_id: str, job_id: str, task_name: str, *, priority: int | None = None
) -> None:
    enqueue(
        session,
        tenant_id=tenant_id,
        task_name=task_name,
        payload={"job_id": str(job_id)},
        job_id=job_id,
        dedupe_key=stage_dedupe_key(task_name, job_id),
        priority=priority,
    )


def outstanding_reviews(session: Any, job_id: str) -> int:
    """Clusters the job is still waiting on: a steward's or the triage worker's."""
    return count_clusters(session, job_id, ClusterStatus.PENDING_REVIEW, ClusterStatus.PENDING_API)


def count_clusters(session: Any, job_id: str, *statuses: ClusterStatus) -> int:
    return int(
        session.execute(
            select(func.count())
            .select_from(Cluster)
            .where(
                Cluster.job_id == job_id,
                Cluster.status.in_([s.value for s in statuses]),
            )
        ).scalar_one()
    )


def load_cluster_members(session: Any, cluster_ids: list[Any]) -> dict[Any, list[tuple[Any, dict]]]:
    """Cluster id -> [(record_id, raw_row)], for master-record selection."""
    rows = session.execute(
        select(ClusterMember.cluster_id, SchoolRecord.id, SchoolRecord.raw_row)
        .join(SchoolRecord, SchoolRecord.id == ClusterMember.record_id)
        .where(ClusterMember.cluster_id.in_(cluster_ids))
    ).all()

    members: dict[Any, list[tuple[Any, dict]]] = {}
    for cluster_id, record_id, raw_row in rows:
        members.setdefault(cluster_id, []).append((record_id, raw_row or {}))
    return members


def route(
    cluster: Cluster,
    members: list[tuple[Any, dict]],
    auto_merge: Decimal,
    min_density: float,
    *,
    floor: Decimal = Decimal("0.800"),
    triage: bool = False,
) -> str:
    """Assign one cluster's status. Returns the band name (one of BANDS)."""
    score = cluster.confidence_score
    if score is None:
        # Never scored: there is no evidence these records are one entity.
        return _separate(cluster, "unscored")

    if cluster.review_reason is not None:
        # Steward constraints contradict each other inside this cluster. No
        # score and no LLM may overrule a steward, so the records stay apart.
        return _separate(cluster, "constraint_conflict")

    if score < floor:
        return _separate(cluster, "below_floor")

    chain = len(members) >= 3 and cluster.density is not None and cluster.density < min_density
    if score >= auto_merge and not chain:
        master = select_master_record(members)
        if master is None:
            return _separate(cluster, "unscored")
        # FIG 2.5's CHECK refuses AUTO_MERGED without a master; set together.
        cluster.master_record_id = master
        cluster.status = ClusterStatus.AUTO_MERGED.value
        _record(cluster, "MERGE", "auto_merged")
        return "auto_merged"

    # Undecided: drop any decision a previous run of the gate recorded.
    cluster.resolution_source = None
    cluster.auto_decision = None
    if triage:
        cluster.status = ClusterStatus.PENDING_API.value
        return "chain_triage" if chain else "triage"
    cluster.status = ClusterStatus.PENDING_REVIEW.value
    return "chain_suspect" if chain else "pending_review"


def _separate(cluster: Cluster, band: str) -> str:
    cluster.status = ClusterStatus.REJECTED.value
    cluster.master_record_id = None
    _record(cluster, "REJECT", band)
    return band


def _record(cluster: Cluster, decision: str, band: str) -> None:
    cluster.resolution_source = "threshold"
    cluster.auto_decision = {
        "tier": 1,
        "decision": decision,
        "band": band,
        "confidence": None if cluster.confidence_score is None else str(cluster.confidence_score),
        "density": cluster.density,
    }
