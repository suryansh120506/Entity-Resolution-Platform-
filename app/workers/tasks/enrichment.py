"""`enrichment` queue — everything that reads the open web (see app/enrichment).

Three tasks share it:

* `enrich_job` — natural-key registry enrichment, fanned out by
  `silver.standardize` alongside the matching chain.
* `triage_job` — Tier 2: settles the clusters the confidence gate left in
  PENDING_API with a web search and an LLM YES/NO per pair.
* `enrich_golden` — Tier 3: fills each resolved golden record's
  `enrichment_targets` with values found verbatim in web snippets.

The queue is capped at one running task platform-wide (topology.py): the
free LLM tier's rate limit is a global constraint, and DDGS bans by IP. The
DDGS spacing is additionally enforced by an advisory lock, so it holds even
for searches made outside this queue.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Any, Generic, TypeVar

import httpx
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert

from app.core.config import Settings, get_settings
from app.core.secrets import SecretsUnavailableError
from app.db.session import sync_engine, tenant_sync_session
from app.enrichment.golden import GoldenEnrichment, enrich_entity, find_hits, merge_into_golden
from app.enrichment.llm_config import LlmConfig, resolve_llm_config, resolve_llm_pool
from app.enrichment.pincode_infer import PincodeDirectory
from app.enrichment.prefetch import finished, search_pool, windowed
from app.enrichment.sources import (
    EntityFields,
    LlmClient,
    LlmExtractor,
    LlmJudge,
    LlmUnusableError,
    PoliteFetcher,
    SearchFn,
    SearchHit,
    ddgs_search,
)
from app.enrichment.throttle import advisory_lock, ask_patiently, polite_search
from app.enrichment.triage import TriageDecision, triage_cluster
from app.enrichment.worker import EnrichmentWorker
from app.ml_pipeline.survivorship import select_master_record
from app.models.cluster import Cluster, ClusterMember
from app.models.enums import ClusterStatus, JobStatus
from app.models.medallion import EnrichmentResult, SilverRecord
from app.models.reference_pincode import ReferencePincode
from app.models.school_record import SchoolRecord
from app.ontology.engine import REQUIRED_FIELDS, EnrichmentTarget, GoldenEnrichmentRules
from app.ontology.registry import load_ontology
from app.queue.registry import RetryLaterError, task
from app.workers.job_state import (
    JobMissingError,
    StaleStageError,
    advance_status,
    load_job,
    record_progress,
)
from app.workers.tasks.resolution import GOLDEN_TASK, count_clusters, enqueue_stage

logger = logging.getLogger(__name__)

STAGE = "enrichment"
TRIAGE_STAGE = "triage"
GOLDEN_STAGE = "golden_enrichment"
#: Progress is written every N clusters, not every one: it is a jobs-row
#: update, and the results page only needs to see it move.
_PROGRESS_EVERY = 10

T = TypeVar("T")


@task("enrichment.enrich_job")
def enrich_job(job_id: str, tenant_id: str) -> dict[str, Any]:
    settings = get_settings()
    with tenant_sync_session(tenant_id) as session:
        try:
            job = load_job(session, job_id)
        except JobMissingError:
            return {"job_id": job_id, "skipped": "job deleted"}

        rules = load_ontology(session, job.ontology, tenant_id).spec.enrichment
        config = resolve_llm_config(session, tenant_id)
        with httpx.Client(
            timeout=settings.enrichment_timeout_seconds, follow_redirects=True
        ) as client:
            llm = LlmExtractor(client, **config.client_kwargs()) if config else None
            worker = EnrichmentWorker(
                rules=rules,
                fetcher=PoliteFetcher(
                    client,
                    user_agent=settings.enrichment_user_agent,
                    min_interval=settings.enrichment_min_interval_seconds,
                ),
                # Throttled like Tier 2/3: this used to call DDGS unspaced.
                search=_web_search(settings),
                llm=llm,
            )
            if not worker.serp_enabled and not any(r.enabled for r in rules.registries):
                record_progress(
                    session,
                    job_id,
                    STAGE,
                    done=0,
                    total=0,
                    details={"skipped": "no enrichment strategy configured"},
                )
                return {"job_id": job_id, "skipped": "no enrichment strategy configured"}

            stats = worker.run(session, tenant_id=tenant_id, job_id=job_id)

        record_progress(
            session, job_id, STAGE, done=stats["keys"], total=stats["keys"], details=stats
        )
        logger.info("Job %s enrichment: %s", job_id, stats)
        return {"job_id": job_id, **stats}


# --- shared web plumbing ----------------------------------------------------------


def _web_search(settings: Settings) -> SearchFn:
    """DDGS, one call per `ddgs_min_interval_seconds` platform-wide, patient
    with rate limits. Every open-web search goes through this."""
    return polite_search(
        ddgs_search,
        interval=settings.ddgs_min_interval_seconds,
        lock=partial(advisory_lock, sync_engine),
    )


#: How often a running task re-reads the tenant's AI settings: one small
#: SELECT every N clusters, so a key saved in Settings takes effect within
#: seconds, not when the task next restarts (which, for a job with thousands
#: of clusters on a rate-limited tier, can be hours).
_CONFIG_REFRESH_EVERY = 5

LlmT = TypeVar("LlmT", bound=LlmClient)


#: A wait is cut into slices this long, so a key saved in Settings is noticed
#: within seconds even in the middle of a five-minute rate-limit pause.
_WAIT_SLICE_S = 15.0


class _LlmHandle(Generic[LlmT]):
    """The LLM a running task is using: switched live when the tenant changes
    their AI tier, rotated across the server's pool when one model's daily
    budget runs out, and the one place that waits on it.

    A switch to a new tier happens only to a config that answers the
    known-answer check; otherwise the task keeps what it has and logs why, so a
    bad key can never stall a job. Decisions keep the label of the model that
    made them (`audit_label` is read per decision), so a job that changes model
    halfway records exactly where.

    **The pool.** Providers like Groq cap each model separately, so Standard AI
    is several free daily budgets, not one. When a model answers "daily limit
    reached" the handle moves to the next model that is not itself cooling
    down, and only sleeps when every model is. (A daily limit is a rolling
    window: the model refills a trickle continuously, so a cooled-down model
    becomes usable again within minutes, not at midnight.)

    While the whole pool waits (a rate limit, an outage) the handle writes a
    status line to `stage_progress["<stage>_waiting"]`, so the dashboard can say
    *why* nothing is moving instead of showing an ETA that is only an
    extrapolation.
    """

    def __init__(
        self,
        client: httpx.Client,
        make: Callable[..., LlmT],
        pool: LlmConfig | Sequence[LlmConfig],
        *,
        session: Any = None,
        tenant_id: str = "",
        job_id: str = "",
        stage: str = "",
    ) -> None:
        self._client = client
        self._make = make
        self._pool: list[LlmConfig] = [pool] if isinstance(pool, LlmConfig) else list(pool)
        self._cooldown: dict[int, float] = {}
        self._index = 0
        self.llm: LlmT = make(client, **self._pool[0].client_kwargs())
        self._session = session
        self._tenant_id = tenant_id
        self._job_id = job_id
        self._status_key = f"{stage}_waiting"
        # Start out assuming a stale note may exist: a previous process killed
        # mid-wait never cleared its own. The first completed call clears it.
        self._noted = session is not None

    @property
    def config(self) -> LlmConfig:
        return self._pool[self._index]

    def ask(self, call: Callable[[], T]) -> T:
        """`call`, patiently: waits out limits and outages, rotates models on a
        daily limit, and adopts a newly saved key between slices of a wait."""
        try:
            return ask_patiently(call, sleep=self._sleep, on_wait=self._on_wait)
        finally:
            self._clear_note()

    def refresh(self, session: Any, tenant_id: str) -> bool:
        """Adopt the tenant's current settings if they changed. True = switched."""
        try:
            latest = resolve_llm_pool(session, tenant_id)
        except SecretsUnavailableError as exc:
            logger.warning("Keeping %s: %s", self.config.audit_label, exc)
            return False
        except Exception:  # noqa: BLE001 - settings the worker cannot read must not kill the job
            logger.exception("Keeping %s: could not read the AI settings", self.config.audit_label)
            return False
        if not latest or latest == self._pool:
            return False
        candidate = self._make(self._client, **latest[0].client_kwargs())
        try:
            # One attempt, no waiting: a refresh must never hold up the job.
            # A rate-limited or unreachable provider is simply tried again at
            # the next refresh.
            candidate.check()
        except (LlmUnusableError, RetryLaterError, httpx.HTTPError) as exc:
            logger.warning(
                "Not switching to %s (%s); staying on %s",
                latest[0].audit_label,
                type(exc).__name__,
                self.config.audit_label,
            )
            return False
        logger.info("AI tier changed: %s -> %s", self.config.audit_label, latest[0].audit_label)
        self._pool, self._index, self._cooldown, self.llm = latest, 0, {}, candidate
        return True

    def _rotate(self, pause: float) -> bool:
        """The current model hit its daily limit: move to one that has not.
        True = switched to a model that can answer now."""
        now = time.monotonic()
        self._cooldown[self._index] = now + pause
        if len(self._pool) < 2:
            return False
        best = min(range(len(self._pool)), key=lambda i: self._cooldown.get(i, 0.0))
        if best == self._index:
            return False
        previous = self.config.audit_label
        self._index = best
        self.llm = self._make(self._client, **self.config.client_kwargs())
        ready = self._cooldown.get(best, 0.0) <= now
        logger.info(
            "%s hit its daily limit; using %s%s",
            previous,
            self.config.audit_label,
            "" if ready else " (also cooling down)",
        )
        return ready

    def _on_wait(self, reason: str, seconds: float) -> bool:
        if "daily" in reason and self._rotate(seconds):
            return True  # another model has budget: nothing to wait for
        self._note_wait(reason, seconds)
        return False

    def _sleep(self, seconds: float) -> None:
        remaining = seconds
        while remaining > 0:
            time.sleep(min(_WAIT_SLICE_S, remaining))
            remaining -= _WAIT_SLICE_S
            if remaining > 0 and self._session is not None:
                self._session.commit()  # end the read transaction before re-reading settings
                if self.refresh(self._session, self._tenant_id):
                    return  # the retry runs at once, on the new tier

    def _note_wait(self, reason: str, seconds: float) -> None:
        if self._session is None:
            return
        until = datetime.now(UTC) + timedelta(seconds=seconds)
        self._write_note(
            {
                "waiting": f"{self.config.display_name}: {reason}",
                "until": until.isoformat(),
                "premium": self.config.premium,
            }
        )
        self._noted = True

    def _clear_note(self) -> None:
        if self._noted:
            self._write_note({"waiting": None})
            self._noted = False

    def _write_note(self, details: dict[str, Any]) -> None:
        # A status line must never fail the call it describes.
        try:
            record_progress(self._session, self._job_id, self._status_key, done=0, details=details)
            self._session.commit()
        except Exception:  # noqa: BLE001
            logger.warning("Could not record the wait status for job %s", self._job_id)
            self._session.rollback()


def _member_entities(
    session: Any, cluster_id: uuid.UUID
) -> tuple[EntityFields, list[EntityFields]]:
    """(master, other members) for one cluster, as their Silver fields.

    The master is the survivorship choice (`select_master_record`), the same
    record a steward's MERGE would keep. The others are ordered most
    complete first, so if the member cap bites it drops the thinnest.
    """
    rows = session.execute(
        select(
            ClusterMember.record_id,
            SchoolRecord.raw_row,
            SilverRecord.name,
            SilverRecord.address,
            SilverRecord.district,
            SilverRecord.state,
            SilverRecord.pincode,
        )
        .join(SchoolRecord, SchoolRecord.id == ClusterMember.record_id)
        .outerjoin(SilverRecord, SilverRecord.record_id == ClusterMember.record_id)
        .where(ClusterMember.cluster_id == cluster_id)
    ).all()
    entities = {
        str(record_id): EntityFields(
            str(record_id), name or "", address or "", district or "", state or "", pincode or ""
        )
        for record_id, _, name, address, district, state, pincode in rows
    }
    master_id = select_master_record([(record_id, raw or {}) for record_id, raw, *_ in rows])
    if master_id is None:
        raise RuntimeError(f"cluster {cluster_id} has no members")
    master = entities.pop(str(master_id))
    others = sorted(
        entities.values(),
        key=lambda e: (-sum(bool(v) for v in e.template_fields().values()), e.record_id),
    )
    return master, others


# --- Tier 2: automated triage -------------------------------------------------------


@task("enrichment.triage_job")
def triage_job(job_id: str, tenant_id: str) -> dict[str, Any]:
    """Decide every PENDING_API cluster of a job, then complete the job.

    Resumable: each cluster is committed as it is decided, and a re-run only
    sees the clusters still in PENDING_API.
    """
    settings = get_settings()
    with tenant_sync_session(tenant_id) as session:
        try:
            job = load_job(session, job_id)
        except JobMissingError:
            return {"job_id": job_id, "skipped": "job deleted"}
        if JobStatus(job.status) not in (JobStatus.RESOLVING, JobStatus.AWAITING_REVIEW):
            return {"job_id": job_id, "skipped": f"job is {job.status}"}
        # The tenant's own key (Premium) or the server's free LLM (Standard).
        pool = resolve_llm_pool(session, tenant_id)
        if not pool:
            # Permanent until someone configures it: fail loudly (retry_job.py
            # resumes from here) rather than separate everything unread.
            raise RuntimeError(
                "Tier 2 triage needs an LLM: set LLM_BASE_URL (Groq or Ollama) or a "
                "Premium AI key in Settings, then run scripts/retry_job.py for this job"
            )

        rules = load_ontology(session, job.ontology, tenant_id).spec.triage
        pending = (
            session.execute(
                select(Cluster.id)
                .where(Cluster.job_id == job_id, Cluster.status == ClusterStatus.PENDING_API.value)
                .order_by(Cluster.confidence_score.desc().nulls_last(), Cluster.id)
            )
            .scalars()
            .all()
        )
        done = _triaged_count(session, job_id)
        total = done + len(pending)
        if done:
            logger.info("Job %s triage: %d/%d already decided; resuming", job_id, done, total)
        session.commit()

        with httpx.Client(
            timeout=settings.enrichment_timeout_seconds, follow_redirects=True
        ) as client:
            handle = _LlmHandle(
                client,
                LlmJudge,
                pool,
                session=session,
                tenant_id=tenant_id,
                job_id=job_id,
                stage=TRIAGE_STAGE,
            )
            handle.ask(handle.llm.check)
            search = _web_search(settings)
            breaker = _Breaker("Tier 2 triage")

            def ask(
                a: EntityFields,
                hits_a: Sequence[SearchHit],
                b: EntityFields,
                hits_b: Sequence[SearchHit],
            ) -> bool | None:
                # No answer for one pair means "keep separate" (never a merge
                # nobody vouched for). Only a run of failures stops the task.
                try:
                    answer = handle.ask(lambda: handle.llm.same_entity(a, hits_a, b, hits_b))
                except LlmUnusableError as exc:
                    breaker.failed(exc)
                    return None
                breaker.ok()
                return answer

            for position, cluster_id in enumerate(pending):
                if position and position % _CONFIG_REFRESH_EVERY == 0:
                    handle.refresh(session, tenant_id)
                master, others = _member_entities(session, cluster_id)
                # Nothing is held open while the web is consulted.
                session.commit()
                decision = triage_cluster(master, others, rules=rules, search=search, judge=ask)
                if _apply_triage(session, cluster_id, decision, model=handle.config.audit_label):
                    done += 1
                if done % _PROGRESS_EVERY == 0:
                    record_progress(session, job_id, TRIAGE_STAGE, done=done, total=total)
                session.commit()

        outcome = _triage_outcome(session, job_id)
        record_progress(session, job_id, TRIAGE_STAGE, done=done, total=total, details=outcome)
        try:
            completed = _complete_if_decided(session, tenant_id, job_id)
        except StaleStageError as stale:
            return {"job_id": job_id, "skipped": str(stale)}
        logger.info("Job %s triage: %s", job_id, outcome)
        return {"job_id": job_id, "completed": completed, **outcome}


def _apply_triage(
    session: Any, cluster_id: uuid.UUID, decision: TriageDecision, *, model: str
) -> bool:
    """Write one decision. False if the cluster was decided meanwhile.

    Re-read under a row lock: the cluster sat in PENDING_API while the web
    was searched, and a steward may have decided it from the review screen.
    Their verdict stands.
    """
    cluster = session.execute(
        select(Cluster).where(Cluster.id == cluster_id).with_for_update()
    ).scalar_one_or_none()
    if cluster is None or cluster.status != ClusterStatus.PENDING_API.value:
        return False

    if decision.excluded:
        # Out before the status flips, exactly as a steward's SPLIT: the
        # records the LLM did not vouch for go back to the pool on their own.
        session.execute(
            delete(ClusterMember).where(
                ClusterMember.cluster_id == cluster_id,
                ClusterMember.record_id.in_([uuid.UUID(r) for r in decision.excluded]),
            )
        )
    if decision.decision == "REJECT":
        # Members stay, as after a steward's REJECT: the cluster's status is
        # what releases them, and the audit trail keeps who was compared.
        cluster.status = ClusterStatus.REJECTED.value
        cluster.master_record_id = None
    else:
        cluster.status = ClusterStatus.RESOLVED.value
        cluster.master_record_id = uuid.UUID(decision.master_id)
    cluster.resolution_source = "llm_triage"
    cluster.auto_decision = {
        "tier": 2,
        **decision.as_json(),
        "model": model,
        "confidence": None if cluster.confidence_score is None else str(cluster.confidence_score),
        "density": cluster.density,
        "decided_at": datetime.now(UTC).isoformat(),
    }
    session.flush()
    return True


def _triaged_count(session: Any, job_id: str) -> int:
    return int(
        session.execute(
            select(func.count())
            .select_from(Cluster)
            .where(Cluster.job_id == job_id, Cluster.resolution_source == "llm_triage")
        ).scalar_one()
    )


def _triage_outcome(session: Any, job_id: str) -> dict[str, int]:
    decision = Cluster.auto_decision["decision"].astext
    rows = session.execute(
        select(decision, func.count())
        .where(Cluster.job_id == job_id, Cluster.resolution_source == "llm_triage")
        .group_by(decision)
    ).all()
    counts = {f"triage_{(value or 'unknown').lower()}": n for value, n in rows}
    return {"triage_merge": 0, "triage_split": 0, "triage_reject": 0, **counts}


def _complete_if_decided(session: Any, tenant_id: str, job_id: str) -> bool:
    """COMPLETED once nothing waits on the triage worker or a steward.

    If stewards still hold PENDING_REVIEW clusters (only possible when the
    gate ran without an LLM), the job stays where it is and
    `jobs.finalize_reviewed` completes it when they finish.
    """
    if count_clusters(session, job_id, ClusterStatus.PENDING_API, ClusterStatus.PENDING_REVIEW):
        return False
    advance_status(session, job_id, JobStatus.RESOLVING)
    advance_status(session, job_id, JobStatus.COMPLETED)
    enqueue_stage(session, tenant_id, job_id, GOLDEN_TASK)
    return True


# --- Tier 3: universal golden-record enrichment ------------------------------------


@task("enrichment.enrich_golden")
def enrich_golden(job_id: str, tenant_id: str) -> dict[str, Any]:
    """Enrich every merged golden record of a completed job.

    Resumable: a cluster is done once `clusters.enrichment` is set, and each
    is committed as it finishes. Results are also cached per stable
    `entity_id`, so re-uploading the same entities searches nothing until
    the cache expires.
    """
    settings = get_settings()
    with tenant_sync_session(tenant_id) as session:
        try:
            job = load_job(session, job_id)
        except JobMissingError:
            return {"job_id": job_id, "skipped": "job deleted"}

        spec = load_ontology(session, job.ontology, tenant_id).spec
        targets = spec.enrichment_targets
        pool = resolve_llm_pool(session, tenant_id)
        skipped = None
        if not targets:
            skipped = f"ontology {spec.name!r} declares no enrichment_targets"
        elif not pool:
            skipped = "no LLM configured (LLM_BASE_URL or a Premium AI key)"
        if skipped:
            record_progress(
                session, job_id, GOLDEN_STAGE, done=0, total=0, details={"skipped": skipped}
            )
            return {"job_id": job_id, "skipped": skipped}

        golden = (
            (Cluster.job_id == job_id)
            & Cluster.status.in_([ClusterStatus.AUTO_MERGED.value, ClusterStatus.RESOLVED.value])
            & Cluster.master_record_id.is_not(None)
        )
        pending = (
            session.execute(
                select(Cluster.id).where(golden, Cluster.enrichment.is_(None)).order_by(Cluster.id)
            )
            .scalars()
            .all()
        )
        total = int(
            session.execute(select(func.count()).select_from(Cluster).where(golden)).scalar_one()
        )
        done = total - len(pending)
        signature = _signature(targets, spec.golden_enrichment)
        stats: Counter[str] = Counter()
        session.commit()

        with httpx.Client(
            timeout=settings.enrichment_timeout_seconds, follow_redirects=True
        ) as client:
            handle = _LlmHandle(
                client,
                LlmExtractor,
                pool,
                session=session,
                tenant_id=tenant_id,
                job_id=job_id,
                stage=GOLDEN_STAGE,
            )
            handle.ask(handle.llm.check)
            search = _web_search(settings)
            directory = _pincode_directory(session, spec)
            session.commit()
            breaker = _Breaker("Tier 3 extraction")

            def extract(
                keys: Sequence[str],
                subject: str,
                hits: Sequence[SearchHit],
                meanings: Mapping[str, str],
            ) -> Mapping[str, Any]:
                # One school the model cannot answer (a malformed document, a
                # reply cut off at the token cap) is that school's "not found",
                # not the end of a job of thousands. A model that fails every
                # time is another matter: the breaker raises for that.
                try:
                    answer = handle.ask(lambda: handle.llm.extract(keys, subject, hits, meanings))
                except LlmUnusableError as exc:
                    breaker.failed(exc)
                    stats["llm_errors"] += 1
                    return {}
                breaker.ok()
                return answer

            def begin(cluster_id: uuid.UUID, executor: ThreadPoolExecutor) -> _Pending | None:
                """Main thread: load one school and start its search."""
                cluster = session.get(Cluster, cluster_id)
                if cluster is None or cluster.enrichment is not None:
                    return None
                entity = _golden_entity(session, cluster)
                result = _cached(session, cluster.entity_id, signature)
                session.commit()
                if result is not None:
                    return _Pending(cluster_id, entity, finished(result), cached=True)
                return _Pending(
                    cluster_id,
                    entity,
                    executor.submit(find_hits, entity, spec.golden_enrichment, search),
                    cached=False,
                )

            with search_pool(settings.ddgs_concurrency) as workers:
                started = windowed(
                    pending,
                    lambda cluster_id: begin(cluster_id, workers),
                    window=_SEARCH_LOOKAHEAD,
                )
                for position, (cluster_id, pending_work) in enumerate(started):
                    if pending_work is None:
                        continue
                    if position and position % _CONFIG_REFRESH_EVERY == 0:
                        handle.refresh(session, tenant_id)
                    cluster = session.get(Cluster, cluster_id)
                    if cluster is None or cluster.enrichment is not None:
                        continue
                    entity = pending_work.entity
                    outcome = pending_work.future.result()
                    cached = pending_work.cached
                    if cached:
                        result = outcome
                    else:
                        result = enrich_entity(
                            entity,
                            targets=targets,
                            rules=spec.golden_enrichment,
                            generic_tokens=spec.generic_name_tokens,
                            extract=extract,
                            found=outcome,
                            is_valid_pincode=_pincode_check(directory, entity),
                        )
                        _store(
                            session,
                            tenant_id,
                            cluster.entity_id,
                            signature,
                            result,
                            ttl_days=spec.enrichment.ttl_days
                            if result.status == "found"
                            else spec.enrichment.negative_ttl_days,
                        )

                    cluster.golden_fields = merge_into_golden(cluster.golden_fields, result.fields)
                    cluster.enrichment = {
                        **result.as_json(),
                        "targets": [t.key for t in targets],
                        "model": handle.config.audit_label,
                        "cached": cached,
                        "enriched_at": datetime.now(UTC).isoformat(),
                    }
                    stats[result.status] += 1
                    stats["values_kept"] += len(result.fields)
                    stats["values_rejected"] += len(result.rejected)
                    for rejected in result.rejected:
                        stats[f"rejected_{rejected.reason}"] += 1
                    done += 1
                    if done % _PROGRESS_EVERY == 0:
                        record_progress(session, job_id, GOLDEN_STAGE, done=done, total=total)
                    session.commit()

        record_progress(session, job_id, GOLDEN_STAGE, done=done, total=total, details=dict(stats))
        logger.info("Job %s golden enrichment: %s", job_id, dict(stats))
        return {"job_id": job_id, "enriched": done, **stats}


#: Searches started ahead of the school being finished. With two search threads
#: and a model call per school, a handful ahead keeps the search lane busy.
_SEARCH_LOOKAHEAD = 6


@dataclass(slots=True)
class _Pending:
    """A school whose web search was started ahead of its turn."""

    cluster_id: uuid.UUID
    entity: EntityFields
    #: `find_hits`'s `(query, hits)`, or (when `cached`) the cached result.
    future: Future[Any]
    cached: bool


class _Breaker:
    """Lets a few model failures pass and stops a run of them.

    A single failed call is a school's problem. The same failure `limit` times
    in a row is the model's (a retired model, a key that stopped working), and
    carrying on would store "not found" for every remaining school; so this
    re-raises the last error and the task fails with its reason, to be retried.
    """

    def __init__(self, what: str, limit: int = 8) -> None:
        self._what = what
        self._limit = limit
        self._run = 0

    def ok(self) -> None:
        self._run = 0

    def failed(self, error: Exception) -> None:
        self._run += 1
        logger.warning("%s: model call failed (%d in a row): %s", self._what, self._run, error)
        if self._run >= self._limit:
            raise error


def _pincode_check(
    directory: PincodeDirectory | None, entity: EntityFields
) -> Callable[[str], bool] | None:
    """Is-this-PIN-in-this-school's-district, or None when there is no directory
    or the school already has a PIN (nothing to recover)."""
    if directory is None or entity.pincode:
        return None
    return partial(directory.has_pin, state=entity.state, district=entity.district)


def _pincode_directory(session: Any, spec: Any) -> PincodeDirectory | None:
    """The India Post directory, for validating a PIN found on the web. None if
    the table was never seeded (scripts/seed_pincodes.py): PIN recovery is then off."""
    rows = session.execute(
        select(
            ReferencePincode.pincode,
            ReferencePincode.office_name,
            ReferencePincode.district_name,
            ReferencePincode.state_name,
        )
    ).all()
    if not rows:
        logger.info("reference_pincodes is empty: not recovering PIN codes from the web")
        return None
    return PincodeDirectory(
        [(r[0], r[1], r[2], r[3]) for r in rows],
        state_aliases=spec.state_aliases,
        district_aliases=spec.district_aliases,
    )


def _golden_entity(session: Any, cluster: Cluster) -> EntityFields:
    """The golden record's identity fields: the master's Silver values, with
    any steward correction in `golden_fields` taking precedence."""
    silver = session.execute(
        select(SilverRecord).where(SilverRecord.record_id == cluster.master_record_id)
    ).scalar_one_or_none()
    fields = {name: (getattr(silver, name, None) or "") for name in REQUIRED_FIELDS}
    for name in REQUIRED_FIELDS:
        corrected = (cluster.golden_fields or {}).get(name)
        if corrected:
            fields[name] = corrected
    return EntityFields(str(cluster.master_record_id), **fields)


def _signature(targets: Sequence[EnrichmentTarget], rules: GoldenEnrichmentRules) -> str:
    """What a cached result answered. A changed target list or query is a
    different question, so its cached answers do not apply."""
    keys = ",".join(f"{t.key}:{t.type}" for t in targets)
    return f"{rules.query}|{rules.max_results}|{keys}"


def _cached(session: Any, entity_id: uuid.UUID, signature: str) -> GoldenEnrichment | None:
    row = session.execute(
        select(EnrichmentResult.fields).where(
            EnrichmentResult.key_type == "entity",
            EnrichmentResult.key_value == str(entity_id),
            EnrichmentResult.strategy == "golden",
            EnrichmentResult.expires_at > datetime.now(UTC),
        )
    ).scalar_one_or_none()
    if not row or row.get("signature") != signature:
        return None
    return GoldenEnrichment.from_json(row["result"])


def _store(
    session: Any,
    tenant_id: str,
    entity_id: uuid.UUID,
    signature: str,
    result: GoldenEnrichment,
    *,
    ttl_days: int,
) -> None:
    now = datetime.now(UTC)
    stmt = insert(EnrichmentResult).values(
        tenant_id=tenant_id,
        key_type="entity",
        key_value=str(entity_id),
        strategy="golden",
        source="ddgs+llm",
        status=result.status,
        fields={"signature": signature, "result": result.as_json()},
        fetched_at=now,
        expires_at=now + timedelta(days=ttl_days),
    )
    session.execute(
        stmt.on_conflict_do_update(
            constraint="uq_enrichment_results_key",
            set_={
                "status": stmt.excluded.status,
                "fields": stmt.excluded.fields,
                "fetched_at": stmt.excluded.fetched_at,
                "expires_at": stmt.excluded.expires_at,
            },
        )
    )
