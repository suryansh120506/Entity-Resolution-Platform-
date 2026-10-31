"""`matching` stage — score candidate pairs, then cluster the matches (FR-04).

Reads the blocking stage's `candidate_pairs`, scores every one with Splink,
and groups records only through pairs the model scores at or above the link
threshold (see `app/ml_pipeline/matching/clustering.py` for how, and for why
cluster confidence is the bottleneck of the strongest chain).

This stage now *creates* the clusters. It used to receive them from
blocking and only score them, which meant a cluster could exist that the
model never believed in — and on a real upload nearly all of them did, each
scored 0.000 at its weakest pair.

**Idempotency.** Unreviewed clusters are deleted and rebuilt from the pairs,
so a redelivery converges. Clusters a steward has already RESOLVED or
REJECTED are never touched, and their records are held out of the rebuild so
a re-run cannot pull a reviewed record into a second cluster.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pandas as pd
from sqlalchemy import delete, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.config import get_settings
from app.db.session import tenant_sync_session
from app.ml_pipeline.blocking.location import usable_pincode
from app.ml_pipeline.fields import canonical_fields
from app.ml_pipeline.matching.clustering import Constraints, ScoredEdge, build_clusters
from app.ml_pipeline.matching.feedback_model import blend_m_probabilities
from app.ml_pipeline.matching.matcher import RecordMatcher, layout_hash
from app.ml_pipeline.matching.name_gate import RecordText, compatible
from app.models.candidate_pair import JobCandidatePair
from app.models.cluster import Cluster, ClusterMember
from app.models.enums import ClusterStatus, JobStatus, ModelType, RecordClassification
from app.models.medallion import PairScore, SplinkLevelStat, SplinkSettings
from app.models.model_version import ModelVersion
from app.models.school_record import SchoolRecord
from app.ontology.registry import load_ontology
from app.queue.registry import task
from app.workers.identity_constraints import (
    build_constraints,
    load_job_identities,
    load_tenant_rules,
)
from app.workers.job_state import StaleStageError, advance_status, load_job, record_progress

logger = logging.getLogger(__name__)

STAGE = "matching"

#: Clusters representing a decision already taken. Rebuilding them would
#: discard the steward's judgement.
_PROTECTED_STATUSES = (ClusterStatus.RESOLVED.value, ClusterStatus.REJECTED.value)

_INSERT_BATCH = 5_000

#: Pair-score bands reported to the results page. Upper bounds, exclusive
#: except the last.
_SCORE_BANDS = (("distinct", 0.5), ("possible", 0.75), ("likely", 0.95), ("certain", 1.01))


@task("matching.probabilistic_match")
def probabilistic_match(job_id: str, tenant_id: str) -> dict[str, Any]:
    """Score this job's candidate pairs and build clusters from the matches."""
    settings = get_settings()
    with tenant_sync_session(tenant_id) as session:
        try:
            advance_status(session, job_id, JobStatus.MATCHING)
            # Status committed now so the dashboard shows "Matching" while it
            # runs; the cluster rebuild below still commits as one unit, so
            # there is never a moment with the old clusters gone and the new
            # ones not yet written.
            session.commit()

            previous = _previous_memberships(session, job_id)
            held_out = _clear_unreviewed_clusters(session, job_id)
            pairs = _load_pairs(session, job_id, held_out)

            if pairs.empty:
                logger.info("Job %s has no candidate pairs to score", job_id)
                record_progress(
                    session, job_id, STAGE, done=0, total=0, details={"pairs_scored": 0}
                )
                return {"job_id": job_id, "pairs_scored": 0, "clusters": 0}

            attributes, texts = _load_records(session, job_id)
            matcher = RecordMatcher(attributes)
            model = prepare_model(
                session, tenant_id, job_id, matcher, settings.feedback_prior_strength
            )
            scored = matcher.score_pairs(pairs)

            edges, vetoed = gated_edges(scored, texts, settings.match_link_threshold)
            constraints = _load_constraints(session, job_id, set(texts) - held_out)
            result = build_clusters(
                edges,
                link_threshold=settings.match_link_threshold,
                max_size=settings.max_cluster_size,
                constraints=constraints,
            )
            forced = constraints.forced_edges(set(texts) - held_out)
            entity_ids = assign_entity_ids(
                [frozenset(planned.members) for planned in result.clusters], previous
            )
            densities = [
                cluster_density(planned.members, [*forced, *edges], settings.match_link_threshold)
                for planned in result.clusters
            ]
            _persist_clusters(
                session,
                job_id,
                tenant_id,
                result.clusters,
                f"{_matcher_version()} | {model['version_tag']}",
                entity_ids=entity_ids,
                densities=densities,
            )
            _persist_pair_scores(session, job_id, tenant_id, matcher, scored)

            details = {
                "pairs_scored": int(len(scored)),
                "pair_bands": _band_counts(scored["confidence_score"]),
                "link_threshold": settings.match_link_threshold,
                "name_vetoed_links": vetoed,
                "model": model,
                "must_link_constraints": len(constraints.must_link),
                "cannot_link_constraints": len(constraints.cannot_link),
                "not_all_together_constraints": len(constraints.not_all_together),
                "natural_key_groups": len(constraints.forced_groups),
                **result.as_dict(),
            }
            record_progress(
                session,
                job_id,
                STAGE,
                done=len(result.clusters),
                total=len(result.clusters),
                details=details,
            )
            return {"job_id": job_id, **details}

        except StaleStageError as stale:
            # A late duplicate of a stage already passed. Not a failure: the
            # work exists, and marking the job FAILED here would let a stray
            # re-dispatch destroy a job sitting in AWAITING_REVIEW.
            logger.info("%s skipped for job %s: %s", "probabilistic_match", job_id, stale)
            return {"job_id": job_id, "skipped": str(stale)}


def prepare_model(
    session: Any,
    tenant_id: str,
    job_id: str,
    matcher: RecordMatcher,
    prior_strength: float,
) -> dict[str, Any]:
    """Make `matcher` scoring-ready: stored model + steward evidence, or EM.

    * **Warm** — the tenant has an active base model fitted under this exact
      comparison layout. Its m/u are adopted, m is updated from
      `splink_level_stats` (O(levels), see feedback_model.py), and only the
      job's duplicate-rate prior is re-counted. No EM.
    * **Cold start** — no active model, or one fitted under a different
      comparison layout (whose gammas would mean something else). EM runs
      once and the result becomes the tenant's active base model.

    The blended model is not stored: it is a pure function of the base model
    and the statistics, recomputed in milliseconds per job, so storing it
    would only create a second copy that could disagree with the first.
    """
    warm = warm_model(session, matcher, prior_strength)
    if warm is not None:
        warm["prior_refit"] = matcher.reestimate_prior()
        return warm

    current = layout_hash(matcher.untrained_layout()[2])
    active = _active_model(session)
    training = matcher.train()
    tag = f"splink-{datetime.now(UTC):%Y%m%dT%H%M%S}-{job_id[:8]}"
    stored = _store_base_model(
        session, tenant_id, tag, matcher.model_dict, current, training, replace=active
    )
    return {
        "source": "cold_start_em",
        "version_tag": tag,
        "em_run": True,
        "em_rules_trained": len(training.get("em_rules_trained", [])),
        "stored_as_active": stored,
        "replaced": active.version_tag if active is not None else None,
    }


def _active_model(session: Any) -> ModelVersion | None:
    return session.execute(
        select(ModelVersion).where(
            ModelVersion.model_type == ModelType.SPLINK.value, ModelVersion.is_active.is_(True)
        )
    ).scalar_one_or_none()


def warm_model(
    session: Any, matcher: RecordMatcher, prior_strength: float
) -> dict[str, Any] | None:
    """Adopt the tenant's base model + steward evidence, if one fits.

    None when the tenant has no active base model for this comparison layout
    — the caller decides whether that means cold-start EM (matching) or
    "nothing to link with yet" (gold linking).
    """
    current = layout_hash(matcher.untrained_layout()[2])
    active = _active_model(session)
    if active is None or active.layout_hash != current or not active.parameters:
        return None
    counts = {
        (stat.comparison_idx, stat.level): stat.match_w
        for stat in session.execute(
            select(SplinkLevelStat).where(SplinkLevelStat.settings_hash == current)
        ).scalars()
    }
    blended, evidence = blend_m_probabilities(
        active.parameters, counts, prior_strength=prior_strength
    )
    matcher.adopt(blended)
    return {
        "source": "stored+feedback" if evidence else "stored",
        "version_tag": active.version_tag,
        "em_run": False,
        "feedback_weight": evidence,
    }


def _store_base_model(
    session: Any,
    tenant_id: str,
    version_tag: str,
    parameters: dict[str, Any],
    layout: str,
    metrics: dict[str, Any],
    *,
    replace: ModelVersion | None,
) -> bool:
    """Make the cold-start fit the tenant's active Splink base model.

    Two jobs for a new tenant can cold-start concurrently; the partial unique
    index (one active model per tenant and type) lets exactly one become
    active. The loser's insert is a no-op and its job still scores with the
    model it just fitted. Returns whether this fit became active.
    """
    if replace is not None:
        replace.is_active = False
        session.flush()
    inserted = session.execute(
        pg_insert(ModelVersion)
        .values(
            tenant_id=tenant_id,
            model_type=ModelType.SPLINK.value,
            version_tag=version_tag,
            is_active=True,
            parameters=parameters,
            layout_hash=layout,
            metrics=metrics,
            trained_at=datetime.now(UTC),
        )
        .on_conflict_do_nothing()
    )
    return bool(inserted.rowcount)


def _load_constraints(session: Any, job_id: str, matchable: set[str]) -> Constraints:
    """Steward rules (identity-level) binding the records this run clusters.

    Held-out records (already in a reviewed cluster) and non-Valid records
    are not in `matchable`, so rules involving only them drop out. Records
    sharing a natural-key value are forced together when the ontology says
    its keys are authoritative.
    """
    identity, key_groups = load_job_identities(session, job_id, matchable)
    job = load_job(session, job_id)
    engine = load_ontology(session, job.ontology, job.tenant_id)
    return build_constraints(
        identity,
        load_tenant_rules(session),
        forced_groups=key_groups if engine.spec.identity.force_merge_on_key else (),
    )


def _previous_memberships(session: Any, job_id: str) -> dict[uuid.UUID, frozenset[str]]:
    """entity_id -> member record ids for the clusters a rebuild will delete."""
    rows = session.execute(
        select(Cluster.entity_id, ClusterMember.record_id)
        .join(ClusterMember, ClusterMember.cluster_id == Cluster.id)
        .where(Cluster.job_id == job_id, Cluster.status.notin_(_PROTECTED_STATUSES))
    ).all()
    grouped: dict[uuid.UUID, set[str]] = {}
    for entity_id, record_id in rows:
        grouped.setdefault(entity_id, set()).add(str(record_id))
    return {entity_id: frozenset(members) for entity_id, members in grouped.items()}


#: Below this overlap a rebuilt cluster is a different entity, not the old one
#: renumbered. 0.5 means "most of the union is shared".
ENTITY_JACCARD_MIN = 0.5


def assign_entity_ids(
    clusters: list[frozenset[str]], previous: dict[uuid.UUID, frozenset[str]]
) -> list[uuid.UUID]:
    """Carry entity ids across a rebuild by member-set Jaccard similarity.

    Greedy maximum-weight matching: all (new, old) pairs at or above the
    threshold, best first, each side used at most once. Greedy is within a
    factor of 2 of the optimal assignment and, at this threshold, exact in
    practice: two old clusters cannot both overlap a new one by more than
    half its union. Ties break on the old id so a re-run is deterministic.
    """
    by_record: dict[str, list[uuid.UUID]] = {}
    for entity_id, members in previous.items():
        for record_id in members:
            by_record.setdefault(record_id, []).append(entity_id)

    candidates: list[tuple[float, str, int, uuid.UUID]] = []
    for index, members in enumerate(clusters):
        touched = {entity_id for record_id in members for entity_id in by_record.get(record_id, ())}
        for entity_id in touched:
            old = previous[entity_id]
            jaccard = len(members & old) / len(members | old)
            if jaccard >= ENTITY_JACCARD_MIN:
                candidates.append((-jaccard, str(entity_id), index, entity_id))

    assigned: dict[int, uuid.UUID] = {}
    used: set[uuid.UUID] = set()
    for _, _, index, entity_id in sorted(candidates):
        if index in assigned or entity_id in used:
            continue
        assigned[index] = entity_id
        used.add(entity_id)
    return [assigned.get(index) or uuid.uuid4() for index in range(len(clusters))]


def cluster_density(members: list[str], edges: list[ScoredEdge], threshold: float) -> float:
    """Share of member pairs joined by a linking edge.

    A cluster's confidence is its bottleneck spanning-tree edge, which is
    blind to chaining: A~B~C~D at 0.95 each scores 0.95 even if A and D
    were never linked. Density exposes that shape (here 3/6 = 0.5).
    """
    n = len(members)
    if n < 2:
        return 1.0
    member_set = set(members)
    linked = {
        frozenset((edge.record_a, edge.record_b))
        for edge in edges
        if edge.score >= threshold and edge.record_a in member_set and edge.record_b in member_set
    }
    return len(linked) / (n * (n - 1) / 2)


def _persist_pair_scores(
    session: Any, job_id: str, tenant_id: str, matcher: RecordMatcher, scored: pd.DataFrame
) -> None:
    """Write every scored pair with its comparison vector (pair_scores.gamma).

    Keyed by the comparison *layout* hash, so steward labels on this job's
    pairs accumulate into the same sufficient statistics as every other job
    scored under the same comparisons.
    """
    names, level_counts, layout = matcher.comparison_layout()
    settings_hash = layout_hash(layout)
    session.execute(
        pg_insert(SplinkSettings)
        .values(
            tenant_id=tenant_id,
            settings_hash=settings_hash,
            comparison_names=names,
            level_counts=level_counts,
            settings=layout,
        )
        .on_conflict_do_nothing(index_elements=["tenant_id", "settings_hash"])
    )
    session.execute(delete(PairScore).where(PairScore.job_id == job_id))

    rows: list[dict[str, Any]] = []
    for a, b, probability, weight, gamma in zip(
        scored["record_id_a"],
        scored["record_id_b"],
        scored["confidence_score"],
        scored["match_weight"],
        scored["gamma"],
        strict=True,
    ):
        if not isinstance(gamma, list):
            continue  # pair the model could not score (absent from predictions)
        left, right = sorted((str(a), str(b)))
        rows.append(
            {
                "job_id": job_id,
                "record_l": uuid.UUID(left),
                "record_r": uuid.UUID(right),
                "tenant_id": tenant_id,
                "settings_hash": settings_hash,
                "match_probability": float(probability),
                "match_weight": float(weight),
                "gamma": gamma,
            }
        )
    for start in range(0, len(rows), _INSERT_BATCH):
        session.execute(insert(PairScore), rows[start : start + _INSERT_BATCH])
    session.flush()


def _clear_unreviewed_clusters(session: Any, job_id: str) -> set[str]:
    """Delete the job's unreviewed clusters; return records in reviewed ones."""
    protected_ids = list(
        session.execute(
            select(Cluster.id).where(
                Cluster.job_id == job_id, Cluster.status.in_(_PROTECTED_STATUSES)
            )
        )
        .scalars()
        .all()
    )
    held_out: set[str] = set()
    if protected_ids:
        held_out = {
            str(record_id)
            for record_id in session.execute(
                select(ClusterMember.record_id).where(ClusterMember.cluster_id.in_(protected_ids))
            ).scalars()
        }

    stale = select(Cluster.id).where(Cluster.job_id == job_id)
    if protected_ids:
        stale = stale.where(Cluster.id.notin_(protected_ids))
    stale_ids = list(session.execute(stale).scalars().all())
    if stale_ids:
        # Members first: the FK cascades, but deleting explicitly keeps the
        # row counts checkable.
        session.execute(delete(ClusterMember).where(ClusterMember.cluster_id.in_(stale_ids)))
        session.execute(delete(Cluster).where(Cluster.id.in_(stale_ids)))
        logger.info("Job %s: cleared %d unreviewed clusters", job_id, len(stale_ids))
    return held_out


def _load_pairs(session: Any, job_id: str, held_out: set[str]) -> pd.DataFrame:
    rows = session.execute(
        select(
            JobCandidatePair.record_id_a,
            JobCandidatePair.record_id_b,
            JobCandidatePair.distance,
            JobCandidatePair.source,
        ).where(JobCandidatePair.job_id == job_id)
    ).all()
    frame = pd.DataFrame(
        [
            {"record_id_a": str(a), "record_id_b": str(b), "distance": d, "source": s}
            for a, b, d, s in rows
            if str(a) not in held_out and str(b) not in held_out
        ],
        columns=["record_id_a", "record_id_b", "distance", "source"],
    )
    return frame


def _load_records(session: Any, job_id: str) -> tuple[pd.DataFrame, dict[str, RecordText]]:
    """Every Valid record: the frame Splink compares, and the name-gate text.

    All Valid records, not only those in a pair: `u` (how often unrelated
    records agree by chance) is estimated by sampling random pairs, and a
    sample drawn only from records blocking already paired would not be
    random. One query for both, because `raw_row` is the expensive column.
    """
    rows = session.execute(
        select(
            SchoolRecord.id,
            SchoolRecord.name_normalized,
            SchoolRecord.address_normalized,
            SchoolRecord.raw_row,
        ).where(
            SchoolRecord.job_id == job_id,
            SchoolRecord.classification == RecordClassification.VALID.value,
        )
    ).all()

    frame_rows: list[dict[str, str | None]] = []
    texts: dict[str, RecordText] = {}
    for record_id, name, address, raw_row in rows:
        # Through the one alias table, like every other stage — this module
        # used to keep its own list of pincode header names.
        fields = canonical_fields(raw_row or {})
        key = str(record_id)
        # Missing values are None, never "": Splink's training and prior
        # estimation block on *equal* values, and SQL treats "" = "" as a
        # match where NULL = NULL is not. On a real upload 29,691 rows shared
        # the placeholder PIN "000000", and as a value it made one ~440M-pair
        # training block — minutes per EM iteration, the same failure the
        # empty-name bug once caused. As NULL it makes none.
        frame_rows.append(
            {
                "record_id": key,
                "name_normalized": (name or "").strip() or None,
                "address_normalized": (address or "").strip() or None,
                "pincode": usable_pincode(fields["pincode"]),
            }
        )
        texts[key] = RecordText.build(
            name or "", address or "", fields["district"], fields["state"]
        )

    frame = pd.DataFrame(
        frame_rows, columns=["record_id", "name_normalized", "address_normalized", "pincode"]
    )
    return frame, texts


def gated_edges(
    scored: pd.DataFrame, texts: dict[str, RecordText], link_threshold: float
) -> tuple[list[ScoredEdge], int]:
    """Scored pairs as edges, minus links the name gate vetoes.

    Only pairs the model would link are checked — the gate can only take a
    link away, never add one, so checking a pair below the threshold is
    work with no possible effect.
    """
    edges: list[ScoredEdge] = []
    vetoed = 0
    for a, b, score in zip(
        scored["record_id_a"],
        scored["record_id_b"],
        scored["confidence_score"].fillna(0.0),
        strict=True,
    ):
        value = float(score)
        if value >= link_threshold:
            left, right = texts.get(a), texts.get(b)
            if left is None or right is None or not compatible(left, right):
                vetoed += 1
                continue
        edges.append(ScoredEdge(a, b, value))
    return edges, vetoed


def _persist_clusters(
    session: Any,
    job_id: str,
    tenant_id: str,
    clusters: list[Any],
    matcher_version: str,
    *,
    entity_ids: list[uuid.UUID],
    densities: list[float],
) -> None:
    """Bulk-insert clusters and members with client-side ids.

    One INSERT per batch rather than one flush per cluster: a large job can
    produce thousands of clusters, and a round trip each is minutes.
    Status starts at PENDING_REVIEW; the confidence gate routes it next.
    """
    cluster_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    for planned, entity_id, density in zip(clusters, entity_ids, densities, strict=True):
        cluster_id = uuid.uuid4()
        cluster_rows.append(
            {
                "id": cluster_id,
                "tenant_id": tenant_id,
                "job_id": job_id,
                "status": ClusterStatus.PENDING_REVIEW.value,
                "confidence_score": Decimal(f"{planned.confidence:.3f}"),
                "matcher_version": matcher_version,
                "entity_id": entity_id,
                "density": density,
                "review_reason": "constraint_conflict" if planned.constraint_conflict else None,
            }
        )
        for record_id in planned.members:
            member_rows.append(
                {
                    "tenant_id": tenant_id,
                    "cluster_id": cluster_id,
                    "record_id": uuid.UUID(record_id),
                    "pairwise_score": Decimal(f"{planned.member_scores[record_id]:.3f}"),
                }
            )

    for start in range(0, len(cluster_rows), _INSERT_BATCH):
        session.execute(insert(Cluster), cluster_rows[start : start + _INSERT_BATCH])
    for start in range(0, len(member_rows), _INSERT_BATCH):
        session.execute(insert(ClusterMember), member_rows[start : start + _INSERT_BATCH])
    session.flush()
    logger.info("Job %s: wrote %d clusters", job_id, len(cluster_rows))


def _band_counts(scores: pd.Series) -> dict[str, int]:
    counts = dict.fromkeys((name for name, _ in _SCORE_BANDS), 0)
    for score in scores.fillna(0.0):
        for name, upper in _SCORE_BANDS:
            if score < upper:
                counts[name] += 1
                break
    return counts


def _matcher_version() -> str:
    try:
        import splink

        return f"splink {splink.__version__}"
    except Exception:
        return "splink"
