"""Cluster detail and steward feedback — FIG 2.10 / FR-08.

`POST /feedback` is the one write in the API that is not only an enqueue,
and everything it does happens in one transaction: record the verdict,
advance the cluster, write the learning signal (feedback event, pair labels,
graph constraints) and enqueue the job finalizer. All or nothing. A recorded
verdict that left the cluster in the queue sends the same work to the next
steward; an advanced cluster with no label loses the training signal FR-08
exists to capture.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select, tuple_
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.api.deps import DbDep, PrincipalDep, get_principal, require_role
from app.core.security import Role
from app.ml_pipeline.cleaning import clean_fields, clean_value
from app.ml_pipeline.fields import canonical_fields
from app.ml_pipeline.survivorship import select_master_record
from app.models.cluster import Cluster, ClusterMember
from app.models.enums import ClusterStatus, FeedbackDecision
from app.models.feedback import HumanFeedback
from app.models.medallion import (
    EntityConstraint,
    EntityGroupConstraint,
    FeedbackEvent,
    PairLabel,
    PairScore,
    SilverRecord,
)
from app.models.school_record import SchoolRecord
from app.queue.core import enqueue_async, stage_dedupe_key
from app.schemas.feedback import FeedbackRequest, FeedbackResponse
from app.schemas.jobs import ClusterDetail, ClusterMemberSummary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/clusters", tags=["clusters"], dependencies=[Depends(get_principal)])

#: Where each verdict leaves the cluster. MERGE and SPLIT both end in
#: RESOLVED because both produce a surviving master — SPLIT resolves the
#: cluster that remains after the excluded records are pulled out of it.
_DECISION_STATUS = {
    FeedbackDecision.MERGE: ClusterStatus.RESOLVED,
    FeedbackDecision.SPLIT: ClusterStatus.RESOLVED,
    FeedbackDecision.REJECT: ClusterStatus.REJECTED,
}


async def _get_cluster_or_404(db: Any, cluster_id: uuid.UUID) -> Cluster:
    """404, not 403, for another tenant's cluster — see jobs.py."""
    cluster = (
        await db.execute(select(Cluster).where(Cluster.id == cluster_id))
    ).scalar_one_or_none()
    if cluster is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Cluster not found")
    return cluster


@router.get("/{cluster_id}", response_model=ClusterDetail, summary="Cluster with members")
async def get_cluster(db: DbDep, cluster_id: uuid.UUID) -> ClusterDetail:
    """One cluster and every record in it — the review screen's payload."""
    cluster = await _get_cluster_or_404(db, cluster_id)

    rows = (
        await db.execute(
            select(ClusterMember.record_id, ClusterMember.pairwise_score, SchoolRecord.raw_row)
            .join(SchoolRecord, SchoolRecord.id == ClusterMember.record_id)
            .where(ClusterMember.cluster_id == cluster_id)
            .order_by(ClusterMember.pairwise_score.desc().nullslast())
        )
    ).all()

    members = [
        ClusterMemberSummary(
            record_id=record_id,
            pairwise_score=score,
            is_master=record_id == cluster.master_record_id,
            raw_row=raw_row or {},
            clean_fields=clean_fields(canonical_fields(raw_row or {})),
        )
        for record_id, score, raw_row in rows
    ]

    # Built explicitly, not via `model_validate(..., update=...)`: `update` is
    # a `model_copy` keyword that `model_validate` rejects at runtime.
    return ClusterDetail(
        id=cluster.id,
        job_id=cluster.job_id,
        status=cluster.status,
        confidence_score=cluster.confidence_score,
        matcher_version=cluster.matcher_version,
        master_record_id=cluster.master_record_id,
        created_at=cluster.created_at,
        member_count=len(members),
        members=members,
        golden_fields=cluster.golden_fields,
    )


@router.post(
    "/{cluster_id}/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record a steward's verdict (FIG 2.10)",
    dependencies=[Depends(require_role(Role.DATA_STEWARD, Role.ORG_ADMIN))],
)
async def submit_feedback(
    db: DbDep,
    principal: PrincipalDep,
    cluster_id: uuid.UUID,
    payload: FeedbackRequest,
) -> FeedbackResponse:
    """Record the verdict, advance the cluster, write the learning signal."""
    cluster = await _get_cluster_or_404(db, cluster_id)

    if cluster.status in {ClusterStatus.RESOLVED.value, ClusterStatus.REJECTED.value}:
        # Two stewards reaching the same cluster is a UI race; the second
        # is told rather than having their verdict quietly replace the first.
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"Cluster is already {cluster.status}; it has been reviewed.",
        )

    members: dict[uuid.UUID, dict[str, Any]] = {
        record_id: raw_row or {}
        for record_id, raw_row in (
            await db.execute(
                select(ClusterMember.record_id, SchoolRecord.raw_row)
                .join(SchoolRecord, SchoolRecord.id == ClusterMember.record_id)
                .where(ClusterMember.cluster_id == cluster_id)
            )
        ).all()
    }
    _validate_records_belong(payload, set(members))

    is_split = payload.decision is FeedbackDecision.SPLIT
    excluded = set(payload.excluded_record_ids or []) if is_split else set()
    kept = set(members) - excluded
    if is_split and (not excluded or not kept):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="A SPLIT must exclude at least one member and keep at least one.",
        )

    # FIG 2.5's CHECK requires a master on every RESOLVED cluster, and MERGE
    # and SPLIT both resolve. The steward's choice if given, otherwise the
    # most complete surviving record. (SPLIT used to flip to RESOLVED with no
    # master whenever none was picked, which the CHECK turned into a 500.)
    master_id: uuid.UUID | None = None
    if payload.decision is not FeedbackDecision.REJECT:
        master_id = payload.master_record_id or select_master_record(
            [(record_id, members[record_id]) for record_id in sorted(kept, key=str)]
        )

    corrections = _clean_corrections(payload.field_corrections)

    feedback = HumanFeedback(
        # From the token, not the body (TRD 2.5), and written explicitly
        # because FIG 2.8's WITH CHECK rejects an INSERT that omits it.
        tenant_id=principal.tenant_id,
        cluster_id=cluster_id,
        decision=payload.decision.value,
        master_record_id=master_id,
        excluded_record_ids=payload.excluded_record_ids,
        model_version_at_review=cluster.matcher_version,
        reviewed_by=principal.subject,
        field_corrections=corrections,
    )
    db.add(feedback)

    if excluded:
        # Out before the status flips, so the cluster is never RESOLVED while
        # still holding records the steward rejected from it.
        await db.execute(
            delete(ClusterMember).where(
                ClusterMember.cluster_id == cluster_id,
                ClusterMember.record_id.in_(excluded),
            )
        )

    if master_id is not None:
        cluster.master_record_id = master_id
    cluster.status = _DECISION_STATUS[payload.decision].value

    if corrections and master_id is not None:
        # Later layers win: the survivor's own cleaned fields, then anything
        # already recorded for this cluster, then the steward's correction.
        cluster.golden_fields = {
            **clean_fields(canonical_fields(members[master_id])),
            **(cluster.golden_fields or {}),
            **corrections,
        }

    await record_learning_signal(
        db,
        tenant_id=principal.tenant_id,
        cluster=cluster,
        decision=payload.decision,
        kept=kept,
        excluded=excluded,
        master_id=master_id,
        actor_id=principal.subject,
    )

    # Same transaction: if this was the job's last open cluster, the job
    # leaves AWAITING_REVIEW as soon as a worker picks this up.
    await enqueue_async(
        db,
        tenant_id=principal.tenant_id,
        task_name="jobs.finalize_reviewed",
        payload={"job_id": str(cluster.job_id)},
        job_id=cluster.job_id,
        dedupe_key=stage_dedupe_key("jobs.finalize_reviewed", cluster.job_id),
    )

    await db.commit()
    await db.refresh(feedback)

    logger.info("Cluster %s %s by %s", cluster_id, payload.decision.value, principal.subject)
    return FeedbackResponse(
        feedback_id=feedback.id,
        cluster_id=cluster_id,
        decision=payload.decision,
        cluster_status=cluster.status,
        created_at=feedback.created_at,
    )


def decision_pairs(
    decision: FeedbackDecision,
    kept: set[uuid.UUID],
    excluded: set[uuid.UUID],
) -> tuple[set[tuple[uuid.UUID, uuid.UUID]], set[tuple[uuid.UUID, uuid.UUID]]]:
    """(match pairs, non-match pairs) a verdict actually asserts, canonically ordered.

    Only what the steward said, never more:

    * MERGE / SPLIT: every kept pair is a match.
    * SPLIT: every kept x excluded pair is a non-match. Excluded x excluded
      is *unknown* — "these do not belong here" says nothing about whether
      they belong with each other — so it yields no label.
    * REJECT of a pair: a non-match. REJECT of three or more says "this
      grouping is wrong", not "no two of these match", so it yields no
      labels; SPLIT is the precise tool for that.
    """

    def ordered(a: uuid.UUID, b: uuid.UUID) -> tuple[uuid.UUID, uuid.UUID]:
        return (a, b) if str(a) < str(b) else (b, a)

    matches: set[tuple[uuid.UUID, uuid.UUID]] = set()
    non_matches: set[tuple[uuid.UUID, uuid.UUID]] = set()
    if decision is FeedbackDecision.REJECT:
        if len(kept) == 2:
            non_matches.add(ordered(*sorted(kept, key=str)))
        return matches, non_matches
    matches = {ordered(a, b) for a, b in combinations(sorted(kept, key=str), 2)}
    non_matches = {ordered(k, e) for k in kept for e in excluded}
    return matches, non_matches


async def record_learning_signal(
    db: Any,
    *,
    tenant_id: uuid.UUID,
    cluster: Cluster,
    decision: FeedbackDecision,
    kept: set[uuid.UUID],
    excluded: set[uuid.UUID],
    master_id: uuid.UUID | None,
    actor_id: str,
) -> None:
    """Feedback event -> pair labels (with gamma) -> graph constraints.

    Pair labels are written only for pairs the matcher scored, because only
    those carry a comparison vector to learn from; the `pair_labels` trigger
    turns each into an O(k) update of `splink_level_stats`. Labels stay
    record-level (they are training data about *these* comparisons);
    constraints are bound to stable identities, so the decision binds every
    later upload of the same schools (see `identity_rules`).
    """
    if decision is FeedbackDecision.REJECT:
        partition = [[str(r)] for r in sorted(kept, key=str)]
    else:
        partition = [sorted(str(r) for r in kept)]
        if excluded:
            partition.append(sorted(str(r) for r in excluded))

    event_id = uuid.uuid4()
    await db.execute(
        pg_insert(FeedbackEvent).values(
            id=event_id,
            tenant_id=tenant_id,
            job_id=cluster.job_id,
            cluster_id=cluster.id,
            actor_type="steward",
            actor_id=actor_id,
            decision=decision.value,
            partition=partition,
            propensity=1.0,
        )
    )

    matches, non_matches = decision_pairs(decision, kept, excluded)
    labelled = {pair: True for pair in matches} | {pair: False for pair in non_matches}
    if labelled:
        scored = (
            await db.execute(
                select(
                    PairScore.record_l, PairScore.record_r, PairScore.settings_hash, PairScore.gamma
                ).where(
                    PairScore.job_id == cluster.job_id,
                    tuple_(PairScore.record_l, PairScore.record_r).in_(list(labelled)),
                )
            )
        ).all()
        label_rows = [
            {
                "tenant_id": tenant_id,
                "record_l": left,
                "record_r": right,
                "source": "steward",
                "label": labelled[(left, right)],
                "weight": 1.0,
                "settings_hash": settings_hash,
                "gamma": gamma,
                "event_id": event_id,
            }
            for left, right, settings_hash, gamma in scored
        ]
        if label_rows:
            stmt = pg_insert(PairLabel).values(label_rows)
            await db.execute(
                stmt.on_conflict_do_update(
                    index_elements=["tenant_id", "record_l", "record_r", "source"],
                    set_={
                        column: stmt.excluded[column]
                        for column in ("label", "weight", "settings_hash", "gamma", "event_id")
                    },
                )
            )

    identity_of = {
        record_id: stable_id
        for record_id, stable_id in (
            await db.execute(
                select(SilverRecord.record_id, SilverRecord.stable_id).where(
                    SilverRecord.record_id.in_(kept | excluded)
                )
            )
        ).all()
        if stable_id
    }
    rules = identity_rules(decision, kept, excluded, master_id, identity_of)
    if rules.unbound:
        logger.info(
            "Cluster %s: %d asserted pairs could not be bound to distinct stable identities",
            cluster.id,
            rules.unbound,
        )

    if rules.pairs:
        stmt = pg_insert(EntityConstraint).values(
            [
                {
                    "tenant_id": tenant_id,
                    "identity_l": left,
                    "identity_r": right,
                    "kind": kind,
                    "event_id": event_id,
                }
                for (left, right), kind in rules.pairs.items()
            ]
        )
        # Latest decision wins: a steward may overturn an earlier one.
        await db.execute(
            stmt.on_conflict_do_update(
                index_elements=["tenant_id", "identity_l", "identity_r"],
                set_={"kind": stmt.excluded.kind, "event_id": stmt.excluded.event_id},
            )
        )
    if rules.group:
        await db.execute(
            pg_insert(EntityGroupConstraint)
            .values(tenant_id=tenant_id, identities=rules.group, event_id=event_id)
            .on_conflict_do_nothing()
        )


@dataclass(frozen=True)
class IdentityRules:
    """What a verdict asserts, bound to stable identities."""

    pairs: dict[tuple[str, str], str]
    group: list[str] | None
    #: Asserted pairs with no rule: a record without a stable id, or two
    #: records sharing one (nothing can separate an identity from itself).
    unbound: int


def identity_rules(
    decision: FeedbackDecision,
    kept: set[uuid.UUID],
    excluded: set[uuid.UUID],
    master_id: uuid.UUID | None,
    identity_of: dict[uuid.UUID, str],
) -> IdentityRules:
    """Translate a verdict on records into rules on identities.

    * MERGE / SPLIT keep-set: must-link, as a star on the master's identity.
    * SPLIT: cannot-link every kept identity with every excluded one.
    * REJECT: two distinct identities -> one cannot-link; three or more ->
      one not-all-together group. "This grouping is wrong" is not "no two of
      these match", so a larger rejection must not become pairwise
      never-merges — but it must stop the same grouping being re-proposed.
    """
    pairs: dict[tuple[str, str], str] = {}
    unbound = 0

    def add(a: uuid.UUID, b: uuid.UUID, kind: str) -> None:
        nonlocal unbound
        left, right = identity_of.get(a), identity_of.get(b)
        if left is None or right is None or left == right:
            unbound += 1
            return
        low, high = sorted((left, right))
        # A cannot-link from the same verdict outranks a must-link implied
        # through a shared identity: the steward's explicit split wins.
        if pairs.get((low, high)) != "cannot_link":
            pairs[(low, high)] = kind

    if decision is FeedbackDecision.REJECT:
        distinct = sorted({identity_of[r] for r in kept if r in identity_of})
        unbound += sum(1 for r in kept if r not in identity_of)
        if len(distinct) == 2:
            pairs[(distinct[0], distinct[1])] = "cannot_link"
            return IdentityRules(pairs, None, unbound)
        return IdentityRules(pairs, distinct if len(distinct) >= 3 else None, unbound)

    for k in sorted(kept, key=str):
        for e in sorted(excluded, key=str):
            add(k, e, "cannot_link")
    if master_id is not None:
        for other in sorted(kept - {master_id}, key=str):
            add(master_id, other, "must_link")
    return IdentityRules(pairs, None, unbound)


def _clean_corrections(corrections: dict[str, str] | None) -> dict[str, str] | None:
    """Corrected values, cleaned the same way every displayed value is.

    A correction that cleans down to nothing is dropped rather than stored
    as a blank that would overwrite a real value in the golden record.
    """
    if not corrections:
        return None
    cleaned = {key: clean_value(value) for key, value in corrections.items()}
    return {key: value for key, value in cleaned.items() if value} or None


def _validate_records_belong(payload: FeedbackRequest, member_ids: set[uuid.UUID]) -> None:
    """Every record named must be in the cluster being reviewed.

    Both foreign keys point at `school_records`, not at this cluster's
    membership, so the database would happily accept a master record from
    someone else's cluster.
    """
    if payload.master_record_id is not None and payload.master_record_id not in member_ids:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="master_record_id is not a member of this cluster",
        )

    stray = set(payload.excluded_record_ids or []) - member_ids
    if stray:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"excluded_record_ids are not members of this cluster: {sorted(map(str, stray))}"
            ),
        )

    if (
        payload.decision is FeedbackDecision.SPLIT
        and payload.master_record_id is not None
        and payload.master_record_id in set(payload.excluded_record_ids or [])
    ):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="master_record_id cannot also be excluded",
        )
