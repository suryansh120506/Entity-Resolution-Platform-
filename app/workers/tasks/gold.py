"""`gold` stage — link an upload against everything the tenant already knows.

Runs after in-upload matching and before the confidence gate. In-upload
matching finds the duplicates *within* this file; this stage finds which of
this file's records are schools the tenant has already seen.

**The pool** (one representative per stable identity, from the tenant's
finished jobs):

* ``gold_master`` — the surviving record of every AUTO_MERGED / RESOLVED
  cluster: decided entities.
* ``released`` — every other Valid record not held by a live cluster:
  singletons, members of REJECTED clusters, records a SPLIT excluded. These
  used to be orphaned for good; now every later upload is matched against
  them, and the steward rules that made them orphans (never-merge,
  not-all-together) are exactly what stops a bad merge being re-proposed.

Deduplicating the pool by stable identity matters twice: the same school
uploaded in four earlier files is one link target rather than four
"ambiguous" ones, and the pool stays proportional to distinct entities, not
to everything ever uploaded.

**The model** is the tenant's warm base model with steward feedback
blended in — the one in-upload matching just used. This stage never runs
EM: with no trained model there is nothing trustworthy to link with, and
the stage records that and steps aside rather than fitting one on whatever
this upload happens to contain.

**Entity persistence.** A cluster whose members auto-link to exactly one
gold entity inherits that entity's id, so the same real-world school
carries one entity_id across uploads.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from typing import Any

import pandas as pd
from sqlalchemy import delete, select, text, update
from sqlalchemy.dialects.postgresql import insert

from app.core.config import get_settings
from app.db.session import tenant_sync_session
from app.ml_pipeline.matching.gold_linking import (
    IncomingRecord,
    LinkCandidate,
    PoolRecord,
    decide_links,
)
from app.ml_pipeline.matching.matcher import RecordMatcher, layout_hash, link_against_pool
from app.ml_pipeline.matching.name_gate import RecordText
from app.models.cluster import Cluster
from app.models.enums import JobStatus
from app.models.medallion import GoldLink
from app.models.tenant import Tenant
from app.ontology.registry import load_ontology
from app.queue.registry import task
from app.workers.identity_constraints import load_tenant_rules
from app.workers.job_state import JobMissingError, load_job, record_progress
from app.workers.tasks.matching import warm_model

logger = logging.getLogger(__name__)

STAGE = "gold_linking"
_INSERT_BATCH = 5_000

_FRAME_COLUMNS = ["record_id", "name_normalized", "address_normalized", "pincode", "stable_id"]

_POOL_SQL = text(
    """
    WITH finished AS (
        SELECT id, created_at FROM jobs
         WHERE id <> :job_id AND status IN ('AWAITING_REVIEW', 'COMPLETED')
    ),
    masters AS (
        SELECT s.record_id, s.stable_id, s.identity_kind, s.name, s.address, s.pincode,
               s.district, s.state, 'gold_master' AS target_kind, c.entity_id,
               c.id AS cluster_id, f.created_at
          FROM clusters c
          JOIN finished f ON f.id = c.job_id
          JOIN silver_records s ON s.record_id = c.master_record_id
         WHERE c.status IN ('AUTO_MERGED', 'RESOLVED') AND s.stable_id IS NOT NULL
    ),
    released AS (
        SELECT s.record_id, s.stable_id, s.identity_kind, s.name, s.address, s.pincode,
               s.district, s.state, 'released' AS target_kind, NULL::uuid AS entity_id,
               NULL::uuid AS cluster_id, f.created_at
          FROM silver_records s
          JOIN finished f ON f.id = s.job_id
          JOIN school_records r ON r.id = s.record_id
         WHERE r.classification = 'Valid' AND s.stable_id IS NOT NULL
           AND NOT EXISTS (
                 SELECT 1 FROM cluster_members m JOIN clusters c ON c.id = m.cluster_id
                  WHERE m.record_id = s.record_id AND c.status <> 'REJECTED')
    )
    SELECT DISTINCT ON (stable_id) *
      FROM (SELECT * FROM masters UNION ALL SELECT * FROM released) pool
     ORDER BY stable_id, (target_kind = 'gold_master') DESC, created_at DESC, record_id
    """
)

_INCOMING_SQL = text(
    """
    SELECT s.record_id, s.stable_id, s.identity_kind, s.name, s.address, s.pincode,
           s.district, s.state
      FROM silver_records s JOIN school_records r ON r.id = s.record_id
     WHERE s.job_id = :job_id AND r.classification = 'Valid'
    """
)


@task("gold.link_job")
def link_job(job_id: str, tenant_id: str) -> dict[str, Any]:
    settings = get_settings()
    with tenant_sync_session(tenant_id) as session:
        try:
            job = load_job(session, job_id)
        except JobMissingError:
            return {"job_id": job_id, "skipped": "job deleted"}
        if JobStatus(job.status) is not JobStatus.MATCHING:
            # Before matching it is premature; after, the gate has run and
            # links would arrive after the job's outcome was decided.
            return {"job_id": job_id, "skipped": f"job is {job.status}"}

        session.execute(delete(GoldLink).where(GoldLink.job_id == job_id))
        incoming_rows = session.execute(_INCOMING_SQL, {"job_id": job_id}).mappings().all()
        pool_rows = session.execute(_POOL_SQL, {"job_id": job_id}).mappings().all()
        details: dict[str, Any] = {"incoming": len(incoming_rows), "pool": len(pool_rows)}

        if not incoming_rows or not pool_rows:
            return _finish(session, job_id, {**details, "skipped": "nothing to link"})

        incoming_frame = _frame(incoming_rows)
        matcher = RecordMatcher(incoming_frame.drop(columns=["stable_id"]))
        model = warm_model(session, matcher, settings.feedback_prior_strength)
        if model is None:
            return _finish(session, job_id, {**details, "skipped": "no trained model yet"})

        scored = link_against_pool(matcher.model_dict, incoming_frame, _frame(pool_rows))
        incoming = _incoming_records(session, job_id, incoming_rows)
        pool = _pool_records(session, pool_rows)
        rules = load_tenant_rules(session)
        tenant = session.execute(select(Tenant).where(Tenant.id == tenant_id)).scalar_one()
        engine = load_ontology(session, job.ontology, tenant_id)

        decisions, stats = decide_links(
            (
                LinkCandidate(
                    str(row["incoming_id"]),
                    str(row["pool_id"]),
                    float(row["match_probability"]),
                    float(row["match_weight"]),
                    [int(g) for g in row["gamma"]],
                )
                for row in scored.to_dict("records")
            ),
            incoming,
            pool,
            must={frozenset((a, b)) for a, b, kind in rules.pairs if kind == "must_link"},
            cannot={frozenset((a, b)) for a, b, kind in rules.pairs if kind == "cannot_link"},
            groups=rules.groups,
            auto_threshold=float(tenant.confidence_auto_merge),
            link_threshold=settings.match_link_threshold,
            force_on_key=engine.spec.identity.force_merge_on_key,
        )

        settings_hash = layout_hash(matcher.comparison_layout()[2])
        rows = [
            {
                "job_id": job_id,
                "record_id": uuid.UUID(d.incoming_id),
                "target_record_id": uuid.UUID(d.pool_id),
                "tenant_id": tenant_id,
                "target_kind": pool[d.pool_id].target_kind,
                "target_entity_id": pool[d.pool_id].entity_id,
                "target_stable_id": pool[d.pool_id].stable_id,
                "match_probability": max(0.0, min(1.0, d.probability)),
                "match_weight": d.weight,
                "gamma": d.gamma,
                "settings_hash": settings_hash if d.gamma is not None else None,
                "status": d.status,
                "reason": d.reason,
            }
            for d in decisions
        ]
        for start in range(0, len(rows), _INSERT_BATCH):
            session.execute(insert(GoldLink), rows[start : start + _INSERT_BATCH])

        inherited = _inherit_entities(session, job_id, decisions, pool)
        return _finish(
            session,
            job_id,
            {
                **details,
                "pairs_scored": int(len(scored)),
                "model": model["version_tag"],
                "entities_inherited": inherited,
                **stats.as_dict(),
            },
        )


def _finish(session: Any, job_id: str, details: dict[str, Any]) -> dict[str, Any]:
    record_progress(
        session,
        job_id,
        STAGE,
        done=details.get("incoming", 0),
        total=details.get("incoming", 0),
        details=details,
    )
    logger.info("Job %s gold linking: %s", job_id, details)
    return {"job_id": job_id, **details}


def _frame(rows: Sequence[Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_id": str(row["record_id"]),
                "name_normalized": row["name"] or None,
                "address_normalized": row["address"] or None,
                "pincode": row["pincode"] or None,
                "stable_id": row["stable_id"],
            }
            for row in rows
        ],
        columns=_FRAME_COLUMNS,
    )


def _text(row: Any) -> RecordText:
    return RecordText.build(
        row["name"] or "", row["address"] or "", row["district"] or "", row["state"] or ""
    )


def _incoming_records(session: Any, job_id: str, rows: Sequence[Any]) -> dict[str, IncomingRecord]:
    """Each record with the identities of its whole in-upload cluster."""
    cluster_of: dict[str, uuid.UUID] = {}
    identities_of: dict[uuid.UUID, set[str]] = {}
    for cluster_id, record_id, stable_id in session.execute(
        text(
            "SELECT m.cluster_id, m.record_id, s.stable_id FROM cluster_members m "
            "JOIN clusters c ON c.id = m.cluster_id "
            "LEFT JOIN silver_records s ON s.record_id = m.record_id "
            "WHERE c.job_id = :job_id AND c.status <> 'REJECTED'"
        ),
        {"job_id": job_id},
    ).all():
        cluster_of[str(record_id)] = cluster_id
        if stable_id:
            identities_of.setdefault(cluster_id, set()).add(stable_id)

    out: dict[str, IncomingRecord] = {}
    for row in rows:
        key = str(row["record_id"])
        own = {row["stable_id"]} if row["stable_id"] else set()
        cluster = cluster_of.get(key)
        out[key] = IncomingRecord(
            record_id=key,
            stable_id=row["stable_id"],
            identity_kind=row["identity_kind"],
            text=_text(row),
            cluster_identities=frozenset(identities_of.get(cluster, set()) | own)
            if cluster is not None
            else frozenset(own),
        )
    return out


def _pool_records(session: Any, rows: Sequence[Any]) -> dict[str, PoolRecord]:
    """Each pool record with the identities of the whole entity behind it."""
    cluster_ids = [row["cluster_id"] for row in rows if row["cluster_id"] is not None]
    members: dict[uuid.UUID, set[str]] = {}
    if cluster_ids:
        for cluster_id, stable_id in session.execute(
            text(
                "SELECT m.cluster_id, s.stable_id FROM cluster_members m "
                "JOIN silver_records s ON s.record_id = m.record_id "
                "WHERE m.cluster_id = ANY(:ids) AND s.stable_id IS NOT NULL"
            ),
            {"ids": cluster_ids},
        ).all():
            members.setdefault(cluster_id, set()).add(stable_id)

    return {
        str(row["record_id"]): PoolRecord(
            record_id=str(row["record_id"]),
            stable_id=row["stable_id"],
            identity_kind=row["identity_kind"],
            target_kind=row["target_kind"],
            entity_id=row["entity_id"],
            text=_text(row),
            entity_identities=frozenset(members.get(row["cluster_id"], set()) | {row["stable_id"]}),
        )
        for row in rows
    }


def _inherit_entities(
    session: Any, job_id: str, decisions: list[Any], pool: dict[str, PoolRecord]
) -> int:
    """Give an in-upload cluster the entity id of the one gold entity its
    members auto-link to. Two different entities is not a merge to make
    silently: the cluster keeps its own id and the links stay for review."""
    targets_of: dict[str, set[uuid.UUID]] = {}
    for d in decisions:
        target = pool[d.pool_id]
        if d.status == "auto_linked" and target.entity_id is not None:
            targets_of.setdefault(d.incoming_id, set()).add(target.entity_id)
    if not targets_of:
        return 0

    by_cluster: dict[uuid.UUID, set[uuid.UUID]] = {}
    for cluster_id, record_id in session.execute(
        text(
            "SELECT m.cluster_id, m.record_id FROM cluster_members m "
            "JOIN clusters c ON c.id = m.cluster_id "
            "WHERE c.job_id = :job_id AND c.status NOT IN ('RESOLVED', 'REJECTED')"
        ),
        {"job_id": job_id},
    ).all():
        for entity in targets_of.get(str(record_id), ()):
            by_cluster.setdefault(cluster_id, set()).add(entity)

    inherited = 0
    for cluster_id, entities in by_cluster.items():
        if len(entities) == 1:
            session.execute(
                update(Cluster)
                .where(Cluster.id == cluster_id)
                .values(entity_id=next(iter(entities)))
            )
            inherited += 1
    return inherited
