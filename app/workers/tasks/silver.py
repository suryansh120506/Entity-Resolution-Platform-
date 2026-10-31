"""`silver` stage — apply the job's ontology to every record (Bronze -> Silver).

Bronze (`school_records.raw_row`) is never edited. This stage writes the
ontology's typed interpretation of each row to `silver_records`, stamped with
the ontology's content hash, so a re-run under an unchanged ontology is a
no-op upsert and a re-run under an edited one visibly supersedes the rows.

Runs inside PREPROCESSING (after ingest's classification, before embedding):
it adds no lifecycle state of its own, it just must not run on a job that
has already moved past it.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.db.session import tenant_sync_session
from app.models.enums import JobStatus
from app.models.medallion import SilverRecord
from app.models.school_record import SchoolRecord
from app.ontology.registry import load_ontology
from app.queue.registry import task
from app.workers.job_state import (
    JobMissingError,
    StaleStageError,
    advance_status,
    load_job,
    record_progress,
)

logger = logging.getLogger(__name__)

STAGE = "silver"
_BATCH = 2_000


@task("silver.standardize")
def standardize(job_id: str, tenant_id: str, batch_size: int = _BATCH) -> dict[str, Any]:
    with tenant_sync_session(tenant_id) as session:
        try:
            advance_status(session, job_id, JobStatus.PREPROCESSING)
            details = _standardize_job(session, job_id, tenant_id, batch_size)
            record_progress(
                session,
                job_id,
                STAGE,
                done=details["records"],
                total=details["records"],
                details=details,
            )
            return {"job_id": job_id, **details}
        except StaleStageError as stale:
            logger.info("standardize skipped for job %s: %s", job_id, stale)
            return {"job_id": job_id, "skipped": str(stale)}


@task("silver.backfill_job")
def backfill_job(job_id: str, tenant_id: str, batch_size: int = _BATCH) -> dict[str, Any]:
    """Standardise (or re-standardise) any job without touching its status.

    For jobs ingested before the Silver layer existed, or after an ontology
    edit: their records gain Silver rows and stable ids, which is what puts
    them into the gold-linking pool of every later upload. Idempotent: the
    upsert overwrites, so a re-run converges.
    """
    with tenant_sync_session(tenant_id) as session:
        try:
            load_job(session, job_id)
        except JobMissingError:
            return {"job_id": job_id, "skipped": "job deleted"}
        details = _standardize_job(session, job_id, tenant_id, batch_size)
        logger.info("Backfilled Silver for job %s: %s", job_id, details)
        return {"job_id": job_id, **details}


def _standardize_job(session: Any, job_id: str, tenant_id: str, batch_size: int) -> dict[str, Any]:
    engine = load_ontology(session, load_job(session, job_id).ontology, tenant_id)
    kinds: Counter[str] = Counter()
    with_keys = 0
    done = 0
    batch: list[dict[str, Any]] = []
    rows = session.execute(
        select(SchoolRecord.id, SchoolRecord.raw_row)
        .where(SchoolRecord.job_id == job_id)
        .execution_options(yield_per=batch_size)
    )
    for record_id, raw_row in rows:
        silver = engine.standardize(raw_row or {})
        identity = engine.identity(str(record_id), silver)
        kinds[identity.kind] += 1
        with_keys += bool(silver.natural_keys)
        batch.append(
            {
                "record_id": record_id,
                "tenant_id": tenant_id,
                "job_id": job_id,
                "ontology": engine.name,
                "ontology_version": engine.version,
                "name": silver.name,
                "name_distinctive": silver.name_distinctive,
                "address": silver.address,
                "locality": silver.locality,
                "district": silver.district,
                "state": silver.state,
                "pincode": silver.pincode,
                "attributes": silver.attributes,
                "natural_keys": silver.natural_keys,
                "stable_id": identity.stable_id,
                "identity_kind": identity.kind,
            }
        )
        if len(batch) >= batch_size:
            done += _upsert(session, batch)
            batch = []
    if batch:
        done += _upsert(session, batch)
    logger.info("Job %s: standardised %d records under %s", job_id, done, engine.name)
    return {
        "records": done,
        "ontology": engine.name,
        "ontology_version": engine.version,
        "records_with_natural_keys": with_keys,
        # How much of the upload has an identity that carries across uploads.
        "identity_kinds": dict(kinds),
    }


def _upsert(session: Any, rows: list[dict[str, Any]]) -> int:
    stmt = insert(SilverRecord).values(rows)
    updatable = {
        column: stmt.excluded[column]
        for column in rows[0]
        if column not in ("record_id", "tenant_id", "job_id")
    }
    session.execute(stmt.on_conflict_do_update(index_elements=["record_id"], set_=updatable))
    return len(rows)
