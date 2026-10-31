"""`blocking` stage — candidate pairs within the same place (FR-03 / FIG 2.5).

Produces *pairs*, not clusters. Grouping records at this stage — which is
what the first version of this pipeline did — decides that records belong
together before anything has scored whether they match; on a real 21,734-row
upload that chained schools across five states into one component. Pairs go
to `candidate_pairs`, and the matching stage scores them and clusters only
the ones it believes.

See `app/ml_pipeline/blocking/location.py` for how pairs are proposed:
exact nearest neighbours within each district and each pincode, in memory.

**Idempotency is delete-and-recreate.** Candidate pairs are wholly derived
from embeddings plus each record's location, so a re-run rebuilds them.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np
from sqlalchemy import delete, insert, select

from app.db.session import tenant_sync_session
from app.ml_pipeline.blocking.location import (
    DEFAULT_MAX_DISTANCE,
    DEFAULT_TOP_K,
    BlockingRecord,
    find_location_candidates,
    location_keys,
    usable_pincode,
)
from app.ml_pipeline.fields import canonical_fields
from app.models.candidate_pair import JobCandidatePair
from app.models.embedding import RecordEmbedding
from app.models.enums import JobStatus
from app.models.reference_pincode import ReferencePincode
from app.models.school_record import SchoolRecord
from app.queue.registry import task
from app.workers.job_state import StaleStageError, advance_status, record_progress

logger = logging.getLogger(__name__)

STAGE = "blocking"

#: Rows per INSERT when persisting pairs.
_INSERT_BATCH = 5_000


@task("blocking.blocking_stage")
def blocking_stage(
    job_id: str,
    tenant_id: str,
    top_k: int = DEFAULT_TOP_K,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> dict[str, Any]:
    """Propose candidate pairs for a job and persist them."""
    with tenant_sync_session(tenant_id) as session:
        try:
            advance_status(session, job_id, JobStatus.BLOCKING)
            # Committed now, not with the work, so the dashboard shows this
            # stage while it runs. Safe on redelivery: re-entering the same
            # status is a no-op (`advance_status`).
            session.commit()

            records = _load_blocking_records(session, job_id)
            pairs, stats = find_location_candidates(records, top_k=top_k, max_distance=max_distance)

            session.execute(delete(JobCandidatePair).where(JobCandidatePair.job_id == job_id))
            rows = [
                {
                    "tenant_id": tenant_id,
                    "job_id": job_id,
                    "record_id_a": pair.record_id_a,
                    "record_id_b": pair.record_id_b,
                    "distance": pair.distance,
                    "source": pair.source,
                }
                for pair in pairs
            ]
            for start in range(0, len(rows), _INSERT_BATCH):
                session.execute(insert(JobCandidatePair), rows[start : start + _INSERT_BATCH])
            session.flush()

            record_progress(
                session,
                job_id,
                STAGE,
                done=len(records),
                total=len(records),
                details=stats.as_dict(),
            )
            logger.info(
                "Job %s: %d candidate pairs from %d records", job_id, len(pairs), len(records)
            )
            return {"job_id": job_id, **stats.as_dict()}

        except StaleStageError as stale:
            # A late duplicate of a stage already passed. Not a failure: the
            # work exists, and marking the job FAILED here would let a stray
            # re-dispatch destroy a job sitting in AWAITING_REVIEW.
            logger.info("%s skipped for job %s: %s", "blocking_stage", job_id, stale)
            return {"job_id": job_id, "skipped": str(stale)}


def _load_blocking_records(session: Any, job_id: str) -> list[BlockingRecord]:
    """Every embedded record of the job, with its location keys resolved."""
    rows = session.execute(
        select(
            RecordEmbedding.record_id,
            RecordEmbedding.name_embedding,
            RecordEmbedding.address_embedding,
            SchoolRecord.raw_row,
            SchoolRecord.name_normalized,
            SchoolRecord.address_normalized,
        )
        .join(SchoolRecord, SchoolRecord.id == RecordEmbedding.record_id)
        .where(SchoolRecord.job_id == job_id)
    ).all()

    fields = [canonical_fields(row.raw_row or {}) for row in rows]
    reference = _reference_locations(
        session, {pin for f in fields if (pin := usable_pincode(f["pincode"]))}
    )

    records: list[BlockingRecord] = []
    for row, row_fields in zip(rows, fields, strict=True):
        record_id, name_vec, address_vec, _raw, name_norm, address_norm = row
        district_keys, pincode, state = location_keys(row_fields, reference)
        records.append(
            BlockingRecord(
                record_id=uuid.UUID(str(record_id)),
                name_vector=np.asarray(name_vec, dtype=np.float32),
                address_vector=np.asarray(address_vec, dtype=np.float32),
                has_name=bool((name_norm or "").strip()),
                has_address=bool((address_norm or "").strip()),
                district_keys=district_keys,
                pincode=pincode,
                state=state,
            )
        )
    return records


def _reference_locations(session: Any, pincodes: set[str]) -> dict[str, tuple[str, str]]:
    """PIN -> (district, state) from `reference_pincodes`, in one query.

    A PIN can cover several post offices; they share a district in the
    government data, so the first row per PIN is taken.
    """
    if not pincodes:
        return {}
    rows = session.execute(
        select(
            ReferencePincode.pincode,
            ReferencePincode.district_name,
            ReferencePincode.state_name,
        ).where(ReferencePincode.pincode.in_(sorted(pincodes)))
    ).all()
    out: dict[str, tuple[str, str]] = {}
    for pincode, district, state in rows:
        out.setdefault(pincode, (district, state))
    return out
