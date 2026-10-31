"""`embeddings` stage — two vectors per Valid record (FIG 2.4 / TRD 2.2).

**Only Valid rows are embedded.** That is the entire point of running the
classifier first (FR-02): embedding is the most expensive step per row, and
spending it on a row that reads "N/A" buys nothing. The partial index
`ix_school_records_tenant_id_job_id_classification` exists to make this
query cheap.

**Idempotent by construction.** `record_embeddings.record_id` is both primary
key and foreign key, so the write is an upsert on a known key (SAD 3.7:
"batches are idempotent, keyed by record_id"). Re-running the stage after a
redelivery overwrites the same rows rather than appending or failing.

**Text comes from the Silver layer.** The job's ontology standardised every
row in `silver.standardize`; this stage embeds `silver_records.name` and
`.address` and copies them to `school_records.*_normalized`, which blocking
and Splink read. So the ontology, not a hard-coded table, decides what
"KV" or "VPO" means to the matcher, and the vector always corresponds to a
string that is also in the database.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert

from app.db.session import tenant_sync_session
from app.ml_pipeline.embedding.encoder import DEFAULT_BATCH_SIZE, SemanticEncoder
from app.models.embedding import RecordEmbedding
from app.models.enums import JobStatus, RecordClassification
from app.models.medallion import SilverRecord
from app.models.school_record import SchoolRecord
from app.queue.registry import task
from app.workers.job_state import StaleStageError, advance_status, record_progress

logger = logging.getLogger(__name__)

STAGE = "embedding"


@task("embeddings.embeddings_stage")
def embeddings_stage(
    job_id: str,
    tenant_id: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Normalise and embed this job's Valid records, in batches of 256."""
    encoder = SemanticEncoder(batch_size=batch_size)

    with tenant_sync_session(tenant_id) as session:
        try:
            advance_status(session, job_id, JobStatus.EMBEDDING)

            # The resume query below joins three tables over a whole job. As
            # a parallel hash join it allocates dynamic shared memory, which
            # a container's default 64 MB /dev/shm cannot supply ("could not
            # resize shared memory segment ... No space left on device").
            # One serial read of a single job is fast enough, and fails the
            # same way nowhere.
            session.execute(text("SET LOCAL max_parallel_workers_per_gather = 0"))
            rows = session.execute(
                select(
                    SchoolRecord,
                    SilverRecord.record_id,
                    SilverRecord.name,
                    SilverRecord.address,
                    RecordEmbedding.embedding_model_version,
                )
                .outerjoin(SilverRecord, SilverRecord.record_id == SchoolRecord.id)
                .outerjoin(RecordEmbedding, RecordEmbedding.record_id == SchoolRecord.id)
                .where(
                    SchoolRecord.job_id == job_id,
                    SchoolRecord.classification == RecordClassification.VALID.value,
                )
            ).all()
            unstandardized = sum(1 for _, silver_id, _, _, _ in rows if silver_id is None)
            if unstandardized:
                # Silver is the only source of matching text; embedding raw
                # rows instead would silently bypass the ontology. Permanent:
                # a pipeline-order bug, not something a retry can fix.
                raise RuntimeError(
                    f"{unstandardized} Valid records have no Silver row; "
                    "silver.standardize must run before embeddings"
                )
            total = len(rows)
            if total == 0:
                logger.warning(
                    "Job %s has no Valid records to embed; the classifier rejected every row",
                    job_id,
                )
                record_progress(session, job_id, STAGE, done=0, total=0)
                return {"job_id": job_id, "embedded": 0}

            model_version = encoder.revision or encoder.model_name
            # Resume, don't redo: a record whose vector was written by this
            # model for exactly this Silver text is already finished. Text is
            # compared too, so an ontology edit (new Silver text) re-embeds
            # only the records it changed. After a crash or an expired lease
            # the retry therefore continues where the last attempt stopped
            # instead of repeating hours of CPU work.
            pending = [
                (record, name or "", address or "")
                for record, _, name, address, stored_version in rows
                if not (
                    stored_version == model_version
                    and (record.name_normalized or "") == (name or "")
                    and (record.address_normalized or "") == (address or "")
                )
            ]
            done = total - len(pending)
            if done:
                logger.info("Job %s: %d/%d already embedded; resuming", job_id, done, total)

            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                done += _embed_batch(session, batch, encoder, model_version)
                # Committed per batch, so a crash keeps the vectors already
                # computed and the next attempt skips them (above).
                session.commit()
                record_progress(session, job_id, STAGE, done=done, total=total)
                session.commit()
                logger.info("Job %s: embedded %d/%d", job_id, done, total)

            return {
                "job_id": job_id,
                "embedded": done,
                "resumed_from": total - len(pending),
                "model_version": model_version,
            }

        except StaleStageError as stale:
            # A late duplicate of a stage already passed. Not a failure: the
            # work exists, and marking the job FAILED here would let a stray
            # re-dispatch destroy a job sitting in AWAITING_REVIEW.
            logger.info("%s skipped for job %s: %s", "embeddings_stage", job_id, stale)
            return {"job_id": job_id, "skipped": str(stale)}


def _embed_batch(
    session: Any,
    batch: Sequence[tuple[SchoolRecord, str, str]],
    encoder: SemanticEncoder,
    model_version: str,
) -> int:
    """Encode one batch of Silver text and upsert it. Returns rows written."""
    names = [name for _, name, _ in batch]
    addresses = [address for _, _, address in batch]

    name_vectors = encoder.encode_batch(names)
    address_vectors = encoder.encode_batch(addresses)

    payload = []
    for (record, name, address), name_vector, address_vector in zip(
        batch, name_vectors, address_vectors, strict=True
    ):
        # Derived columns on the immutable source row (TRD 2.3) — `raw_row`
        # itself is never touched.
        record.name_normalized = name
        record.address_normalized = address
        payload.append(
            {
                "record_id": record.id,
                "tenant_id": record.tenant_id,
                "name_embedding": name_vector.tolist(),
                "address_embedding": address_vector.tolist(),
                "embedding_model_version": model_version,
            }
        )

    statement = insert(RecordEmbedding).values(payload)
    session.execute(
        statement.on_conflict_do_update(
            index_elements=[RecordEmbedding.record_id],
            set_={
                "name_embedding": statement.excluded.name_embedding,
                "address_embedding": statement.excluded.address_embedding,
                "embedding_model_version": statement.excluded.embedding_model_version,
            },
        )
    )
    return len(payload)
