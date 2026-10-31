"""`ingest` stage — read the CSV, persist rows, classify them (FR-01 / FR-02).

One task rather than two despite spanning two queues in TRD 2.6. Classification
reads only the row itself, needs no embedding and no other record, so splitting
it out would buy a second queue hop and a second pass over the same rows to
gain nothing. The `classify` queue stays declared for the retraining path,
which re-classifies existing rows without re-reading a CSV.

**`raw_row` is written once and never updated** (TRD 2.3). Everything else on
`school_records` is derived from it, so a re-run re-derives rather than
re-reads, and the tenant's original upload is always recoverable.

When the user mapped their columns on the upload screen (`jobs.column_mapping`),
the mapping is applied here, once, as each row is stored: mapped columns are
written under their canonical names (app/ml_pipeline/column_mapping.py). The
original file is untouched in storage, so file + mapping reproduce every row.

**Idempotency.** Re-running against a job that already has records does not
append a second copy — the stage detects them and re-classifies in place.
That is the case lease expiry makes routine: a worker that finished the
insert and died before completing the task gets the same task again.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from app.core.config import get_settings
from app.db.session import tenant_sync_session
from app.ml_pipeline.classification.classifier import DataQualityClassifier
from app.ml_pipeline.column_mapping import (
    ColumnMapping,
    alias_tables,
    apply_mapping,
    missing_headers,
)
from app.ml_pipeline.fields import canonical_fields
from app.ml_pipeline.quality import issue_counts, record_issues
from app.ml_pipeline.utils import validate_pincode
from app.models.enums import JobStatus, LabelSource
from app.models.school_record import SchoolRecord
from app.ontology.registry import load_ontology
from app.queue.registry import task
from app.storage import local_copy
from app.workers.job_state import StaleStageError, advance_status, load_job, record_progress

logger = logging.getLogger(__name__)

#: Rows per INSERT batch. Large enough that the round trip is amortised, small
#: enough that one bad batch does not roll back an hour of work.
DEFAULT_CHUNK_SIZE = 1_000

STAGE = "ingest"


RowTransform = Callable[[dict[str, Any]], dict[str, Any]]


class MappedColumnMissingError(ValueError):
    """A column the user mapped is not in the file. Permanent: re-upload."""


def read_header(csv_path: Path) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        return next(csv.reader(handle), [])


def iter_csv_chunks(csv_path: Path, chunk_size: int) -> Iterator[list[dict[str, Any]]]:
    """Stream the CSV in chunks, never holding the whole file in memory.

    A 50k-row school CSV is small, but the stage should not acquire a memory
    profile proportional to a tenant's upload — the failure arrives later, on
    someone else's bigger file, as an OOM with no obvious cause.
    """
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path} has no header row")

        chunk: list[dict[str, Any]] = []
        for row in reader:
            # `None` keys appear when a data line has more fields than the
            # header. Dropped rather than persisted under a null key, which
            # is not representable in JSONB.
            chunk.append({k: v for k, v in row.items() if k is not None})
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


@task("ingest.ingest_and_classify")
def ingest_and_classify(
    job_id: str,
    tenant_id: str,
    csv_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Load `csv_path` into `school_records` and classify every row.

    `csv_path` is passed in rather than read off the job row: object-storage
    retrieval is the upload route's concern, and a worker that resolved the
    location itself would need storage credentials it otherwise does not.
    """
    with tenant_sync_session(tenant_id) as session:
        try:
            advance_status(session, job_id, JobStatus.PREPROCESSING)

            existing = session.execute(
                select(func.count()).select_from(SchoolRecord).where(SchoolRecord.job_id == job_id)
            ).scalar_one()

            if existing:
                logger.info(
                    "Job %s already has %d records; re-classifying in place rather "
                    "than re-reading the CSV",
                    job_id,
                    existing,
                )
                total_rows = existing
            else:
                # s3:// sources (presigned direct uploads) are downloaded to
                # a temp file; local paths (multipart fallback) are read in place.
                transform = _row_transform(session, job_id, tenant_id)
                with local_copy(csv_path, max_bytes=get_settings().max_upload_bytes) as path:
                    total_rows = _insert_rows(
                        session, job_id, tenant_id, path, chunk_size, transform
                    )

            job = record_progress(session, job_id, STAGE, done=total_rows, total=total_rows)
            job.total_rows = total_rows

            classified, issues_by_class = _classify_job_records(session, job_id, chunk_size)
            record_progress(
                session,
                job_id,
                "classify",
                done=classified,
                total=total_rows,
                details={"issues_by_class": issues_by_class},
            )
            return {"job_id": job_id, "rows": total_rows, "classified": classified}

        except StaleStageError as stale:
            # A late duplicate of a stage already passed. Not a failure: the
            # work exists, and marking the job FAILED here would let a stray
            # re-dispatch destroy a job sitting in AWAITING_REVIEW.
            logger.info("%s skipped for job %s: %s", "ingest_and_classify", job_id, stale)
            return {"job_id": job_id, "skipped": str(stale)}


def _row_transform(
    session: Any, job_id: str, tenant_id: str
) -> tuple[ColumnMapping, RowTransform] | None:
    """The job's column mapping as a row function, or None when there is none."""
    job = load_job(session, job_id)
    if not job.column_mapping:
        return None
    mapping = ColumnMapping.model_validate(job.column_mapping)
    spec = load_ontology(session, job.ontology, tenant_id).spec
    field_aliases, key_headers = alias_tables(
        spec.fields, {key: rule.headers for key, rule in spec.natural_keys.items()}
    )

    def transform(row: dict[str, Any]) -> dict[str, Any]:
        return apply_mapping(
            row, mapping, field_aliases=field_aliases, natural_key_headers=key_headers
        )

    return mapping, transform


def _insert_rows(
    session: Any,
    job_id: str,
    tenant_id: str,
    csv_path: Path,
    chunk_size: int,
    mapped: tuple[ColumnMapping, RowTransform] | None = None,
) -> int:
    """Persist the CSV (re-keyed by the column mapping, if any). Returns the row count.

    `tenant_id` is written explicitly even though the session is already
    bound to that tenant. SAD 3.4 is explicit that RLS is "the backstop for
    the query that forgets to, not a replacement for writing it" — and the
    WITH CHECK half of the FIG 2.8 policy rejects an INSERT that omits the
    column outright, so leaving it to be inferred is not an option either.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    transform: RowTransform | None = None
    if mapped is not None:
        mapping, transform = mapped
        missing = missing_headers(mapping, read_header(csv_path))
        if missing:
            raise MappedColumnMissingError(
                f"Mapped column(s) {missing} are not in the file's header row. "
                "Re-upload and map the columns again."
            )

    total = 0
    for chunk in iter_csv_chunks(csv_path, chunk_size):
        rows = [transform(row) for row in chunk] if transform else chunk
        session.bulk_insert_mappings(
            SchoolRecord,
            [{"tenant_id": tenant_id, "job_id": job_id, "raw_row": row} for row in rows],
        )
        total += len(chunk)
        # Flushed per chunk so a large upload does not accumulate one enormous
        # transaction, and so `stage_progress` can be believed mid-run.
        session.flush()
        record_progress(session, job_id, STAGE, done=total)
        logger.info("Job %s: inserted %d rows", job_id, total)
    return total


def _classify_job_records(
    session: Any, job_id: str, chunk_size: int
) -> tuple[int, dict[str, dict[str, int]]]:
    """Classify every record of a job, writing classification + provenance.

    Cold start: with no trained classifier for this tenant, the model is
    fitted on pseudo-labels derived from this job's own rows, which is what
    `DataQualityClassifier` is built for. Those rows are stamped
    `LabelSource.PSEUDO` so a later retrain can discard them without
    contaminating steward-labelled ground truth (TRD 2.2, migration 0003).

    Also returns, per class, how many records carry each data-quality issue
    (`app/ml_pipeline/quality.py`) — the "why" behind the counts.
    """
    records = (
        session.execute(select(SchoolRecord).where(SchoolRecord.job_id == job_id)).scalars().all()
    )
    if not records:
        return 0, {}

    rows = [record.raw_row for record in records]

    classifier = DataQualityClassifier()
    # `db=session` lets the pincode feature consult `reference_pincodes`; the
    # blocker's own cache warming makes that one query for the batch.
    classifier.train_on_pseudo_labels(rows, db=session)
    predictions = classifier.predict_batch(rows, db=session)

    issues: dict[str, list[list[str]]] = {}
    for record, (_, prediction) in zip(records, predictions.iterrows(), strict=True):
        record.classification = prediction["classification"]
        record.label_source = LabelSource.PSEUDO.value

        fields = canonical_fields(record.raw_row or {})
        pincode = fields["pincode"].strip()
        # The classifier already warmed the pincode cache for this batch, so
        # this is a dict lookup, not a query.
        known = validate_pincode(pincode, session) if pincode else None
        issues.setdefault(record.classification, []).append(
            record_issues(fields, pincode_known=known)
        )

    session.flush()
    logger.info("Job %s: classified %d records", job_id, len(records))
    return len(records), {cls: issue_counts(found) for cls, found in issues.items()}
