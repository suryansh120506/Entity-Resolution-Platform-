"""Job routes — upload, poll, review queue, export (TRD 2.5 / FIG 2.9).

Every route here is `async def` and every route here is I/O only. SAD 3.2's
rule — "FastAPI never imports pandas, torch, or xgboost in a request path" —
is kept structurally rather than by review:

  * The upload route enqueues by **task name** (`app.queue.core.enqueue_async`)
    rather than importing the task function. Importing
    `app.workers.tasks.ingest` would pull xgboost and pandas into the API
    process, and the packaging split that makes that rule enforceable (the
    API image does not install the `[ml]` extra) would turn into an
    ImportError at start-up instead.
  * The CSV is streamed to disk in chunks and never parsed here. Parsing is
    the ingest worker's job; a route that read the file to count its rows
    would block the event loop for every other request on the process.
  * The export streams out of a server-side cursor rather than materialising
    50k rows (FR-09).

Tenant isolation is the `get_db` dependency, which binds `app.tenant_id` from
the verified token before the route body runs. No route filters on tenant
itself — RLS is the backstop and the bound session is the mechanism.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from sqlalchemy import delete as sql_delete
from sqlalchemy import func, or_, select, tuple_

from app.api.deps import DbDep, PrincipalDep, get_principal, require_role
from app.api.pagination import InvalidCursorError, decode_cursor, encode_cursor
from app.core.config import get_settings
from app.core.security import Role
from app.enrichment.golden import WEB_PINCODE_KEY
from app.enrichment.llm_config import llm_available, load_settings_row_async
from app.enrichment.pincode_infer import PincodeDirectory
from app.exports.client import build_layout, merge_natural_keys, resolve_pincode
from app.ml_pipeline.cleaning import clean_fields
from app.ml_pipeline.column_mapping import ColumnMapping
from app.ml_pipeline.fields import REQUIRED_FIELDS, canonical_fields
from app.ml_pipeline.quality import ISSUE_LABELS, record_issues
from app.models.candidate_pair import JobCandidatePair
from app.models.cluster import Cluster, ClusterMember
from app.models.embedding import RecordEmbedding
from app.models.enums import ClusterStatus, JobStatus, RecordClassification
from app.models.job import Job
from app.models.medallion import SilverRecord
from app.models.reference_pincode import ReferencePincode
from app.models.school_record import SchoolRecord
from app.models.tenant_settings import TenantOntology
from app.ontology import OntologyEngine, available_ontologies
from app.ontology.registry import engine_from_spec, load_ontology_async, ontology_exists
from app.queue.core import enqueue_async, stage_dedupe_key
from app.queue.topology import PIPELINE_ENTRYPOINT
from app.schemas.jobs import (
    ClusterPage,
    ClusterStatusCounts,
    ClusterSummary,
    JobAccepted,
    JobFunnel,
    JobPage,
    JobStats,
    JobStatusResponse,
    JobSummary,
    OntologyOption,
    PresignedUpload,
    PresignRequest,
    RecordItem,
    RecordPage,
    UploadOptions,
)
from app.storage import (
    S3_SCHEME,
    StorageNotConfiguredError,
    UploadMissingError,
    delete_object,
    object_size,
    presign_upload,
    read_head,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1/jobs",
    tags=["jobs"],
    # Applied at the router so a new route cannot be added unauthenticated by
    # forgetting a decorator. `get_db` re-resolves the same principal.
    dependencies=[Depends(get_principal)],
)

#: Upload read size. 1 MiB balances syscall count against the transient
#: buffer held per in-flight upload.
_UPLOAD_CHUNK = 1024 * 1024

#: Rows per fetch from the export cursor.
_EXPORT_FETCH = 1_000

#: The post office directory, built once per (ontology version, row count) and
#: shared by every export: 160k rows take a few seconds to index, and it only
#: changes when someone re-seeds the table.
_PINCODE_DIRECTORIES: dict[tuple[str, int], PincodeDirectory] = {}


async def _pincode_directory(db: Any, engine: Any) -> PincodeDirectory | None:
    """The directory for filling blank PINs, or None if `reference_pincodes`
    was never seeded (the export then leaves blank PINs blank)."""
    count = int((await db.execute(select(func.count()).select_from(ReferencePincode))).scalar_one())
    if not count:
        return None
    key = (engine.version, count)
    if key not in _PINCODE_DIRECTORIES:
        rows = (
            (
                await db.execute(
                    select(
                        ReferencePincode.pincode,
                        ReferencePincode.office_name,
                        ReferencePincode.district_name,
                        ReferencePincode.state_name,
                    )
                )
            )
            .tuples()
            .all()
        )
        _PINCODE_DIRECTORIES.clear()
        _PINCODE_DIRECTORIES[key] = await run_in_threadpool(
            PincodeDirectory,
            rows,
            state_aliases=engine.spec.state_aliases,
            district_aliases=engine.spec.district_aliases,
        )
    return _PINCODE_DIRECTORIES[key]


#: What a record with no cluster is called in the export (FR-09).
UNRESOLVED_SINGLETON = "UNRESOLVED_SINGLETON"


async def _get_job_or_404(db: Any, job_id: uuid.UUID) -> Job:
    """Load a job within the caller's tenant.

    Under RLS another tenant's job is simply invisible, so this returns 404
    rather than 403 — and that is the correct answer, not a convenient one:
    403 would confirm the id exists, which is exactly the enumeration FR-10
    forbids.
    """
    job = (await db.execute(select(Job).where(Job.id == job_id))).scalar_one_or_none()
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job


# --- Upload --------------------------------------------------------------------


@router.post(
    "",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a CSV and start a resolution job (FR-01)",
)
async def create_job(
    db: DbDep,
    principal: PrincipalDep,
    file: Annotated[UploadFile, File(description="CSV of records to resolve")],
    ontology: Annotated[str, Form(description="Ontology to standardise with")] = "education",
    column_mapping: Annotated[
        str | None,
        Form(description='JSON column mapping, e.g. {"fields": {"name": "schl_nm"}}'),
    ] = None,
) -> JobAccepted:
    """Spool the upload, create the job row, enqueue, return 202.

    202 rather than 201: the job row exists, but the work it represents has
    not started. FR-01 wants the id back immediately, and a 201 would imply
    the resource is complete.
    """
    settings = get_settings()

    engine = await _require_ontology(db, principal.tenant_id, ontology)
    mapping = _parse_mapping(column_mapping, engine)

    # tenant_id from the verified token, written explicitly: the bound
    # session governs what this INSERT is *allowed* to write (FIG 2.8's WITH
    # CHECK), it does not supply the value.
    job = Job(
        tenant_id=principal.tenant_id,
        status=JobStatus.PENDING.value,
        created_by=principal.subject,
        original_filename=_display_filename(file.filename),
        ontology=ontology,
        column_mapping=mapping,
    )
    db.add(job)
    await db.flush()

    destination = (settings.upload_dir / str(principal.tenant_id) / f"{job.id}.csv").resolve()
    try:
        total_bytes = await _spool_upload(file, destination, settings.max_upload_bytes)
    except ValueError as exc:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    job.source_uri = str(destination)

    await _enqueue_ingest(db, job, principal.tenant_id)
    await db.commit()

    logger.info(
        "Job %s accepted (%d bytes) for tenant %s", job.id, total_bytes, principal.tenant_id
    )

    return JobAccepted(job_id=job.id, status=job.status, total_bytes=total_bytes)


async def _enqueue_ingest(db: Any, job: Job, tenant_id: uuid.UUID) -> None:
    """Enqueue the pipeline in the caller's transaction (transactional outbox):
    the job and the work to run it commit together or not at all."""
    await enqueue_async(
        db,
        tenant_id=tenant_id,
        task_name=PIPELINE_ENTRYPOINT,
        payload={"job_id": str(job.id), "csv_path": job.source_uri},
        job_id=job.id,
        dedupe_key=stage_dedupe_key(PIPELINE_ENTRYPOINT, job.id),
    )


async def _require_ontology(db: Any, tenant_id: uuid.UUID, name: str) -> OntologyEngine:
    """The compiled ontology a new job will use: shipped, or this tenant's own."""
    if not await ontology_exists(db, name, tenant_id):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown data domain {name!r}; shipped: {available_ontologies()}",
        )
    return await load_ontology_async(db, name, tenant_id)


def _parse_mapping(
    raw: str | ColumnMapping | None, engine: OntologyEngine
) -> dict[str, Any] | None:
    """Validate a column mapping against the domain it will be applied with.

    Whether the mapped columns exist is checked at ingestion, against the
    file's real header row: the API never reads the file on a direct upload.
    """
    if raw is None or raw == "":
        return None
    try:
        mapping = raw if isinstance(raw, ColumnMapping) else ColumnMapping.model_validate_json(raw)
    except ValueError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid column mapping: {exc}"
        ) from exc
    unknown = sorted(set(mapping.natural_keys) - set(engine.spec.natural_keys))
    if unknown:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{engine.name!r} has no natural key(s) {unknown}",
        )
    return mapping.model_dump(exclude_none=True)


@router.get(
    "/upload-options",
    response_model=UploadOptions,
    summary="Ontologies and upload mode for the upload screen",
)
async def upload_options(db: DbDep, principal: PrincipalDep) -> UploadOptions:
    settings = get_settings()
    options = [
        OntologyOption.from_engine(OntologyEngine.load(name)) for name in available_ontologies()
    ]
    # Domain ontologies first, the catch-all last.
    options.sort(key=lambda o: (o.name == "generic", o.label))
    # Then this tenant's own, newest first.
    custom = (
        await db.execute(
            select(TenantOntology.spec)
            .where(TenantOntology.tenant_id == principal.tenant_id)
            .order_by(TenantOntology.created_at.desc())
        )
    ).scalars()
    options += [OntologyOption.from_engine(engine_from_spec(spec), custom=True) for spec in custom]
    ai_row = await load_settings_row_async(db, principal.tenant_id)
    return UploadOptions(
        ontologies=options,
        default_ontology="education",
        custom_domains_available=llm_available(ai_row, settings),
        direct_upload=settings.object_storage_enabled,
        max_upload_bytes=settings.max_upload_bytes,
    )


@router.post(
    "/uploads",
    response_model=PresignedUpload,
    status_code=status.HTTP_201_CREATED,
    summary="Create a job and a presigned URL to PUT its CSV directly to storage",
)
async def create_direct_upload(
    db: DbDep, principal: PrincipalDep, payload: PresignRequest
) -> PresignedUpload:
    """Step 1 of 2. The browser then PUTs the file to `upload_url` and calls
    `POST /v1/jobs/{job_id}/uploaded`. The bytes never pass through the API.

    The job exists from this point, PENDING with no task: until the upload
    is confirmed there is nothing to process. The object key is chosen here,
    under the tenant's prefix, so a client cannot write anywhere else.
    """
    settings = get_settings()
    if not settings.object_storage_enabled:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="Direct upload is not configured; use multipart POST /v1/jobs.",
        )
    if payload.size_bytes > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Upload exceeds the {settings.max_upload_bytes} byte limit",
        )
    if not payload.filename.lower().endswith(".csv"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Only .csv files")
    engine = await _require_ontology(db, principal.tenant_id, payload.ontology)
    mapping = _parse_mapping(payload.column_mapping, engine)

    job = Job(
        tenant_id=principal.tenant_id,
        status=JobStatus.PENDING.value,
        created_by=principal.subject,
        original_filename=_display_filename(payload.filename),
        ontology=payload.ontology,
        column_mapping=mapping,
        stage_progress={"upload": {"expected_bytes": payload.size_bytes, "mode": "direct"}},
    )
    db.add(job)
    await db.flush()

    presigned = await run_in_threadpool(
        presign_upload, principal.tenant_id, job.id, payload.size_bytes
    )
    job.source_uri = presigned.source_uri
    await db.commit()

    return PresignedUpload(
        job_id=job.id,
        upload_url=presigned.url,
        method=presigned.method,
        headers=presigned.headers,
        expires_in=presigned.expires_in,
    )


@router.post(
    "/{job_id}/uploaded",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Confirm a direct upload and start the pipeline",
)
async def confirm_direct_upload(
    db: DbDep, principal: PrincipalDep, job_id: uuid.UUID
) -> JobAccepted:
    """Step 2 of 2: verify the object, then enqueue — in one transaction.

    The stored size must equal the size that was signed. S3/R2 already
    refuse a PUT whose Content-Length differs from the signature, so this is
    the second, independent check: it does not trust that the provider (or
    an emulator, or a proxy in front of it) enforced the first.
    """
    settings = get_settings()
    job = await _get_job_or_404(db, job_id)
    source = job.source_uri or ""
    if JobStatus(job.status) is not JobStatus.PENDING or not source.startswith(S3_SCHEME):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"Job is {job.status}; only a pending direct upload can be confirmed.",
        )

    try:
        size = await run_in_threadpool(object_size, source)
    except UploadMissingError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT, detail="The file has not been uploaded yet."
        ) from exc

    expected = (job.stage_progress or {}).get("upload", {}).get("expected_bytes")
    if size != expected or size > settings.max_upload_bytes:
        await run_in_threadpool(delete_object, source)
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Uploaded {size} bytes but {expected} were declared; upload discarded.",
        )

    job.stage_progress = {
        **(job.stage_progress or {}),
        "upload": {"expected_bytes": expected, "mode": "direct", "confirmed_bytes": size},
    }
    await _enqueue_ingest(db, job, principal.tenant_id)
    await db.commit()
    logger.info("Job %s direct upload confirmed (%d bytes)", job.id, size)
    return JobAccepted(job_id=job.id, status=job.status, total_bytes=size)


def _display_filename(name: str | None) -> str | None:
    """The upload's own name, safe to show — never used as a path.

    Browsers send a bare name, but some clients send a full path, and
    nothing stops a name carrying control characters. Basename only,
    printable characters only, bounded length.
    """
    if not name:
        return None
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = "".join(ch for ch in base if ch.isprintable()).strip()
    return cleaned[:255] or None


async def _spool_upload(file: UploadFile, destination: Any, max_bytes: int) -> int:
    """Stream an upload to disk without holding it in memory.

    The size limit is enforced *while* streaming, not after. A limit checked
    once the file has landed is a limit that already paid for the disk, and
    on a shared volume that is how one tenant fills it for everyone.
    """
    await run_in_threadpool(destination.parent.mkdir, parents=True, exist_ok=True)

    total = 0
    handle = await run_in_threadpool(destination.open, "wb")
    try:
        while chunk := await file.read(_UPLOAD_CHUNK):
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"Upload exceeds the {max_bytes} byte limit")
            # Disk writes are blocking; off the loop they go.
            await run_in_threadpool(handle.write, chunk)
    except ValueError:
        await run_in_threadpool(handle.close)
        await run_in_threadpool(destination.unlink, True)
        raise
    else:
        await run_in_threadpool(handle.close)
    return total


@router.get("", response_model=JobPage, summary="This tenant's jobs, newest first")
async def list_jobs(
    db: DbDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: Annotated[str | None, Query(description="Opaque; from next_cursor")] = None,
) -> JobPage:
    """One page of this tenant's jobs.

    Exists so the dashboard survives a navigation or a page reload. Without
    it, "what's uploaded" lived only in the browser tab's React state —
    accurate for exactly as long as that tab stayed on `/dashboard`, and gone
    the moment a steward followed the review-queue link and came back. Jobs
    were never lost; there was simply no way to ask the API for them again.
    `ix_jobs_tenant_id_created_at` exists for precisely this query.
    """
    query = (
        select(Job)
        .order_by(Job.created_at.desc(), Job.id.desc())
        .limit(limit + 1)  # one extra: its existence is the "more pages" signal
    )

    if cursor:
        try:
            position = decode_cursor(cursor)
        except InvalidCursorError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        # Descending page order, so the next page is strictly *older* than
        # the cursor — the mirror image of the review queue's `>` (FIG 2.9's
        # queue pages oldest-first; this dashboard pages newest-first).
        query = query.where(tuple_(Job.created_at, Job.id) < (position.created_at, position.id))

    rows = list((await db.execute(query)).scalars().all())
    has_more = len(rows) > limit
    page = rows[:limit]

    next_cursor = encode_cursor(page[-1].created_at, page[-1].id) if has_more and page else None
    return JobPage(
        items=[JobStatusResponse.model_validate(job) for job in page], next_cursor=next_cursor
    )


#: Statuses in which a worker may be writing this job's rows right now.
#: Deleting underneath one would fail its inserts on foreign keys; PENDING is
#: deliberately absent, so a job whose worker never started can be removed.
_ACTIVE_STATUSES = frozenset(
    {
        JobStatus.PREPROCESSING.value,
        JobStatus.EMBEDDING.value,
        JobStatus.BLOCKING.value,
        JobStatus.MATCHING.value,
        JobStatus.RESOLVING.value,
    }
)


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a job and everything derived from it",
    dependencies=[Depends(require_role(Role.DATA_ENGINEER, Role.DATA_STEWARD, Role.ORG_ADMIN))],
)
async def delete_job(db: DbDep, principal: PrincipalDep, job_id: uuid.UUID) -> None:
    """Remove a job, its records, clusters, pairs, feedback and uploaded file.

    One DELETE on `jobs`: every child table cascades from it. A statement
    rather than `session.delete(job)` so the ORM does not load 50k children
    just to delete them. Any of the job's tasks still queued will find no
    row and skip (`JobMissingError`), rather than fail.
    """
    job = await _get_job_or_404(db, job_id)
    if job.status in _ACTIVE_STATUSES:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"Job is still processing ({job.status.lower()}); delete it once it finishes.",
        )

    source = job.source_uri
    await db.execute(sql_delete(Job).where(Job.id == job_id))
    await db.commit()

    # Best effort, after the commit: a leftover file is a storage cost, while
    # a deleted file under a job that failed to delete is a broken job.
    try:
        if source and source.startswith(S3_SCHEME):
            await run_in_threadpool(delete_object, source)
        else:
            upload = get_settings().upload_dir / str(principal.tenant_id) / f"{job_id}.csv"
            await run_in_threadpool(upload.unlink, True)
    except Exception:
        logger.warning("Could not remove upload %s for deleted job %s", source, job_id)
    logger.info("Job %s deleted by %s", job_id, principal.subject)


@router.get("/{job_id}/summary", response_model=JobSummary, summary="Where every row ended up")
async def get_job_summary(db: DbDep, job_id: uuid.UUID) -> JobSummary:
    """The results page in one request: funnel, bands, and the why.

    Counted in SQL, like `/stats`. The per-issue breakdown is read from what
    the classify stage recorded rather than recomputed per request — it
    needs every row's fields, and the stage already had them in hand.
    """
    job = await _get_job_or_404(db, job_id)

    class_rows = await db.execute(
        select(SchoolRecord.classification, func.count())
        .where(SchoolRecord.job_id == job_id)
        .group_by(SchoolRecord.classification)
    )
    by_class: dict[str | None, int] = dict(class_rows.tuples().all())
    embedded = (
        await db.execute(
            select(func.count())
            .select_from(RecordEmbedding)
            .join(SchoolRecord, SchoolRecord.id == RecordEmbedding.record_id)
            .where(SchoolRecord.job_id == job_id)
        )
    ).scalar_one()
    pairs = (
        await db.execute(
            select(func.count())
            .select_from(JobCandidatePair)
            .where(JobCandidatePair.job_id == job_id)
        )
    ).scalar_one()

    cluster_counts = ClusterStatusCounts()
    for cluster_status, count in (
        await db.execute(
            select(Cluster.status, func.count())
            .where(Cluster.job_id == job_id)
            .group_by(Cluster.status)
        )
    ).all():
        if hasattr(cluster_counts, cluster_status):
            setattr(cluster_counts, cluster_status, count)

    in_live_clusters = (
        await db.execute(
            select(func.count(func.distinct(ClusterMember.record_id)))
            .select_from(ClusterMember)
            .join(Cluster, Cluster.id == ClusterMember.cluster_id)
            .where(Cluster.job_id == job_id, Cluster.status != ClusterStatus.REJECTED.value)
        )
    ).scalar_one()
    merged_away = (
        await db.execute(
            select(func.count())
            .select_from(ClusterMember)
            .join(Cluster, Cluster.id == ClusterMember.cluster_id)
            .where(
                Cluster.job_id == job_id,
                Cluster.status.in_([ClusterStatus.AUTO_MERGED.value, ClusterStatus.RESOLVED.value]),
                ClusterMember.record_id != Cluster.master_record_id,
            )
        )
    ).scalar_one()

    valid = by_class.get(RecordClassification.VALID.value, 0)
    progress = job.stage_progress or {}

    def details(stage: str) -> dict[str, Any] | None:
        entry = progress.get(stage)
        return entry.get("details") if isinstance(entry, dict) else None

    return JobSummary(
        job=JobStatusResponse.model_validate(job),
        funnel=JobFunnel(
            total_rows=job.total_rows or sum(by_class.values()),
            valid=valid,
            missing_context=by_class.get(RecordClassification.MISSING_CONTEXT.value, 0),
            junk=by_class.get(RecordClassification.JUNK.value, 0),
            unclassified=by_class.get(None, 0),
            embedded=embedded,
            candidate_pairs=pairs,
            records_in_clusters=in_live_clusters,
            unique_records=max(0, valid - in_live_clusters),
            clusters=sum(n for n in cluster_counts.model_dump().values()),
        ),
        clusters_by_status=cluster_counts,
        records_merged_away=merged_away,
        issues_by_class=(details("classify") or {}).get("issues_by_class", {}),
        issue_labels=ISSUE_LABELS,
        blocking=details("blocking"),
        matching=details("matching"),
    )


#: `view` -> the records it selects. "unique" is a Valid record in no cluster
#: that is still live — a REJECTED cluster's members count as unique, since
#: rejecting says "these are different schools".
_RECORD_VIEWS = ("needs_info", "junk", "unique", "all")


@router.get("/{job_id}/records", response_model=RecordPage, summary="Browse a job's rows")
async def list_records(
    db: DbDep,
    job_id: uuid.UUID,
    view: Annotated[str, Query(pattern="^(needs_info|junk|unique|all)$")] = "all",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: Annotated[str | None, Query(description="Opaque; from next_cursor")] = None,
) -> RecordPage:
    """One keyset page of rows, cleaned, with the reasons each was set aside."""
    await _get_job_or_404(db, job_id)

    live_member = (
        select(ClusterMember.record_id)
        .join(Cluster, Cluster.id == ClusterMember.cluster_id)
        .where(
            ClusterMember.record_id == SchoolRecord.id,
            Cluster.status != ClusterStatus.REJECTED.value,
        )
        .exists()
    )

    query = (
        select(
            SchoolRecord.id,
            SchoolRecord.classification,
            SchoolRecord.raw_row,
            SchoolRecord.created_at,
            Cluster.id,
            Cluster.status,
        )
        .outerjoin(ClusterMember, ClusterMember.record_id == SchoolRecord.id)
        .outerjoin(Cluster, Cluster.id == ClusterMember.cluster_id)
        .where(SchoolRecord.job_id == job_id)
        .order_by(SchoolRecord.created_at, SchoolRecord.id)
        .limit(limit + 1)
    )
    if view == "needs_info":
        query = query.where(
            SchoolRecord.classification == RecordClassification.MISSING_CONTEXT.value
        )
    elif view == "junk":
        query = query.where(SchoolRecord.classification == RecordClassification.JUNK.value)
    elif view == "unique":
        query = query.where(
            SchoolRecord.classification == RecordClassification.VALID.value, ~live_member
        )

    if cursor:
        try:
            position = decode_cursor(cursor)
        except InvalidCursorError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        query = query.where(
            tuple_(SchoolRecord.created_at, SchoolRecord.id) > (position.created_at, position.id)
        )

    rows = (await db.execute(query)).all()
    has_more = len(rows) > limit
    page = rows[:limit]

    fields = [canonical_fields(raw_row or {}) for _, _, raw_row, *_ in page]
    known = await _known_pincodes(db, {f["pincode"].strip() for f in fields})

    items: list[RecordItem] = []
    for row, row_fields in zip(page, fields, strict=True):
        record_id, classification, _raw, _created, cluster_id, cluster_status = row
        pin = row_fields["pincode"].strip()
        items.append(
            RecordItem(
                record_id=record_id,
                classification=classification,
                clean_fields=clean_fields(row_fields),
                issues=record_issues(row_fields, pincode_known=(pin in known) if pin else None),
                cluster_id=cluster_id,
                cluster_status=cluster_status,
            )
        )

    last = page[-1] if page else None
    next_cursor = encode_cursor(last[3], last[0]) if has_more and last is not None else None
    return RecordPage(items=items, next_cursor=next_cursor)


async def _known_pincodes(db: Any, pincodes: set[str]) -> set[str]:
    """Which of these PINs exist in `reference_pincodes` — one query."""
    candidates = sorted(p for p in pincodes if p)
    if not candidates:
        return set()
    rows = await db.execute(
        select(ReferencePincode.pincode).where(ReferencePincode.pincode.in_(candidates)).distinct()
    )
    return set(rows.scalars().all())


# --- Polling -------------------------------------------------------------------


@router.get("/{job_id}", response_model=JobStatusResponse, summary="Job status")
async def get_job(db: DbDep, job_id: uuid.UUID) -> Job:
    return await _get_job_or_404(db, job_id)


@router.get("/{job_id}/stats", response_model=JobStats, summary="Job statistics")
async def get_job_stats(db: DbDep, job_id: uuid.UUID) -> JobStats:
    """Aggregates for one job, counted in the database.

    Three grouped counts rather than three table loads. This is a polled
    endpoint, and the difference at 50k records is a millisecond of SQL
    against a megabyte over the wire, repeated every few seconds.
    """
    job = await _get_job_or_404(db, job_id)

    classification_rows = (
        await db.execute(
            select(SchoolRecord.classification, func.count())
            .where(SchoolRecord.job_id == job_id)
            .group_by(SchoolRecord.classification)
        )
    ).all()

    cluster_rows = (
        await db.execute(
            select(Cluster.status, func.count(), func.avg(Cluster.confidence_score))
            .where(Cluster.job_id == job_id)
            .group_by(Cluster.status)
        )
    ).all()

    clustered = (
        await db.execute(
            select(func.count(func.distinct(ClusterMember.record_id)))
            .select_from(ClusterMember)
            .join(Cluster, Cluster.id == ClusterMember.cluster_id)
            .where(Cluster.job_id == job_id)
        )
    ).scalar_one()

    counts = ClusterStatusCounts()
    weighted_total = 0.0
    scored = 0
    for cluster_status, count, mean in cluster_rows:
        if hasattr(counts, cluster_status):
            setattr(counts, cluster_status, count)
        if mean is not None:
            weighted_total += float(mean) * count
            scored += count

    return JobStats(
        job_id=job.id,
        status=job.status,
        total_rows=job.total_rows,
        records_by_classification={
            (c if c is not None else "UNCLASSIFIED"): n for c, n in classification_rows
        },
        clusters_by_status=counts,
        clusters_total=sum(n for _, n, _ in cluster_rows),
        records_clustered=clustered,
        mean_confidence=round(weighted_total / scored, 4) if scored else None,
    )


# --- Review queue --------------------------------------------------------------


@router.get(
    "/{job_id}/clusters",
    response_model=ClusterPage,
    summary="Review queue, keyset-paginated (FIG 2.9)",
)
async def list_clusters(
    db: DbDep,
    job_id: uuid.UUID,
    cluster_status: Annotated[
        ClusterStatus, Query(alias="status", description="Band to page through")
    ] = ClusterStatus.PENDING_REVIEW,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: Annotated[str | None, Query(description="Opaque; from next_cursor")] = None,
) -> ClusterPage:
    """One page of the queue, ordered oldest-first by `(created_at, id)`.

    See `app/api/pagination.py` for why this is keyset rather than OFFSET —
    the short version is that stewards resolve clusters while paging, and
    OFFSET silently skips the rows that shift underneath them.
    """
    await _get_job_or_404(db, job_id)

    query = (
        select(Cluster)
        .where(Cluster.job_id == job_id, Cluster.status == cluster_status.value)
        .order_by(Cluster.created_at, Cluster.id)
        .limit(limit + 1)  # one extra: its existence is the "more pages" signal
    )

    if cursor:
        try:
            position = decode_cursor(cursor)
        except InvalidCursorError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        # `tuple_()`, not a bare Python tuple. `(a, b) > (c, d)` on ORM
        # columns is a *Python* tuple comparison: it evaluates `a == c` for
        # truthiness, which SQLAlchemy refuses — so page 1 would work and
        # every page after it would 500. This emits a real SQL row-value
        # comparison, which is also what lets Postgres serve the page as one
        # range scan of ix_clusters_review_queue.
        query = query.where(
            tuple_(Cluster.created_at, Cluster.id) > (position.created_at, position.id)
        )

    rows = list((await db.execute(query)).scalars().all())
    has_more = len(rows) > limit
    page = rows[:limit]

    member_counts = await _member_counts(db, [c.id for c in page])
    items = [_summarise(cluster, member_counts.get(cluster.id, 0)) for cluster in page]

    next_cursor = encode_cursor(page[-1].created_at, page[-1].id) if has_more and page else None
    return ClusterPage(items=items, next_cursor=next_cursor)


def _summarise(cluster: Cluster, member_count: int) -> ClusterSummary:
    """Build the wire model explicitly.

    Not `model_validate(cluster, update=...)` — `update` is a `model_copy`
    keyword and `model_validate` rejects it at runtime. Constructing the
    fields by hand also means adding a column to `clusters` cannot silently
    widen the API's response.
    """
    return ClusterSummary(
        id=cluster.id,
        job_id=cluster.job_id,
        status=cluster.status,
        confidence_score=cluster.confidence_score,
        matcher_version=cluster.matcher_version,
        master_record_id=cluster.master_record_id,
        created_at=cluster.created_at,
        member_count=member_count,
    )


async def _member_counts(db: Any, cluster_ids: list[uuid.UUID]) -> dict[uuid.UUID, int]:
    """Member counts for a page, in one query rather than N."""
    if not cluster_ids:
        return {}
    rows = (
        await db.execute(
            select(ClusterMember.cluster_id, func.count())
            .where(ClusterMember.cluster_id.in_(cluster_ids))
            .group_by(ClusterMember.cluster_id)
        )
    ).all()
    return dict(rows)


# --- Export --------------------------------------------------------------------


@router.get("/{job_id}/export", summary="Dirty ID -> Master ID mapping (FR-09)")
async def export_job(db: DbDep, job_id: uuid.UUID) -> StreamingResponse:
    """Stream the mapping CSV straight out of Postgres.

    `StreamingResponse` over a server-side cursor, so memory is one fetch
    window rather than the whole job. Building this with `.all()` and
    `"\\n".join(...)` works fine against a test fixture and falls over on the
    50k-row export it exists for.

    Every record appears, including those in no cluster — a row that resolved
    to nothing is a result, and an export that silently omitted it would not
    reconcile against the tenant's input file.
    """
    job = await _get_job_or_404(db, job_id)

    async def rows() -> AsyncIterator[str]:
        buffer = io.StringIO()
        writer = csv.writer(buffer)

        def take() -> str:
            value = buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)
            return value

        writer.writerow(
            ["dirty_id", "master_id", "cluster_id", "cluster_status", "confidence_score"]
        )
        yield take()

        query = (
            select(
                SchoolRecord.id,
                Cluster.master_record_id,
                Cluster.id,
                Cluster.status,
                Cluster.confidence_score,
            )
            .select_from(SchoolRecord)
            .outerjoin(ClusterMember, ClusterMember.record_id == SchoolRecord.id)
            .outerjoin(Cluster, Cluster.id == ClusterMember.cluster_id)
            .where(SchoolRecord.job_id == job_id)
            .order_by(SchoolRecord.id)
            .execution_options(yield_per=_EXPORT_FETCH)
        )

        result = await db.stream(query)
        async for record_id, master_id, cluster_id, cluster_status, score in result:
            writer.writerow(
                [
                    str(record_id),
                    str(master_id) if master_id else UNRESOLVED_SINGLETON,
                    str(cluster_id) if cluster_id else "",
                    cluster_status or UNRESOLVED_SINGLETON,
                    f"{score:.3f}" if score is not None else "",
                ]
            )
            yield take()

    return StreamingResponse(
        rows(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="job_{job.id}_mapping.csv"'},
    )


#: Columns of the cleaned export, in order.
CLEANED_COLUMNS = (
    "source_id",
    "record_id",
    "outcome",
    "classification",
    "issues",
    *(f"clean_{f}" for f in REQUIRED_FIELDS),
    "cluster_id",
    "cluster_status",
    "confidence_score",
    "master_record_id",
    *(f"canonical_{f}" for f in REQUIRED_FIELDS),
)

#: What happened to a row, in words a data owner can filter on.
_OUTCOME_BY_CLUSTER = {
    ClusterStatus.PENDING_REVIEW.value: "awaiting_review",
    ClusterStatus.PENDING_API.value: "awaiting_triage",
    ClusterStatus.REJECTED.value: "unique",
}


@router.get("/{job_id}/export/cleaned", summary="Every row, cleaned, with its outcome")
async def export_cleaned(db: DbDep, job_id: uuid.UUID) -> StreamingResponse:
    """The cleaned dataset: one row per uploaded row, nothing dropped.

    Each row carries its own cleaned fields *and* its canonical fields — the
    surviving record's values (a steward's corrections included) when it was
    merged into a cluster, its own otherwise. Keeping only rows whose
    `outcome` is `unique` or `kept_master` gives the deduplicated list.
    """
    job = await _get_job_or_404(db, job_id)

    # Merged clusters' survivors, loaded up front: there are at most a few
    # thousand, and the stream below needs them for every member row.
    # Tier 3's targets for this job's domain become `enriched_<key>` columns,
    # filled from the golden record of the row's cluster. An ontology that
    # no longer loads just exports without them.
    try:
        targets = [
            t.key
            for t in (
                await load_ontology_async(db, job.ontology, job.tenant_id)
            ).spec.enrichment_targets
        ]
    except ValueError:
        targets = []

    merged = (
        await db.execute(
            select(Cluster.id, Cluster.master_record_id, Cluster.golden_fields).where(
                Cluster.job_id == job_id,
                Cluster.status.in_([ClusterStatus.AUTO_MERGED.value, ClusterStatus.RESOLVED.value]),
            )
        )
    ).all()
    master_ids = {master for _, master, _ in merged if master is not None}
    master_rows: dict[Any, dict[str, Any]] = {}
    if master_ids:
        found = await db.execute(
            select(SchoolRecord.id, SchoolRecord.raw_row).where(SchoolRecord.id.in_(master_ids))
        )
        master_rows = dict(found.tuples().all())
    canonical_by_cluster: dict[Any, dict[str, str]] = {}
    for cluster_id, master, golden in merged:
        base = clean_fields(canonical_fields(master_rows.get(master) or {}))
        canonical_by_cluster[cluster_id] = {**base, **(golden or {})}

    known = set((await db.execute(select(ReferencePincode.pincode).distinct())).scalars().all())

    async def rows() -> AsyncIterator[str]:
        buffer = io.StringIO()
        writer = csv.writer(buffer)

        def take() -> str:
            value = buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)
            return value

        writer.writerow([*CLEANED_COLUMNS, "decided_by", *(f"enriched_{k}" for k in targets)])
        yield take()

        query = (
            select(
                SchoolRecord.id,
                SchoolRecord.classification,
                SchoolRecord.raw_row,
                Cluster.id,
                Cluster.status,
                Cluster.confidence_score,
                Cluster.master_record_id,
                Cluster.resolution_source,
            )
            .select_from(SchoolRecord)
            .outerjoin(ClusterMember, ClusterMember.record_id == SchoolRecord.id)
            .outerjoin(Cluster, Cluster.id == ClusterMember.cluster_id)
            .where(SchoolRecord.job_id == job_id)
            .order_by(SchoolRecord.created_at, SchoolRecord.id)
            .execution_options(yield_per=_EXPORT_FETCH)
        )

        result = await db.stream(query)
        async for (
            record_id,
            classification,
            raw_row,
            cluster_id,
            c_status,
            score,
            master,
            source,
        ) in result:
            raw = raw_row or {}
            fields = canonical_fields(raw)
            own = clean_fields(fields)
            pin = fields["pincode"].strip()
            issues = record_issues(fields, pincode_known=(pin in known) if pin else None)

            canonical = canonical_by_cluster.get(cluster_id, own)
            outcome = _outcome(classification, record_id, c_status, master)

            writer.writerow(
                [
                    _source_id(raw),
                    str(record_id),
                    outcome,
                    classification or "",
                    ";".join(issues),
                    *(own[f] for f in REQUIRED_FIELDS),
                    str(cluster_id) if cluster_id else "",
                    c_status or "",
                    f"{score:.3f}" if score is not None else "",
                    str(master) if master else "",
                    *(canonical.get(f, "") for f in REQUIRED_FIELDS),
                    _decided_by(c_status, source),
                    *(canonical.get(k, "") for k in targets),
                ]
            )
            yield take()

    stem = (job.original_filename or f"job_{job.id}").rsplit(".", 1)[0]
    return StreamingResponse(
        rows(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{_safe_stem(stem)}_cleaned.csv"'},
    )


GOLD_COLUMNS = (
    "entity_id",
    "stable_id",
    "record_id",
    "source_id",
    "kind",
    "decided_by",
    "confidence_score",
    "cluster_status",
    "member_count",
    # `name` is the cleaned original; `name_standardized` is the ontology's
    # fully expanded form ("SR SEC" -> "SENIOR SECONDARY"), upper-cased like the
    # source files it is compared with.
    *(c for f in REQUIRED_FIELDS for c in ((f, "name_standardized") if f == "name" else (f,))),
)
_MERGED = (ClusterStatus.AUTO_MERGED.value, ClusterStatus.RESOLVED.value)
_UNDECIDED = (ClusterStatus.PENDING_REVIEW.value, ClusterStatus.PENDING_API.value)


@router.get("/{job_id}/export/gold", summary="The final deduplicated, enriched database")
async def export_gold(
    db: DbDep,
    job_id: uuid.UUID,
    allow_partial: Annotated[
        bool, Query(description="Download before the job is COMPLETED; undecided rows are flagged.")
    ] = False,
) -> StreamingResponse:
    """The Gold layer: one row per real-world entity, nothing duplicated.

    * `kind=merged`     the master of a merged cluster, with its golden fields
                        and the values Tier 3 found on the web (`enriched_*`).
    * `kind=unique`     a record that matched nothing, or that triage kept
                        separate. Not enriched: Tier 3 covers merged records.
    * `kind=unresolved` a member of a cluster still awaiting a decision. Only
                        present in a partial download; each member is its own row.

    `decided_by` says who settled the row's cluster: `threshold` (Tier 1),
    `llm_triage` (Tier 2) or `steward`. `enrichment_sources` lists where each
    web value was found. Rows that needed more information or were junk are
    not Gold; they are in the cleaned export.
    """
    job = await _get_job_or_404(db, job_id)
    if JobStatus(job.status) is not JobStatus.COMPLETED and not allow_partial:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=(
                f"Job is {job.status}, not COMPLETED. Wait for it to finish, or download "
                "a partial export (allow_partial=true) in which undecided rows are flagged."
            ),
        )
    try:
        engine: OntologyEngine | None = await load_ontology_async(db, job.ontology, job.tenant_id)
    except ValueError:
        engine = None
    targets = [t.key for t in engine.spec.enrichment_targets] if engine else []

    clusters = (
        await db.execute(
            select(
                Cluster.id,
                Cluster.entity_id,
                Cluster.master_record_id,
                Cluster.golden_fields,
                Cluster.enrichment,
                Cluster.resolution_source,
                Cluster.status,
                Cluster.confidence_score,
            ).where(
                Cluster.job_id == job_id,
                Cluster.status.in_(_MERGED),
                Cluster.master_record_id.is_not(None),
            )
        )
    ).all()
    master_ids = {row.master_record_id for row in clusters}
    master_rows: dict[Any, dict[str, Any]] = {}
    stable_ids: dict[Any, str] = {}
    sizes: dict[Any, int] = {}
    if clusters:
        master_rows = dict(
            (
                await db.execute(
                    select(SchoolRecord.id, SchoolRecord.raw_row).where(
                        SchoolRecord.id.in_(master_ids)
                    )
                )
            )
            .tuples()
            .all()
        )
        stable_ids = {
            record_id: stable or ""
            for record_id, stable in (
                await db.execute(
                    select(SilverRecord.record_id, SilverRecord.stable_id).where(
                        SilverRecord.record_id.in_(master_ids)
                    )
                )
            ).all()
        }
        sizes = dict(
            (
                await db.execute(
                    select(ClusterMember.cluster_id, func.count())
                    .join(Cluster, Cluster.id == ClusterMember.cluster_id)
                    .where(Cluster.job_id == job_id, Cluster.status.in_(_MERGED))
                    .group_by(ClusterMember.cluster_id)
                )
            )
            .tuples()
            .all()
        )

    header = [
        *GOLD_COLUMNS,
        *(f"enriched_{k}" for k in targets),
        "enrichment_status",
        "enrichment_sources",
    ]

    async def rows() -> AsyncIterator[str]:
        buffer = io.StringIO()
        writer = csv.writer(buffer)

        def take() -> str:
            value = buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)
            return value

        writer.writerow(header)
        yield take()

        for row in clusters:
            raw = master_rows.get(row.master_record_id) or {}
            golden = {**clean_fields(canonical_fields(raw)), **(row.golden_fields or {})}
            enrichment = row.enrichment or {}
            sources = "; ".join(
                f"{key}={found.get('url', '')}"
                for key, found in (enrichment.get("fields") or {}).items()
            )
            writer.writerow(
                [
                    str(row.entity_id),
                    stable_ids.get(row.master_record_id, ""),
                    str(row.master_record_id),
                    _source_id(raw),
                    "merged",
                    _decided_by(row.status, row.resolution_source),
                    f"{row.confidence_score:.3f}" if row.confidence_score is not None else "",
                    row.status,
                    sizes.get(row.id, 1),
                    # `golden` already carries a steward's corrected name, so a
                    # correction is standardised, not the master's old spelling.
                    *_gold_cells(golden, _standardized_name(engine, golden.get("name", ""))),
                    *(golden.get(k, "") for k in targets),
                    enrichment.get("status", ""),
                    sources,
                ]
            )
            yield take()

        # Everything that is its own entity: unmatched records, records triage
        # kept separate (REJECTED clusters), and, in a partial download, the
        # members of clusters nobody has decided yet.
        query = (
            select(
                SchoolRecord.id,
                SchoolRecord.raw_row,
                SilverRecord.stable_id,
                Cluster.status,
                Cluster.confidence_score,
                Cluster.resolution_source,
            )
            .select_from(SchoolRecord)
            .outerjoin(SilverRecord, SilverRecord.record_id == SchoolRecord.id)
            .outerjoin(ClusterMember, ClusterMember.record_id == SchoolRecord.id)
            .outerjoin(Cluster, Cluster.id == ClusterMember.cluster_id)
            .where(
                SchoolRecord.job_id == job_id,
                SchoolRecord.classification == RecordClassification.VALID.value,
                or_(Cluster.id.is_(None), Cluster.status.notin_(_MERGED)),
            )
            .order_by(SchoolRecord.created_at, SchoolRecord.id)
            .execution_options(yield_per=_EXPORT_FETCH)
        )
        result = await db.stream(query)
        async for record_id, raw_row, stable_id, c_status, score, source in result:
            raw = raw_row or {}
            own = clean_fields(canonical_fields(raw))
            writer.writerow(
                [
                    "",
                    stable_id or "",
                    str(record_id),
                    _source_id(raw),
                    "unresolved" if c_status in _UNDECIDED else "unique",
                    _decided_by(c_status, source),
                    f"{score:.3f}" if score is not None else "",
                    c_status or "",
                    1,
                    *_gold_cells(own, _standardized_name(engine, own["name"])),
                    *("" for _ in targets),
                    "",
                    "",
                ]
            )
            yield take()

    stem = (job.original_filename or f"job_{job.id}").rsplit(".", 1)[0]
    suffix = "gold" if JobStatus(job.status) is JobStatus.COMPLETED else "gold_partial"
    return StreamingResponse(
        rows(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{_safe_stem(stem)}_{suffix}.csv"'},
    )


@router.get("/{job_id}/export/client", summary="The final database, ready to hand to a client")
async def export_client(db: DbDep, job_id: uuid.UUID) -> StreamingResponse:
    """One row per entity, readable, with the client's own columns kept.

    The raw dump (`/export/gold`) is for engineers; this is the file a
    business user opens in Excel. Formatting lives in app/exports/client.py.
    Only for a COMPLETED job: a client file with undecided duplicates in it
    would be wrong without saying so, and it has no column to say so in.
    """
    job = await _get_job_or_404(db, job_id)
    if JobStatus(job.status) is not JobStatus.COMPLETED:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=(
                f"Job is {job.status}. The final database is ready when processing "
                "finishes; the raw dump can be downloaded now."
            ),
        )
    try:
        engine = await load_ontology_async(db, job.ontology, job.tenant_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    # Every row of one upload has the same keys (one CSV header), so one
    # row tells which columns exist; the file itself tells their order.
    raw_keys = list(
        (
            await db.execute(
                select(SchoolRecord.raw_row).where(SchoolRecord.job_id == job_id).limit(1)
            )
        ).scalar_one_or_none()
        or {}
    )
    file_headers = await run_in_threadpool(_file_header, job.source_uri)
    layout = build_layout(
        engine.spec, mapping=job.column_mapping, raw_keys=raw_keys, file_headers=file_headers
    )
    directory = await _pincode_directory(db, engine)

    clusters = (
        await db.execute(
            select(Cluster.id, Cluster.master_record_id, Cluster.golden_fields).where(
                Cluster.job_id == job_id,
                Cluster.status.in_(_MERGED),
                Cluster.master_record_id.is_not(None),
            )
        )
    ).all()
    masters: dict[Any, dict[str, Any]] = {}
    keys_by_cluster: dict[Any, dict[str, Any]] = {}
    pins_by_cluster: dict[Any, list[str]] = {}
    if clusters:
        masters = dict(
            (
                await db.execute(
                    select(SchoolRecord.id, SchoolRecord.raw_row).where(
                        SchoolRecord.id.in_({c.master_record_id for c in clusters})
                    )
                )
            )
            .tuples()
            .all()
        )
        # Identifiers from every member, master first: one record may carry
        # the UDISE code and another the affiliation number.
        master_of = {c.id: c.master_record_id for c in clusters}
        member_keys: dict[Any, list[tuple[bool, dict[str, Any]]]] = {}
        member_pins: dict[Any, list[tuple[bool, str]]] = {}
        for cluster_id, record_id, natural_keys, member_pin in (
            await db.execute(
                select(
                    ClusterMember.cluster_id,
                    ClusterMember.record_id,
                    SilverRecord.natural_keys,
                    SilverRecord.pincode,
                )
                .join(SilverRecord, SilverRecord.record_id == ClusterMember.record_id)
                .where(ClusterMember.cluster_id.in_(list(master_of)))
            )
        ).all():
            not_master = record_id != master_of[cluster_id]
            member_keys.setdefault(cluster_id, []).append((not_master, natural_keys or {}))
            member_pins.setdefault(cluster_id, []).append((not_master, member_pin or ""))
        keys_by_cluster = {
            cid: merge_natural_keys(keys for _, keys in sorted(found, key=lambda m: m[0]))
            for cid, found in member_keys.items()
        }
        pins_by_cluster = {
            cid: [pin for _, pin in sorted(found, key=lambda m: m[0])]
            for cid, found in member_pins.items()
        }

    async def rows() -> AsyncIterator[str]:
        buffer = io.StringIO()
        writer = csv.writer(buffer)

        def take() -> str:
            value = buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)
            return value

        # BOM: Excel opens UTF-8 correctly only with one (Devanagari, ₹).
        buffer.write("﻿")
        writer.writerow(layout.header())
        yield take()

        for cluster in clusters:
            raw = masters.get(cluster.master_record_id) or {}
            golden = {**clean_fields(canonical_fields(raw)), **(cluster.golden_fields or {})}
            golden["pincode"] = resolve_pincode(
                golden.get("pincode", ""),
                address=golden.get("address", ""),
                district=golden.get("district", ""),
                state=golden.get("state", ""),
                web=golden.get(WEB_PINCODE_KEY, ""),
                siblings=pins_by_cluster.get(cluster.id, ()),
                directory=directory,
            )
            writer.writerow(
                layout.row(
                    name=engine.display_name(golden.get("name", "")),
                    fields=golden,
                    natural_keys=keys_by_cluster.get(cluster.id, {}),
                    enriched=golden,
                    raw=raw,
                )
            )
            yield take()

        # Entities of their own: unmatched records and records kept separate.
        query = (
            select(SchoolRecord.raw_row, SilverRecord.natural_keys)
            .select_from(SchoolRecord)
            .outerjoin(SilverRecord, SilverRecord.record_id == SchoolRecord.id)
            .outerjoin(ClusterMember, ClusterMember.record_id == SchoolRecord.id)
            .outerjoin(Cluster, Cluster.id == ClusterMember.cluster_id)
            .where(
                SchoolRecord.job_id == job_id,
                SchoolRecord.classification == RecordClassification.VALID.value,
                or_(Cluster.id.is_(None), Cluster.status.notin_(_MERGED)),
            )
            .order_by(SchoolRecord.created_at, SchoolRecord.id)
            .execution_options(yield_per=_EXPORT_FETCH)
        )
        async for raw_row, natural_keys in await db.stream(query):
            raw = raw_row or {}
            own = clean_fields(canonical_fields(raw))
            own["pincode"] = resolve_pincode(
                own["pincode"],
                address=own["address"],
                district=own["district"],
                state=own["state"],
                directory=directory,
            )
            writer.writerow(
                layout.row(
                    name=engine.display_name(own["name"]),
                    fields=own,
                    natural_keys=natural_keys or {},
                    enriched={},
                    raw=raw,
                )
            )
            yield take()

    stem = (job.original_filename or f"job_{job.id}").rsplit(".", 1)[0]
    return StreamingResponse(
        rows(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{_safe_stem(stem)}_final.csv"'},
    )


def _file_header(source_uri: str | None) -> list[str] | None:
    """The uploaded file's header row, for column order. None if unreadable:
    the export then orders the client's columns alphabetically."""
    if not source_uri:
        return None
    try:
        head = read_head(source_uri).decode("utf-8-sig", errors="replace")
        return next(csv.reader(io.StringIO(head)), None)
    except (OSError, ValueError, StorageNotConfiguredError):
        return None
    except Exception:  # noqa: BLE001 - column order is cosmetic; never fail the export on it
        logger.warning("Could not read the header of %s for column order", source_uri)
        return None


def _outcome(
    classification: str | None, record_id: Any, cluster_status: str | None, master: Any
) -> str:
    if classification == RecordClassification.JUNK.value:
        return "excluded_junk"
    if classification == RecordClassification.MISSING_CONTEXT.value:
        return "needs_more_info"
    if cluster_status in (ClusterStatus.AUTO_MERGED.value, ClusterStatus.RESOLVED.value):
        return "kept_master" if record_id == master else "duplicate_of_master"
    if cluster_status is None:
        return "unique"
    return _OUTCOME_BY_CLUSTER.get(cluster_status, "unique")


_INDIAN_PIN = re.compile(r"[1-9]\d{5}")


def _gold_cells(values: dict[str, str], standardized_name: str) -> list[str]:
    """The identity columns of a Gold row, in GOLD_COLUMNS order."""
    cells: list[str] = []
    for field in REQUIRED_FIELDS:
        cells.append(_gold_value(field, values.get(field, "")))
        if field == "name":
            cells.append(standardized_name)
    return cells


def _standardized_name(engine: OntologyEngine | None, name: str) -> str:
    """The fully resolved entity name: abbreviations expanded by the ontology,
    in capitals like the source files. Blank without an ontology, rather than
    a half-expanded guess.

    Built from the displayed name with `display_name`, not taken from the
    Silver name: Silver is a matching key and also collapses synonyms
    ("vidya mandir" -> "school"), which would rename the school.
    """
    if engine is None or not name:
        return ""
    return engine.display_name(name).upper()


def _gold_value(field: str, value: str) -> str:
    """A field as the final database should hold it. A PIN that is not a real
    six-digit Indian PIN ("000000", "NA", a phone fragment) is a placeholder
    and is left blank rather than published as if it were an address."""
    if field == "pincode" and not _INDIAN_PIN.fullmatch(value.strip()):
        return ""
    return value


def _decided_by(cluster_status: str | None, source: str | None) -> str:
    """Who settled the row's cluster: 'threshold' (Tier 1), 'llm_triage'
    (Tier 2), 'steward', or '' when nothing has been decided."""
    if source:
        return source
    if cluster_status in (ClusterStatus.RESOLVED.value, ClusterStatus.REJECTED.value):
        return "steward"
    return ""


def _source_id(raw_row: dict[str, Any]) -> str:
    """The tenant's own row id, if their file had one (an `id` column)."""
    for key, value in raw_row.items():
        if str(key).strip().lower() in {"id", "record_id", "recordid", "row_id"}:
            return "" if value is None else str(value)
    return ""


def _safe_stem(stem: str) -> str:
    """A filename stem safe inside a quoted Content-Disposition header."""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_. " else "_" for ch in stem).strip()
    return cleaned[:120] or "export"
