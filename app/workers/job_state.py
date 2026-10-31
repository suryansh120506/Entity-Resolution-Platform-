"""Job status and progress transitions — FIG 2.2.

Every pipeline stage begins by claiming the job and ends by handing it on, so
the transition rules live here rather than being written out six times. Three
properties matter:

**Illegal transitions are refused, not logged.** `jobs.status` is what the
frontend polls and what stall detection reads (TRD 2.7). A stage that wrote
EMBEDDING over a COMPLETED job would resurrect a finished job in the UI with
no trace of what did it.

**Re-entering the status you already hold is a no-op, not an error.** A task
whose lease expires is redelivered (app/queue/core.py), so a worker that dies
after doing the work but before acking will have its task redelivered and the
stage will run again. That redelivery is the normal case, not a bug, and it
must not fail on the status it already set.

**Progress is merged, never replaced.** `stage_progress` accumulates one entry
per stage across the whole run; a stage that assigned the whole column would
erase the history of the stages before it.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.enums import JobStatus
from app.models.job import Job

logger = logging.getLogger(__name__)

#: The pipeline order from FIG 2.2. A job moves forward through this list;
#: FAILED is reachable from anywhere and COMPLETED only from the end.
PIPELINE_ORDER: tuple[JobStatus, ...] = (
    JobStatus.PENDING,
    JobStatus.PREPROCESSING,
    JobStatus.EMBEDDING,
    JobStatus.BLOCKING,
    JobStatus.MATCHING,
    JobStatus.AWAITING_REVIEW,
    JobStatus.RESOLVING,
    JobStatus.COMPLETED,
)

_ORDER_INDEX = {status: index for index, status in enumerate(PIPELINE_ORDER)}


class JobStateError(RuntimeError):
    """An illegal status transition was attempted."""


class StaleStageError(JobStateError):
    """This stage has already been passed; the task is a late duplicate.

    Distinguished from the general case because the correct response is the
    opposite one. A genuinely illegal transition is a bug and should fail the
    job loudly; a *stale* one means the pipeline has already moved on, and
    failing the job for it would let a stray re-dispatch destroy a job that
    was sitting healthily in AWAITING_REVIEW waiting for a steward.

    Stages catch this and return without doing anything.
    """


class JobMissingError(StaleStageError):
    """The job row does not exist — most often because a user deleted it.

    A stale-stage case rather than a hard failure: a deleted job's tasks are
    still sitting in Redis, and each one surfacing later as an unhandled
    error (then failing again inside `fail_job_out_of_band`, which cannot
    mark a row FAILED that is not there) is noise, not a diagnosis. The
    stages' existing StaleStageError handling returns quietly.
    """


def load_job(session: Session, job_id: str) -> Job:
    """Fetch a job, or raise.

    The session must be tenant-bound, so a job belonging to another tenant is
    invisible here and surfaces as "not found" rather than as a leak.
    """
    job = session.execute(select(Job).where(Job.id == job_id)).scalar_one_or_none()
    if job is None:
        raise JobMissingError(
            f"Job {job_id} not found for the bound tenant. Either it was deleted, "
            "or this task was dispatched with the wrong tenant_id."
        )
    return job


def advance_status(session: Session, job_id: str, to_status: JobStatus) -> Job:
    """Move a job forward to `to_status`.

    Returns the job unchanged if it is already there (see the module docstring
    on redelivery). Raises `JobStateError` on a backwards move or on any move
    out of a terminal state.
    """
    job = load_job(session, job_id)
    current = JobStatus(job.status)

    if current == to_status:
        logger.info("Job %s already at %s; treating as a redelivery", job_id, to_status)
        return job

    if current in JobStatus.terminal():
        # Stale, not illegal: any stage arriving after the job finished is by
        # definition a late duplicate. Raised as the general JobStateError it
        # used to be, a stray re-dispatch of a COMPLETED job fell through to
        # the stage's failure handler and marked a finished job FAILED.
        raise StaleStageError(
            f"Job {job_id} is {current}, which is terminal. Refusing to move it "
            f"to {to_status} — a finished job reappearing as running is worse "
            "than a task that declines to run."
        )

    if to_status is not JobStatus.FAILED:
        current_index = _ORDER_INDEX.get(current)
        target_index = _ORDER_INDEX.get(to_status)
        if current_index is None or target_index is None:
            raise JobStateError(f"{current} -> {to_status} is not a pipeline transition.")
        if target_index < current_index:
            raise StaleStageError(
                f"Job {job_id} is already at {current}; refusing to move it back to "
                f"{to_status}. This task is a late duplicate of a stage the "
                "pipeline has already passed."
            )

    job.status = to_status.value
    if to_status in JobStatus.terminal():
        job.completed_at = datetime.now(UTC)

    session.flush()
    logger.info("Job %s: %s -> %s", job_id, current, to_status)
    return job


def mark_failed(session: Session, job_id: str, error: str) -> Job:
    """Terminal failure, with the reason recorded in `stage_progress`.

    The reason goes in the job row because TRD 2.7 requires a stalled or
    failed job to be diagnosable from Postgres alone. `dead_letter_tasks`
    holds the traceback; this holds the one line an operator reads first.
    """
    job = load_job(session, job_id)
    if JobStatus(job.status) is JobStatus.FAILED:
        return job
    if JobStatus(job.status) is JobStatus.COMPLETED:
        # Nothing that runs after completion can un-complete a job; a failure
        # reported now belongs to a stray task, not to this job's result.
        logger.warning("Ignoring late failure for completed job %s: %s", job_id, error)
        return job

    job.stage_progress = {
        **(job.stage_progress or {}),
        "error": {"message": error[:2000], "at": datetime.now(UTC).isoformat()},
    }
    job.status = JobStatus.FAILED.value
    job.completed_at = datetime.now(UTC)
    session.flush()
    logger.error("Job %s FAILED: %s", job_id, error)
    return job


def fail_job_out_of_band(
    failed_session: Session, tenant_id: str, job_id: str, message: str
) -> None:
    """Mark a job FAILED in a *fresh* transaction.

    Every stage calls this from its `except` block. Two things have to happen
    in the right order, and getting them backwards deadlocks a worker
    permanently:

    1. **Roll the failing transaction back first.** It has almost certainly
       already run `advance_status`, so it holds a row lock on this exact
       `jobs` row. Opening a second connection and updating the same row
       while the first still holds that lock is a self-deadlock — and
       Postgres cannot break it, because the first session is waiting on
       `ClientRead` rather than on a lock, so there is no cycle for the
       deadlock detector to find. It hangs until something is killed.

    2. **Then write the failure on a new session.** The rollback in step 1
       is exactly why this cannot reuse the caller's session: the status
       would be discarded along with the work that failed, leaving a job
       stuck at a stage with nothing in Postgres explaining why — the
       situation TRD 2.7 exists to prevent.

    Swallows its own errors. It runs while another exception is propagating,
    and masking the original failure with a secondary one throws away the
    diagnosis.
    """
    from app.db.session import tenant_sync_session

    try:
        # Step 1. Releases the row lock the fresh session is about to need.
        failed_session.rollback()
    except Exception:
        logger.exception("Could not roll back the failed transaction for job %s", job_id)

    try:
        with tenant_sync_session(tenant_id) as session:
            mark_failed(session, job_id, message)
    except JobMissingError:
        logger.info("Job %s no longer exists; nothing to mark failed", job_id)
    except Exception:
        logger.exception("Could not record failure for job %s", job_id)


def record_progress(
    session: Session,
    job_id: str,
    stage: str,
    *,
    done: int,
    total: int | None = None,
    details: dict[str, Any] | None = None,
) -> Job:
    """Merge one stage's counters into `stage_progress`.

    Each entry carries `started_at` (kept from the stage's first report) and
    `updated_at`, so a client can compute a rate — and from it an ETA —
    from server time rather than from when it happened to start polling.
    `details` holds stage-specific summaries (issue counts, blocking stats)
    for the results page.

    Assigns a new dict rather than mutating the existing one: SQLAlchemy does
    not track in-place mutation of a JSONB value, so a mutated dict is a
    silently discarded write.
    """
    job = load_job(session, job_id)
    progress = dict(job.stage_progress or {})
    existing = progress.get(stage)
    previous: dict[str, Any] = existing if isinstance(existing, dict) else {}
    now = datetime.now(UTC).isoformat()

    # A count that went backwards is a stage re-running from scratch
    # (redelivery after a crash), and a rate measured from the first run's
    # start would understate its speed.
    restarted = isinstance(previous.get("done"), int) and done < previous["done"]

    entry: dict[str, Any] = {"done": done}
    if total is not None:
        entry["total"] = total
    entry["started_at"] = now if restarted else (previous.get("started_at") or now)
    entry["updated_at"] = now
    if details is not None:
        entry["details"] = details
    elif "details" in previous:
        entry["details"] = previous["details"]

    progress[stage] = entry
    job.stage_progress = progress
    session.flush()
    return job
