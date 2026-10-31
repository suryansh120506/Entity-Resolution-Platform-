"""Postgres task queue: every state transition, as one SQL statement each.

Two planes, deliberately:

* **Enqueue** runs on the caller's tenant-bound data-plane session, inside
  the caller's own transaction. The task row commits or rolls back with the
  work that produced it (a transactional outbox), and FIG 2.8's WITH CHECK
  makes it impossible to enqueue work for another tenant.
* **Claim / complete / retry / reap** run on the BYPASSRLS control-plane
  session: a worker serves every tenant, and `er_app` has had UPDATE and
  DELETE on `task_queue` revoked (migration 0006).

Correctness rests on three things:

1. `FOR UPDATE SKIP LOCKED` in the claim, so concurrent workers partition
   the ready set instead of serialising on it.
2. Fencing on `locked_by`: completion and retry only touch a row the caller
   still holds. A worker whose lease was reaped and re-claimed elsewhere gets
   rowcount 0 and backs off instead of clobbering the new holder.
3. Successors are inserted in the same transaction that marks the task
   succeeded, guarded by the live-dedupe unique index, so a stage hands off
   exactly once however often it is redelivered.
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.queue.topology import PIPELINE, QUEUES, QueueSpec, queue_for, spec_for


@dataclass(frozen=True, slots=True)
class ClaimedTask:
    id: uuid.UUID
    tenant_id: uuid.UUID
    job_id: uuid.UUID | None
    queue: str
    task_name: str
    payload: dict[str, Any]
    attempts: int
    max_attempts: int

    @property
    def final_attempt(self) -> bool:
        return self.attempts >= self.max_attempts


# --- Enqueue (data plane, caller's transaction) --------------------------------

_ENQUEUE_SQL = text(
    """
    INSERT INTO task_queue
        (tenant_id, job_id, queue, task_name, payload, priority, max_attempts,
         run_at, dedupe_key)
    VALUES
        (:tenant_id, :job_id, :queue, :task_name, CAST(:payload AS jsonb), :priority,
         :max_attempts, coalesce(CAST(:run_at AS timestamptz), now()), :dedupe_key)
    ON CONFLICT (tenant_id, dedupe_key)
        WHERE dedupe_key IS NOT NULL AND status IN ('queued', 'running')
        DO NOTHING
    RETURNING id
    """
)


def _enqueue_params(
    *,
    tenant_id: uuid.UUID | str,
    task_name: str,
    payload: dict[str, Any],
    job_id: uuid.UUID | str | None,
    dedupe_key: str | None,
    run_at: datetime | None,
    priority: int | None,
) -> dict[str, Any]:
    spec = spec_for(task_name)
    return {
        "tenant_id": str(tenant_id),
        "job_id": None if job_id is None else str(job_id),
        "queue": queue_for(task_name),
        "task_name": task_name,
        "payload": json.dumps(payload, separators=(",", ":"), default=str),
        "priority": spec.priority if priority is None else priority,
        "max_attempts": spec.max_attempts,
        "run_at": run_at,
        "dedupe_key": dedupe_key,
    }


def enqueue(
    session: Session,
    *,
    tenant_id: uuid.UUID | str,
    task_name: str,
    payload: dict[str, Any],
    job_id: uuid.UUID | str | None = None,
    dedupe_key: str | None = None,
    run_at: datetime | None = None,
    priority: int | None = None,
) -> uuid.UUID | None:
    """Insert a task in the caller's transaction. None = deduplicated."""
    params = _enqueue_params(
        tenant_id=tenant_id,
        task_name=task_name,
        payload=payload,
        job_id=job_id,
        dedupe_key=dedupe_key,
        run_at=run_at,
        priority=priority,
    )
    return session.execute(_ENQUEUE_SQL, params).scalar_one_or_none()


async def enqueue_async(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID | str,
    task_name: str,
    payload: dict[str, Any],
    job_id: uuid.UUID | str | None = None,
    dedupe_key: str | None = None,
    run_at: datetime | None = None,
    priority: int | None = None,
) -> uuid.UUID | None:
    params = _enqueue_params(
        tenant_id=tenant_id,
        task_name=task_name,
        payload=payload,
        job_id=job_id,
        dedupe_key=dedupe_key,
        run_at=run_at,
        priority=priority,
    )
    return (await session.execute(_ENQUEUE_SQL, params)).scalar_one_or_none()


def stage_dedupe_key(task_name: str, job_id: uuid.UUID | str) -> str:
    return f"{task_name}:{job_id}"


# --- Claim (control plane) ------------------------------------------------------

_CLAIM_SQL = text(
    """
    UPDATE task_queue t
       SET status = 'running',
           locked_by = :worker_id,
           lease_expires_at = now() + make_interval(secs => :lease_seconds),
           attempts = t.attempts + 1,
           started_at = now()
      FROM (
            SELECT id
              FROM task_queue
             WHERE status = 'queued'
               AND queue = :queue
               AND run_at <= now()
               AND (CAST(:tenant_id AS uuid) IS NULL OR tenant_id = CAST(:tenant_id AS uuid))
             ORDER BY priority DESC, run_at, enqueued_at
             LIMIT :limit
               FOR UPDATE SKIP LOCKED
           ) ready
     WHERE t.id = ready.id
    RETURNING t.id, t.tenant_id, t.job_id, t.queue, t.task_name, t.payload,
              t.attempts, t.max_attempts
    """
)


def claim(
    session: Session,
    *,
    worker_id: str,
    queue: str,
    limit: int,
    tenant_id: uuid.UUID | str | None = None,
) -> list[ClaimedTask]:
    """Lease up to `limit` ready tasks from `queue`. Commit to publish the lease.

    `tenant_id` restricts the claim to one tenant: draining one tenant's
    backlog by hand, or a test that must not execute anyone else's work.

    A capped queue takes a transaction-scoped advisory lock first, so two
    workers cannot both read "1 running, cap 2" and both claim: the count and
    the claim are serialised per queue, and only for capped queues.
    """
    if limit <= 0:
        return []
    spec: QueueSpec = QUEUES[queue]
    if spec.max_running is not None:
        session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
            {"key": f"er_task_queue:{queue}"},
        )
        running = session.execute(
            text("SELECT count(*) FROM task_queue WHERE queue = :queue AND status = 'running'"),
            {"queue": queue},
        ).scalar_one()
        limit = min(limit, spec.max_running - running)
        if limit <= 0:
            return []

    rows = session.execute(
        _CLAIM_SQL,
        {
            "worker_id": worker_id,
            "queue": queue,
            "limit": limit,
            "lease_seconds": spec.lease_seconds,
            "tenant_id": None if tenant_id is None else str(tenant_id),
        },
    ).mappings()
    return [ClaimedTask(**row) for row in rows]


# --- Lease maintenance ----------------------------------------------------------


def heartbeat(session: Session, *, task_id: uuid.UUID, worker_id: str, lease_seconds: int) -> bool:
    """Extend a held lease. False means the lease was lost: stop treating the
    task as ours (another worker may already be running it)."""
    result = session.execute(
        text(
            "UPDATE task_queue SET lease_expires_at = now() + make_interval(secs => :lease) "
            "WHERE id = :id AND status = 'running' AND locked_by = :worker_id"
        ),
        {"id": task_id, "worker_id": worker_id, "lease": lease_seconds},
    )
    return result.rowcount == 1  # type: ignore[attr-defined]


# --- Completion -----------------------------------------------------------------

_SUCCESSOR_SQL = text(
    """
    INSERT INTO task_queue
        (tenant_id, job_id, queue, task_name, payload, priority, max_attempts, dedupe_key)
    SELECT :tenant_id, CAST(:job_id AS uuid), :queue, :task_name, CAST(:payload AS jsonb),
           :priority, :max_attempts, :dedupe_key
     WHERE CAST(:job_id AS uuid) IS NULL
        OR EXISTS (SELECT 1 FROM jobs WHERE id = CAST(:job_id AS uuid) AND status <> 'FAILED')
    ON CONFLICT (tenant_id, dedupe_key)
        WHERE dedupe_key IS NOT NULL AND status IN ('queued', 'running')
        DO NOTHING
    """
)


def complete(session: Session, *, task: ClaimedTask, worker_id: str) -> bool:
    """Mark succeeded and enqueue pipeline successors, atomically.

    Successors carry only the job id: stages communicate through Postgres,
    not through payloads, so a redelivered stage reads committed state rather
    than a message that may have been produced twice. They are not enqueued
    for a job that has FAILED (or been deleted) in the meantime.
    """
    marked = session.execute(
        text(
            "UPDATE task_queue SET status = 'succeeded', finished_at = now(), "
            "locked_by = NULL, lease_expires_at = NULL "
            "WHERE id = :id AND status = 'running' AND locked_by = :worker_id"
        ),
        {"id": task.id, "worker_id": worker_id},
    )
    if marked.rowcount != 1:  # type: ignore[attr-defined]
        return False

    for successor in PIPELINE.get(task.task_name, ()):
        spec = spec_for(successor)
        job_id = None if task.job_id is None else str(task.job_id)
        session.execute(
            _SUCCESSOR_SQL,
            {
                "tenant_id": task.tenant_id,
                "job_id": job_id,
                "queue": queue_for(successor),
                "task_name": successor,
                "payload": json.dumps({"job_id": job_id} if job_id else {}),
                "priority": spec.priority,
                "max_attempts": spec.max_attempts,
                "dedupe_key": stage_dedupe_key(successor, job_id) if job_id else None,
            },
        )
    return True


def retry_delay(spec: QueueSpec, attempt: int) -> float:
    """Exponential (x4) with +-25% jitter, so a burst of failures does not
    come back as a synchronised burst of retries."""
    base = min(spec.backoff_cap_seconds, spec.backoff_base_seconds * (4 ** max(0, attempt - 1)))
    return base * random.uniform(0.75, 1.25)


def retry(
    session: Session,
    *,
    task: ClaimedTask,
    worker_id: str,
    error: str,
    countdown: float,
) -> bool:
    result = session.execute(
        text(
            "UPDATE task_queue SET status = 'queued', locked_by = NULL, lease_expires_at = NULL, "
            "run_at = now() + make_interval(secs => :countdown), last_error = :error "
            "WHERE id = :id AND status = 'running' AND locked_by = :worker_id"
        ),
        {"id": task.id, "worker_id": worker_id, "error": error[:4000], "countdown": countdown},
    )
    return result.rowcount == 1  # type: ignore[attr-defined]


def fail(
    session: Session,
    *,
    task: ClaimedTask,
    worker_id: str,
    error: str,
    traceback_text: str | None,
) -> bool:
    """Terminal failure plus a dead-letter row, in one transaction."""
    result = session.execute(
        text(
            "UPDATE task_queue SET status = 'failed', finished_at = now(), locked_by = NULL, "
            "lease_expires_at = NULL, last_error = :error "
            "WHERE id = :id AND status = 'running' AND locked_by = :worker_id"
        ),
        {"id": task.id, "worker_id": worker_id, "error": error[:4000]},
    )
    if result.rowcount != 1:  # type: ignore[attr-defined]
        return False
    _dead_letter(session, task, error, traceback_text)
    return True


def _dead_letter(session: Session, task: ClaimedTask, error: str, tb: str | None) -> None:
    session.execute(
        text(
            "INSERT INTO dead_letter_tasks "
            "(tenant_id, job_id, task_name, queue, task_id, payload, error, traceback, "
            " retries_exhausted) "
            "SELECT :tenant_id, CAST(:job_id AS uuid), :task_name, :queue, :task_id, "
            "       CAST(:payload AS jsonb), :error, :traceback, :attempts "
            # The job may have been deleted mid-run; its FK would reject the row.
            " WHERE CAST(:job_id AS uuid) IS NULL "
            "    OR EXISTS (SELECT 1 FROM jobs WHERE id = CAST(:job_id AS uuid))"
        ),
        {
            "tenant_id": task.tenant_id,
            "job_id": None if task.job_id is None else str(task.job_id),
            "task_name": task.task_name,
            "queue": task.queue,
            "task_id": str(task.id),
            "payload": json.dumps(task.payload, default=str),
            "error": error[:4000],
            "traceback": tb,
            "attempts": task.attempts,
        },
    )


# --- Recovery and housekeeping --------------------------------------------------

_REAP_SQL = text(
    """
    WITH expired AS (
        SELECT id
          FROM task_queue
         WHERE status = 'running' AND lease_expires_at < now()
           AND (CAST(:exclude AS text) IS NULL OR locked_by <> CAST(:exclude AS text))
         ORDER BY lease_expires_at
         LIMIT :limit
           FOR UPDATE SKIP LOCKED
    )
    UPDATE task_queue t
       SET status = CASE WHEN t.attempts < t.max_attempts THEN 'queued' ELSE 'failed' END,
           finished_at = CASE WHEN t.attempts < t.max_attempts THEN NULL ELSE now() END,
           run_at = now(),
           locked_by = NULL,
           lease_expires_at = NULL,
           last_error = 'lease expired; worker ' || t.locked_by || ' presumed dead'
      FROM expired
     WHERE t.id = expired.id
    RETURNING t.id, t.tenant_id, t.job_id, t.queue, t.task_name, t.payload,
              t.attempts, t.max_attempts, t.status
    """
)


def reap_expired(
    session: Session, *, limit: int = 500, exclude_worker: str | None = None
) -> list[tuple[ClaimedTask, str]]:
    """Recover tasks whose worker stopped heartbeating. Requeued while they
    have attempts left; otherwise failed and dead-lettered.

    `exclude_worker` is the caller's own id. A worker that is running this
    code is alive, so it is never "presumed dead", and its own tasks are left
    to its heartbeat. The distinction matters after a stall that freezes the
    whole process (a laptop sleeping, a long pause): every lease lapses, and
    on waking the reaper can run before the heartbeat thread does. Reaping
    its own task then burns an attempt and starts a duplicate run of work
    that was about to carry on. A heartbeat does not check expiry, so it
    simply revives the lease. Tasks of any *other* worker are still reaped.
    """
    reaped: list[tuple[ClaimedTask, str]] = []
    for row in session.execute(_REAP_SQL, {"limit": limit, "exclude": exclude_worker}).mappings():
        data = dict(row)
        outcome = data.pop("status")
        task = ClaimedTask(**data)
        if outcome == "failed":
            _dead_letter(session, task, "lease expired on final attempt", None)
        reaped.append((task, outcome))
    return reaped


def purge_finished(session: Session, *, succeeded_days: int = 7, failed_days: int = 30) -> int:
    result = session.execute(
        text(
            "DELETE FROM task_queue WHERE "
            "(status IN ('succeeded', 'cancelled') "
            "   AND finished_at < now() - make_interval(days => :ok_days)) "
            "OR (status = 'failed' AND finished_at < now() - make_interval(days => :bad_days))"
        ),
        {"ok_days": succeeded_days, "bad_days": failed_days},
    )
    return int(result.rowcount)  # type: ignore[attr-defined]


def enqueue_reviewed_job_finalizers(session: Session) -> int:
    """Safety-net sweep: every AWAITING_REVIEW job with nothing left to review
    gets a `jobs.finalize_reviewed` task. The feedback route enqueues the same
    task event-driven; this catches whatever that path missed."""
    result = session.execute(
        text(
            """
            INSERT INTO task_queue
                (tenant_id, job_id, queue, task_name, payload, priority, max_attempts, dedupe_key)
            SELECT j.tenant_id, j.id, 'jobs', 'jobs.finalize_reviewed',
                   jsonb_build_object('job_id', j.id::text), :priority, :max_attempts,
                   'jobs.finalize_reviewed:' || j.id::text
              FROM jobs j
             WHERE j.status = 'AWAITING_REVIEW'
               AND NOT EXISTS (
                     SELECT 1 FROM clusters c
                      WHERE c.job_id = j.id AND c.status IN ('PENDING_REVIEW', 'PENDING_API'))
            ON CONFLICT (tenant_id, dedupe_key)
                WHERE dedupe_key IS NOT NULL AND status IN ('queued', 'running')
                DO NOTHING
            """
        ),
        {"priority": QUEUES["jobs"].priority, "max_attempts": QUEUES["jobs"].max_attempts},
    )
    return int(result.rowcount)  # type: ignore[attr-defined]
