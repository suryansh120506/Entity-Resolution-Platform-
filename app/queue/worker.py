"""Queue worker: `python -m app.queue.worker [--queues a,b] [--concurrency N] [--burst]`.

One process, a thread pool of task slots, one LISTEN connection.

* **Wake-ups** come from `pg_notify` (fired by an insert/requeue trigger), so
  an idle worker costs one sleeping socket rather than a polling loop. The
  poll interval is only a fallback, for delayed retries whose `run_at`
  arrives with no insert to announce it.
* **Leases, not acks.** A claimed row carries `lease_expires_at`; a
  heartbeat thread extends it while the task runs. A worker that dies stops
  heartbeating, the reaper (run by every live worker) requeues the row, and
  the stage — idempotent by design — runs again.
* **Liveness is the LISTEN connection itself.** It is opened on the
  data-plane role with application_name `er-worker:<id>`, so `/system/status`
  counts live workers from `pg_stat_activity` with no heartbeat table: a
  worker that dies takes its connection, and its entry, with it.
* **Failure policy lives here, not in the stages.** A stage raises; the
  worker decides retry vs. terminal, and only on a terminal failure marks the
  job FAILED. (Stages used to mark FAILED before re-raising for a retry,
  which made every retry a no-op against an already-failed job.)
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import psycopg
from sqlalchemy.engine import make_url

from app.core.config import get_settings
from app.db.session import platform_session, tenant_sync_session
from app.queue import core
from app.queue.core import ClaimedTask
from app.queue.registry import REGISTRY, RetryLaterError, TaskSpec, load_task_modules
from app.queue.topology import QUEUES, WORKER_APPLICATION_NAME_PREFIX

logger = logging.getLogger(__name__)

NOTIFY_CHANNEL = "er_task_queue"


def _new_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"


def _libpq_dsn(sqlalchemy_url: str) -> str:
    return (
        make_url(sqlalchemy_url).set(drivername="postgresql").render_as_string(hide_password=False)
    )


class _Heartbeat(threading.Thread):
    """Extends one task's lease every lease/3 until stopped or lost."""

    def __init__(self, task: ClaimedTask, worker_id: str) -> None:
        super().__init__(name=f"heartbeat-{task.id}", daemon=True)
        self.task = task
        self.worker_id = worker_id
        self.lease = QUEUES[task.queue].lease_seconds
        self.lost = threading.Event()
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.wait(self.lease / 3):
            try:
                with platform_session() as session:
                    held = core.heartbeat(
                        session,
                        task_id=self.task.id,
                        worker_id=self.worker_id,
                        lease_seconds=self.lease,
                    )
            except Exception:
                # A missed beat is survivable (the lease has 2/3 left); a
                # crashed heartbeat thread would silently forfeit the task.
                logger.exception("Heartbeat failed for task %s", self.task.id)
                continue
            if not held:
                logger.warning("Lost lease on task %s (%s)", self.task.id, self.task.task_name)
                self.lost.set()
                return

    def stop(self) -> None:
        self._stop_event.set()


class Worker:
    def __init__(
        self,
        *,
        queues: list[str] | None = None,
        concurrency: int = 4,
        poll_interval: float = 5.0,
        reap_interval: float = 30.0,
        worker_id: str | None = None,
        tenant_id: uuid.UUID | str | None = None,
    ) -> None:
        unknown = set(queues or ()) - QUEUES.keys()
        if unknown:
            raise ValueError(f"Unknown queues: {sorted(unknown)}")
        # Highest-priority queue claims first when slots are scarce.
        self.queues = sorted(queues or QUEUES, key=lambda q: -QUEUES[q].priority)
        #: Serve only this tenant's tasks (manual drains, tests). None = all.
        self.tenant_id = tenant_id
        self.concurrency = concurrency
        self.poll_interval = poll_interval
        self.reap_interval = reap_interval
        self.worker_id = worker_id or _new_worker_id()

        self._stopping = threading.Event()
        self._slot_freed = threading.Event()
        self._in_flight: set[Future[str]] = set()
        self._lock = threading.Lock()
        self._listen: psycopg.Connection[Any] | None = None
        self._last_reap = 0.0
        self._last_purge = 0.0

    # --- task execution -----------------------------------------------------

    def execute(self, task: ClaimedTask) -> str:
        """Run one claimed task to an outcome. Never raises."""
        spec = REGISTRY.get(task.task_name)
        if spec is None:
            self._finalize_failure(task, f"No task registered as {task.task_name!r}", None)
            return "failed"

        heartbeat = _Heartbeat(task, self.worker_id)
        heartbeat.start()
        try:
            spec.fn(tenant_id=str(task.tenant_id), **task.payload)
        except RetryLaterError as deferred:
            return self._retry_or_fail(task, str(deferred), deferred.countdown, None)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc()
            if self._retryable(spec, exc):
                delay = core.retry_delay(QUEUES[task.queue], task.attempts)
                logger.warning("Task %s %s failed transiently: %s", task.id, task.task_name, exc)
                return self._retry_or_fail(task, message, delay, tb)
            logger.exception("Task %s %s failed permanently", task.id, task.task_name)
            self._finalize_failure(task, message, tb)
            return "failed"
        finally:
            heartbeat.stop()

        with platform_session() as session:
            completed = core.complete(session, task=task, worker_id=self.worker_id)
        if not completed:
            # Reaped and re-claimed while we ran. The other holder owns the
            # outcome and the hand-off; ours is discarded, and the stage's
            # idempotence makes the duplicate run harmless.
            logger.warning("Task %s finished after losing its lease; outcome discarded", task.id)
            return "lost"
        return "succeeded"

    @staticmethod
    def _retryable(spec: TaskSpec, exc: BaseException) -> bool:
        return isinstance(exc, spec.retry_on)

    def _retry_or_fail(
        self, task: ClaimedTask, message: str, countdown: float, tb: str | None
    ) -> str:
        if task.final_attempt:
            self._finalize_failure(task, f"{message} (attempts exhausted)", tb)
            return "failed"
        with platform_session() as session:
            core.retry(
                session, task=task, worker_id=self.worker_id, error=message, countdown=countdown
            )
        return "retried"

    def _finalize_failure(self, task: ClaimedTask, message: str, tb: str | None) -> None:
        with platform_session() as session:
            owned = core.fail(
                session, task=task, worker_id=self.worker_id, error=message, traceback_text=tb
            )
        if owned:
            _mark_job_failed(task, message)

    # --- main loop ----------------------------------------------------------

    def run(self) -> None:
        load_task_modules()
        logger.info(
            "Worker %s serving %s with %d slots", self.worker_id, self.queues, self.concurrency
        )
        executor = ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="task")
        try:
            while not self._stopping.is_set():
                try:
                    self._housekeeping()
                    self._fill_slots(executor)
                    self._wait()
                except psycopg.OperationalError:
                    logger.exception("Database unavailable; backing off")
                    self._close_listen()
                    self._stopping.wait(self.poll_interval)
                except Exception:
                    logger.exception("Worker loop error")
                    self._stopping.wait(self.poll_interval)
        finally:
            logger.info(
                "Worker %s draining %d in-flight tasks", self.worker_id, len(self._in_flight)
            )
            executor.shutdown(wait=True)
            self._close_listen()

    def stop(self, *_: Any) -> None:
        self._stopping.set()
        self._slot_freed.set()

    def run_until_idle(self, *, max_tasks: int | None = None) -> dict[str, int]:
        """Burst mode: execute ready tasks inline until none are claimable.

        Single-threaded and deterministic — what tests and one-off backfills
        want. Tasks whose `run_at` is in the future are not waited for.
        """
        load_task_modules()
        outcomes: dict[str, int] = {}
        executed = 0
        while max_tasks is None or executed < max_tasks:
            with platform_session() as session:
                core.reap_expired(session, exclude_worker=self.worker_id)
            task = self._claim_one()
            if task is None:
                break
            outcome = self.execute(task)
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            executed += 1
        return outcomes

    def _claim_one(self) -> ClaimedTask | None:
        for queue in self.queues:
            with platform_session() as session:
                claimed = core.claim(
                    session,
                    worker_id=self.worker_id,
                    queue=queue,
                    limit=1,
                    tenant_id=self.tenant_id,
                )
            if claimed:
                return claimed[0]
        return None

    def _fill_slots(self, executor: ThreadPoolExecutor) -> None:
        for queue in self.queues:
            with self._lock:
                free = self.concurrency - len(self._in_flight)
            if free <= 0:
                return
            with platform_session() as session:
                claimed = core.claim(
                    session,
                    worker_id=self.worker_id,
                    queue=queue,
                    limit=free,
                    tenant_id=self.tenant_id,
                )
            for task in claimed:
                future = executor.submit(self.execute, task)
                with self._lock:
                    self._in_flight.add(future)
                future.add_done_callback(self._on_done)

    def _on_done(self, future: Future[str]) -> None:
        with self._lock:
            self._in_flight.discard(future)
        self._slot_freed.set()

    def _wait(self) -> None:
        with self._lock:
            saturated = len(self._in_flight) >= self.concurrency
        if saturated:
            # Nothing to claim into; only a finishing task changes that.
            self._slot_freed.wait(self.poll_interval)
            self._slot_freed.clear()
            return
        conn = self._ensure_listen()
        # Any notification means "something may be claimable"; the payload
        # (the queue name) is not needed because every queue is re-checked.
        for _ in conn.notifies(timeout=self.poll_interval, stop_after=1):
            pass

    def _housekeeping(self) -> None:
        now = time.monotonic()
        if now - self._last_reap >= self.reap_interval:
            self._last_reap = now
            with platform_session() as session:
                reaped = core.reap_expired(session, exclude_worker=self.worker_id)
                finalizers = core.enqueue_reviewed_job_finalizers(session)
            for task, outcome in reaped:
                logger.warning("Reaped task %s (%s): %s", task.id, task.task_name, outcome)
                if outcome == "failed":
                    _mark_job_failed(task, "Task lease expired on its final attempt")
            if finalizers:
                logger.info("Enqueued %d reviewed-job finalizers", finalizers)
        if now - self._last_purge >= 3600:
            self._last_purge = now
            with platform_session() as session:
                purged = core.purge_finished(session)
            if purged:
                logger.info("Purged %d finished task rows", purged)

    def _ensure_listen(self) -> psycopg.Connection[Any]:
        if self._listen is None or self._listen.closed:
            self._listen = psycopg.connect(
                _libpq_dsn(get_settings().sync_database_url),
                autocommit=True,
                application_name=f"{WORKER_APPLICATION_NAME_PREFIX}{self.worker_id}"[:63],
            )
            self._listen.execute(f"LISTEN {NOTIFY_CHANNEL}")
        return self._listen

    def _close_listen(self) -> None:
        if self._listen is not None:
            try:
                self._listen.close()
            finally:
                self._listen = None


def _mark_job_failed(task: ClaimedTask, message: str) -> None:
    """Terminal task failure -> FAILED job, in a fresh tenant transaction.

    The task's own session has already rolled back and closed by now, so
    this cannot self-deadlock on the jobs row the stage had locked.
    """
    job_id = task.payload.get("job_id") or (str(task.job_id) if task.job_id else None)
    if not job_id:
        return
    from app.workers.job_state import JobMissingError, mark_failed

    try:
        with tenant_sync_session(task.tenant_id) as session:
            mark_failed(session, str(job_id), f"{task.task_name}: {message}")
    except JobMissingError:
        logger.info("Job %s no longer exists; nothing to mark failed", job_id)
    except Exception:
        logger.exception("Could not record failure for job %s", job_id)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ER-PLATFORM Postgres queue worker")
    parser.add_argument("--queues", default=",".join(QUEUES), help="comma-separated")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--burst", action="store_true", help="drain ready tasks, then exit")
    parser.add_argument("--tenant", type=uuid.UUID, default=None, help="serve one tenant only")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s",
    )
    worker = Worker(
        queues=[q.strip() for q in args.queues.split(",") if q.strip()],
        concurrency=args.concurrency,
        poll_interval=args.poll_interval,
        tenant_id=args.tenant,
    )
    if args.burst:
        logger.info("Burst run finished: %s", worker.run_until_idle())
        return

    signal.signal(signal.SIGINT, worker.stop)
    signal.signal(signal.SIGTERM, worker.stop)
    worker.run()


if __name__ == "__main__":
    main()
