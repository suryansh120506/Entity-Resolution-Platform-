"""Queue topology and pipeline order, as data.

Dependency-light on purpose: the API imports this to route an enqueue by task
name, and must not pull in pandas/torch/xgboost by doing so (SAD 3.2). Task
*implementations* live in `app.workers.tasks.*` and register themselves via
`app.queue.registry.task`; nothing here imports them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from sqlalchemy.exc import InterfaceError, InternalError, OperationalError


@dataclass(frozen=True, slots=True)
class QueueSpec:
    #: Platform-wide cap on concurrently running tasks from this queue, across
    #: every worker process. Enforced at claim time under an advisory lock, so
    #: it is exact, not per-process. None = bounded only by worker slots.
    max_running: int | None = None
    max_attempts: int = 3
    #: Retry delay is backoff_base * 4**(attempt-1) seconds, capped, +-25% jitter.
    backoff_base_seconds: float = 2.0
    backoff_cap_seconds: float = 300.0
    #: Lease length. A heartbeat extends it every lease/3 while the task runs,
    #: so this bounds how long a *dead* worker's task stays invisible, not how
    #: long a task may run.
    lease_seconds: int = 120
    priority: int = 0


QUEUES: Final[dict[str, QueueSpec]] = {
    "ingest": QueueSpec(priority=9),
    "silver": QueueSpec(priority=9),
    "embeddings": QueueSpec(priority=6, max_running=2),
    "blocking": QueueSpec(priority=6),
    "matching": QueueSpec(priority=6),
    "gold": QueueSpec(priority=6),
    "resolution": QueueSpec(priority=6),
    # Outbound scraping. One at a time platform-wide: politeness toward
    # public registries and the free LLM tier's rate limit are global
    # constraints, and a per-process limiter cannot enforce a global one.
    "enrichment": QueueSpec(priority=1, max_running=1, max_attempts=4, backoff_base_seconds=15.0),
    "jobs": QueueSpec(priority=3),
}

#: Successors enqueued atomically with a task's completion. The pipeline,
#: in one place: "what runs after matching?" is answered here.
PIPELINE: Final[dict[str, tuple[str, ...]]] = {
    "ingest.ingest_and_classify": ("silver.standardize",),
    # Enrichment fans out beside the matching chain rather than gating it:
    # it is network-bound and rate-limited, and matching does not need it.
    "silver.standardize": ("embeddings.embeddings_stage", "enrichment.enrich_job"),
    "embeddings.embeddings_stage": ("blocking.blocking_stage",),
    "blocking.blocking_stage": ("matching.probabilistic_match",),
    # In-upload dedup first, then link the upload against the gold layer,
    # then the confidence gate.
    "matching.probabilistic_match": ("gold.link_job",),
    "gold.link_job": ("resolution.resolve_by_confidence",),
}

PIPELINE_ENTRYPOINT: Final[str] = "ingest.ingest_and_classify"

#: Prefix of the application_name a worker's LISTEN connection carries.
#: `/system/status` counts live workers by it in pg_stat_activity.
WORKER_APPLICATION_NAME_PREFIX: Final[str] = "er-worker:"

#: Failures worth retrying — an allowlist, never `Exception`. A permanent
#: error (missing upload, bad header, illegal state transition) cannot win on
#: a second attempt; retrying it only buries the real diagnosis under the
#: different error the retry hits. Deliberately not `OSError`:
#: FileNotFoundError is one, and it is the canonical permanent failure.
TRANSIENT_ERRORS: Final[tuple[type[BaseException], ...]] = (
    OperationalError,
    InterfaceError,
    InternalError,
    ConnectionError,
    TimeoutError,
)


class UnknownTaskError(LookupError):
    pass


def queue_for(task_name: str) -> str:
    """Queue a task runs on: its name's prefix, which must be a declared queue."""
    prefix, _, rest = task_name.partition(".")
    if not rest or prefix not in QUEUES:
        raise UnknownTaskError(f"{task_name!r} does not route to a declared queue")
    return prefix


def spec_for(task_name: str) -> QueueSpec:
    return QUEUES[queue_for(task_name)]
