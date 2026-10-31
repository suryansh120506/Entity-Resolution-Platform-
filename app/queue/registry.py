"""Task registration.

`@task` records a function under a task name and returns it unchanged, so a
stage stays an ordinary function: tests call it directly, the worker calls it
with the row's payload. There is no proxy object and no `self`.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from app.queue.topology import TRANSIENT_ERRORS, queue_for

F = TypeVar("F", bound=Callable[..., Any])

#: Modules whose import registers tasks. The worker imports these at start;
#: the API never does.
TASK_MODULES: tuple[str, ...] = (
    "app.workers.tasks.ingest",
    "app.workers.tasks.silver",
    "app.workers.tasks.embeddings",
    "app.workers.tasks.blocking",
    "app.workers.tasks.matching",
    "app.workers.tasks.gold",
    "app.workers.tasks.resolution",
    "app.workers.tasks.enrichment",
    "app.workers.tasks.jobs",
)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    queue: str
    fn: Callable[..., Any]
    retry_on: tuple[type[BaseException], ...]


REGISTRY: dict[str, TaskSpec] = {}


def task(
    name: str, *, retry_on: tuple[type[BaseException], ...] = TRANSIENT_ERRORS
) -> Callable[[F], F]:
    queue = queue_for(name)

    def register(fn: F) -> F:
        existing = REGISTRY.get(name)
        if existing is not None and existing.fn is not fn:
            raise ValueError(f"Task {name!r} is registered twice")
        REGISTRY[name] = TaskSpec(name=name, queue=queue, fn=fn, retry_on=retry_on)
        return fn

    return register


def load_task_modules() -> dict[str, TaskSpec]:
    for module in TASK_MODULES:
        importlib.import_module(module)
    return REGISTRY


class RetryLaterError(Exception):
    """Raised by a task to be re-run after `countdown` seconds.

    Not an error and not counted as a failure reason: a rate limit or a
    not-yet-ready dependency. Still consumes an attempt, so a task that
    defers forever eventually dead-letters instead of spinning.
    """

    def __init__(self, countdown: float, reason: str = "") -> None:
        super().__init__(reason or f"retry in {countdown:.0f}s")
        self.countdown = max(0.0, countdown)
