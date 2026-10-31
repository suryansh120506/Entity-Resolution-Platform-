"""Keep a few web searches running ahead of the work that consumes them.

Tier 3 does, per school: a web search (3-10 s of waiting on DuckDuckGo), an
optional LLM call (a few seconds, often rationed), and a database write. Done
in that order, one school at a time, the search and the model never overlap:
the pipeline waits on one while the other sits idle. Starting the next few
searches before the current school is finished hides the slower of the two
behind the other.

The search function is still the throttled one (`polite_search`), so however
many run at once, they *start* at most one per `ddgs_min_interval_seconds`
platform-wide. This module only decides when to start, never how fast.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def windowed(items: Iterable[T], start: Callable[[T], R], window: int) -> Iterator[tuple[T, R]]:
    """Yield `(item, start(item))` in order, with `window` items started ahead.

    `start` runs on the consuming thread (it may touch the database session);
    what it returns is typically a `Future` of work it handed to a pool. The
    first result is yielded only once `window` items have been started, or the
    input is exhausted.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    started: deque[tuple[T, R]] = deque()
    for item in items:
        started.append((item, start(item)))
        if len(started) >= window:
            yield started.popleft()
    while started:
        yield started.popleft()


@contextmanager
def search_pool(workers: int) -> Iterator[ThreadPoolExecutor]:
    """A thread pool for searches that never leaves work queued: if the
    consumer stops (an error, a deferred task), searches not yet begun are
    dropped rather than run for nobody. A search already in flight is not
    waited for: it may be in a long rate-limit pause, and the task should end
    now, not hours from now."""
    pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="ddgs")
    try:
        yield pool
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def finished(value: R) -> Future[R]:
    """An already-finished future, for work that needed no search (a cache hit)."""
    future: Future[R] = Future()
    future.set_result(value)
    return future
