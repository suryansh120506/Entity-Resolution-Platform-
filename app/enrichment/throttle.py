"""Rate-limit safety for open-web search: one DDGS call every N seconds,
across every worker process, and patience when the provider pushes back.

Why a database lock and not a `time.sleep` alone: a sleep spaces out the
calls of one process. Two workers (or two tasks on the enrichment queue)
each sleeping two seconds still hit DuckDuckGo twice every two seconds from
the same IP, and DDGS bans by IP. The lock is a Postgres session-level
advisory lock, so every process on every machine that shares the database
takes its turn, and a crashed holder releases it when its connection drops.

Why ride out rate limits in-process: a `RetryLaterError` re-queues the task,
and every re-queue spends one of the queue's attempts. Triage of a big job
makes thousands of calls; a handful of 429s over two hours is normal and
must not dead-letter the job. Pausing inside the task (the lease heartbeat
keeps running) costs nothing but wall-clock.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import TypeVar

import httpx
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.enrichment.sources import SearchFn, SearchHit
from app.queue.registry import RetryLaterError

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Advisory lock id for "a DDGS call is in flight". Arbitrary but fixed:
#: every process must agree on it. ASCII "ERDDGS".
SEARCH_LOCK_KEY = 0x455244444753

LockFactory = Callable[[], AbstractContextManager[None]]


@contextmanager
def advisory_lock(engine: Engine, key: int = SEARCH_LOCK_KEY) -> Iterator[None]:
    """Hold a Postgres session-level advisory lock for the block.

    Blocks until the lock is free. Released explicitly on exit, and by
    Postgres itself if the process dies holding it.
    """
    with engine.connect() as connection:
        connection.execute(text("SELECT pg_advisory_lock(:key)"), {"key": key})
        try:
            yield
        finally:
            connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})
            connection.commit()


class SearchThrottle:
    """Space search *starts* `interval` seconds apart, platform-wide.

    Each caller holds the lock for `interval` seconds and only then releases it
    and makes its call, so two calls anywhere never start less than `interval`
    apart: the request rate DuckDuckGo sees is capped at one per `interval`
    however many callers there are. The call itself runs outside the lock.

    That is the difference from holding the lock *through* the call, which is
    what this used to do: a search takes 3-10 s, so "one every 2 s" was really
    one every 5-12 s, and a job of thousands of schools spent days waiting on
    searches that could have overlapped. Callers that want the overlap run
    searches on a few threads (see `app/enrichment/prefetch.py`); a single
    serial caller behaves exactly as before.
    """

    def __init__(
        self,
        *,
        interval: float,
        lock: LockFactory,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval = interval
        self._lock = lock
        self._sleep = sleep

    def wrap(self, search: SearchFn) -> SearchFn:
        def throttled(query: str, max_results: int) -> list[SearchHit]:
            with self._lock():
                self._sleep(self.interval)
            return search(query, max_results)

        return throttled


def patiently(
    call: Callable[[], T],
    *,
    attempts: int = 5,
    max_pause: float = 300.0,
    sleep: Callable[[float], None] = time.sleep,
    what: str = "call",
    budget: float | None = None,
) -> T:
    """Run `call`, pausing and retrying when it raises `RetryLaterError`.

    Each pause is the provider's own Retry-After (at least 5 s), doubled per
    consecutive refusal and capped at `max_pause`. After `attempts`
    refusals in a row the error propagates, and the queue defers the task.
    With `budget` (seconds) it keeps going until the pauses add up to that
    instead: how a long outage or ban is waited out without failing the job.
    """
    waited = 0.0
    for attempt in range(10**9 if budget is not None else attempts):
        try:
            return call()
        except RetryLaterError as refused:
            pause = min(max_pause, max(refused.countdown, 5.0) * (2 ** min(attempt, 20)))
            waited += pause
            if budget is not None:
                if waited > budget:
                    raise
            elif attempt == attempts - 1:
                raise
            logger.warning("%s rate-limited (%s); pausing %.0fs", what, refused, pause)
            sleep(pause)
    raise AssertionError("unreachable")  # pragma: no cover


#: How long one LLM call may wait out a provider's rate limit before the task
#: gives up and defers. A daily token cap refills over hours, and the wait is
#: *progress being paced*, not a failure: failing the job after a few of them
#: (what routing each wait through the queue's attempt counter did) throws away
#: a day's work to save nothing. The lease heartbeat keeps the task alive.
RATE_LIMIT_BUDGET_S = 6 * 3600.0
#: How long a lost connection is waited out before the task defers, backing off
#: 30 s, 60 s, ... 300 s between tries. Counted in *time*, not tries: a count
#: of tries (8, about 27 minutes) failed a job on a two-hour Wi-Fi outage, and
#: waiting costs nothing but the wait (the lease heartbeat keeps the task alive,
#: and every finished decision is already saved).
OUTAGE_BUDGET_S = 6 * 3600.0

#: Returns True when it dealt with the wait itself (e.g. moved to another source
#: of capacity), so there is nothing to sleep for.
WaitHook = Callable[[str, float], bool | None]


def ask_patiently(
    call: Callable[[], T],
    *,
    sleep: Callable[[float], None] = time.sleep,
    on_wait: WaitHook | None = None,
    rate_limit_budget: float = RATE_LIMIT_BUDGET_S,
    outage_budget: float = OUTAGE_BUDGET_S,
) -> T:
    """Run an LLM call, waiting out rate limits and outages in-process.

    * `RetryLaterError` (a provider 429): wait exactly what the provider says
      (its Retry-After, not a guess that doubles), for up to
      `rate_limit_budget` seconds in all.
    * `httpx.HTTPError` (no connection, a timeout, a provider 5xx): back off
      and retry for up to `outage_budget` seconds in all.

    Each call has its own budgets. When one is spent the error propagates as a
    `RetryLaterError` and the queue defers the task, which is the only case
    that costs an attempt. `on_wait(reason, seconds)` runs before each wait, so
    the caller can show *why* progress has stopped, or return True to skip the
    wait because it found capacity elsewhere.
    """
    waited = 0.0
    down_for = 0.0
    tries = 0
    while True:
        try:
            return call()
        except RetryLaterError as limited:
            pause = min(max(limited.countdown, 5.0), 900.0)
            waited += pause
            if waited > rate_limit_budget:
                raise
            reason = str(limited)
        except httpx.HTTPError as down:
            tries += 1
            pause = min(30.0 * 2 ** (tries - 1), 300.0)
            down_for += pause
            if down_for > outage_budget:
                raise RetryLaterError(
                    60.0, f"LLM unreachable: {type(down).__name__}: {down}"
                ) from down
            reason = f"LLM unreachable ({type(down).__name__}); retrying"
        if on_wait is not None and on_wait(reason, pause) is True:
            waited = 0.0
            continue
        logger.warning("%s; waiting %.0fs", reason, pause)
        sleep(pause)


def polite_search(
    search: SearchFn,
    *,
    interval: float,
    lock: LockFactory,
    sleep: Callable[[float], None] = time.sleep,
) -> SearchFn:
    """`search`, started at most once per `interval` platform-wide, and patient
    with rate limits. What every Tier 2 and Tier 3 search goes through."""
    throttled = SearchThrottle(interval=interval, lock=lock, sleep=sleep).wrap(search)

    def call(query: str, max_results: int) -> list[SearchHit]:
        # A time budget, not a count: a lost connection or a ban is waited out
        # for hours (see OUTAGE_BUDGET_S) instead of failing the job after ~12 minutes.
        return patiently(
            lambda: throttled(query, max_results),
            sleep=sleep,
            what="DDGS search",
            budget=OUTAGE_BUDGET_S,
        )

    return call
