"""A running task adopts a tenant's new AI tier without restarting."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

import app.workers.tasks.enrichment as tasks
from app.core.secrets import SecretsUnavailableError
from app.enrichment.llm_config import LlmConfig, premium_config
from app.enrichment.sources import EntityFields, LlmUnusableError
from app.queue.registry import RetryLaterError

STANDARD = LlmConfig(
    provider="openai_compatible",
    base_url="https://api.groq.com/openai/v1",
    model="qwen/qwen3.8-27b",
    api_key="gsk_free",
    premium=False,
)
PREMIUM = premium_config("anthropic", "claude-haiku-5-5", "sk-ant-tenant-key-123456")


class FakeLlm:
    checks: list[str] = []
    fail: Exception | None = None

    def __init__(self, client: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def check(self) -> None:
        FakeLlm.checks.append(self.kwargs["model"])
        if FakeLlm.fail:
            raise FakeLlm.fail


@pytest.fixture(autouse=True)
def _reset() -> None:
    FakeLlm.checks, FakeLlm.fail = [], None


def _handle(latest: Any, monkeypatch: pytest.MonkeyPatch) -> tasks._LlmHandle[Any]:
    monkeypatch.setattr(
        tasks, "resolve_llm_pool", lambda session, tenant: [latest] if latest else []
    )
    return tasks._LlmHandle(httpx.Client(), FakeLlm, STANDARD)  # type: ignore[type-var,arg-type]


def test_a_new_premium_key_is_adopted_after_it_passes_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _handle(PREMIUM, monkeypatch)

    assert handle.refresh(None, "t") is True
    assert (
        handle.config == PREMIUM
        and handle.config.audit_label == "premium:anthropic/claude-haiku-5-5"
    )
    assert handle.llm.kwargs["api_key"] == "sk-ant-tenant-key-123456"
    assert FakeLlm.checks == ["claude-haiku-5-5"]


def test_unchanged_settings_cost_no_provider_call(monkeypatch: pytest.MonkeyPatch) -> None:
    handle = _handle(STANDARD, monkeypatch)

    assert handle.refresh(None, "t") is False
    assert FakeLlm.checks == []


@pytest.mark.parametrize(
    "failure",
    [LlmUnusableError("HTTP 401"), RetryLaterError(30.0, "429")],
)
def test_a_key_that_fails_the_check_never_replaces_a_working_llm(
    failure: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle = _handle(PREMIUM, monkeypatch)
    FakeLlm.fail = failure

    assert handle.refresh(None, "t") is False
    assert handle.config == STANDARD


def test_removing_every_llm_keeps_the_task_on_what_it_has(monkeypatch: pytest.MonkeyPatch) -> None:
    handle = _handle(None, monkeypatch)

    assert handle.refresh(None, "t") is False and handle.config == STANDARD


def test_an_undecryptable_stored_key_keeps_the_task_going(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(session: Any, tenant: Any) -> None:
        raise SecretsUnavailableError("key could not be decrypted")

    monkeypatch.setattr(tasks, "resolve_llm_pool", boom)
    handle = tasks._LlmHandle(httpx.Client(), FakeLlm, STANDARD)  # type: ignore[type-var,arg-type]

    assert handle.refresh(None, "t") is False and handle.config == STANDARD


def test_switching_back_to_standard_works_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tasks, "resolve_llm_pool", lambda session, tenant: [STANDARD])
    handle = tasks._LlmHandle(httpx.Client(), FakeLlm, PREMIUM)  # type: ignore[type-var,arg-type]

    assert handle.refresh(None, "t") is True and handle.config.audit_label.startswith("standard:")


# --- Patience: rate limits and outages ----------------------------------------------------


from app.enrichment.sources import rate_limit_reason  # noqa: E402
from app.enrichment.throttle import ask_patiently  # noqa: E402


def _script(*outcomes: Any) -> Any:
    steps = iter(outcomes)

    def call() -> Any:
        outcome = next(steps)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return call


def test_a_rate_limit_waits_what_the_provider_says_not_a_doubling_guess() -> None:
    waits: list[float] = []
    # Far more consecutive 429s than the old five-attempt limit allowed.
    call = _script(*[RetryLaterError(88.0, "daily limit reached")] * 12, "ok")

    assert ask_patiently(call, sleep=waits.append) == "ok"
    assert waits == [88.0] * 12


def test_a_rate_limit_that_never_clears_eventually_defers_the_task() -> None:
    call = _script(*[RetryLaterError(600.0, "daily limit reached")] * 100)

    with pytest.raises(RetryLaterError):
        ask_patiently(call, sleep=lambda s: None, rate_limit_budget=3600.0)


def test_a_success_resets_the_rate_limit_budget() -> None:
    call = _script(*[RetryLaterError(600.0, "x")] * 5, "a", *[RetryLaterError(600.0, "x")] * 5, "b")

    first = ask_patiently(call, sleep=lambda s: None, rate_limit_budget=3600.0)
    second = ask_patiently(call, sleep=lambda s: None, rate_limit_budget=3600.0)

    assert (first, second) == ("a", "b")


def test_a_network_outage_is_ridden_out_for_about_twenty_five_minutes() -> None:
    """The overnight failure: `getaddrinfo failed` burned all four task
    attempts in under four minutes and failed the job."""
    waits: list[float] = []
    down = httpx.ConnectError("[Errno 11001] getaddrinfo failed")
    call = _script(*[down] * 8, "back")

    assert ask_patiently(call, sleep=waits.append) == "back"
    assert waits == [30, 60, 120, 240, 300, 300, 300, 300]
    assert sum(waits) / 60 == pytest.approx(27.5)


def test_an_outage_that_outlasts_the_budget_defers_instead_of_failing() -> None:
    call = _script(*[httpx.ConnectError("down")] * 100)

    with pytest.raises(RetryLaterError, match="unreachable"):
        ask_patiently(call, sleep=lambda s: None, outage_budget=1000.0)


def test_a_two_hour_internet_outage_is_waited_out_not_failed() -> None:
    """The real incident (25 Oct): the connection dropped at about 12:00 and
    came back after about two hours. Counting tries gave up after 27 minutes,
    three times over, and failed the job; counting time does not."""
    waits: list[float] = []
    outage = [httpx.ConnectError("[Errno 11001] getaddrinfo failed")] * 30  # 30 x 5 min and more

    result = ask_patiently(_script(*outage, "back online"), sleep=waits.append)

    assert result == "back online"
    assert sum(waits) >= 2 * 3600, "it really did wait through two hours"
    assert max(waits) == 300.0, "never backing off further than five minutes"


def test_a_search_outage_is_waited_out_for_hours_too() -> None:
    """The same outage hit the web search first; it gave up after ~12 minutes."""
    from app.enrichment.throttle import patiently

    waits: list[float] = []
    call = _script(*[RetryLaterError(60.0, "search unavailable")] * 40, "results")

    assert patiently(call, sleep=waits.append, budget=6 * 3600.0) == "results"
    assert sum(waits) >= 2 * 3600


def test_a_search_outage_longer_than_the_budget_still_defers() -> None:
    from app.enrichment.throttle import patiently

    call = _script(*[RetryLaterError(60.0, "banned")] * 1000)

    with pytest.raises(RetryLaterError):
        patiently(call, sleep=lambda s: None, budget=3600.0)


def test_each_wait_says_why() -> None:
    reasons: list[tuple[str, float]] = []
    call = _script(RetryLaterError(30.0, "daily limit reached"), httpx.ReadTimeout("slow"), "ok")

    ask_patiently(call, sleep=lambda s: None, on_wait=lambda r, s: reasons.append((r, s)))

    assert reasons[0] == ("daily limit reached", 30.0)
    assert "ReadTimeout" in reasons[1][0] and reasons[1][1] == 30.0


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("Rate limit reached ... on tokens per day (TPD): Limit 200000", "daily limit reached"),
        ("... on requests per day (RPD) ...", "daily limit reached"),
        ("... on tokens per minute (TPM) ...", "per-minute limit reached"),
        ("Too many requests", "rate limit reached"),
    ],
)
def test_a_429_names_the_limit_that_was_hit(body: str, expected: str) -> None:
    assert rate_limit_reason(body) == expected


# --- The handle: status line and live switching during a wait --------------------------


class _Session:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def _waiting_handle(monkeypatch: pytest.MonkeyPatch, config: LlmConfig = STANDARD) -> Any:
    written: list[tuple[str, dict[str, Any]]] = []

    def record(session: Any, job: str, stage: str, *, done: int, details: dict[str, Any]) -> None:
        written.append((stage, details))

    monkeypatch.setattr(tasks, "record_progress", record)
    monkeypatch.setattr(tasks.time, "sleep", lambda s: None)
    handle = tasks._LlmHandle(
        httpx.Client(),
        FakeLlm,
        config,
        session=_Session(),
        tenant_id="t",
        job_id="j",
        stage="triage",  # type: ignore[type-var,arg-type]
    )
    handle.written = written
    return handle


def test_a_wait_is_reported_on_the_job_and_cleared_when_it_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tasks, "resolve_llm_pool", lambda session, tenant: [STANDARD])
    handle = _waiting_handle(monkeypatch)

    result = handle.ask(_script(RetryLaterError(20.0, "daily limit reached"), "decided"))

    assert result == "decided"
    first, last = handle.written[0], handle.written[-1]
    assert first[0] == "triage_waiting"
    assert first[1]["waiting"] == "Standard AI: daily limit reached"
    assert first[1]["premium"] is False and first[1]["until"]
    assert last[1] == {"waiting": None}


def test_a_failing_status_write_never_fails_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tasks, "resolve_llm_pool", lambda session, tenant: [STANDARD])
    handle = _waiting_handle(monkeypatch)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("db hiccup")

    monkeypatch.setattr(tasks, "record_progress", boom)

    assert handle.ask(_script(RetryLaterError(20.0, "x"), "decided")) == "decided"
    assert handle._session.rollbacks >= 1


def test_a_key_saved_mid_wait_is_adopted_at_once_and_ends_the_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The user saves a Premium key while the worker sleeps through a
    five-minute pause: it must not sleep the other four and a half."""
    handle = _waiting_handle(monkeypatch)
    monkeypatch.setattr(tasks, "resolve_llm_pool", lambda session, tenant: [PREMIUM])
    slept: list[float] = []
    monkeypatch.setattr(tasks.time, "sleep", slept.append)

    handle._sleep(300.0)

    assert handle.config == PREMIUM
    assert slept == [15.0], "woke after one slice and switched"


def test_without_a_change_the_whole_wait_is_slept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tasks, "resolve_llm_pool", lambda session, tenant: [STANDARD])
    handle = _waiting_handle(monkeypatch)
    slept: list[float] = []
    monkeypatch.setattr(tasks.time, "sleep", slept.append)

    handle._sleep(40.0)

    assert slept == [15.0, 15.0, 10.0]


def test_settings_the_worker_cannot_read_never_kill_a_running_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E.g. a provider this worker's code does not know yet: the old code
    raised an AssertionError out of the task and burned an attempt."""

    def unknown_provider(session: Any, tenant: Any) -> None:
        raise AssertionError("provider 'newcomer' is not in PROVIDERS")

    monkeypatch.setattr(tasks, "resolve_llm_pool", unknown_provider)
    handle = tasks._LlmHandle(httpx.Client(), FakeLlm, STANDARD)  # type: ignore[type-var,arg-type]

    assert handle.refresh(None, "t") is False and handle.config == STANDARD


def test_a_wait_names_who_is_answering(monkeypatch: pytest.MonkeyPatch) -> None:
    cerebras = premium_config("cerebras", "qwen-3.8-27b", "csk-tenant-secret-123456")
    monkeypatch.setattr(tasks, "resolve_llm_pool", lambda session, tenant: [cerebras])
    handle = _waiting_handle(monkeypatch, cerebras)

    handle.ask(_script(RetryLaterError(20.0, "daily limit reached"), "ok"))

    note = handle.written[0][1]
    assert note["waiting"] == "Cerebras: daily limit reached" and note["premium"] is True


# --- The model pool: several free daily budgets on one key -------------------------------

from dataclasses import replace  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.enrichment.llm_config import standard_pool  # noqa: E402
from app.enrichment.sources import LlmJudge  # noqa: E402


def _pool_handle(monkeypatch: pytest.MonkeyPatch, *models: str) -> Any:
    slept: list[float] = []
    monkeypatch.setattr(tasks.time, "sleep", slept.append)
    pool = [replace(STANDARD, model=m) for m in models]
    handle = tasks._LlmHandle(httpx.Client(), FakeLlm, pool)  # type: ignore[type-var,arg-type]
    handle.slept = slept
    return handle


def test_a_daily_limit_moves_to_the_next_model_without_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _pool_handle(monkeypatch, "qwen", "gpt-oss-120b", "gpt-oss-20b")
    tried: list[str] = []

    def call() -> str:
        model = handle.llm.kwargs["model"]
        tried.append(model)
        if model == "qwen":
            raise RetryLaterError(90.0, "daily limit reached")
        return f"answered by {model}"

    assert handle.ask(call) == "answered by gpt-oss-120b"
    assert tried == ["qwen", "gpt-oss-120b"]
    assert handle.slept == [], "another model had budget, so nothing was waited for"
    assert handle.config.model == "gpt-oss-120b", "and it stays there until that one runs out"


def test_the_pool_only_sleeps_when_every_model_is_cooling_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _pool_handle(monkeypatch, "a", "b")
    attempts = {"n": 0}

    def call() -> str:
        attempts["n"] += 1
        if attempts["n"] <= 3:  # a, then b, then a again: all cooling
            raise RetryLaterError(30.0, "daily limit reached")
        return "ok"

    assert handle.ask(call) == "ok"
    assert handle.slept, "with nowhere left to go it waited like a single model"


def test_a_per_minute_limit_never_rotates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a *daily* limit means this model is spent; a per-minute one clears
    in seconds, and hopping models on it would scatter one job across all of them."""
    handle = _pool_handle(monkeypatch, "a", "b")
    attempts = {"n": 0}

    def call() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RetryLaterError(20.0, "per-minute limit reached")
        return handle.llm.kwargs["model"]

    assert handle.ask(call) == "a"
    assert handle.slept


def test_a_hook_that_finds_capacity_skips_the_wait_and_resets_the_budget() -> None:
    sleeps: list[float] = []
    call = _script(*[RetryLaterError(600.0, "daily limit reached")] * 40, "ok")

    result = ask_patiently(
        call, sleep=sleeps.append, on_wait=lambda reason, seconds: True, rate_limit_budget=1000.0
    )

    assert result == "ok" and sleeps == [], "40 x 600s would have blown a 1000s budget"


def test_a_saved_premium_key_replaces_the_whole_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    handle = _pool_handle(monkeypatch, "a", "b")
    monkeypatch.setattr(tasks, "resolve_llm_pool", lambda session, tenant: [PREMIUM])

    assert handle.refresh(None, "t") is True
    assert handle.config == PREMIUM
    assert handle._pool == [PREMIUM], "your own key never shares a task with the free pool"


def _settings(**overrides: Any) -> Settings:
    base = {
        "llm_base_url": "https://api.groq.com/openai/v1",
        "llm_api_key": "gsk_x",
        "llm_model": "qwen/qwen3.8-27b",
    }
    return Settings(_env_file=None, **{**base, **overrides})  # type: ignore[arg-type]


def test_the_server_pool_is_the_primary_model_then_the_extras_in_order() -> None:
    pool = standard_pool(
        _settings(llm_pool_models="openai/gpt-oss-120b, openai/gpt-oss-20b ,qwen/qwen3.8-27b")
    )

    assert [c.model for c in pool] == [
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ], "blanks trimmed, the primary model not repeated"
    assert all(c.api_key == "gsk_x" and not c.premium for c in pool)
    assert standard_pool(_settings(llm_base_url=None)) == []


def test_a_reasoning_model_gets_low_effort_and_room_to_think() -> None:
    qwen, oss = standard_pool(_settings(llm_pool_models="openai/gpt-oss-120b"))

    assert (qwen.reply_tokens, dict(qwen.extra)) == (3, {})
    assert (oss.reply_tokens, dict(oss.extra)) == (400, {"reasoning_effort": "low"})


def test_the_reasoning_models_request_is_what_was_measured_to_work() -> None:
    """gpt-oss with the old 3-token budget answered nothing at all; this is the
    request that returned a correct NO in about 525 tokens."""
    seen: list[httpx.Request] = []
    handler = lambda request: (  # noqa: E731
        seen.append(request)
        or httpx.Response(200, json={"choices": [{"message": {"content": "NO"}}]})
    )
    oss = standard_pool(_settings(llm_pool_models="openai/gpt-oss-20b"))[1]

    verdict = LlmJudge(
        httpx.Client(transport=httpx.MockTransport(handler)), **oss.client_kwargs()
    ).same_entity(
        EntityFields("1", name="army public school"),
        [],
        EntityFields("2", name="army public school"),
        [],
    )

    body = json.loads(seen[0].content)
    assert verdict is False
    assert (body["model"], body["reasoning_effort"], body["max_tokens"]) == (
        "openai/gpt-oss-20b",
        "low",
        400,
    )


def test_a_stale_wait_note_from_a_killed_process_is_cleared_by_the_first_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tasks, "resolve_llm_pool", lambda session, tenant: [STANDARD])
    handle = _waiting_handle(monkeypatch)

    assert handle.ask(_script("fine")) == "fine"

    assert handle.written == [("triage_waiting", {"waiting": None})]
