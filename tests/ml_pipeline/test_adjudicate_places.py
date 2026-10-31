"""scripts/adjudicate_places.py: the answer parser, and the guards on spending.

No network and no model: the LLM is a stub with a `chat` method.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest

from app.enrichment.sources import LlmUnusableError
from app.queue.registry import RetryLaterError

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "adjudicate_places.py"
_spec = importlib.util.spec_from_file_location("adjudicate_places", _SCRIPT)
assert _spec and _spec.loader
ap = importlib.util.module_from_spec(_spec)
sys.modules["adjudicate_places"] = ap
_spec.loader.exec_module(ap)  # scripts/ is not a package


class Stub:
    def __init__(self, *replies, delay=0.0):
        self.replies = list(replies)
        self.delay = delay  # a real call takes time; an instant stub finishes before any abort
        self.calls = 0

    def chat(self, system, user, *, json_mode, max_tokens=None):
        self.calls += 1
        time.sleep(self.delay)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("YES - same school, abbreviation only", ("YES", "same school, abbreviation only")),
        ("no - different town", ("NO", "different town")),
        ("UNSURE", ("UNSURE", "")),
        ("**YES** - same", ("YES", "same")),
        ("Probably the same school", ("UNSURE", "unparseable reply")),
        ("", ("UNSURE", "unparseable reply")),
    ],
)
def test_parse_answer(text, expected):
    assert ap.parse_answer(text) == expected


def test_only_a_clear_yes_or_no_decides():
    assert ap.final_verdict("YES") == "confirmed_by_ai"
    assert ap.final_verdict("NO") == "different_school"
    assert ap.final_verdict("UNSURE") == "review"


def test_prompt_carries_both_records_and_the_rule_about_towns():
    ours = {"addr1": "Sector 16", "addr2": "", "locality": "Rohini, Delhi", "pin": "110085"}
    row = {"name": "DPS", "google_name": "Delhi Public School", "google_address": "Rohini 110085", "google_status": "OPERATIONAL"}
    text = ap.prompt_for(ours, row)
    assert "Sector 16, Rohini, Delhi" in text and "Delhi Public School" in text and "different town" in text


def test_a_rate_limit_is_waited_out(monkeypatch):
    monkeypatch.setattr(ap.time, "sleep", lambda s: None)
    llm = Stub(RetryLaterError(5, "rate"), "YES - ok")
    assert ap.ask(llm, "q") == {"answer": "YES", "reason": "ok"}
    assert llm.calls == 2


def test_an_unreadable_reply_is_a_failure_not_a_cached_verdict(tmp_path):
    with pytest.raises(ValueError, match="unreadable"):
        ap.ask(Stub(""), "q")
    cache: dict = {}
    llm = Stub("", delay=0.02)  # a model that returns no text (it spent its tokens thinking)
    with pytest.raises(ap.AdjudicateAbort, match="in a row failed"):
        ap.adjudicate(llm, {f"k{i}": f"q{i}" for i in range(100)}, cache, tmp_path / "c.jsonl", 100, 2)
    assert cache == {} and llm.calls < 100


def test_cap_cache_and_resume(tmp_path):
    prompts = {f"k{i}": f"q{i}" for i in range(10)}
    cache: dict = {}
    path = tmp_path / "c.jsonl"
    llm = Stub("NO - different")
    ap.adjudicate(llm, prompts, cache, path, max_calls=4, workers=3)
    assert llm.calls == 4 and len(cache) == 4
    again = ap.load_cache(path)
    assert set(again) == set(cache)
    ap.adjudicate(llm, prompts, again, path, max_calls=100, workers=3)
    assert llm.calls == 10  # only the six it did not have


def test_an_unusable_model_stops_the_run(tmp_path):
    llm = Stub(LlmUnusableError("HTTP 401"), delay=0.02)
    with pytest.raises(ap.AdjudicateAbort, match="unusable"):
        ap.adjudicate(llm, {f"k{i}": f"q{i}" for i in range(50)}, {}, tmp_path / "c.jsonl", 50, 4)
    assert llm.calls < 50
