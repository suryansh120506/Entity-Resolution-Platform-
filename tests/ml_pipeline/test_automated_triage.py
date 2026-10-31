"""Automated triage (Tier 2) and universal golden enrichment (Tier 3), with no
network and no live LLM.

The property that matters most is negative, as in test_enrichment.py: no
value the LLM returns reaches a golden record unless it is verbatim in a
snippet that is about this entity.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from typing import Any

import httpx
import pytest

from app.enrichment.golden import enrich_entity, merge_into_golden
from app.enrichment.grounding import VerifiedValue, occurs_verbatim, verify_extraction
from app.enrichment.sources import (
    EntityFields,
    LlmExtractor,
    LlmJudge,
    SearchHit,
    parse_yes_no,
)
from app.enrichment.throttle import SearchThrottle, patiently, polite_search
from app.enrichment.triage import triage_cluster
from app.ontology import OntologyEngine
from app.ontology.engine import (
    EnrichmentTarget,
    GoldenEnrichmentRules,
    OntologySpec,
    TriageRules,
    render_query,
)
from app.queue.registry import RetryLaterError

GENERIC = ["school", "public", "senior", "secondary", "government", "number"]
SCHOOL = EntityFields(
    record_id="r1",
    name="nalanda senior secondary school",
    address="sector 15",
    district="faridabad",
    state="haryana",
    pincode="121004",
)


def _targets(*items: Any) -> list[EnrichmentTarget]:
    return [EnrichmentTarget.model_validate(item) for item in items]


def _hit(title: str, snippet: str, url: str = "https://example.org/page") -> SearchHit:
    return SearchHit(title=title, url=url, snippet=snippet)


ABOUT_NALANDA = _hit(
    "Nalanda Senior Secondary School, Faridabad",
    "Principal: Dr. Anita Sharma. Call +91 98765 43210 or write to info@nalandaschool.in.",
    url="https://nalandaschool.in/contact",
)


# --- Ontology: enrichment_targets ------------------------------------------------------


def test_a_bare_target_key_gets_its_type_from_its_name() -> None:
    targets = _targets("principal_name", "ceo_name", "website", "contact_email", "phone", "board")

    assert [t.type for t in targets] == ["person", "person", "url", "email", "phone", "text"]


def test_a_target_may_declare_its_type_and_meaning() -> None:
    (target,) = _targets({"key": "revenue", "type": "number", "description": "annual revenue"})

    assert (target.key, target.type, target.description) == ("revenue", "number", "annual revenue")


def test_a_target_may_not_overwrite_an_identity_field() -> None:
    with pytest.raises(ValueError, match="identity field"):
        _targets("name")


def test_a_search_template_may_only_use_canonical_fields() -> None:
    with pytest.raises(ValueError, match="principal"):
        TriageRules(query='"{name}" {principal}')


def test_every_shipped_ontology_loads_with_its_targets() -> None:
    corporate = OntologyEngine.load("corporate").spec
    healthcare = OntologyEngine.load("healthcare").spec
    education = OntologyEngine.load("education").spec

    assert {"ceo_name", "revenue"} <= {t.key for t in corporate.enrichment_targets}
    assert "bed_count" in {t.key for t in healthcare.enrichment_targets}
    assert "principal_name" in {t.key for t in education.enrichment_targets}
    assert OntologyEngine.load("generic").spec.enrichment_targets == []


def test_duplicate_targets_are_refused() -> None:
    raw = OntologyEngine.load("generic").spec.model_dump(by_alias=True)
    raw["enrichment_targets"] = ["website", "website"]

    with pytest.raises(ValueError, match="more than once"):
        OntologySpec.model_validate(raw)


def test_editing_targets_does_not_make_silver_rows_stale() -> None:
    """The version marks Silver output; what to enrich is not Silver output."""
    spec = OntologyEngine.load("education").spec
    edited = spec.model_copy(update={"enrichment_targets": _targets("website")})

    assert OntologyEngine(edited).version == OntologyEngine(spec).version


def test_a_query_renders_without_empty_quotes_or_none() -> None:
    query = render_query('"{name}" {district} {state} school', {"name": "", "state": "delhi"})

    assert query == "delhi school"


# --- The hallucination check -------------------------------------------------------------


TARGETS = _targets("principal_name", "phone", "email", "website", "board")


def _verify(extracted: dict[str, Any], hits: list[SearchHit] | None = None) -> tuple[Any, Any]:
    return verify_extraction(TARGETS, extracted, hits or [ABOUT_NALANDA], SCHOOL, GENERIC)


def test_values_found_verbatim_in_a_snippet_about_the_entity_are_kept() -> None:
    verified, rejected = _verify(
        {
            "principal_name": {"value": "Dr. Anita Sharma", "snippet_index": 0},
            "phone": "+91 98765 43210",
            "email": "info@nalandaschool.in",
        }
    )

    assert rejected == []
    assert verified["principal_name"].value == "Dr. Anita Sharma"
    assert verified["phone"].url == "https://nalandaschool.in/contact"
    assert "Anita Sharma" in verified["principal_name"].evidence


def test_an_invented_value_is_rejected() -> None:
    verified, rejected = _verify({"principal_name": "Dr. Rajesh Kumar"})

    assert verified == {}
    assert [(r.key, r.reason) for r in rejected] == [("principal_name", "not_in_snippets")]


def test_a_reformatted_value_is_not_verbatim() -> None:
    """The snippet says '+91 98765 43210'; the LLM 'tidied' it."""
    verified, rejected = _verify({"phone": "9876543210"})

    assert verified == {}
    assert rejected[0].reason == "not_in_snippets"


def test_a_value_must_sit_on_word_boundaries() -> None:
    hit = _hit("Nalanda Senior Secondary School", "Principal Ramesh Gupta, Nalanda school")

    verified, rejected = _verify({"principal_name": "Ram Gupta"}, [hit])
    assert verified == {}

    verified, _ = _verify({"principal_name": "Ramesh Gupta"}, [hit])
    assert verified["principal_name"].value == "Ramesh Gupta"


def test_a_value_stitched_across_title_and_snippet_is_not_verbatim() -> None:
    hit = _hit("Nalanda School principal Anita", "Sharma joined in 2019. Nalanda, Faridabad.")

    verified, rejected = _verify({"principal_name": "Anita Sharma"}, [hit])

    assert verified == {}
    assert rejected[0].reason == "not_in_snippets"


def test_case_and_spacing_do_not_count_as_changes() -> None:
    verified, _ = _verify({"principal_name": "DR.  ANITA   SHARMA"})

    assert "principal_name" in verified


def test_a_value_copied_from_another_entitys_snippet_is_rejected() -> None:
    """Search results mix schools: verbatim, but not this school's principal."""
    other = _hit(
        "Modern Public School, Delhi", "Principal: Dr. Meera Iyer of Modern Public School."
    )

    verified, rejected = _verify({"principal_name": "Dr. Meera Iyer"}, [ABOUT_NALANDA, other])

    assert verified == {}
    assert rejected[0].reason == "not_attributed"


def test_an_all_generic_name_is_attributed_by_pin_or_district() -> None:
    entity = EntityFields(
        "r2",
        name="government senior secondary school number 1",
        district="faridabad",
        state="haryana",
        pincode="121004",
    )
    elsewhere = _hit("Government Senior Secondary School No. 1", "Principal: Sunil Rao. Rohtak.")
    here = _hit("Government Senior Secondary School No. 1", "Principal: Sunil Rao, Faridabad.")
    (target,) = _targets("principal_name")

    assert (
        verify_extraction([target], {"principal_name": "Sunil Rao"}, [elsewhere], entity, GENERIC)[
            0
        ]
        == {}
    )
    assert (
        "principal_name"
        in verify_extraction([target], {"principal_name": "Sunil Rao"}, [here], entity, GENERIC)[0]
    )


def test_echoing_the_entitys_own_name_is_rejected() -> None:
    verified, rejected = _verify({"board": "Nalanda Senior Secondary School"})

    assert verified == {}
    assert rejected[0].reason == "echoes_entity"


@pytest.mark.parametrize(
    ("key", "value"),
    [("email", "write to us"), ("phone", "call the office"), ("principal_name", "12345")],
)
def test_a_value_of_the_wrong_shape_is_rejected_before_grounding(key: str, value: str) -> None:
    hit = _hit("Nalanda Senior Secondary School", f"{value} Nalanda Faridabad")

    verified, rejected = _verify({key: value}, [hit])

    assert verified == {}
    assert rejected[0].reason == "wrong_shape"


@pytest.mark.parametrize("nothing", [None, "", "null", "N/A", "Not found", "unknown"])
def test_not_found_is_not_a_rejection(nothing: Any) -> None:
    assert _verify({"principal_name": nothing}) == ({}, [])


def test_keys_the_llm_added_are_never_read() -> None:
    verified, rejected = _verify({"principal_name": "Dr. Anita Sharma", "bank_account": "123"})

    assert set(verified) == {"principal_name"}
    assert rejected == []


def test_a_website_may_be_grounded_in_the_results_own_link() -> None:
    verified, _ = _verify({"website": "nalandaschool.in"})

    assert verified["website"].url == "https://nalandaschool.in/contact"


def test_verbatim_matching_respects_edges_only_where_the_value_has_them() -> None:
    assert occurs_verbatim("+91 98765 43210", "call+91 98765 43210.")
    assert not occurs_verbatim("ram", "ramesh")


# --- Tier 3 end to end, with fakes ----------------------------------------------------


def test_golden_enrichment_keeps_only_verified_values_for_any_domain() -> None:
    """A corporate ontology's targets run through the same code as a school's."""
    company = EntityFields(
        "c1", name="acme widgets private limited", district="pune", state="maharashtra"
    )
    hit = _hit(
        "Acme Widgets Pvt Ltd | Pune",
        "Acme Widgets, led by CEO Priya Nair, reported revenue of Rs 120 crore in FY24.",
    )
    seen: dict[str, Any] = {}

    def extract(keys: Any, subject: str, hits: Any, meanings: Any) -> dict[str, Any]:
        seen.update(keys=list(keys), meanings=dict(meanings))
        return {"ceo_name": "Priya Nair", "revenue": "Rs 120 crore", "employee_count": "5000"}

    result = enrich_entity(
        company,
        targets=_targets(
            "ceo_name",
            {"key": "revenue", "type": "number", "description": "annual revenue"},
            {"key": "employee_count", "type": "number"},
        ),
        rules=GoldenEnrichmentRules(query='"{name}" {district}'),
        generic_tokens=["private", "limited"],
        search=lambda query, n: [hit],
        extract=extract,
    )

    assert result.status == "found"
    assert {k: v.value for k, v in result.fields.items()} == {
        "ceo_name": "Priya Nair",
        "revenue": "Rs 120 crore",
    }
    assert [(r.key, r.reason) for r in result.rejected] == [("employee_count", "not_in_snippets")]
    assert seen["meanings"]["revenue"] == "annual revenue"


def test_no_search_results_never_reaches_the_llm() -> None:
    def extract(*args: Any) -> dict[str, Any]:
        raise AssertionError("the LLM must not be asked about nothing")

    result = enrich_entity(
        SCHOOL,
        targets=TARGETS,
        rules=GoldenEnrichmentRules(),
        generic_tokens=GENERIC,
        search=lambda query, n: [],
        extract=extract,
    )
    assert result.status == "not_found"


def test_web_values_never_overwrite_what_the_golden_record_has() -> None:
    found = {
        "website": VerifiedValue("https://web.example", "u", "e"),
        "board": VerifiedValue("CBSE", "u", "e"),
    }

    merged = merge_into_golden({"website": "https://steward.example"}, found)

    assert merged == {"website": "https://steward.example", "board": "CBSE"}


# --- LLM clients ----------------------------------------------------------------------


def _llm_client(reply: str, captured: list[dict[str, Any]]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_the_judge_asks_for_yes_or_no_and_nothing_else() -> None:
    captured: list[dict[str, Any]] = []
    judge = LlmJudge(_llm_client("YES", captured), base_url="http://llm", model="m", api_key=None)

    other = EntityFields("r2", name="nalanda sr sec school", district="faridabad")
    assert judge.same_entity(SCHOOL, [ABOUT_NALANDA], other, []) is True

    body = captured[0]
    assert body["temperature"] == 0 and body["max_tokens"] == 3
    assert "response_format" not in body
    assert "Reply ONLY with YES or NO" in body["messages"][0]["content"]
    assert "(no search results)" in body["messages"][1]["content"]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("YES", True),
        ("no", False),
        (" Yes. ", True),
        ("NO\n", False),
        ("Yes, because the address matches", None),
        ("Probably", None),
        ("", None),
    ],
)
def test_only_a_bare_yes_or_no_is_an_answer(reply: str, expected: bool | None) -> None:
    assert parse_yes_no(reply) is expected


def test_extraction_returns_exactly_the_requested_keys() -> None:
    reply = json.dumps({"website": {"value": "x.in"}, "injected": "evil"})
    captured: list[dict[str, Any]] = []
    llm = LlmExtractor(_llm_client(reply, captured), base_url="http://llm", model="m", api_key=None)

    out = llm.extract(["website", "board"], "Nalanda", [ABOUT_NALANDA], {"board": "exam board"})

    assert out == {"website": {"value": "x.in"}, "board": None}
    assert captured[0]["response_format"] == {"type": "json_object"}
    assert "- board: exam board" in captured[0]["messages"][1]["content"]


# --- Tier 2 decisions -----------------------------------------------------------------


def _member(record_id: str) -> EntityFields:
    return EntityFields(record_id, name=f"school {record_id}", district="faridabad")


def _triage(answers: dict[str, bool | None], members: int = 1, **rules: Any) -> Any:
    searched: list[str] = []

    def search(query: str, n: int) -> list[SearchHit]:
        searched.append(query)
        return [_hit(query, "a result")]

    def judge(a: Any, hits_a: Any, b: EntityFields, hits_b: Any) -> bool | None:
        return answers[b.record_id]

    decision = triage_cluster(
        _member("m"),
        [_member(f"o{i}") for i in range(members)],
        rules=TriageRules(**rules),
        search=search,
        judge=judge,
    )
    return decision, searched


def test_a_yes_merges_the_pair() -> None:
    decision, searched = _triage({"o0": True})

    assert decision.decision == "MERGE"
    assert decision.kept == {"m", "o0"} and decision.excluded == frozenset()
    assert len(searched) == 2


def test_a_no_keeps_the_pair_apart() -> None:
    decision, _ = _triage({"o0": False})

    assert decision.decision == "REJECT"
    assert decision.excluded == {"o0"}


def test_an_unparseable_reply_keeps_the_pair_apart() -> None:
    decision, _ = _triage({"o0": None})

    assert decision.decision == "REJECT"
    assert decision.verdicts[0].reason == "llm_unparseable"


def test_mixed_verdicts_split_the_cluster() -> None:
    decision, searched = _triage({"o0": True, "o1": False, "o2": True}, members=3)

    assert decision.decision == "SPLIT"
    assert decision.kept == {"m", "o0", "o2"} and decision.excluded == {"o1"}
    assert len(searched) == 4, "the master is searched once, not once per pair"


def test_members_past_the_cap_are_separated_unverified() -> None:
    decision, searched = _triage({"o0": True, "o1": True}, members=3, max_members_checked=2)

    assert decision.excluded == {"o2"}
    assert decision.verdicts[-1].reason == "over_member_cap"
    assert len(searched) == 3


def test_with_no_web_evidence_the_llm_is_not_asked() -> None:
    def judge(*args: Any) -> bool:
        raise AssertionError("nothing to read, so nothing to ask")

    decision = triage_cluster(
        _member("m"), [_member("o0")], rules=TriageRules(), search=lambda q, n: [], judge=judge
    )

    assert decision.decision == "REJECT"
    assert decision.verdicts[0].reason == "no_web_evidence"


def test_a_decision_records_its_evidence() -> None:
    decision, _ = _triage({"o0": True})
    audit = decision.as_json()

    assert audit["decision"] == "MERGE"
    assert audit["verdicts"][0]["verdict"] == "YES"
    assert audit["verdicts"][0]["urls"] == ["https://example.org/page"]


# --- Rate-limit safety ----------------------------------------------------------------


def test_the_interval_is_held_inside_the_lock_before_the_call_runs_outside_it() -> None:
    """Starts are spaced by the lock; the search itself does not hold it, so
    searches can overlap without ever starting closer than `interval`."""
    events: list[str] = []

    @contextmanager
    def lock() -> Any:
        events.append("lock")
        yield
        events.append("unlock")

    def search(query: str, n: int) -> list[SearchHit]:
        events.append(f"search {query}")
        return []

    throttled = SearchThrottle(interval=2.0, lock=lock, sleep=lambda s: events.append(f"sleep {s}"))
    throttled.wrap(search)("a", 3)
    throttled.wrap(search)("b", 3)

    assert events == [
        "lock", "sleep 2.0", "unlock", "search a",
        "lock", "sleep 2.0", "unlock", "search b",
    ]  # fmt: skip


def test_a_failed_search_is_not_retried_faster_than_the_interval() -> None:
    sleeps: list[float] = []

    @contextmanager
    def lock() -> Any:
        yield

    def broken(query: str, n: int) -> list[SearchHit]:
        raise RuntimeError("boom")

    throttled = SearchThrottle(interval=2.0, lock=lock, sleep=sleeps.append).wrap(broken)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            throttled("q", 1)
    assert sleeps == [2.0, 2.0]


def test_the_lock_serialises_concurrent_searches() -> None:
    gate = threading.Lock()
    inside = 0
    overlap = False

    @contextmanager
    def lock() -> Any:
        with gate:
            yield

    def search(query: str, n: int) -> list[SearchHit]:
        nonlocal inside, overlap
        inside += 1
        overlap = overlap or inside > 1
        inside -= 1
        return []

    throttled = SearchThrottle(interval=0.01, lock=lock).wrap(search)
    threads = [threading.Thread(target=throttled, args=(str(i), 1)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not overlap


def test_rate_limits_are_ridden_out_with_growing_pauses() -> None:
    pauses: list[float] = []
    calls = iter([RetryLaterError(10.0), RetryLaterError(10.0), "ok"])

    def call() -> str:
        outcome = next(calls)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert patiently(call, sleep=pauses.append) == "ok"
    assert pauses == [10.0, 20.0]


def test_a_persistent_rate_limit_defers_the_task() -> None:
    def call() -> str:
        raise RetryLaterError(120.0, "banned")

    with pytest.raises(RetryLaterError):
        patiently(call, attempts=3, sleep=lambda s: None)


def test_polite_search_composes_spacing_and_patience() -> None:
    sleeps: list[float] = []
    attempts = iter([RetryLaterError(5.0), [ABOUT_NALANDA]])

    @contextmanager
    def lock() -> Any:
        yield

    def search(query: str, n: int) -> list[SearchHit]:
        outcome = next(attempts)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    hits = polite_search(search, interval=2.0, lock=lock, sleep=sleeps.append)("q", 3)

    assert hits == [ABOUT_NALANDA]
    assert sleeps == [2.0, 5.0, 2.0], "interval before each call, back-off between them"


# --- The LLM startup check ------------------------------------------------------------


def _scripted_client(status: int, body: Any) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(status, json=body))
    )


def test_a_retired_model_fails_the_check_with_the_providers_message() -> None:
    from app.enrichment.sources import LlmUnusableError

    error = {"error": {"message": "The model `old` does not exist", "code": "model_not_found"}}
    judge = LlmJudge(_scripted_client(404, error), base_url="http://llm", model="old", api_key="k")

    with pytest.raises(LlmUnusableError, match="model_not_found"):
        judge.check()


def test_a_model_that_answers_nothing_fails_the_check() -> None:
    """Reasoning models burn the 3-token cap thinking and reply ''. Read as
    'no answer', that would quietly separate every cluster."""
    from app.enrichment.sources import LlmUnusableError

    empty = {"choices": [{"message": {"content": ""}}]}
    judge = LlmJudge(_scripted_client(200, empty), base_url="http://llm", model="r", api_key="k")

    with pytest.raises(LlmUnusableError, match="non-reasoning"):
        judge.check()


def test_a_working_model_passes_the_check() -> None:
    yes = {"choices": [{"message": {"content": "YES"}}]}

    LlmJudge(_scripted_client(200, yes), base_url="http://llm", model="m", api_key="k").check()


def test_a_rate_limit_is_deferred_not_treated_as_unusable() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(429, headers={"Retry-After": "30"}, json={})
        )
    )

    with pytest.raises(RetryLaterError):
        LlmJudge(client, base_url="http://llm", model="m", api_key="k").check()
