"""Tier 3 without waste: patterns instead of LLM calls, PINs from the directory
or the web (never guessed), overlapping searches, and survivable model errors."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from app.enrichment.golden import WEB_PINCODE_KEY, enrich_entity
from app.enrichment.patterns import extract_by_pattern, find_pincode
from app.enrichment.prefetch import search_pool, windowed
from app.enrichment.sources import EntityFields, SearchHit
from app.ontology.engine import EnrichmentTarget, GoldenEnrichmentRules

GENERIC = ["school", "public", "senior", "secondary"]
SCHOOL = EntityFields("r1", "nalanda public school", "sector 15", "faridabad", "haryana", "")
TARGETS = [
    EnrichmentTarget.model_validate(t)
    for t in (
        "principal_name",
        {"key": "board", "pattern": r"\b(CBSE|ICSE)\b"},
        {
            "key": "cbse_affiliation_no",
            "type": "number",
            "pattern": r"(?i)\baff(?:iliation)?\.?\s*(?:no\.?|number)?\s*[:#]?\s*(\d{6,7})\b",
        },
    )
]


def _hit(text: str, title: str = "Nalanda Public School, Faridabad, Haryana") -> SearchHit:
    return SearchHit(title, "https://example.org/x", text)


def test_a_pattern_target_reads_the_value_nearest_the_schools_name() -> None:
    hits = [_hit("Nalanda Public School Faridabad. CBSE affiliation no: 530123. Other: DPS ICSE")]

    out = extract_by_pattern(TARGETS, hits, SCHOOL, GENERIC)

    assert out["board"]["value"] == "CBSE"
    assert out["cbse_affiliation_no"]["value"] == "530123"
    assert "principal_name" not in out, "non-pattern targets are left to the LLM"


def test_a_snippet_about_another_place_gives_nothing() -> None:
    hits = [_hit("Nalanda Public School Jaipur Rajasthan. CBSE affiliation no: 111222", "Nalanda")]

    assert extract_by_pattern(TARGETS, hits, SCHOOL, GENERIC) == {}


def test_a_value_far_from_the_name_is_not_taken() -> None:
    far = "Nalanda Public School Faridabad Haryana. " + "x " * 400 + "ICSE"

    assert "board" not in extract_by_pattern(TARGETS, [_hit(far)], SCHOOL, GENERIC)


def test_pattern_targets_cost_no_llm_call_and_other_targets_still_do() -> None:
    asked: list[list[str]] = []

    def extract(keys: Any, subject: str, hits: Any, meanings: Any) -> dict[str, Any]:
        asked.append(list(keys))
        return {}

    hit = _hit("Nalanda Public School Faridabad Haryana. Principal: Dr A Rao. CBSE aff no 530123")
    result = enrich_entity(
        SCHOOL,
        targets=TARGETS,
        rules=GoldenEnrichmentRules(),
        generic_tokens=GENERIC,
        search=lambda q, n: [hit],
        extract=extract,
    )

    assert asked == [["principal_name"]]
    assert {k: v.value for k, v in result.fields.items()} == {
        "board": "CBSE",
        "cbse_affiliation_no": "530123",
    }


def test_no_llm_call_when_the_snippets_cannot_hold_the_value() -> None:
    def extract(*args: Any) -> dict[str, Any]:
        raise AssertionError("nothing here to ask about")

    only_pattern = [t for t in TARGETS if t.pattern]
    enrich_entity(
        SCHOOL,
        targets=[*only_pattern, EnrichmentTarget.model_validate("email")],
        rules=GoldenEnrichmentRules(),
        generic_tokens=GENERIC,
        search=lambda q, n: [_hit("Nalanda Public School Faridabad Haryana, a CBSE school")],
        extract=extract,
    )


def test_a_quoted_search_with_no_hits_is_retried_once_without_quotes() -> None:
    queries: list[str] = []

    def search(query: str, n: int) -> list[SearchHit]:
        queries.append(query)
        return []

    enrich_entity(
        SCHOOL,
        targets=TARGETS,
        rules=GoldenEnrichmentRules(query='"{name}" {district}'),
        generic_tokens=GENERIC,
        search=search,
        extract=lambda *a: {},
    )

    assert queries == ['"nalanda public school" faridabad', " nalanda public school  faridabad"]


# --- PIN codes ---------------------------------------------------------------------------


def test_a_pin_is_taken_from_the_web_only_if_the_directory_lists_it_for_the_district() -> None:
    hit = _hit("Nalanda Public School, Sector 15, Faridabad, Haryana - 121004")

    assert find_pincode([hit], SCHOOL, GENERIC, is_valid=lambda p: p == "121004").value == "121004"
    assert find_pincode([hit], SCHOOL, GENERIC, is_valid=lambda p: False) is None


def test_two_different_pins_near_the_name_give_no_answer() -> None:
    hit = _hit("Nalanda Public School Faridabad Haryana 121004 and also 121006")

    assert find_pincode([hit], SCHOOL, GENERIC, is_valid=lambda p: True) is None


def test_enrichment_stores_a_found_pin_beside_not_over_the_golden_pincode() -> None:
    hit = _hit("Nalanda Public School, Faridabad, Haryana 121004")
    result = enrich_entity(
        SCHOOL,
        targets=TARGETS,
        rules=GoldenEnrichmentRules(),
        generic_tokens=GENERIC,
        search=lambda q, n: [hit],
        extract=lambda *a: {},
        is_valid_pincode=lambda p: True,
    )

    assert result.fields[WEB_PINCODE_KEY].value == "121004"


# --- Overlap and resilience ------------------------------------------------------------


def test_windowed_keeps_work_started_ahead_and_preserves_order() -> None:
    started: list[int] = []

    def start(item: int) -> int:
        started.append(item)
        return item * 10

    out = []
    for item, value in windowed(range(5), start, window=3):
        out.append((item, value, len(started)))

    assert [(i, v) for i, v, _ in out] == [(0, 0), (1, 10), (2, 20), (3, 30), (4, 40)]
    assert out[0][2] == 3, "three were started before the first was handed back"


def test_searches_on_the_pool_overlap_in_time() -> None:
    live = peak = 0
    gate = threading.Lock()

    def slow(_: int) -> None:
        nonlocal live, peak
        with gate:
            live += 1
            peak = max(peak, live)
        time.sleep(0.05)
        with gate:
            live -= 1

    with search_pool(3) as pool:
        for f in [pool.submit(slow, i) for i in range(6)]:
            f.result()

    assert peak >= 2


def test_one_unanswerable_school_does_not_end_the_run() -> None:
    from app.workers.tasks.enrichment import _Breaker

    breaker = _Breaker("t", limit=3)
    err = RuntimeError("boom")
    breaker.failed(err)
    breaker.failed(err)
    breaker.ok()
    breaker.failed(err)
    breaker.failed(err)  # never three in a row, so never raises
    with pytest.raises(RuntimeError):
        breaker.failed(err)
