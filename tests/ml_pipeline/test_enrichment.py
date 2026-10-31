"""Enrichment sources against mocked HTTP: no network, no live LLM.

The property that matters most is negative: nothing the LLM says survives
unless it is literally in the snippets it was shown.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.enrichment.sources import (
    LlmExtractor,
    PoliteFetcher,
    SearchHit,
    clean_value,
    grounded,
    registry_lookup,
    serp_llm_lookup,
)
from app.ontology.engine import RegistryAdapter
from app.queue.registry import RetryLaterError

ADAPTER = RegistryAdapter(
    name="test_registry",
    key="cbse_affiliation_no",
    url="https://registry.example/school/{key}",
    fields={"principal_name": ".principal", "email": "a.mail"},
    not_found_selector=".no-record",
    enabled=True,
)


def _fetcher(handler: Any) -> PoliteFetcher:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return PoliteFetcher(client, user_agent="test-agent", min_interval=0.0)


# --- Field validation ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "raw", "expected"),
    [
        ("email", "mailto:Principal@KVS.gov.in", "principal@kvs.gov.in"),
        ("email", "not an email", None),
        ("phone", "+91 98765 43210", "9876543210"),
        ("phone", "12345", None),
        ("website", "kvsangathan.nic.in", "https://kvsangathan.nic.in"),
        ("principal_name", "Dr. Anita Sharma", "Dr. Anita Sharma"),
        ("principal_name", "12345", None),
    ],
)
def test_values_are_shaped_or_dropped(field: str, raw: str, expected: str | None) -> None:
    assert clean_value(field, raw) == expected


def test_grounding_accepts_only_text_that_is_actually_there() -> None:
    corpus = ["Principal: Dr. Anita Sharma, email principal@school.in, ph 98765-43210"]

    assert grounded("principal_name", "Anita Sharma", corpus)
    assert grounded("phone", "9876543210", corpus)
    assert grounded("email", "principal@school.in", corpus)
    assert grounded("principal_name", "Anil Sharma", corpus) is None
    assert grounded("email", "admin@school.in", corpus) is None


# --- Strategy 1: registry --------------------------------------------------------------


def test_registry_fields_come_from_the_declared_selectors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        assert request.headers["User-Agent"] == "test-agent"
        assert request.url.path == "/school/2130176"
        return httpx.Response(
            200,
            html='<div class="principal">Mr. R. K. Verma</div>'
            '<a class="mail" href="mailto:kv@school.in">mail us</a>',
        )

    outcome = registry_lookup(_fetcher(handler), ADAPTER, "2130176")

    assert outcome.status == "found"
    assert outcome.fields["principal_name"].value == "Mr. R. K. Verma"
    assert outcome.fields["email"].value == "kv@school.in"


def test_the_registry_says_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, html='<p class="no-record">No school</p>')

    assert registry_lookup(_fetcher(handler), ADAPTER, "1").status == "not_found"


def test_robots_txt_is_honoured() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /school/\n")
        return httpx.Response(200, html="<p>should never be fetched</p>")

    outcome = registry_lookup(_fetcher(handler), ADAPTER, "2130176")

    assert outcome.status == "blocked"
    assert calls == ["/robots.txt"]


def test_an_unreachable_robots_txt_means_disallow() -> None:
    """RFC 9309: a 5xx robots.txt is treated as complete disallow."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    assert registry_lookup(_fetcher(handler), ADAPTER, "1").status == "blocked"


def test_a_429_defers_the_task_instead_of_failing_it() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(429, headers={"Retry-After": "30"})

    with pytest.raises(RetryLaterError) as deferred:
        registry_lookup(_fetcher(handler), ADAPTER, "1")
    assert deferred.value.countdown == 30


# --- Strategy 2: SERP + LLM ------------------------------------------------------------

HITS = [
    SearchHit(
        "Kendriya Vidyalaya Jalahalli - Contact",
        "https://kvjalahalli.example/contact",
        "Principal Shri Mohan Rao. Email: kvjalahalli@example.in Phone 080-2345 6789",
    )
]


def _llm(answer: dict[str, Any], seen: list[dict[str, Any]] | None = None) -> LlmExtractor:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if seen is not None:
            seen.append(body)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(answer)}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return LlmExtractor(client, base_url="http://llm.local/v1", model="m", api_key=None)


def test_grounded_llm_values_are_kept_with_provenance() -> None:
    seen: list[dict[str, Any]] = []
    llm = _llm(
        {
            "principal_name": {"value": "Mohan Rao", "snippet_index": 0},
            "email": {"value": "kvjalahalli@example.in", "snippet_index": 0},
            "phone": {"value": "080 2345 6789", "snippet_index": 0},
            "website": {"value": None, "snippet_index": None},
        },
        seen,
    )

    outcome = serp_llm_lookup(
        lambda q, n: HITS,
        llm,
        query="q",
        subject="KV Jalahalli",
        fields=["principal_name", "email", "phone", "website"],
        max_results=3,
    )

    assert outcome.status == "found"
    assert outcome.fields["email"].url == "https://kvjalahalli.example/contact"
    assert outcome.fields["phone"].value == "08023456789"
    assert "website" not in outcome.fields
    request = seen[0]
    assert request["temperature"] == 0
    assert request["response_format"] == {"type": "json_object"}


def test_hallucinated_values_are_discarded() -> None:
    llm = _llm(
        {
            "principal_name": {"value": "Dr. Invented Person"},
            "email": {"value": "principal@kvjalahalli.gov.in"},
        }
    )

    outcome = serp_llm_lookup(
        lambda q, n: HITS,
        llm,
        query="q",
        subject="s",
        fields=["principal_name", "email"],
        max_results=3,
    )

    assert outcome.status == "not_found"
    assert outcome.fields == {}


def test_no_search_results_short_circuits_the_llm() -> None:
    def explode(request: httpx.Request) -> httpx.Response:
        raise AssertionError("LLM called with nothing to read")

    llm = LlmExtractor(
        httpx.Client(transport=httpx.MockTransport(explode)),
        base_url="http://llm.local/v1",
        model="m",
        api_key=None,
    )

    outcome = serp_llm_lookup(
        lambda q, n: [], llm, query="q", subject="s", fields=["email"], max_results=3
    )
    assert outcome.status == "not_found"


def test_malformed_llm_output_yields_nothing_rather_than_crashing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "sure! here:"}}]})

    llm = LlmExtractor(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="http://llm.local/v1",
        model="m",
        api_key="k",
    )

    outcome = serp_llm_lookup(
        lambda q, n: HITS, llm, query="q", subject="s", fields=["email"], max_results=3
    )
    assert outcome.status == "not_found"
