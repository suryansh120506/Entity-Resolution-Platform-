"""Column mapping, custom-domain generation, BYOK providers and secret handling.

No network: providers are httpx MockTransports, the LLM's drafts are dicts.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
import pytest

from app.core.secrets import decrypt_secret, encrypt_secret, key_hint, scrub
from app.enrichment.llm_config import LlmConfig, config_for, llm_available, premium_config
from app.enrichment.sources import LlmJudge, LlmUnusableError
from app.ml_pipeline.column_mapping import ColumnMapping, alias_tables, apply_mapping
from app.ml_pipeline.fields import canonical_fields
from app.models.tenant_settings import TenantAiSettings
from app.ontology import OntologyEngine
from app.ontology.engine import OntologyError
from app.ontology.generator import (
    build_spec,
    draft_prompt,
    new_ontology_name,
    pattern_min_length,
    sanitize_draft,
)
from app.ontology.registry import engine_from_spec

GSTIN = r"\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]"


# --- Column mapping -----------------------------------------------------------------------


def _tables(ontology: str = "education") -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    spec = OntologyEngine.load(ontology).spec
    return alias_tables(spec.fields, {k: r.headers for k, r in spec.natural_keys.items()})


def _apply(row: dict[str, Any], mapping: ColumnMapping) -> dict[str, Any]:
    fields, keys = _tables()
    return apply_mapping(row, mapping, field_aliases=fields, natural_key_headers=keys)


def test_mapped_columns_are_stored_under_their_canonical_names() -> None:
    mapping = ColumnMapping(fields={"name": "schl_nm", "pincode": "pin_cd"}, record_id="uid")

    out = _apply({"schl_nm": "DPS", "pin_cd": "110065", "uid": "A1", "fee": "9000"}, mapping)

    assert out == {"name": "DPS", "pincode": "110065", "id": "A1", "fee": "9000"}
    assert canonical_fields(out)["name"] == "DPS"


def test_an_unmapped_column_cannot_outrank_the_users_choice() -> None:
    """The file has a `name` column (the principal's) and `schl_nm` (the
    school's). The user mapped schl_nm. JSONB does not keep key order, so
    the competing column is renamed rather than merely placed second."""
    mapping = ColumnMapping(fields={"name": "schl_nm"})

    out = _apply({"name": "Mrs Rao", "school": "x", "schl_nm": "DPS", "phone": "1"}, mapping)

    assert out["name"] == "DPS"
    assert out["original name"] == "Mrs Rao" and out["original school"] == "x"
    assert out["phone"] == "1"
    assert canonical_fields(out)["name"] == "DPS"
    assert OntologyEngine.load("education").standardize(out).name == "delhi public school"


def test_fields_left_unmapped_are_still_recognised_by_their_aliases() -> None:
    out = _apply(
        {"schl_nm": "DPS", "district": "south delhi"}, ColumnMapping(fields={"name": "schl_nm"})
    )

    assert out["district"] == "south delhi"


def test_a_natural_key_is_written_under_a_header_the_ontology_reads() -> None:
    mapping = ColumnMapping(fields={"name": "n"}, natural_keys={"udise_code": "u_code"})
    out = _apply({"n": "DPS", "u_code": "07250123456"}, mapping)

    assert OntologyEngine.load("education").standardize(out).natural_keys == {
        "udise_code": "07250123456"
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"fields": {"address": "a"}},  # name is required
        {"fields": {"name": "x", "address": "x"}},  # one column, two fields
        {"fields": {"name": "x"}, "record_id": "x"},
        {"fields": {"name": " "}},
        {"fields": {"name": "x", "phone": "p"}},  # not a field
    ],
)
def test_malformed_mappings_are_refused(payload: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ColumnMapping.model_validate(payload)


# --- Custom domains: the sanitiser -----------------------------------------------------------


@pytest.mark.parametrize(
    ("pattern", "length"),
    [
        (GSTIN, 15),
        (r"\d{11}", 11),
        (r"[A-Z]{3}\d{1,4}", 4),
        (r"(a+)+$", None),  # catastrophic backtracking: outside the grammar
        (r".*", None),
        (r"\w+", None),
        (r"[^0-9]{5}", None),
        (r"", None),
    ],
)
def test_identifier_patterns_are_held_to_a_tiny_grammar(pattern: str, length: int | None) -> None:
    assert pattern_min_length(pattern) == length


def _draft(**overrides: Any) -> dict[str, Any]:
    return {
        "label": "Car dealerships",
        "description": "Authorised car dealerships",
        "entity_noun": "car dealership",
        "name_headers": ["dealer_name"],
        "abbreviations": {"mtrs": "motors", "pvt": "private"},
        "generic_name_tokens": ["motors", "showroom"],
        "natural_keys": [
            {"key": "gstin", "label": "GSTIN", "headers": ["gst_no"], "pattern": GSTIN}
        ],
        "enrichment_targets": [{"key": "brand", "description": "car brand sold"}],
        **overrides,
    }


def test_a_clean_draft_survives_and_builds_a_valid_ontology() -> None:
    draft, notes = sanitize_draft(_draft(), description="Car dealerships in India")
    spec = build_spec(draft, new_ontology_name(draft.label))
    engine = engine_from_spec(spec)

    assert notes == [
        "Key 'GSTIN' has no column in your file; map one on the next step or it will not be used"
    ]
    assert engine.name.startswith("custom_car_dealerships_")
    assert engine.normalize_name("Sharma Mtrs Pvt Ltd") == "sharma motors private ltd"
    assert "dealername" in engine.spec.fields["name"]
    assert engine.spec.golden_enrichment.query.endswith("car dealership")
    # Indian geography comes from the shipped base, not from the LLM.
    assert engine.spec.state_aliases


def test_generated_keys_never_force_a_merge() -> None:
    draft, _ = sanitize_draft(_draft(), description="d")

    spec = build_spec(draft, "custom_x_abc123")

    assert spec["identity"] == {"keys": ["gstin"], "force_merge_on_key": False}


@pytest.mark.parametrize(
    ("key", "why"),
    [
        ({"key": "regno", "pattern": r"(\d+)+"}, "not a plain format"),
        ({"key": "short_id", "pattern": r"\d{4}"}, "too short"),
        ({"key": "mobile_no", "pattern": r"\d{10}"}, "contact details"),
        ({"key": "dealer_pin", "pattern": r"\d{6}"}, "contact details"),
    ],
)
def test_unsafe_or_meaningless_keys_are_dropped_with_a_reason(
    key: dict[str, str], why: str
) -> None:
    draft, notes = sanitize_draft(_draft(natural_keys=[key]), description="d")

    assert draft.natural_keys == []
    assert any(why in note for note in notes), notes


def test_a_key_format_the_files_own_values_contradict_is_dropped() -> None:
    rows = [{"gst_no": v} for v in ("29ABCDE1234F1Z5", "N/A-12", "12345", "abc")]

    draft, notes = sanitize_draft(_draft(), description="d", sample_rows=rows)

    assert draft.natural_keys == []
    assert any("sample values" in n for n in notes)


def test_a_key_format_the_files_values_agree_with_is_kept() -> None:
    rows = [{"GST No": v} for v in ("29ABCDE1234F1Z5", "07AAACB1234C1ZV", "33-AABCT3518Q1ZW")]

    draft, notes = sanitize_draft(_draft(), description="d", sample_rows=rows)

    assert [k.key for k in draft.natural_keys] == ["gstin"]
    assert notes == []


def test_abbreviation_chains_and_self_expansions_are_dropped() -> None:
    draft, notes = sanitize_draft(
        _draft(
            abbreviations={"auto": "auto motors", "mtrs": "motors", "am": "auto mtrs", "x": "y"}
        ),
        description="d",
    )

    # "auto" expands into itself, "am" into another abbreviation; "x" is too short.
    assert draft.abbreviations == {"mtrs": "motors"}
    assert any("auto" in n and "am" in n for n in notes)


def test_identity_fields_are_never_enrichment_targets() -> None:
    draft, _ = sanitize_draft(
        _draft(enrichment_targets=[{"key": "name"}, {"key": "pincode"}, {"key": "brand"}]),
        description="d",
    )

    assert [t.key for t in draft.enrichment_targets] == ["brand"]


def test_garbage_from_the_llm_yields_an_empty_but_valid_draft() -> None:
    draft, _ = sanitize_draft(
        {"label": 7, "abbreviations": "nope", "natural_keys": "x", "enrichment_targets": [1, None]},
        description="Wholesale pharmacies",
    )

    assert draft.label == "7" and draft.natural_keys == [] and draft.enrichment_targets == []
    build_spec(draft, "custom_pharma_abc123")  # still a valid ontology


def test_the_prompt_bounds_what_is_sent_to_the_provider() -> None:
    rows = [{"col": "x" * 500} for _ in range(50)]

    _, user = draft_prompt("Car dealers", ["col"], rows)

    assert user.count("x" * 80) == 15 and "x" * 81 not in user


def test_a_spec_that_fails_validation_is_refused_on_build() -> None:
    draft, _ = sanitize_draft(_draft(), description="d")

    with pytest.raises(OntologyError):
        build_spec(draft, "Not A Valid Name")


# --- BYOK: providers on the wire ------------------------------------------------------------


def _recording(reply: dict[str, Any], seen: list[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=reply)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_anthropic_is_spoken_natively() -> None:
    seen: list[httpx.Request] = []
    reply = {"content": [{"type": "text", "text": "YES"}]}
    config = premium_config("anthropic", "claude-haiku-5-5", "sk-ant-secret-key-123456")

    LlmJudge(_recording(reply, seen), **config.client_kwargs()).check()

    request = seen[0]
    body = json.loads(request.content)
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "sk-ant-secret-key-123456"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in request.headers
    assert body["system"].startswith("Reply ONLY")
    assert body["max_tokens"] == 3
    # Claude models answer HTTP 400 to `temperature`; it must not be sent.
    assert "temperature" not in body
    # Haiku 5.5 thinks by default and, given 3 tokens, returned no text (42% of the
    # triage verdicts on a real job); thinking must be off for a YES/NO.
    assert body["thinking"] == {"type": "disabled"}
    assert [m["role"] for m in body["messages"]] == ["user"]


def test_openai_gets_max_completion_tokens() -> None:
    seen: list[httpx.Request] = []
    reply = {"choices": [{"message": {"content": "YES"}}]}
    config = premium_config("openai", "gpt-4.1-mini", "sk-proj-secret-key-123456")

    LlmJudge(_recording(reply, seen), **config.client_kwargs()).check()

    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://api.openai.com/v1/chat/completions"
    assert body["max_completion_tokens"] == 3 and "max_tokens" not in body


def test_a_rejected_key_is_never_quoted_back() -> None:
    key = "sk-proj-THISISASECRETKEY987654"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401, json={"error": {"message": f"Incorrect API key provided: {key}"}}
        )

    config = premium_config("openai", "gpt-4.1-mini", key)
    judge = LlmJudge(httpx.Client(transport=httpx.MockTransport(handler)), **config.client_kwargs())

    with pytest.raises(LlmUnusableError) as caught:
        judge.check()
    assert key not in str(caught.value) and "[redacted]" in str(caught.value)


def test_a_config_never_shows_its_key_in_a_repr_or_audit_label() -> None:
    config = premium_config("anthropic", "claude-haiku-5-5", "sk-ant-secret-key-123456")

    assert "secret" not in repr(config)
    assert config.audit_label == "premium:anthropic/claude-haiku-5-5"


# --- BYOK: resolution and secrets ----------------------------------------------------------


@pytest.fixture
def encryption_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from cryptography.fernet import Fernet

    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "secrets_encryption_key", Fernet.generate_key().decode())


@pytest.mark.usefixtures("encryption_key")
def test_a_premium_tenant_runs_on_its_own_key() -> None:
    row = TenantAiSettings(
        mode="premium",
        provider="anthropic",
        model="claude-haiku-5-5",
        api_key_ciphertext=encrypt_secret("sk-ant-tenant-key-123456"),
    )

    config = config_for(row)

    assert isinstance(config, LlmConfig) and config.premium
    assert config.api_key == "sk-ant-tenant-key-123456"
    assert llm_available(row)


def test_a_standard_tenant_runs_on_the_servers_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "llm_base_url", "http://groq.example/v1")
    config = config_for(TenantAiSettings(mode="standard"))

    assert config is not None and not config.premium and config.provider == "openai_compatible"
    monkeypatch.setattr(get_settings(), "llm_base_url", None)
    assert config_for(None) is None and not llm_available(None)


@pytest.mark.usefixtures("encryption_key")
def test_secrets_round_trip_and_ciphertext_reveals_nothing() -> None:
    token = encrypt_secret("sk-ant-tenant-key-123456")

    assert "sk-ant" not in token and decrypt_secret(token) == "sk-ant-tenant-key-123456"
    assert key_hint("sk-ant-tenant-key-123456") == "…3456"


def test_a_key_quoted_in_part_is_still_scrubbed() -> None:
    key = "sk-proj-ABCDEFGHIJKLMNOP"
    text = f"bad key {key[:8]}****{key[-6:]}"

    assert not re.search(r"sk-proj-|KLMNOP", scrub(text, key))


# --- Drafts cut off by the token limit ----------------------------------------------------


def test_a_runaway_draft_keeps_everything_it_completed() -> None:
    """Seen live: the model looped "holder": "holder", ... until max_tokens."""
    from app.ontology.generator import parse_draft

    cut = (
        '{"label": "Car dealerships", "natural_keys": [{"key": "gstin", "pattern": "x"}], '
        '"abbreviations": {"pvt": "private", "holder": "holder", "rack": "ra'
    )

    assert parse_draft(cut) == {
        "label": "Car dealerships",
        "natural_keys": [{"key": "gstin", "pattern": "x"}],
        "abbreviations": {"pvt": "private", "holder": "holder"},
    }
    draft, _ = sanitize_draft(parse_draft(cut), description="d")
    assert draft.abbreviations == {"pvt": "private"}, "self-maps never survive"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('Here you go:\n```json\n{"label": "A"}\n```', {"label": "A"}),
        ('{"label": "A", "x": "brace } in a string"}', {"label": "A", "x": "brace } in a string"}),
        ("no json at all", {}),
        ('{"label": "unterminated', {}),
    ],
)
def test_drafts_are_parsed_from_whatever_wraps_them(text: str, expected: dict[str, Any]) -> None:
    from app.ontology.generator import parse_draft

    assert parse_draft(text) == expected


def test_the_important_fields_are_asked_for_before_the_open_ended_lists() -> None:
    _, user = draft_prompt("Car dealers", [], [])
    order = [
        user.index(f'"{key}"')
        for key in ("natural_keys", "enrichment_targets", "abbreviations", "generic_name_tokens")
    ]

    assert order == sorted(order)
    assert "at most 25" in user


# --- Cerebras: a free bring-your-own-key provider ---------------------------------------------


def test_cerebras_is_spoken_with_reasoning_switched_off() -> None:
    """Qwen on Cerebras reasons by default; a reasoning model spends the
    3-token YES/NO budget thinking and answers nothing, so every pair would be
    silently read as 'no answer'. The request must say reasoning_effort none."""
    seen: list[httpx.Request] = []
    reply = {"choices": [{"message": {"content": "YES"}}]}
    config = premium_config("cerebras", "qwen-3.8-27b", "csk-tenant-secret-123456")

    LlmJudge(_recording(reply, seen), **config.client_kwargs()).check()

    request, body = seen[0], json.loads(seen[0].content)
    assert str(request.url) == "https://api.cerebras.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer csk-tenant-secret-123456"
    assert body["reasoning_effort"] == "none"
    assert body["max_tokens"] == 3 and "max_completion_tokens" not in body, "one alias only"
    assert body["temperature"] == 0 and body["model"] == "qwen-3.8-27b"


def test_extra_fields_can_never_override_what_the_caller_asked_for() -> None:
    seen: list[httpx.Request] = []
    reply = {"choices": [{"message": {"content": "YES"}}]}
    client = LlmJudge(
        _recording(reply, seen),
        base_url="http://llm",
        model="m",
        api_key="k",
        extra_body={"temperature": 1.0, "max_tokens": 9999, "reasoning_effort": "none"},
    )

    client.check()

    body = json.loads(seen[0].content)
    assert body["temperature"] == 0 and body["max_tokens"] == 3
    assert body["reasoning_effort"] == "none"


def test_a_cerebras_config_is_labelled_by_who_it_is_not_by_its_wire_format() -> None:
    config = premium_config("cerebras", "qwen-3.8-27b", "csk-tenant-secret-123456")

    assert config.provider == "openai_compatible", "the request format"
    assert config.audit_label == "premium:cerebras/qwen-3.8-27b"
    assert config.display_name == "Cerebras"
    assert "secret" not in repr(config)
    assert premium_config("anthropic", "claude-haiku-5-5", "k" * 12).display_name == "Anthropic"


def test_the_free_provider_is_marked_free_and_the_paid_ones_are_not() -> None:
    from app.enrichment.llm_config import PROVIDERS

    assert PROVIDERS["cerebras"].free and "1M tokens" in PROVIDERS["cerebras"].note
    assert not PROVIDERS["openai"].free and not PROVIDERS["anthropic"].free
    assert all(info.key_url.startswith("https://") for info in PROVIDERS.values())
