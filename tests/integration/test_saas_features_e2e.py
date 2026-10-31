"""BYOK AI, smart column mapping and custom data domains, end to end.

Real API, real database, real queue and pipeline; only the LLM and the web
are fakes. Each test answers one user-facing question: is my key stored
safely and used, does my mapping decide what my columns mean, and does a
domain drafted from a sentence actually drive the Silver layer.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

# Same directory, no package (see test_medallion_e2e.py).
from test_automation_e2e import _api_client, _thresholds
from test_phase2_e2e import write_variant_csv
from test_pipeline_e2e import stub_encoder  # noqa: F401

from app.core.config import get_settings
from app.enrichment.sources import (
    EntityFields,
    LlmClient,
    LlmExtractor,
    LlmJudge,
    LlmUnusableError,
    SearchHit,
)
from app.models.enums import JobStatus
from app.queue.worker import Worker

pytestmark = pytest.mark.integration

TENANT_KEY = "sk-ant-api03-TENANTSECRETKEY-0123456789"


@pytest.fixture
def encryption_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from cryptography.fernet import Fernet

    monkeypatch.setattr(get_settings(), "secrets_encryption_key", Fernet.generate_key().decode())


@pytest.fixture
def provider_accepts(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """The provider's known-answer check, faked: records what was verified."""
    import app.api.v1.routes.settings as routes

    verified: list[Any] = []
    monkeypatch.setattr(routes, "_verify", verified.append)
    return verified


def _upload(client: Any, csv_bytes: bytes, ontology: str, mapping: dict | None) -> uuid.UUID:
    data = {"ontology": ontology}
    if mapping is not None:
        data["column_mapping"] = json.dumps(mapping)
    response = client.post(
        "/v1/jobs", files={"file": ("upload.csv", csv_bytes, "text/csv")}, data=data
    )
    assert response.status_code == 202, response.text
    return uuid.UUID(response.json()["job_id"])


def _drain(tenant: uuid.UUID) -> dict[str, int]:
    return Worker(tenant_id=tenant).run_until_idle()


def _job(owner_engine: Engine, job_id: uuid.UUID) -> Any:
    with Session(owner_engine) as session:
        return session.execute(
            text("SELECT status, stage_progress, column_mapping, ontology FROM jobs WHERE id = :j"),
            {"j": job_id},
        ).one()


# --- 1. Hybrid AI tier ----------------------------------------------------------------------


@pytest.mark.usefixtures("encryption_key", "assert_no_leaked_rows")
def test_a_premium_key_is_verified_encrypted_and_never_returned(
    e2e_tenant: uuid.UUID, owner_engine: Engine, provider_accepts: list[Any]
) -> None:
    client = _api_client(e2e_tenant)

    assert client.get("/v1/settings/ai").json()["mode"] == "standard"
    response = client.put(
        "/v1/settings/ai",
        json={"mode": "premium", "provider": "anthropic", "api_key": TENANT_KEY},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "premium" and body["key_hint"] == "…6789"
    assert body["model"] == "claude-haiku-5-5", "the provider's default model"
    assert TENANT_KEY not in response.text
    assert provider_accepts[0].api_key == TENANT_KEY, "the key itself was checked"
    assert TENANT_KEY not in client.get("/v1/settings/ai").text

    with Session(owner_engine) as session:
        stored = session.execute(
            text(
                "SELECT api_key_ciphertext, verified_at FROM tenant_ai_settings "
                "WHERE tenant_id = :t"
            ),
            {"t": e2e_tenant},
        ).one()
    assert stored.api_key_ciphertext and TENANT_KEY not in stored.api_key_ciphertext
    assert stored.verified_at is not None

    # Changing only the model re-verifies with the stored key.
    again = client.put("/v1/settings/ai", json={"mode": "premium", "model": "claude-sonnet-5-5"})
    assert again.status_code == 200 and provider_accepts[1].model == "claude-sonnet-5-5"
    assert provider_accepts[1].api_key == TENANT_KEY

    removed = client.delete("/v1/settings/ai/key").json()
    assert removed["mode"] == "standard" and removed["key_hint"] is None


@pytest.mark.usefixtures("encryption_key", "assert_no_leaked_rows")
def test_a_key_the_provider_refuses_is_not_stored_and_not_echoed(
    e2e_tenant: uuid.UUID, owner_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.api.v1.routes.settings as routes

    def refuse(config: Any) -> None:
        raise LlmUnusableError(f"HTTP 401: invalid x-api-key {config.api_key}")

    monkeypatch.setattr(routes, "_verify", refuse)
    response = _api_client(e2e_tenant).put(
        "/v1/settings/ai", json={"mode": "premium", "provider": "openai", "api_key": TENANT_KEY}
    )

    assert response.status_code == 422
    assert TENANT_KEY not in response.text and "[redacted]" in response.text
    with Session(owner_engine) as session:
        assert not session.execute(
            text(
                "SELECT count(*) FROM tenant_ai_settings WHERE mode = 'premium' AND tenant_id = :t"
            ),
            {"t": e2e_tenant},
        ).scalar_one()


@pytest.mark.usefixtures("encryption_key", "assert_no_leaked_rows")
def test_only_an_org_admin_may_change_the_ai_tier(
    e2e_tenant: uuid.UUID, provider_accepts: list[Any]
) -> None:
    steward = _api_client(e2e_tenant, role="data_steward")

    assert steward.get("/v1/settings/ai").status_code == 200
    assert steward.put("/v1/settings/ai", json={"mode": "standard"}).status_code == 403
    assert steward.delete("/v1/settings/ai/key").status_code == 403
    assert provider_accepts == []


@pytest.mark.usefixtures("stub_encoder", "encryption_key", "assert_no_leaked_rows")
def test_a_premium_tenants_triage_runs_on_its_own_key_even_without_a_server_llm(
    e2e_tenant: uuid.UUID,
    owner_engine: Engine,
    tmp_path: Any,
    provider_accepts: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The server has no LLM (the suite default). The tenant's own key is
    enough to turn Tier 2 on, and every decision records which tier made it."""
    import app.workers.tasks.enrichment as tasks

    seen_keys: list[str] = []

    def same_entity(self: Any, a: EntityFields, ha: Any, b: EntityFields, hb: Any) -> bool:
        seen_keys.append(self._api_key)
        return True

    monkeypatch.setattr(LlmClient, "check", lambda self: None)
    monkeypatch.setattr(LlmJudge, "same_entity", same_entity)
    monkeypatch.setattr(
        LlmExtractor, "extract", lambda self, *a, **k: seen_keys.append(self._api_key) or {}
    )
    monkeypatch.setattr(
        tasks, "_web_search", lambda s: lambda q, n: [SearchHit(q, "https://x.example", q)]
    )
    assert get_settings().llm_base_url is None
    client = _api_client(e2e_tenant)
    assert (
        client.put(
            "/v1/settings/ai",
            json={"mode": "premium", "provider": "anthropic", "api_key": TENANT_KEY},
        ).status_code
        == 200
    )
    _thresholds(e2e_tenant, auto_merge="1.000", floor="0.000")
    csv_path = tmp_path / "variants.csv"
    write_variant_csv(csv_path)

    job_id = _upload(client, csv_path.read_bytes(), "education", None)
    outcomes = _drain(e2e_tenant)

    assert outcomes.get("failed", 0) == 0, outcomes
    assert _job(owner_engine, job_id).status == JobStatus.COMPLETED.value
    assert seen_keys and set(seen_keys) == {TENANT_KEY}
    with Session(owner_engine) as session:
        models = set(
            session.execute(
                text(
                    "SELECT auto_decision->>'model' FROM clusters WHERE job_id = :j "
                    "AND resolution_source = 'llm_triage'"
                ),
                {"j": job_id},
            ).scalars()
        )
    assert models == {"premium:anthropic/claude-haiku-5-5"}


# --- 2. Smart column mapping ------------------------------------------------------------------


def _messy_csv(tmp_path: Any) -> bytes:
    """The variant fixture with unrecognisable headers, plus a decoy `name`
    column holding the principal's name."""
    path = tmp_path / "variants.csv"
    write_variant_csv(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    header = "schl_nm,addr_ln,dist_nm,st_cd,pin_cd,u_code,name"
    body = [f"{line},Mrs Principal {i}" for i, line in enumerate(lines[1:])]
    return ("\n".join([header, *body]) + "\n").encode("utf-8")


MAPPING = {
    "fields": {
        "name": "schl_nm",
        "address": "addr_ln",
        "district": "dist_nm",
        "state": "st_cd",
        "pincode": "pin_cd",
    },
    "natural_keys": {"udise_code": "u_code"},
}


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_a_mapping_decides_what_messy_columns_mean(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    client = _api_client(e2e_tenant)

    job_id = _upload(client, _messy_csv(tmp_path), "education", MAPPING)
    outcomes = _drain(e2e_tenant)

    assert outcomes.get("failed", 0) == 0, outcomes
    job = _job(owner_engine, job_id)
    assert job.column_mapping == MAPPING
    with Session(owner_engine) as session:
        names = session.execute(
            text("SELECT s.name, s.natural_keys FROM silver_records s WHERE s.job_id = :j"),
            {"j": job_id},
        ).all()
        stored = session.execute(
            text("SELECT raw_row FROM school_records WHERE job_id = :j LIMIT 1"), {"j": job_id}
        ).scalar_one()
    real = [n for n, _ in names if n and "primary" in n]
    assert real, "the mapped school-name column was not used"
    assert not any(n and "principal" in n for n, _ in names), "the decoy `name` column won"
    assert any(keys.get("udise_code") == "29281200612" for _, keys in names)
    assert stored["original name"].startswith("Mrs Principal")


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_a_mapping_naming_a_column_the_file_lacks_fails_the_job_with_the_reason(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    client = _api_client(e2e_tenant)
    mapping = {"fields": {"name": "school_title"}}

    job_id = _upload(client, _messy_csv(tmp_path), "education", mapping)
    _drain(e2e_tenant)

    job = _job(owner_engine, job_id)
    assert job.status == JobStatus.FAILED.value
    assert "school_title" in job.stage_progress["error"]["message"]


@pytest.mark.usefixtures("assert_no_leaked_rows")
def test_mappings_are_validated_against_the_domain(e2e_tenant: uuid.UUID) -> None:
    client = _api_client(e2e_tenant)
    bad_key = {"fields": {"name": "a"}, "natural_keys": {"gstin": "g"}}
    no_name = {"fields": {"address": "a"}}

    for mapping in (bad_key, no_name):
        response = client.post(
            "/v1/jobs",
            files={"file": ("x.csv", b"a,g\n1,2\n", "text/csv")},
            data={"ontology": "education", "column_mapping": json.dumps(mapping)},
        )
        assert response.status_code == 422, response.text


@pytest.mark.usefixtures("assert_no_leaked_rows")
def test_upload_options_carry_the_aliases_the_mapping_screen_suggests_from(
    e2e_tenant: uuid.UUID,
) -> None:
    options = _api_client(e2e_tenant).get("/v1/jobs/upload-options").json()
    education = next(o for o in options["ontologies"] if o["name"] == "education")

    assert "schoolname" in education["field_aliases"]["name"]
    assert "pin" in education["field_aliases"]["pincode"]
    assert {k["key"] for k in education["natural_keys"]} == {"udise_code", "cbse_affiliation_no"}


# --- 3. Custom data domains ------------------------------------------------------------------


DEALER_DRAFT = {
    "label": "Car dealerships",
    "description": "Authorised car dealerships",
    "entity_noun": "car dealership",
    "name_headers": ["dealer_name"],
    "abbreviations": {"mtrs": "motors", "pvt": "private", "auto": "auto"},
    "generic_name_tokens": ["motors", "showroom", "dealers"],
    "natural_keys": [
        {
            "key": "gstin",
            "label": "GSTIN",
            "headers": ["gst_no"],
            "pattern": r"\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]",
        },
        {"key": "phone_no", "label": "Phone", "headers": ["phone"], "pattern": r"\d{10}"},
    ],
    "enrichment_targets": [
        {"key": "brand", "description": "car brand sold"},
        {"key": "name"},
    ],
}

DEALER_ROWS = [
    {"dealer_name": "Sharma Mtrs Pvt Ltd", "gst_no": "29ABCDE1234F1Z5", "phone": "9876543210"},
    {"dealer_name": "Kapoor Auto", "gst_no": "07AAACB1234C1ZV", "phone": "9876543211"},
    {"dealer_name": "Mehta Mtrs", "gst_no": "33AABCT3518Q1ZW", "phone": "9876543212"},
]


@pytest.fixture
def fake_drafter(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    import app.api.v1.routes.ontologies as routes

    calls: list[Any] = []

    def draft(config: Any, payload: Any) -> dict:
        calls.append((config, payload))
        return json.loads(json.dumps(DEALER_DRAFT))

    monkeypatch.setattr(get_settings(), "llm_base_url", "http://fake-llm.invalid/v1")
    monkeypatch.setattr(routes, "_draft", draft)
    return calls


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_a_domain_drafted_from_a_sentence_drives_the_silver_layer(
    e2e_tenant: uuid.UUID, owner_engine: Engine, fake_drafter: list[Any]
) -> None:
    client = _api_client(e2e_tenant)

    generated = client.post(
        "/v1/ontologies/generate",
        json={
            "description": "Car dealerships in India",
            "headers": ["dealer_name", "gst_no", "phone", "city"],
            "sample_rows": DEALER_ROWS,
        },
    )
    assert generated.status_code == 200, generated.text
    body = generated.json()
    draft = body["draft"]
    # The sanitiser's work is visible to the user, with reasons.
    assert [k["key"] for k in draft["natural_keys"]] == ["gstin"]
    assert any("contact details" in n for n in body["notes"])
    assert [t["key"] for t in draft["enrichment_targets"]] == ["brand"]
    assert "auto" not in draft["abbreviations"]
    assert body["drafted_by"].startswith("standard:")
    assert fake_drafter[0][1].sample_rows == DEALER_ROWS

    saved = client.post(
        "/v1/ontologies", json={"draft": draft, "prompt": "Car dealerships in India"}
    )
    assert saved.status_code == 201, saved.text
    name = saved.json()["name"]
    assert name.startswith("custom_car_dealerships_")

    options = client.get("/v1/jobs/upload-options").json()
    custom = next(o for o in options["ontologies"] if o["name"] == name)
    assert custom["custom"] is True and "dealername" in custom["field_aliases"]["name"]
    assert options["custom_domains_available"] is True

    csv_bytes = (
        "dealer_name,addr,city,state,pin,gst_no\n"
        + "".join(
            f"{r['dealer_name']},MG Road,Bangalore,Karnataka,5600{i:02d},{r['gst_no']}\n"
            for i, r in enumerate(DEALER_ROWS * 3)
        )
        # Junk rows: the cold-start classifier needs more than one class.
        + "".join(f"N/A,,,,000000,\nTEST {i},,,,,\n" for i in range(4))
    ).encode()
    mapping = {
        "fields": {
            "name": "dealer_name",
            "address": "addr",
            "district": "city",
            "state": "state",
            "pincode": "pin",
        },
        "natural_keys": {"gstin": "gst_no"},
    }
    # The domain is saved; the pipeline itself runs without an LLM (no
    # triage, no web enrichment) so this test is about the Silver layer only.
    from app.core.config import get_settings as settings_now

    settings_now().llm_base_url = None
    job_id = _upload(client, csv_bytes, name, mapping)
    outcomes = _drain(e2e_tenant)

    assert outcomes.get("failed", 0) == 0, outcomes
    with Session(owner_engine) as session:
        silver = session.execute(
            text("SELECT ontology, name, natural_keys FROM silver_records WHERE job_id = :j"),
            {"j": job_id},
        ).all()
    assert {row.ontology for row in silver} == {name}
    assert "sharma motors private ltd" in {row.name for row in silver}
    assert {row.natural_keys.get("gstin") for row in silver} >= {"29ABCDE1234F1Z5"}


@pytest.mark.usefixtures("assert_no_leaked_rows")
def test_a_saved_draft_is_re_sanitised_the_browser_is_not_trusted(
    e2e_tenant: uuid.UUID,
) -> None:
    tampered = {
        **DEALER_DRAFT,
        "natural_keys": [{"key": "evil", "headers": ["x"], "pattern": r"(a+)+$"}],
        "abbreviations": {"a": "b", "loop": "loop forever"},
    }

    response = _api_client(e2e_tenant).post("/v1/ontologies", json={"draft": tampered})

    assert response.status_code == 201, response.text
    with Session(_engine()) as session:
        spec = session.execute(
            text("SELECT spec FROM tenant_ontologies WHERE name = :n"),
            {"n": response.json()["name"]},
        ).scalar_one()
    assert spec["natural_keys"] == {}
    assert "loop" not in spec["text"]["name_abbreviations"]


def _engine() -> Engine:
    from app.db.session import _get_platform_engine

    return _get_platform_engine()


@pytest.mark.usefixtures("assert_no_leaked_rows")
def test_another_tenants_domain_cannot_be_used(
    two_e2e_tenants: tuple[uuid.UUID, uuid.UUID], fake_drafter: list[Any]
) -> None:
    owner, other = two_e2e_tenants
    saved = _api_client(owner).post("/v1/ontologies", json={"draft": DEALER_DRAFT})
    name = saved.json()["name"]

    response = _api_client(other).post(
        "/v1/jobs",
        files={"file": ("x.csv", b"name\nx\n", "text/csv")},
        data={"ontology": name},
    )

    assert response.status_code == 422
    names = {
        o["name"] for o in _api_client(other).get("/v1/jobs/upload-options").json()["ontologies"]
    }
    assert name not in names


@pytest.mark.usefixtures("assert_no_leaked_rows")
def test_without_any_llm_a_domain_cannot_be_drafted(e2e_tenant: uuid.UUID) -> None:
    response = _api_client(e2e_tenant).post(
        "/v1/ontologies/generate", json={"description": "Car dealerships in India"}
    )

    assert response.status_code == 409 and "Premium" in response.json()["detail"]


@pytest.mark.usefixtures("encryption_key", "assert_no_leaked_rows")
def test_a_free_cerebras_key_is_accepted_stored_and_offered_with_its_terms(
    e2e_tenant: uuid.UUID, owner_engine: Engine, provider_accepts: list[Any]
) -> None:
    client = _api_client(e2e_tenant)
    csk = "csk-tenantsecretkey0123456789"

    listed = client.get("/v1/settings/ai").json()["providers"]
    cerebras = next(p for p in listed if p["id"] == "cerebras")
    assert cerebras["free"] is True and cerebras["default_model"] == "qwen-3.8-27b"
    assert "1M tokens" in cerebras["note"] and cerebras["key_url"].startswith("https://")
    assert not any(p["free"] for p in listed if p["id"] in {"openai", "anthropic"})

    response = client.put(
        "/v1/settings/ai", json={"mode": "premium", "provider": "cerebras", "api_key": csk}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["provider"], body["model"], body["key_hint"]) == (
        "cerebras",
        "qwen-3.8-27b",
        "…6789",
    )
    assert csk not in response.text
    verified = provider_accepts[0]
    assert verified.vendor == "cerebras" and verified.api_key == csk
    assert dict(verified.extra) == {"reasoning_effort": "none"}, "thinking is off when verified"
    with Session(owner_engine) as session:
        provider = session.execute(
            text("SELECT provider FROM tenant_ai_settings WHERE tenant_id = :t"), {"t": e2e_tenant}
        ).scalar_one()
    assert provider == "cerebras"
