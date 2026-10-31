"""Automated triage and golden enrichment against the live database.

The web and the LLM are fakes (no network in the suite); everything else is
real: the gate, the queue hand-offs, the row locks, the audit columns, the
enrichment cache and the job lifecycle. The question each test answers is
whether a job now finishes with *no* human, and leaves a trail that says
who decided what on which evidence.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Any

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

# Same directory, no package (see test_medallion_e2e.py).
from test_phase2_e2e import write_variant_csv
from test_pipeline_e2e import _make_job, _run_pipeline, stub_encoder  # noqa: F401

from app.core.config import get_settings
from app.db.session import sync_engine, tenant_sync_session
from app.enrichment.sources import EntityFields, LlmClient, LlmExtractor, LlmJudge, SearchHit
from app.enrichment.throttle import advisory_lock
from app.models.enums import JobStatus

pytestmark = pytest.mark.integration

PRINCIPAL = "Dr. Asha Rao"
PHONE = "080 2345 6789"


@pytest.fixture
def fake_web(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """An LLM is 'configured'; search and both LLM calls are deterministic.

    The judge says YES unless either record is spelled "Scool", so a
    four-member cluster must come out SPLIT. The extractor returns one value
    that is in the snippet (principal, phone) and one it made up (email).
    """
    import app.workers.tasks.enrichment as tasks

    monkeypatch.setattr(get_settings(), "llm_base_url", "http://fake-llm.invalid/v1")
    calls = {"search": 0, "judge": 0, "extract": 0}

    def search(query: str, max_results: int) -> list[SearchHit]:
        calls["search"] += 1
        subject = query.split('"')[1] if '"' in query else query
        return [
            SearchHit(
                title=f"{subject} | Bangalore",
                url="https://schools.example/" + str(abs(hash(subject))),
                snippet=f"{subject}. Principal: {PRINCIPAL}. Office: {PHONE}.",
            )
        ]

    def same_entity(self: Any, a: EntityFields, hits_a: Any, b: EntityFields, hits_b: Any) -> bool:
        calls["judge"] += 1
        return "scool" not in a.name and "scool" not in b.name

    def extract(self: Any, keys: Any, subject: str, hits: Any, meanings: Any = None) -> dict:
        calls["extract"] += 1
        return {
            "principal_name": {"value": PRINCIPAL},
            "phone": PHONE,
            "email": "invented@nowhere.example",
            "board": None,
        }

    monkeypatch.setattr(tasks, "_web_search", lambda settings: search)
    monkeypatch.setattr(LlmClient, "check", lambda self: None)
    monkeypatch.setattr(LlmJudge, "same_entity", same_entity)
    monkeypatch.setattr(LlmExtractor, "extract", extract)
    return calls


def _thresholds(tenant: uuid.UUID, auto_merge: str, floor: str) -> None:
    with tenant_sync_session(tenant) as session:
        session.execute(
            text(
                "UPDATE tenants SET confidence_auto_merge = :a, confidence_api_floor = :f "
                "WHERE id = :t"
            ),
            {"a": auto_merge, "f": floor, "t": tenant},
        )


def _upload(tenant: uuid.UUID, csv_path: Any) -> uuid.UUID:
    with tenant_sync_session(tenant) as session:
        job_id = _make_job(session, tenant)
    _run_pipeline(job_id, tenant, csv_path)
    return job_id


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_a_job_completes_with_no_human_and_enriched_golden_records(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any, fake_web: dict[str, int]
) -> None:
    # Nothing clears the auto-merge bar and nothing falls under the floor:
    # every cluster has to go through Tier 2.
    _thresholds(e2e_tenant, auto_merge="1.000", floor="0.000")
    csv_path = tmp_path / "variants.csv"
    write_variant_csv(csv_path)

    job_id = _upload(e2e_tenant, csv_path)

    with Session(owner_engine) as session:
        status, progress = session.execute(
            text("SELECT status, stage_progress FROM jobs WHERE id = :j"), {"j": job_id}
        ).one()
        open_clusters = session.execute(
            text(
                "SELECT count(*) FROM clusters WHERE job_id = :j "
                "AND status IN ('PENDING_REVIEW', 'PENDING_API')"
            ),
            {"j": job_id},
        ).scalar_one()
        decided = session.execute(
            text(
                "SELECT c.id, c.status, c.resolution_source, c.auto_decision, "
                "array_agg(s.name ORDER BY s.name) FILTER (WHERE s.name IS NOT NULL) "
                "FROM clusters c LEFT JOIN cluster_members m ON m.cluster_id = c.id "
                "LEFT JOIN silver_records s ON s.record_id = m.record_id "
                "WHERE c.job_id = :j GROUP BY c.id"
            ),
            {"j": job_id},
        ).all()

    assert status == JobStatus.COMPLETED.value
    assert open_clusters == 0
    triaged = [row for row in decided if row.resolution_source == "llm_triage"]
    assert triaged, "the middle band was not sent to Tier 2"
    assert fake_web["judge"] > 0

    for row in triaged:
        assert row.auto_decision["tier"] == 2
        assert row.auto_decision["verdicts"], "a decision without its evidence"
        # The audit trail keeps what the LLM read, not just where it came from.
        for verdict in row.auto_decision["verdicts"]:
            assert verdict["query"]
            assert "Principal:" in verdict["snippets"][0]["snippet"]
        assert row.auto_decision["master_snippets"]
        if row.status == "RESOLVED":
            # The judge said NO to every "Scool": none may survive in a merge.
            assert not any("scool" in name for name in row[4] or []), row[4]
    assert any(row.auto_decision["decision"] == "SPLIT" for row in triaged)
    assert progress["triage"]["details"]["triage_split"] >= 1

    # Tier 3 ran on the completed job: verified values in, the invented one out.
    with Session(owner_engine) as session:
        golden = session.execute(
            text(
                "SELECT golden_fields, enrichment FROM clusters WHERE job_id = :j "
                "AND status IN ('AUTO_MERGED', 'RESOLVED')"
            ),
            {"j": job_id},
        ).all()
        cached = session.execute(
            text(
                "SELECT count(*) FROM enrichment_results WHERE tenant_id = :t "
                "AND strategy = 'golden' AND key_type = 'entity'"
            ),
            {"t": e2e_tenant},
        ).scalar_one()

    assert golden, "no golden records to enrich"
    for fields, enrichment in golden:
        assert fields["principal_name"] == PRINCIPAL
        assert fields["phone"] == PHONE
        assert "email" not in fields
        assert {
            "key": "email",
            "value": "invented@nowhere.example",
            "reason": "not_in_snippets",
        } in (enrichment["rejected"])
        assert enrichment["fields"]["principal_name"]["url"].startswith("https://schools.example/")
    assert cached == len(golden)


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_tier1_separates_below_the_floor_without_asking_anyone(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any, fake_web: dict[str, int]
) -> None:
    # Everything scores under a floor of 1.000: all of it is Tier 1's call.
    _thresholds(e2e_tenant, auto_merge="1.000", floor="1.000")
    csv_path = tmp_path / "variants.csv"
    write_variant_csv(csv_path)

    job_id = _upload(e2e_tenant, csv_path)

    with Session(owner_engine) as session:
        status = session.execute(
            text("SELECT status FROM jobs WHERE id = :j"), {"j": job_id}
        ).scalar_one()
        bands = session.execute(
            text(
                "SELECT status, resolution_source, auto_decision->>'band', count(*) "
                "FROM clusters WHERE job_id = :j GROUP BY 1, 2, 3"
            ),
            {"j": job_id},
        ).all()

    assert status == JobStatus.COMPLETED.value
    assert fake_web["judge"] == 0, "Tier 1 decisions must not consult the LLM"
    # Natural-key clusters may still score exactly 1.000; everything else is
    # kept apart by the threshold, with the band recorded.
    for cluster_status, source, band, _ in bands:
        assert source == "threshold"
        assert (cluster_status, band) in {
            ("REJECTED", "below_floor"),
            ("AUTO_MERGED", "auto_merged"),
        }


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_without_an_llm_the_middle_band_waits_for_a_steward(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    """No LLM configured (the suite default): nothing is separated unread."""
    _thresholds(e2e_tenant, auto_merge="1.000", floor="0.000")
    csv_path = tmp_path / "variants.csv"
    write_variant_csv(csv_path)

    job_id = _upload(e2e_tenant, csv_path)

    with Session(owner_engine) as session:
        status, details = session.execute(
            text("SELECT status, stage_progress->'resolution'->'details' FROM jobs WHERE id = :j"),
            {"j": job_id},
        ).one()
    assert status == JobStatus.AWAITING_REVIEW.value
    assert details["automated_triage"] is False
    assert details["pending_review"] + details["chain_suspect"] > 0


def test_the_search_lock_serialises_across_connections() -> None:
    """Two threads, two pooled connections: one search at a time, platform-wide."""
    held: list[tuple[float, float]] = []

    def hold() -> None:
        with advisory_lock(sync_engine):
            start = time.monotonic()
            time.sleep(0.3)
            held.append((start, time.monotonic()))

    threads = [threading.Thread(target=hold) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    (a_start, a_end), (b_start, b_end) = sorted(held)
    assert b_start >= a_end, "the two holders overlapped"


# --- The Gold export, through the real API ---------------------------------------------


def _api_client(tenant: uuid.UUID, role: str = "org_admin") -> Any:
    from datetime import UTC, datetime, timedelta

    import jwt
    from fastapi.testclient import TestClient

    from app.main import app

    settings = get_settings()
    token = jwt.encode(
        {
            "sub": "export-test",
            "tenant_id": str(tenant),
            "role": role,
            "aud": settings.jwt_audience,
            "iss": settings.jwt_issuer,
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    client = TestClient(app, backend_options={"loop_factory": asyncio.SelectorEventLoop})
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client


def _csv_rows(body: str) -> list[dict[str, str]]:
    import csv
    import io

    return list(csv.DictReader(io.StringIO(body)))


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_the_gold_export_is_one_row_per_entity_with_enrichment_and_audit(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any, fake_web: dict[str, int]
) -> None:
    _thresholds(e2e_tenant, auto_merge="1.000", floor="0.000")
    csv_path = tmp_path / "variants.csv"
    write_variant_csv(csv_path)
    job_id = _upload(e2e_tenant, csv_path)

    response = _api_client(e2e_tenant).get(f"/v1/jobs/{job_id}/export/gold")

    assert response.status_code == 200
    assert response.headers["content-disposition"].endswith('_gold.csv"')
    rows = _csv_rows(response.text)
    assert {"enriched_principal_name", "enriched_phone", "decided_by", "kind"} <= set(rows[0])

    record_ids = [r["record_id"] for r in rows]
    assert len(record_ids) == len(set(record_ids)), "a record appears twice"

    merged = [r for r in rows if r["kind"] == "merged"]
    assert merged, "no merged golden records"
    for row in merged:
        assert row["decided_by"] in {"threshold", "llm_triage"}
        assert row["enriched_principal_name"] == PRINCIPAL
        assert row["enriched_phone"] == PHONE
        assert row["enriched_email"] == "", "an ungrounded value reached the export"
        assert "principal_name=https://schools.example/" in row["enrichment_sources"]
        assert int(row["member_count"]) >= 2

    # Every Valid record is either a merged master, an unmatched record, or
    # a duplicate folded into a master: duplicates must not be rows.
    with Session(owner_engine) as session:
        folded = set(
            session.execute(
                text(
                    "SELECT m.record_id::text FROM cluster_members m "
                    "JOIN clusters c ON c.id = m.cluster_id WHERE c.job_id = :j "
                    "AND c.status IN ('AUTO_MERGED', 'RESOLVED') "
                    "AND m.record_id IS DISTINCT FROM c.master_record_id"
                ),
                {"j": job_id},
            ).scalars()
        )
        valid = session.execute(
            text(
                "SELECT count(*) FROM school_records WHERE job_id = :j AND classification = 'Valid'"
            ),
            {"j": job_id},
        ).scalar_one()
    assert folded and not folded & set(record_ids)
    assert len(rows) == valid - len(folded)
    assert {r["kind"] for r in rows} <= {"merged", "unique"}


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_the_gold_export_waits_for_completion_unless_partial_is_allowed(
    e2e_tenant: uuid.UUID, tmp_path: Any
) -> None:
    """No LLM (the suite default): the middle band waits for a steward, so the
    job is AWAITING_REVIEW. Refused by default; flagged when partial."""
    _thresholds(e2e_tenant, auto_merge="1.000", floor="0.000")
    csv_path = tmp_path / "variants.csv"
    write_variant_csv(csv_path)
    job_id = _upload(e2e_tenant, csv_path)
    client = _api_client(e2e_tenant)

    refused = client.get(f"/v1/jobs/{job_id}/export/gold")
    partial = client.get(f"/v1/jobs/{job_id}/export/gold?allow_partial=true")

    assert refused.status_code == 409 and "allow_partial" in refused.json()["detail"]
    assert partial.status_code == 200
    assert partial.headers["content-disposition"].endswith('_gold_partial.csv"')
    kinds = {r["kind"] for r in _csv_rows(partial.text)}
    assert "unresolved" in kinds


def test_the_gold_export_is_not_reachable_without_a_token() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app, backend_options={"loop_factory": asyncio.SelectorEventLoop})
    assert client.get(f"/v1/jobs/{uuid.uuid4()}/export/gold").status_code == 401


# --- The client-ready export ------------------------------------------------------------


def _variant_csv_with_client_columns(tmp_path: Any) -> Any:
    """The variant fixture plus two columns of the client's own: `fee`, which
    nothing standard holds, and `website`, which collides with a web-found
    column of the same name."""
    path = tmp_path / "variants.csv"
    write_variant_csv(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    out = [lines[0] + ",fee,website"] + [
        f"{line},9000,old{i}.example" for i, line in enumerate(lines[1:])
    ]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_the_client_export_reads_like_a_report_and_keeps_the_clients_columns(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any, fake_web: dict[str, int]
) -> None:
    _thresholds(e2e_tenant, auto_merge="1.000", floor="0.000")
    job_id = _upload(e2e_tenant, _variant_csv_with_client_columns(tmp_path))

    response = _api_client(e2e_tenant).get(f"/v1/jobs/{job_id}/export/client")

    assert response.status_code == 200, response.text
    assert response.headers["content-disposition"].endswith('_final.csv"')
    assert response.content.startswith("﻿".encode()), "Excel needs the BOM for UTF-8"
    rows = _csv_rows(response.text.lstrip("﻿"))
    header = list(rows[0])
    assert header == [
        "School Name",
        "Address",
        "District",
        "State",
        "PIN Code",
        "UDISE Code",
        "Affiliation Number",
        "Principal Name",
        "Website",
        "Phone",
        "Email",
        "Board",
        "Established Year",
        "fee",
        "website (original)",
    ]

    names = {r["School Name"] for r in rows}
    # Which spelling survives as the master is a tie-break on record id, and the
    # fixture's judge splits "Scool" away from its group (its own row), so Unit 0
    # is one or two rows, always one of the three spellings and never anything else.
    unit_0 = {n for n in names if n.endswith("Unit 0")}
    assert unit_0 and unit_0 <= {
        f"Government Higher Primary {spelling} Unit 0" for spelling in ("School", "Schol", "Scool")
    }, sorted(names)[:6]
    assert all(r["fee"] == "9000" for r in rows), "the client's own column is kept"
    assert all(r["website (original)"].startswith("old") for r in rows)
    assert {r["District"] for r in rows} <= {"Bangalore", "Mysuru"}
    assert {r["State"] for r in rows} == {"Karnataka"}
    # Two unrelated-looking rows share one UDISE code: one entity, keyed.
    assert sum(1 for r in rows if r["UDISE Code"] == "29281200612") == 1

    enriched = [r for r in rows if r["Principal Name"]]
    assert enriched and {r["Principal Name"] for r in enriched} == {PRINCIPAL}
    assert all(r["Email"] == "" for r in rows), "an ungrounded value reached the client"

    gold = _csv_rows(_api_client(e2e_tenant).get(f"/v1/jobs/{job_id}/export/gold").text)
    assert len(rows) == len(gold), "one row per entity, same as the raw dump"


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_the_client_export_waits_for_a_finished_job(e2e_tenant: uuid.UUID, tmp_path: Any) -> None:
    """No LLM: the middle band waits for a steward; nothing client-ready yet."""
    _thresholds(e2e_tenant, auto_merge="1.000", floor="0.000")
    csv_path = tmp_path / "variants.csv"
    write_variant_csv(csv_path)
    job_id = _upload(e2e_tenant, csv_path)

    response = _api_client(e2e_tenant).get(f"/v1/jobs/{job_id}/export/client")

    assert response.status_code == 409
    assert "raw dump" in response.json()["detail"]
