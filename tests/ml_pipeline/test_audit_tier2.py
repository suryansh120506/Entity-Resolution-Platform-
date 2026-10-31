"""The Tier 2 audit: what is sampled, and that the evidence is shown."""

from __future__ import annotations

import csv
import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from app.enrichment.sources import EntityFields, SearchHit
from app.enrichment.triage import triage_cluster
from app.ontology.engine import TriageRules

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_tier2.py"
_spec = importlib.util.spec_from_file_location("audit_tier2", _SCRIPT)
assert _spec and _spec.loader
audit = importlib.util.module_from_spec(_spec)
sys.modules["audit_tier2"] = audit  # dataclasses look their module up here
_spec.loader.exec_module(audit)  # scripts/ is not a package


def _cluster(verdicts: list[tuple[str, str | None]], snippets: bool = True) -> dict[str, Any]:
    master = str(uuid.uuid4())
    return {
        "id": uuid.uuid4(),
        "job_id": uuid.uuid4(),
        "confidence": "0.900",
        "auto_decision": {
            "tier": 2,
            "model": "m",
            "master_record_id": master,
            "master_query": "q-master",
            "master_snippets": [{"title": "T", "url": "https://a", "snippet": "S"}]
            if snippets
            else [],
            "master_urls": ["https://a"],
            "verdicts": [
                {
                    "record_id": str(uuid.uuid4()),
                    "verdict": verdict,
                    "reason": "r",
                    "query": "q-other",
                    "snippets": [{"title": "T2", "url": "https://b", "snippet": "S2"}]
                    if snippets
                    else [],
                    "urls": ["https://b"],
                }
                for _, verdict in verdicts
            ],
        },
    }


def test_only_yes_and_no_verdicts_are_the_llms_judgement() -> None:
    cluster = _cluster([("a", "YES"), ("b", "NO"), ("c", None), ("d", None)])

    pairs = audit.collect_pairs([cluster])

    assert sorted(p.verdict for p in pairs) == ["NO", "YES"]


def test_a_sample_is_half_yes_and_half_no() -> None:
    pairs = audit.collect_pairs(
        [_cluster([("x", "YES")]) for _ in range(40)] + [_cluster([("x", "NO")]) for _ in range(25)]
    )

    sample, available = audit.draw_sample(pairs, per_side=15, seed=1)

    assert [p.verdict for p in sample].count("YES") == 15
    assert [p.verdict for p in sample].count("NO") == 15
    assert available == {"YES": 40, "NO": 25}


def test_a_shortfall_is_reported_not_padded() -> None:
    pairs = audit.collect_pairs(
        [_cluster([("x", "YES")]) for _ in range(20)] + [_cluster([("x", "NO")])] * 3
    )

    sample, available = audit.draw_sample(pairs, per_side=15, seed=1)

    assert [p.verdict for p in sample].count("NO") == 3
    assert available["NO"] == 3


def test_the_draw_is_random_but_reproducible() -> None:
    pairs = audit.collect_pairs([_cluster([("x", "YES"), ("y", "NO")]) for _ in range(60)])

    first, _ = audit.draw_sample(pairs, 10, seed=7)
    again, _ = audit.draw_sample(pairs, 10, seed=7)
    other, _ = audit.draw_sample(pairs, 10, seed=8)

    assert first == again
    assert first != other


def test_unstored_decisions_are_flagged_so_they_can_be_skipped() -> None:
    old = audit.collect_pairs([_cluster([("x", "YES")], snippets=False)])
    new = audit.collect_pairs([_cluster([("x", "YES")])])

    assert not old[0].has_snippets and new[0].has_snippets


def test_the_report_and_csv_show_the_records_the_verdict_and_the_snippets(tmp_path: Path) -> None:
    pair = audit.collect_pairs([_cluster([("x", "NO")])])[0]
    names = {
        pair.master_id: {
            "name": "dps",
            "address": "mathura road",
            "district": "delhi",
            "state": "delhi",
            "pincode": "110065",
        },
        pair.other_id: {
            "name": "dps",
            "address": "sector 45",
            "district": "gurgaon",
            "state": "haryana",
            "pincode": "122003",
        },
    }

    report = audit.render_report([pair], names)
    audit.write_csv(tmp_path / "a.csv", [pair], names)
    (row,) = list(csv.DictReader((tmp_path / "a.csv").open(encoding="utf-8-sig")))

    assert "LLM said NO" in report and "mathura road" in report and "sector 45" in report
    assert "S2" in report and "q-other" in report
    assert row["llm_verdict"] == "NO" and row["your_verdict"] == ""
    assert row["a_address"] == "mathura road" and row["b_district"] == "gurgaon"
    assert "S2" in row["b_snippets"] and "https://b" in row["b_snippets"]


def test_triage_stores_the_query_and_snippets_the_llm_was_shown() -> None:
    def search(query: str, n: int) -> list[SearchHit]:
        return [SearchHit(f"title {query}", "https://x.example", "the snippet text " * 50)]

    decision = triage_cluster(
        EntityFields("m", name="alpha school"),
        [EntityFields("o", name="alpha school")],
        rules=TriageRules(),
        search=search,
        judge=lambda *args: True,
    )
    audit_json = decision.as_json()

    assert audit_json["master_query"] == '"alpha school"'
    assert audit_json["master_snippets"][0]["url"] == "https://x.example"
    (verdict,) = audit_json["verdicts"]
    assert verdict["query"] == '"alpha school"'
    assert 0 < len(verdict["snippets"][0]["snippet"]) <= 400, "snippets are kept, truncated"


@pytest.mark.parametrize("flag", ["--per-side", "--seed", "--csv", "--job", "--include-unstored"])
def test_the_script_documents_its_options(flag: str) -> None:
    from click.testing import CliRunner

    result = CliRunner().invoke(audit.main, ["--help"])

    assert flag in result.output
