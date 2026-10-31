"""Spot-check the Tier 2 LLM verdicts: a random sample, with the evidence.

Each Tier 2 decision is a set of pair verdicts (the cluster's master record
against one other member: YES or NO). This draws a random sample of those
pairs, half YES and half NO, and shows for each:

* the two records (name, address, district, state, PIN) side by side,
* the LLM's exact YES / NO,
* the search queries and the snippets the LLM was shown.

It prints a readable report and writes a CSV with an empty `your_verdict`
column: fill in "agree" or "disagree" per row, and the agreement rate is the
measured accuracy of the LLM on this data.

    python scripts/audit_tier2.py                    # 30 pairs: 15 YES, 15 NO
    python scripts/audit_tier2.py --job <job_id> --per-side 25 --seed 7
    python scripts/audit_tier2.py --csv audit.csv --no-print

Only decisions made after snippets began to be stored carry them; earlier ones
have links but not the text the LLM read, so they are skipped unless you pass
`--include-unstored`. The sample is random, not the first N: the earliest
decisions are the highest-confidence (easiest) ones and would flatter the LLM.
"""

from __future__ import annotations

import csv
import random
import textwrap
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
from sqlalchemy import text

from app.db.session import platform_session

_FIELDS = ("name", "address", "district", "state", "pincode")


@dataclass(frozen=True)
class AuditPair:
    cluster_id: str
    job_id: str
    confidence: str | None
    model: str | None
    verdict: str  # "YES" | "NO"
    master_id: str
    other_id: str
    master_query: str = ""
    other_query: str = ""
    master_snippets: tuple[dict[str, str], ...] = ()
    other_snippets: tuple[dict[str, str], ...] = ()
    master_urls: tuple[str, ...] = ()
    other_urls: tuple[str, ...] = ()
    names: dict[str, dict[str, str]] = field(default_factory=dict, compare=False)

    @property
    def has_snippets(self) -> bool:
        return bool(self.master_snippets or self.other_snippets)


def collect_pairs(clusters: list[dict[str, Any]]) -> list[AuditPair]:
    """Flatten Tier 2 cluster decisions into one entry per YES/NO pair verdict.

    Verdicts that are not a YES or a NO (no web evidence, an unreadable reply,
    a member over the cap) are not the LLM's judgement and are left out.
    """
    pairs: list[AuditPair] = []
    for cluster in clusters:
        decision = cluster["auto_decision"] or {}
        for verdict in decision.get("verdicts") or []:
            if verdict.get("verdict") not in ("YES", "NO"):
                continue
            pairs.append(
                AuditPair(
                    cluster_id=str(cluster["id"]),
                    job_id=str(cluster["job_id"]),
                    confidence=None
                    if cluster["confidence"] is None
                    else str(cluster["confidence"]),
                    model=decision.get("model"),
                    verdict=verdict["verdict"],
                    master_id=str(decision.get("master_record_id")),
                    other_id=str(verdict["record_id"]),
                    master_query=decision.get("master_query", ""),
                    other_query=verdict.get("query", ""),
                    master_snippets=tuple(decision.get("master_snippets") or ()),
                    other_snippets=tuple(verdict.get("snippets") or ()),
                    master_urls=tuple(decision.get("master_urls") or ()),
                    other_urls=tuple(verdict.get("urls") or ()),
                )
            )
    return pairs


def draw_sample(
    pairs: list[AuditPair], per_side: int, seed: int | None = None
) -> tuple[list[AuditPair], dict[str, int]]:
    """`per_side` random YES pairs and `per_side` random NO pairs.

    Returns the sample (YES first, then NO) and how many of each were
    available, so a shortfall is reported rather than silently accepted.
    """
    rng = random.Random(seed)
    sample: list[AuditPair] = []
    available: dict[str, int] = {}
    for side in ("YES", "NO"):
        pool = sorted(
            (p for p in pairs if p.verdict == side), key=lambda p: (p.cluster_id, p.other_id)
        )
        available[side] = len(pool)
        sample.extend(rng.sample(pool, min(per_side, len(pool))))
    return sample, available


def _load_clusters(job_id: uuid.UUID | None) -> list[dict[str, Any]]:
    with platform_session() as session:
        rows = session.execute(
            text(
                "SELECT id, job_id, confidence_score, auto_decision FROM clusters "
                "WHERE resolution_source = 'llm_triage' "
                "AND (CAST(:job AS uuid) IS NULL OR job_id = CAST(:job AS uuid))"
            ),
            {"job": str(job_id) if job_id else None},
        ).all()
    return [{"id": r[0], "job_id": r[1], "confidence": r[2], "auto_decision": r[3]} for r in rows]


def _load_names(record_ids: set[str]) -> dict[str, dict[str, str]]:
    if not record_ids:
        return {}
    with platform_session() as session:
        rows = session.execute(
            text(
                "SELECT record_id::text, name, address, district, state, pincode "
                "FROM silver_records WHERE record_id = ANY(CAST(:ids AS uuid[]))"
            ),
            {"ids": sorted(record_ids)},
        ).all()
    return {r[0]: dict(zip(_FIELDS, (v or "" for v in r[1:]), strict=True)) for r in rows}


def _record_line(record: dict[str, str]) -> str:
    place = ", ".join(
        v for v in (record.get("district"), record.get("state"), record.get("pincode")) if v
    )
    return f"{record.get('name', '?')}  |  {record.get('address', '')}  |  {place}"


def _snippet_lines(snippets: tuple[dict[str, str], ...], urls: tuple[str, ...]) -> list[str]:
    if snippets:
        return [f"- {s['title']}: {s['snippet']}  <{s['url']}>" for s in snippets]
    if urls:
        return [f"- (text not stored) <{u}>" for u in urls]
    return ["- (no search results)"]


def render_report(sample: list[AuditPair], names: dict[str, dict[str, str]]) -> str:
    out: list[str] = []
    for i, pair in enumerate(sample, 1):
        a, b = names.get(pair.master_id, {}), names.get(pair.other_id, {})
        out.append(
            f"[{i}/{len(sample)}]  LLM said {pair.verdict}   (cluster {pair.cluster_id[:8]}, "
            f"score {pair.confidence}, model {pair.model})"
        )
        out.append(f"  A: {_record_line(a)}")
        out.append(f"  B: {_record_line(b)}")
        for label, query, snippets, urls in (
            ("A", pair.master_query, pair.master_snippets, pair.master_urls),
            ("B", pair.other_query, pair.other_snippets, pair.other_urls),
        ):
            out.append(f"  Search for {label}: {query or '(not stored)'}")
            for line in _snippet_lines(snippets, urls):
                out.append(
                    textwrap.fill(line, 110, initial_indent="    ", subsequent_indent="      ")
                )
        out.append("")
    return "\n".join(out)


def write_csv(path: Path, sample: list[AuditPair], names: dict[str, dict[str, str]]) -> None:
    def flat(snippets: tuple[dict[str, str], ...], urls: tuple[str, ...]) -> str:
        if snippets:
            return "\n".join(f"{s['title']} :: {s['snippet']} <{s['url']}>" for s in snippets)
        return "\n".join(f"(text not stored) {u}" for u in urls)

    header = [
        "llm_verdict",
        "your_verdict",
        "cluster_id",
        "job_id",
        "confidence",
        "model",
        *(f"a_{f}" for f in _FIELDS),
        *(f"b_{f}" for f in _FIELDS),
        "a_search_query",
        "a_snippets",
        "b_search_query",
        "b_snippets",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:  # BOM: opens cleanly in Excel
        writer = csv.writer(handle)
        writer.writerow(header)
        for pair in sample:
            a, b = names.get(pair.master_id, {}), names.get(pair.other_id, {})
            writer.writerow(
                [
                    pair.verdict,
                    "",
                    pair.cluster_id,
                    pair.job_id,
                    pair.confidence,
                    pair.model,
                    *(a.get(f, "") for f in _FIELDS),
                    *(b.get(f, "") for f in _FIELDS),
                    pair.master_query,
                    flat(pair.master_snippets, pair.master_urls),
                    pair.other_query,
                    flat(pair.other_snippets, pair.other_urls),
                ]
            )


@click.command()
@click.option("--job", "job_id", type=click.UUID, default=None, help="Limit to one job.")
@click.option("--per-side", default=15, show_default=True, help="Pairs to draw for YES and for NO.")
@click.option("--seed", type=int, default=None, help="Fix the draw so a sample can be reproduced.")
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(path_type=Path),
    default=Path("tier2_audit.csv"),
    show_default=True,
)
@click.option("--no-print", is_flag=True, help="Write the CSV only.")
@click.option(
    "--include-unstored",
    is_flag=True,
    help="Also sample decisions made before snippets were stored (links only).",
)
def main(
    job_id: uuid.UUID | None,
    per_side: int,
    seed: int | None,
    csv_path: Path,
    no_print: bool,
    include_unstored: bool,
) -> None:
    clusters = _load_clusters(job_id)
    pairs = collect_pairs(clusters)
    stored = [p for p in pairs if p.has_snippets]
    click.echo(
        f"{len(clusters)} Tier 2 decisions, {len(pairs)} YES/NO pair verdicts, "
        f"{len(stored)} with the snippets stored."
    )
    pool = pairs if include_unstored else stored
    if not pool:
        raise click.ClickException(
            "Nothing to audit yet: no decision with stored snippets. Let triage run a little "
            "longer, or pass --include-unstored to sample the earlier ones (links only)."
        )

    sample, available = draw_sample(pool, per_side, seed)
    for side in ("YES", "NO"):
        got = sum(1 for p in sample if p.verdict == side)
        if got < per_side:
            click.echo(
                f"Only {available[side]} {side} verdicts available; sampled {got}.", err=True
            )

    names = _load_names({r for p in sample for r in (p.master_id, p.other_id)})
    if not no_print:
        click.echo("")
        click.echo(render_report(sample, names))
    write_csv(csv_path, sample, names)
    click.echo(f"Wrote {len(sample)} pairs to {csv_path.resolve()}")
    click.echo("Fill in `your_verdict` (agree / disagree) per row; agreement = the LLM's accuracy.")


if __name__ == "__main__":
    main()
