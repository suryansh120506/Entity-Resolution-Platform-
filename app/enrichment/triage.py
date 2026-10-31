"""Tier 2: settle a cluster the matcher was unsure about, with no human.

The cluster's master (its most complete record, `select_master_record`) is
searched once; every other member is searched and put to the LLM as a pair
with the master: "same entity, YES or NO?". A member joins the golden
record only on a YES.

Why the master against *each* member, not just "the top two records": a
cluster flagged as a suspected chain (A~B~C~D, A never linked to D) is
exactly the case where two members agreeing says nothing about the rest.
Checking every member against one anchor turns a chain into a star the LLM
actually vouched for, edge by edge. For a two-record cluster, by far the
common case, it is the same single question.

Everything short of a clear YES keeps records apart: a NO, an unparseable
reply, no web results for either record, or a member past the
`max_members_checked` cap. A wrong merge hides a real entity inside
another; a missed duplicate is one extra row. So doubt resolves to SEPARATE.

Outcome, in the steward's vocabulary so downstream code needs no new path:

    every member YES          -> MERGE   (RESOLVED, master kept)
    some YES, some not        -> SPLIT   (RESOLVED; the rest released)
    no YES                    -> REJECT  (REJECTED; all released)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from app.enrichment.sources import EntityFields, SearchFn, SearchHit
from app.ontology.engine import TriageRules, render_query

Decision = Literal["MERGE", "SPLIT", "REJECT"]
JudgeFn = Callable[
    [EntityFields, Sequence[SearchHit], EntityFields, Sequence[SearchHit]], bool | None
]

#: URLs kept per record in the audit trail.
_AUDIT_URLS = 3
#: The snippets the LLM read are stored too, truncated, so `scripts/audit_tier2.py`
#: can show exactly what a verdict was based on. Search results change from
#: one day to the next; a link alone cannot reproduce what the LLM saw.
_SNIPPET_CHARS = 400


def snippets_json(hits: Sequence[SearchHit]) -> list[dict[str, str]]:
    return [
        {"title": h.title[:200], "url": h.url, "snippet": h.snippet[:_SNIPPET_CHARS]} for h in hits
    ]


@dataclass(frozen=True, slots=True)
class PairVerdict:
    record_id: str
    #: True = YES, False = NO, None = no usable answer.
    same: bool | None
    reason: Literal["llm_yes", "llm_no", "llm_unparseable", "no_web_evidence", "over_member_cap"]
    urls: tuple[str, ...] = ()
    #: What this record was searched for and what came back: the evidence the
    #: LLM saw for the second record of the pair.
    query: str = ""
    snippets: tuple[dict[str, str], ...] = ()

    def as_json(self) -> dict[str, Any]:
        verdict = {True: "YES", False: "NO", None: None}[self.same]
        return {
            "record_id": self.record_id,
            "verdict": verdict,
            "reason": self.reason,
            "urls": list(self.urls),
            "query": self.query,
            "snippets": list(self.snippets),
        }


@dataclass(frozen=True, slots=True)
class TriageDecision:
    decision: Decision
    master_id: str
    kept: frozenset[str]
    excluded: frozenset[str]
    verdicts: tuple[PairVerdict, ...] = field(default_factory=tuple)
    master_urls: tuple[str, ...] = ()
    master_query: str = ""
    master_snippets: tuple[dict[str, str], ...] = ()

    def as_json(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "master_record_id": self.master_id,
            "master_urls": list(self.master_urls),
            "master_query": self.master_query,
            "master_snippets": list(self.master_snippets),
            "verdicts": [v.as_json() for v in self.verdicts],
        }


def triage_cluster(
    master: EntityFields,
    others: Sequence[EntityFields],
    *,
    rules: TriageRules,
    search: SearchFn,
    judge: JudgeFn,
) -> TriageDecision:
    """Verify each member against the master; keep only the YESes."""
    checked = others[: rules.max_members_checked]
    over_cap = others[rules.max_members_checked :]

    def look_up(entity: EntityFields) -> tuple[str, list[SearchHit]]:
        query = render_query(rules.query, entity.template_fields())
        return query, search(query, rules.max_results)

    master_query, master_hits = look_up(master)
    verdicts: list[PairVerdict] = []
    for other in checked:
        other_query, other_hits = look_up(other)
        urls = tuple(h.url for h in other_hits[:_AUDIT_URLS])
        snippets = tuple(snippets_json(other_hits))
        if not master_hits and not other_hits:
            # Nothing for the LLM to read: asking anyway would get a guess.
            verdicts.append(
                PairVerdict(other.record_id, None, "no_web_evidence", urls, other_query, snippets)
            )
            continue
        same = judge(master, master_hits, other, other_hits)
        reason = {True: "llm_yes", False: "llm_no", None: "llm_unparseable"}[same]
        verdicts.append(
            PairVerdict(other.record_id, same, reason, urls, other_query, snippets)  # type: ignore[arg-type]
        )
    verdicts.extend(PairVerdict(o.record_id, None, "over_member_cap") for o in over_cap)

    kept = frozenset({master.record_id} | {v.record_id for v in verdicts if v.same is True})
    excluded = frozenset(o.record_id for o in others) - kept
    decision: Decision = "REJECT" if len(kept) == 1 else ("SPLIT" if excluded else "MERGE")
    return TriageDecision(
        decision=decision,
        master_id=master.record_id,
        kept=kept,
        excluded=excluded,
        verdicts=tuple(verdicts),
        master_urls=tuple(h.url for h in master_hits[:_AUDIT_URLS]),
        master_query=master_query,
        master_snippets=tuple(snippets_json(master_hits)),
    )
