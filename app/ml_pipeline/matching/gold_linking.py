"""Deciding which incoming→gold links to keep. Pure: no database, no Splink.

Splink `link_only` proposes scored pairs (incoming record, pool record).
This module turns them into decisions, in a fixed order:

1. **Identity forces.** A shared natural key (same UDISE code) or a steward
   must-link between the two identities links them whatever the score —
   Splink may not even have proposed the pair, so must-link partners present
   in the pool are added here.
2. **Steward vetoes.** A cannot-link between any identity on the incoming
   side (the record's whole in-upload cluster) and any on the gold side (the
   target's whole entity), or a not-all-together group the link would
   complete, removes a scored link. A *forced* link that hits a veto is a
   contradiction between decisions: kept, but only for review.
3. **Name gate.** The same veto in-upload matching applies (gender, kind,
   branch number, chain names), for score-based links only.
4. **Thresholds.** ≥ the tenant's auto-merge threshold: auto_linked;
   ≥ the link threshold: review; below: dropped.
5. **Ambiguity.** An incoming record auto-linked to two *different* gold
   identities cannot be both; its score-based auto links drop to review.
"""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

from app.ml_pipeline.matching.name_gate import RecordText, compatible


@dataclass(frozen=True, slots=True)
class LinkCandidate:
    incoming_id: str
    pool_id: str
    probability: float
    weight: float | None
    gamma: list[int] | None


@dataclass(frozen=True, slots=True)
class IncomingRecord:
    record_id: str
    stable_id: str | None
    identity_kind: str | None
    text: RecordText
    #: Identities of the record's whole in-upload cluster (itself if none):
    #: linking the record links its cluster, so vetoes are checked for all.
    cluster_identities: frozenset[str]


@dataclass(frozen=True, slots=True)
class PoolRecord:
    record_id: str
    stable_id: str
    identity_kind: str
    target_kind: str
    entity_id: uuid.UUID | None
    text: RecordText
    #: Identities of the whole gold entity the record represents.
    entity_identities: frozenset[str]


@dataclass(frozen=True, slots=True)
class LinkDecision:
    incoming_id: str
    pool_id: str
    probability: float
    weight: float | None
    gamma: list[int] | None
    status: str
    reason: str


@dataclass
class LinkStats:
    candidates: int = 0
    forced: int = 0
    blocked_by_constraint: int = 0
    name_vetoed: int = 0
    below_threshold: int = 0
    ambiguous_demoted: int = 0
    statuses: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, object]:
        return {
            "candidates": self.candidates,
            "forced": self.forced,
            "blocked_by_constraint": self.blocked_by_constraint,
            "name_vetoed": self.name_vetoed,
            "below_threshold": self.below_threshold,
            "ambiguous_demoted": self.ambiguous_demoted,
            **{f"links_{status}": count for status, count in self.statuses.items()},
        }


def decide_links(
    candidates: Iterable[LinkCandidate],
    incoming: Mapping[str, IncomingRecord],
    pool: Mapping[str, PoolRecord],
    *,
    must: set[frozenset[str]],
    cannot: set[frozenset[str]],
    groups: Iterable[frozenset[str]],
    auto_threshold: float,
    link_threshold: float,
    force_on_key: bool,
) -> tuple[list[LinkDecision], LinkStats]:
    groups = list(groups)
    stats = LinkStats()
    by_pair: dict[tuple[str, str], LinkCandidate] = {}
    for candidate in candidates:
        if candidate.incoming_id in incoming and candidate.pool_id in pool:
            by_pair[(candidate.incoming_id, candidate.pool_id)] = candidate

    # Must-link partners Splink's blocking never put side by side.
    pool_by_identity = {record.stable_id: record for record in pool.values()}
    for record in incoming.values():
        if record.stable_id is None:
            continue
        for pair in must:
            if record.stable_id in pair:
                (partner,) = pair - {record.stable_id}
                target = pool_by_identity.get(partner)
                if target is not None and (record.record_id, target.record_id) not in by_pair:
                    by_pair[(record.record_id, target.record_id)] = LinkCandidate(
                        record.record_id, target.record_id, 1.0, None, None
                    )

    decisions: list[LinkDecision] = []
    for (incoming_id, pool_id), candidate in by_pair.items():
        stats.candidates += 1
        source, target = incoming[incoming_id], pool[pool_id]
        reason = _forced_reason(source, target, must, force_on_key)
        if reason != "score":
            stats.forced += 1

        if _vetoed(source, target, cannot, groups):
            if reason == "score":
                stats.blocked_by_constraint += 1
                continue
            decisions.append(_decision(candidate, "review", reason))
            continue

        if reason != "score":
            decisions.append(_decision(candidate, "auto_linked", reason))
            continue
        if not compatible(source.text, target.text):
            stats.name_vetoed += 1
            continue
        if candidate.probability >= auto_threshold:
            decisions.append(_decision(candidate, "auto_linked", reason))
        elif candidate.probability >= link_threshold:
            decisions.append(_decision(candidate, "review", reason))
        else:
            stats.below_threshold += 1

    decisions = _demote_ambiguous(decisions, pool, stats)
    stats.statuses.update(d.status for d in decisions)
    return decisions, stats


def _forced_reason(
    source: IncomingRecord, target: PoolRecord, must: set[frozenset[str]], force_on_key: bool
) -> str:
    if (
        force_on_key
        and source.identity_kind == "natural_key"
        and target.identity_kind == "natural_key"
        and source.stable_id == target.stable_id
    ):
        return "natural_key"
    if source.stable_id and frozenset((source.stable_id, target.stable_id)) in must:
        return "must_link"
    return "score"


def _vetoed(
    source: IncomingRecord,
    target: PoolRecord,
    cannot: set[frozenset[str]],
    groups: list[frozenset[str]],
) -> bool:
    left, right = source.cluster_identities, target.entity_identities
    if any(frozenset((a, b)) in cannot for a in left for b in right if a != b):
        return True
    joined = left | right
    return any(group <= joined for group in groups)


def _decision(candidate: LinkCandidate, status: str, reason: str) -> LinkDecision:
    return LinkDecision(
        candidate.incoming_id,
        candidate.pool_id,
        candidate.probability,
        candidate.weight,
        candidate.gamma,
        status,
        reason,
    )


def _demote_ambiguous(
    decisions: list[LinkDecision], pool: Mapping[str, PoolRecord], stats: LinkStats
) -> list[LinkDecision]:
    auto_targets: dict[str, set[str]] = {}
    for d in decisions:
        if d.status == "auto_linked":
            auto_targets.setdefault(d.incoming_id, set()).add(pool[d.pool_id].stable_id)
    ambiguous = {incoming for incoming, ids in auto_targets.items() if len(ids) > 1}
    if not ambiguous:
        return decisions
    out: list[LinkDecision] = []
    for d in decisions:
        if d.incoming_id in ambiguous and d.status == "auto_linked" and d.reason == "score":
            stats.ambiguous_demoted += 1
            out.append(replace(d, status="review"))
        else:
            out.append(d)
    return out
