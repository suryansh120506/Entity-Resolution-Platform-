"""Loading steward rules for a set of records, at the identity level.

Shared by in-upload matching and gold linking: both need the same answer to
"which steward decisions bind these records?", and both must ignore rules
whose identities are not all present (a rule that cannot be completed in
this graph constrains nothing in it).

Steward rules are human-scale data — one row per decision — so the tenant's
whole rule set is loaded and filtered in Python. That stays correct at any
upload size, where an `IN (...)` over every identity of a 50k-row upload
would exceed the driver's bind-parameter limit.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from app.ml_pipeline.matching.clustering import Constraints
from app.models.medallion import EntityConstraint, EntityGroupConstraint, SilverRecord


@dataclass(frozen=True, slots=True)
class TenantRules:
    pairs: list[tuple[str, str, str]]
    groups: list[frozenset[str]]


def load_tenant_rules(session: Any) -> TenantRules:
    """Every identity-level rule of the bound tenant (RLS scopes the read)."""
    pairs = [
        (left, right, kind)
        for left, right, kind in session.execute(
            select(EntityConstraint.identity_l, EntityConstraint.identity_r, EntityConstraint.kind)
        ).all()
    ]
    groups = [
        frozenset(identities)
        for identities in session.execute(select(EntityGroupConstraint.identities)).scalars()
    ]
    return TenantRules(pairs=pairs, groups=groups)


def load_job_identities(
    session: Any, job_id: Any, records: set[str]
) -> tuple[dict[str, str], list[frozenset[str]]]:
    """(record -> stable_id, record groups sharing a natural-key value).

    Records without a stable id (Silver not yet backfilled) are simply
    absent from the map: they act as their own identity, which binds them to
    no rule — the safe default for a record we cannot identify.
    """
    identity: dict[str, str] = {}
    by_key: dict[tuple[str, str], set[str]] = {}
    for record_id, stable_id, natural_keys in session.execute(
        select(SilverRecord.record_id, SilverRecord.stable_id, SilverRecord.natural_keys).where(
            SilverRecord.job_id == job_id
        )
    ).all():
        key = str(record_id)
        if key not in records:
            continue
        if stable_id:
            identity[key] = stable_id
        for key_type, value in (natural_keys or {}).items():
            by_key.setdefault((key_type, str(value)), set()).add(key)
    return identity, [frozenset(g) for g in by_key.values() if len(g) >= 2]


def build_constraints(
    identity: dict[str, str],
    rules: TenantRules,
    *,
    forced_groups: Iterable[frozenset[str]] = (),
) -> Constraints:
    """The subset of the tenant's rules that can bind this record set."""
    present = set(identity.values())
    must: list[tuple[str, str]] = []
    cannot: list[tuple[str, str]] = []
    for left, right, kind in rules.pairs:
        if left in present and right in present:
            (must if kind == "must_link" else cannot).append((left, right))
    groups = [g for g in rules.groups if g <= present]
    return Constraints.from_pairs(
        must, cannot, groups=groups, identity=identity, forced_groups=forced_groups
    )
