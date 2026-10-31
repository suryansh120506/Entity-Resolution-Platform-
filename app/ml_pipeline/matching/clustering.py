"""Clusters from scored pairs — link only what the matcher believes.

The earlier pipeline grouped records into clusters straight from *blocking*
pairs and only scored them afterwards. Blocking pairs say "worth comparing",
not "the same school", so clusters formed that way chained unrelated schools
together and then, correctly, scored the chain at its weakest pair — which
is why nearly every cluster in a real run read 0.000. The fix is the order:
score every candidate pair first, and let only pairs the model scores as
more-likely-than-not matches join records together.

**Cluster confidence is the bottleneck of the strongest chain.** A cluster
is a transitive closure over its links, so it is exactly as trustworthy as
the weakest link it cannot do without: the minimum edge on the *maximum*
spanning tree. Taking the minimum over every internal edge instead would let
one redundant weak edge sink a cluster already held together by strong ones;
taking the mean would let strong edges paper over the one weak join the
cluster actually depends on.

**Oversized components are tightened, not truncated.** A component above
`max_size` is re-split at progressively stricter thresholds until every piece
fits; only if even the strictest threshold cannot split it is it refused —
and reported, never silently dropped.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

#: Pairs at or above this match probability link two records into a cluster.
#: 0.5 is the Fellegi-Sunter decision point — "more likely a match than
#: not". Below it the model's own verdict is "different schools", and
#: sending those to a human would fill the queue with pairs it has already
#: answered.
DEFAULT_LINK_THRESHOLD = 0.5

#: Larger than any plausible real duplicate group, small enough that a
#: steward can still review it as one screen.
DEFAULT_MAX_CLUSTER_SIZE = 50

#: Stricter thresholds tried, in order, on a component above the size cap.
TIGHTENING_THRESHOLDS: tuple[float, ...] = (0.8, 0.9, 0.95, 0.99)


@dataclass(frozen=True)
class ScoredEdge:
    record_a: str
    record_b: str
    score: float
    #: A steward must-link: kept whatever its score and never cut.
    forced: bool = False


@dataclass(frozen=True)
class Constraints:
    """Steward decisions, bound to stable identities rather than records.

    `identity` maps each record to its stable id; a record absent from it is
    its own identity (which is how record-level callers and tests use this).
    Rules are stated between identities and apply to every record carrying
    them, so a never-merge decided on one upload binds the next.

    * `must_link` / `cannot_link`: identity pairs.
    * `not_all_together`: identity groups that may never all share one
      cluster — a rejected group of three or more. Any proper subset may.
    * `forced_groups`: record sets that must merge (shared natural key).
    """

    identity: Mapping[str, str] = field(default_factory=dict)
    must_link: frozenset[frozenset[str]] = frozenset()
    cannot_link: frozenset[frozenset[str]] = frozenset()
    not_all_together: tuple[frozenset[str], ...] = ()
    forced_groups: tuple[frozenset[str], ...] = ()

    @classmethod
    def from_pairs(
        cls,
        must: Iterable[tuple[str, str]],
        cannot: Iterable[tuple[str, str]],
        *,
        groups: Iterable[Iterable[str]] = (),
        identity: Mapping[str, str] | None = None,
        forced_groups: Iterable[Iterable[str]] = (),
    ) -> Constraints:
        return cls(
            identity=dict(identity or {}),
            must_link=frozenset(frozenset(p) for p in must if p[0] != p[1]),
            cannot_link=frozenset(frozenset(p) for p in cannot if p[0] != p[1]),
            not_all_together=tuple(
                g for g in (frozenset(group) for group in groups) if len(g) >= 2
            ),
            forced_groups=tuple(
                g for g in (frozenset(group) for group in forced_groups) if len(g) >= 2
            ),
        )

    def identity_of(self, record: str) -> str:
        return self.identity.get(record, record)

    def partners(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for pair in self.cannot_link:
            a, b = tuple(pair)
            out.setdefault(a, set()).add(b)
            out.setdefault(b, set()).add(a)
        return out

    def forced_edges(self, records: Iterable[str]) -> list[ScoredEdge]:
        """Star edges that realise every must-link and forced group.

        Must-links are transitive over identities (A=B, B=C ⇒ A=B=C), and an
        identity can have several records in one upload, so the closure is
        taken over identities first and then expanded to the records present.
        A star (first record to each other) is the fewest edges that connect
        a group; the spanning tree needs nothing more.
        """
        uf = _UnionFind()
        for pair in self.must_link:
            a, b = tuple(pair)
            uf.union(a, b)
        by_identity: dict[str, list[str]] = {}
        for record in sorted(set(records)):
            by_identity.setdefault(self.identity_of(record), []).append(record)

        groups: list[list[str]] = []
        for identities in uf.groups():
            present = [r for identity in sorted(identities) for r in by_identity.get(identity, ())]
            if len(present) >= 2:
                groups.append(present)
        groups.extend(sorted(g) for g in self.forced_groups)

        edges: list[ScoredEdge] = []
        seen: set[frozenset[str]] = set()
        for group in groups:
            hub = group[0]
            for other in group[1:]:
                key = frozenset((hub, other))
                if other != hub and key not in seen:
                    seen.add(key)
                    a, b = sorted(key)
                    edges.append(ScoredEdge(a, b, 1.0, forced=True))
        return edges


NO_CONSTRAINTS = Constraints()


@dataclass
class PlannedCluster:
    """One cluster to persist."""

    members: list[str]
    confidence: float
    #: Each member's strongest link inside the cluster — the evidence it
    #: belongs at all. Distinct from `confidence`, which is the cluster's.
    member_scores: dict[str, float]
    #: Steward must-links forced together records that a steward cannot-link
    #: says are distinct. Unresolvable by the algorithm; a human decides.
    constraint_conflict: bool = False


@dataclass
class ClusteringResult:
    clusters: list[PlannedCluster] = field(default_factory=list)
    links: int = 0
    tightened_components: int = 0
    #: Records a tightened component shed because none of their links met the
    #: stricter threshold. Counted so "left unclustered" is never silent.
    released_records: int = 0
    refused_sizes: list[int] = field(default_factory=list)
    #: Model links not taken because they would join a cannot-link pair.
    cut_links: int = 0
    #: Steward must-links applied as forced edges.
    forced_links: int = 0
    constraint_conflicts: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "clusters": len(self.clusters),
            "links": self.links,
            "tightened_components": self.tightened_components,
            "released_records": self.released_records,
            "refused_components": len(self.refused_sizes),
            "records_in_refused_components": sum(self.refused_sizes),
            "cut_links": self.cut_links,
            "forced_links": self.forced_links,
            "constraint_conflicts": self.constraint_conflicts,
        }


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self._parent.setdefault(item, item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: str, right: str) -> bool:
        a, b = self.find(left), self.find(right)
        if a == b:
            return False
        self._parent[b] = a
        return True

    def groups(self) -> list[list[str]]:
        out: dict[str, list[str]] = {}
        for item in list(self._parent):
            out.setdefault(self.find(item), []).append(item)
        return list(out.values())


@dataclass
class _Component:
    members: list[str]
    edges: list[ScoredEdge]
    conflict: bool = False


def _grouped(edges: Sequence[ScoredEdge], constraints: Constraints) -> tuple[list[_Component], int]:
    """Constraint-aware connected components: Kruskal over the maximum
    spanning forest, refusing any join that would break a steward rule.

    Edges are taken strongest first, forced edges before all others. Each
    root carries its records, the stable identities they carry, and the
    union of those identities' cannot-link partners, so the two checks on a
    join are cheap set operations on the smaller side:

    * cannot-link — does the smaller side hold an identity the larger side
      is forbidden? (Partners are symmetric, so one direction suffices.)
    * not-all-together — would the joined identity set cover a whole
      rejected group? Only groups touching the smaller side can newly
      complete, so only those are checked.

    A model edge failing either is *cut*: the sides stay separate and the
    tree is built from the next-best edges. That is the cut a post-hoc pass
    over the finished spanning tree would make, applied when the edge is
    considered, so no component ever has to be re-split after the fact.

    A forced edge is never cut. If forced edges themselves break a rule, the
    stewards' decisions contradict each other (or a shared natural key
    contradicts one); the component is kept and flagged for a human.

    Returns (components with ≥2 members and their internal edges, cuts).
    """
    partners = constraints.partners()
    group_index: dict[str, list[frozenset[str]]] = {}
    for group in constraints.not_all_together:
        for identity in group:
            group_index.setdefault(identity, []).append(group)

    parent: dict[str, str] = {}
    members: dict[str, set[str]] = {}
    identities: dict[str, set[str]] = {}
    forbidden: dict[str, set[str]] = {}
    conflicted: set[str] = set()

    def find(x: str) -> str:
        if x not in parent:
            parent[x] = x
            members[x] = {x}
            identity = constraints.identity_of(x)
            identities[x] = {identity}
            forbidden[x] = set(partners.get(identity, ()))
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def violates(small: str, large: str) -> bool:
        if not identities[small].isdisjoint(forbidden[large]):
            return True
        candidates = {g for i in identities[small] for g in group_index.get(i, ())}
        if not candidates:
            return False
        joined = identities[small] | identities[large]
        return any(group <= joined for group in candidates)

    cut = 0
    accepted: list[ScoredEdge] = []
    for edge in sorted(edges, key=lambda e: (not e.forced, -e.score, e.record_a, e.record_b)):
        a, b = find(edge.record_a), find(edge.record_b)
        if a == b:
            accepted.append(edge)
            continue
        small, large = (a, b) if len(members[a]) <= len(members[b]) else (b, a)
        broken = violates(small, large)
        if broken and not edge.forced:
            cut += 1
            continue
        parent[small] = large
        members[large] |= members.pop(small)
        identities[large] |= identities.pop(small)
        forbidden[large] |= forbidden.pop(small)
        if broken or small in conflicted:
            conflicted.add(large)
        conflicted.discard(small)
        accepted.append(edge)

    by_root: dict[str, _Component] = {}
    for edge in accepted:
        root = find(edge.record_a)
        component = by_root.get(root)
        if component is None:
            component = by_root[root] = _Component(sorted(members[root]), [], root in conflicted)
        component.edges.append(edge)
    return [c for c in by_root.values() if len(c.members) > 1], cut


def _plan(component: _Component) -> PlannedCluster:
    """Confidence (max-spanning-tree bottleneck) and per-member scores."""
    uf = _UnionFind()
    bottleneck = 1.0
    member_scores = dict.fromkeys(component.members, 0.0)
    for edge in sorted(component.edges, key=lambda e: (not e.forced, -e.score)):
        if uf.union(edge.record_a, edge.record_b):
            bottleneck = min(bottleneck, edge.score)
        for record in (edge.record_a, edge.record_b):
            member_scores[record] = max(member_scores[record], edge.score)
    return PlannedCluster(
        members=component.members,
        confidence=bottleneck,
        member_scores=member_scores,
        constraint_conflict=component.conflict,
    )


def build_clusters(
    edges: Iterable[ScoredEdge],
    *,
    link_threshold: float = DEFAULT_LINK_THRESHOLD,
    max_size: int = DEFAULT_MAX_CLUSTER_SIZE,
    constraints: Constraints = NO_CONSTRAINTS,
) -> ClusteringResult:
    """Group records linked by edges at or above `link_threshold`, honouring
    steward rules: must-link (forced, any score), cannot-link and
    not-all-together (never joined), all stated over stable identities."""
    edges = list(edges)
    universe = (
        {r for e in edges for r in (e.record_a, e.record_b)}
        | set(constraints.identity)
        | {r for group in constraints.forced_groups for r in group}
    )
    if not constraints.identity:
        # Record-level mode: every record is its own identity, so must-link
        # members are records even when no model edge mentions them.
        universe |= {r for pair in constraints.must_link for r in pair}
    forced = constraints.forced_edges(universe)
    forced_pairs = {frozenset((e.record_a, e.record_b)) for e in forced}
    links = [
        e
        for e in edges
        if e.score >= link_threshold
        and e.record_a != e.record_b
        and frozenset((e.record_a, e.record_b)) not in forced_pairs
    ]
    result = ClusteringResult(links=len(links), forced_links=len(forced))

    components, cut = _grouped([*forced, *links], constraints)
    result.cut_links += cut
    for component in components:
        if len(component.members) <= max_size:
            result.clusters.append(_plan(component))
            continue

        for threshold in TIGHTENING_THRESHOLDS:
            pieces, piece_cut = _grouped(
                [e for e in component.edges if e.forced or e.score >= threshold], constraints
            )
            if all(len(piece.members) <= max_size for piece in pieces):
                result.tightened_components += 1
                result.cut_links += piece_cut
                result.released_records += len(component.members) - sum(
                    len(p.members) for p in pieces
                )
                result.clusters.extend(_plan(piece) for piece in pieces)
                break
        else:
            result.refused_sizes.append(len(component.members))

    result.constraint_conflicts = sum(1 for c in result.clusters if c.constraint_conflict)
    return result
