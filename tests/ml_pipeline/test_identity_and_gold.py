"""Stable identity, identity-level rules, and gold-link decisions. No database."""

from __future__ import annotations

import uuid

import pytest

from app.api.v1.routes.clusters import identity_rules
from app.ml_pipeline.matching.clustering import Constraints, ScoredEdge, build_clusters
from app.ml_pipeline.matching.gold_linking import (
    IncomingRecord,
    LinkCandidate,
    PoolRecord,
    decide_links,
)
from app.ml_pipeline.matching.name_gate import RecordText
from app.models.enums import FeedbackDecision
from app.ontology import OntologyEngine
from app.ontology.identity import derive_identity


def _identity(**overrides: object) -> tuple[str, str]:
    fields: dict[str, object] = {
        "ontology": "education",
        "record_id": "r1",
        "natural_keys": {},
        "key_priority": ["udise_code", "cbse_affiliation_no"],
        "name": "saint josephs convent school",
        "name_distinctive": "josephs",
        "address": "near bus stand",
        "pincode": "175101",
        "district": "kullu",
        "state": "himachal pradesh",
    }
    fields.update(overrides)
    identity = derive_identity(**fields)  # type: ignore[arg-type]
    return identity.stable_id, identity.kind


# --- Derivation ------------------------------------------------------------------------


def test_the_strongest_natural_key_roots_the_identity() -> None:
    sid, kind = _identity(natural_keys={"cbse_affiliation_no": "2130176", "udise_code": "1" * 11})

    assert (sid, kind) == (f"nk:udise_code:{'1' * 11}", "natural_key")


def test_without_a_key_the_identity_is_a_deterministic_fingerprint() -> None:
    first, kind = _identity()
    again, _ = _identity(record_id="some-other-record")

    assert kind == "fingerprint" and first == again and first.startswith("fp:education:")


def test_the_fingerprint_ignores_volatile_fields_but_not_identifying_ones() -> None:
    base, _ = _identity()

    assert _identity(address="opposite the bus stand")[0] == base, "address is not identifying"
    assert _identity(pincode="175102")[0] != base
    assert _identity(name="saint marys convent school", name_distinctive="marys")[0] != base


def test_without_a_pin_district_and_state_anchor_the_fingerprint() -> None:
    sid, kind = _identity(pincode=None)

    assert kind == "fingerprint" and sid != _identity()[0]


def test_an_all_generic_name_needs_the_address_to_be_safe() -> None:
    """ "Government Primary School" in one PIN code is several schools."""
    generic = {"name": "government primary school", "name_distinctive": None}
    a, kind_a = _identity(**generic, address="ward 3")
    b, _ = _identity(**generic, address="ward 7")

    assert kind_a == "fingerprint" and a != b
    assert _identity(**generic, address=None)[1] == "record"


def test_no_name_or_no_anchor_means_no_stable_identity() -> None:
    assert _identity(name=None) == ("rec:r1", "record")
    assert _identity(pincode=None, district=None)[1] == "record"


def test_the_engine_uses_the_ontology_key_priority() -> None:
    engine = OntologyEngine.load("education")
    silver = engine.standardize(
        {"name": "KV Jalahalli", "udise": "29281200612", "remarks": "CBSE Affiliation No. 830013"}
    )

    assert engine.identity("r1", silver).stable_id == "nk:udise_code:29281200612"


# --- Verdicts become identity rules ----------------------------------------------------

R = [uuid.UUID(int=i) for i in range(1, 7)]


def test_split_writes_never_merges_between_distinct_identities_only() -> None:
    identity = {R[0]: "fp:a", R[1]: "fp:a", R[2]: "fp:b"}
    rules = identity_rules(FeedbackDecision.SPLIT, {R[0], R[1]}, {R[2]}, R[0], identity)

    assert rules.pairs == {("fp:a", "fp:b"): "cannot_link"}
    assert rules.unbound == 1, "R0~R1 share an identity: no rule can separate or join them"


def test_merge_writes_a_must_link_star_on_the_master_identity() -> None:
    identity = {R[0]: "fp:a", R[1]: "fp:b", R[2]: "fp:c"}
    rules = identity_rules(FeedbackDecision.MERGE, {R[0], R[1], R[2]}, set(), R[0], identity)

    assert rules.pairs == {("fp:a", "fp:b"): "must_link", ("fp:a", "fp:c"): "must_link"}


def test_rejecting_two_identities_is_a_never_merge() -> None:
    identity = {R[0]: "fp:a", R[1]: "fp:b", R[2]: "fp:a"}
    rules = identity_rules(FeedbackDecision.REJECT, {R[0], R[1], R[2]}, set(), None, identity)

    assert rules.pairs == {("fp:a", "fp:b"): "cannot_link"} and rules.group is None


def test_rejecting_three_identities_is_one_not_all_together_group() -> None:
    identity = {R[0]: "fp:c", R[1]: "fp:a", R[2]: "fp:b"}
    rules = identity_rules(FeedbackDecision.REJECT, {R[0], R[1], R[2]}, set(), None, identity)

    assert rules.pairs == {} and rules.group == ["fp:a", "fp:b", "fp:c"]


def test_records_without_an_identity_bind_nothing() -> None:
    rules = identity_rules(FeedbackDecision.SPLIT, {R[0]}, {R[1]}, R[0], {R[0]: "fp:a"})

    assert rules.pairs == {} and rules.unbound == 1


# --- Identity-level clustering ------------------------------------------------------------


def test_a_never_merge_binds_every_record_of_both_identities() -> None:
    identity = {"a1": "X", "a2": "X", "b1": "Y"}
    edges = [ScoredEdge("a1", "b1", 0.99), ScoredEdge("a2", "b1", 0.98)]
    result = build_clusters(
        edges, constraints=Constraints.from_pairs([], [("X", "Y")], identity=identity)
    )

    assert result.clusters == [] and result.cut_links == 2


def test_not_all_together_cuts_only_the_join_that_completes_the_group() -> None:
    identity = {"a": "X", "b": "Y", "c": "Z"}
    edges = [ScoredEdge("a", "b", 0.99), ScoredEdge("b", "c", 0.98), ScoredEdge("a", "c", 0.97)]
    result = build_clusters(
        edges,
        constraints=Constraints.from_pairs([], [], groups=[("X", "Y", "Z")], identity=identity),
    )

    assert [c.members for c in result.clusters] == [["a", "b"]]
    assert result.cut_links == 2


def test_a_must_link_merges_all_records_of_both_identities() -> None:
    identity = {"a1": "X", "a2": "X", "b1": "Y", "z": "Z"}
    result = build_clusters(
        [], constraints=Constraints.from_pairs([("X", "Y")], [], identity=identity)
    )

    assert [c.members for c in result.clusters] == [["a1", "a2", "b1"]]


def test_a_shared_natural_key_forces_a_merge_and_a_conflicting_rule_flags_it() -> None:
    identity = {"a": "X", "b": "Y"}
    result = build_clusters(
        [],
        constraints=Constraints.from_pairs(
            [], [("X", "Y")], identity=identity, forced_groups=[("a", "b")]
        ),
    )

    (cluster,) = result.clusters
    assert cluster.members == ["a", "b"] and cluster.constraint_conflict


# --- Gold-link decisions ---------------------------------------------------------------------


def _text(name: str) -> RecordText:
    return RecordText.build(name, "main road", "kullu", "himachal pradesh")


def _in(
    rid: str,
    sid: str,
    kind: str = "fingerprint",
    cluster: set[str] | None = None,
    name: str = "saint josephs school",
) -> IncomingRecord:
    return IncomingRecord(rid, sid, kind, _text(name), frozenset(cluster or {sid}))


def _pool(
    rid: str,
    sid: str,
    kind: str = "fingerprint",
    entity: set[str] | None = None,
    name: str = "saint josephs school",
) -> PoolRecord:
    return PoolRecord(
        rid, sid, kind, "gold_master", uuid.uuid4(), _text(name), frozenset(entity or {sid})
    )


def _decide(
    candidates: list[LinkCandidate],
    incoming: list[IncomingRecord],
    pool: list[PoolRecord],
    **rules: object,
) -> tuple[list, object]:
    return decide_links(
        candidates,
        {r.record_id: r for r in incoming},
        {p.record_id: p for p in pool},
        must=rules.get("must", set()),  # type: ignore[arg-type]
        cannot=rules.get("cannot", set()),  # type: ignore[arg-type]
        groups=rules.get("groups", []),  # type: ignore[arg-type]
        auto_threshold=0.95,
        link_threshold=0.5,
        force_on_key=True,
    )


def C(i: str, p: str, prob: float) -> LinkCandidate:  # noqa: N802 - notation
    return LinkCandidate(i, p, prob, 1.0, [3, 3, 3])


def test_thresholds_split_auto_review_and_drop() -> None:
    decisions, stats = _decide(
        [C("n1", "g1", 0.99), C("n2", "g2", 0.7), C("n3", "g3", 0.2)],
        [_in("n1", "fp:1"), _in("n2", "fp:2"), _in("n3", "fp:3")],
        [_pool("g1", "fp:1"), _pool("g2", "fp:2"), _pool("g3", "fp:3")],
    )

    assert {(d.incoming_id, d.status) for d in decisions} == {
        ("n1", "auto_linked"),
        ("n2", "review"),
    }
    assert stats.below_threshold == 1  # type: ignore[attr-defined]


def test_a_shared_natural_key_links_whatever_the_score() -> None:
    key = "nk:udise_code:29281200612"
    decisions, _ = _decide(
        [C("n1", "g1", 0.01)],
        [_in("n1", key, "natural_key", name="sunrise academy")],
        [_pool("g1", key, "natural_key", name="old mission building")],
    )

    assert [(d.status, d.reason) for d in decisions] == [("auto_linked", "natural_key")]


def test_a_steward_never_merge_blocks_a_confident_link() -> None:
    decisions, stats = _decide(
        [C("n1", "g1", 0.99)],
        [_in("n1", "fp:a")],
        [_pool("g1", "fp:b")],
        cannot={frozenset(("fp:a", "fp:b"))},
    )

    assert decisions == [] and stats.blocked_by_constraint == 1  # type: ignore[attr-defined]


def test_a_never_merge_with_any_member_of_the_gold_entity_blocks_the_link() -> None:
    decisions, _ = _decide(
        [C("n1", "g1", 0.99)],
        [_in("n1", "fp:a")],
        [_pool("g1", "fp:b", entity={"fp:b", "fp:c"})],
        cannot={frozenset(("fp:a", "fp:c"))},
    )

    assert decisions == []


def test_a_link_completing_a_rejected_group_is_blocked() -> None:
    decisions, _ = _decide(
        [C("n1", "g1", 0.99)],
        [_in("n1", "fp:a", cluster={"fp:a", "fp:b"})],
        [_pool("g1", "fp:c")],
        groups=[frozenset(("fp:a", "fp:b", "fp:c"))],
    )

    assert decisions == []


def test_a_must_link_adds_a_pair_blocking_never_proposed() -> None:
    decisions, _ = _decide(
        [], [_in("n1", "fp:a")], [_pool("g1", "fp:b")], must={frozenset(("fp:a", "fp:b"))}
    )

    assert [(d.pool_id, d.status, d.reason) for d in decisions] == [
        ("g1", "auto_linked", "must_link")
    ]


def test_a_forced_link_against_a_veto_goes_to_review_not_auto() -> None:
    decisions, _ = _decide(
        [],
        [_in("n1", "fp:a")],
        [_pool("g1", "fp:b")],
        must={frozenset(("fp:a", "fp:b"))},
        cannot={frozenset(("fp:a", "fp:b"))},
    )

    assert [d.status for d in decisions] == ["review"]


def test_the_name_gate_vetoes_score_links() -> None:
    decisions, stats = _decide(
        [C("n1", "g1", 0.99)],
        [_in("n1", "fp:a", name="government girls school ramnagar")],
        [_pool("g1", "fp:b", name="government boys school ramnagar")],
    )

    assert decisions == [] and stats.name_vetoed == 1  # type: ignore[attr-defined]


def test_two_confident_targets_are_ambiguous_and_go_to_review() -> None:
    decisions, stats = _decide(
        [C("n1", "g1", 0.99), C("n1", "g2", 0.98)],
        [_in("n1", "fp:a")],
        [_pool("g1", "fp:b"), _pool("g2", "fp:c")],
    )

    assert {d.status for d in decisions} == {"review"}
    assert stats.ambiguous_demoted == 2  # type: ignore[attr-defined]


@pytest.mark.parametrize("forced_kind", ["natural_key"])
def test_an_identity_forced_link_survives_ambiguity(forced_kind: str) -> None:
    key = "nk:udise_code:29281200612"
    decisions, _ = _decide(
        [C("n1", "g1", 0.5), C("n1", "g2", 0.99)],
        [_in("n1", key, forced_kind)],
        [_pool("g1", key, forced_kind), _pool("g2", "fp:other")],
    )

    by_target = {d.pool_id: d.status for d in decisions}
    assert by_target == {"g1": "auto_linked", "g2": "review"}
