"""Unit tests for the confidence gate, survivorship, stable entity ids, chain
density, and the pair labels a steward's verdict actually asserts."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

import pytest

from app.api.v1.routes.clusters import decision_pairs
from app.ml_pipeline.matching.clustering import ScoredEdge
from app.ml_pipeline.survivorship import select_master_record
from app.models.cluster import Cluster
from app.models.enums import ClusterStatus, FeedbackDecision, JobStatus
from app.workers.tasks.matching import assign_entity_ids, cluster_density
from app.workers.tasks.resolution import route, statuses_after_gate

AUTO_MERGE = Decimal("0.950")
FLOOR = Decimal("0.800")
MIN_DENSITY = 0.6
PAIR = [("rec-1", {"name": "A", "address": "x"}), ("rec-2", {"name": "B"})]
TRIO = [*PAIR, ("rec-3", {"name": "C"})]


def _cluster(score: str | None, density: float | None = 1.0) -> Cluster:
    cluster = Cluster()
    cluster.status = ClusterStatus.PENDING_REVIEW.value
    cluster.confidence_score = None if score is None else Decimal(score)
    cluster.density = density
    return cluster


# --- Gate ----------------------------------------------------------------------


def test_high_confidence_auto_merges_with_the_most_complete_master() -> None:
    cluster = _cluster("0.990")

    assert route(cluster, PAIR, AUTO_MERGE, MIN_DENSITY) == "auto_merged"
    assert cluster.status == ClusterStatus.AUTO_MERGED.value
    assert cluster.master_record_id == "rec-1"


def test_the_threshold_is_inclusive() -> None:
    assert route(_cluster("0.950"), PAIR, AUTO_MERGE, MIN_DENSITY) == "auto_merged"


def test_an_auto_merge_records_that_the_threshold_decided() -> None:
    cluster = _cluster("0.990")
    route(cluster, PAIR, AUTO_MERGE, MIN_DENSITY, triage=True)

    assert cluster.resolution_source == "threshold"
    assert cluster.auto_decision == {
        "tier": 1,
        "decision": "MERGE",
        "band": "auto_merged",
        "confidence": "0.990",
        "density": 1.0,
    }


@pytest.mark.parametrize("triage", [False, True])
def test_below_the_floor_is_kept_separate_with_no_review(triage: bool) -> None:
    """Tier 1: under 0.80 the records stay apart, LLM or no LLM."""
    cluster = _cluster("0.799")

    assert (
        route(cluster, PAIR, AUTO_MERGE, MIN_DENSITY, floor=FLOOR, triage=triage) == "below_floor"
    )
    assert cluster.status == ClusterStatus.REJECTED.value
    assert cluster.master_record_id is None
    assert cluster.resolution_source == "threshold"
    assert cluster.auto_decision is not None
    assert cluster.auto_decision["band"] == "below_floor"


def test_the_floor_is_inclusive() -> None:
    cluster = _cluster("0.800")

    assert route(cluster, PAIR, AUTO_MERGE, MIN_DENSITY, floor=FLOOR, triage=True) == "triage"


def test_the_middle_band_goes_to_automated_triage() -> None:
    cluster = _cluster("0.900")

    assert route(cluster, PAIR, AUTO_MERGE, MIN_DENSITY, floor=FLOOR, triage=True) == "triage"
    assert cluster.status == ClusterStatus.PENDING_API.value
    assert cluster.resolution_source is None


def test_without_an_llm_the_middle_band_still_goes_to_a_human() -> None:
    """Triage needs an LLM; a missing setting must not silently separate."""
    cluster = _cluster("0.900")

    assert route(cluster, PAIR, AUTO_MERGE, MIN_DENSITY, floor=FLOOR) == "pending_review"
    assert cluster.status == ClusterStatus.PENDING_REVIEW.value


def test_unscored_is_kept_separate() -> None:
    """No score is no evidence these records are one entity."""
    cluster = _cluster(None)

    assert route(cluster, PAIR, AUTO_MERGE, MIN_DENSITY, triage=True) == "unscored"
    assert cluster.status == ClusterStatus.REJECTED.value


def test_a_suspected_chain_goes_to_triage_whatever_its_score() -> None:
    """Bottleneck score 0.99, but only half the member pairs ever linked."""
    cluster = _cluster("0.990", density=0.5)

    assert route(cluster, TRIO, AUTO_MERGE, MIN_DENSITY, triage=True) == "chain_triage"
    assert cluster.status == ClusterStatus.PENDING_API.value
    assert cluster.master_record_id is None


def test_without_an_llm_a_suspected_chain_goes_to_a_human() -> None:
    cluster = _cluster("0.990", density=0.5)

    assert route(cluster, TRIO, AUTO_MERGE, MIN_DENSITY) == "chain_suspect"
    assert cluster.status == ClusterStatus.PENDING_REVIEW.value
    assert cluster.master_record_id is None


def test_rerouting_an_undecided_cluster_clears_the_previous_decision() -> None:
    cluster = _cluster("0.990")
    route(cluster, PAIR, AUTO_MERGE, MIN_DENSITY)
    cluster.confidence_score = Decimal("0.900")

    route(cluster, PAIR, AUTO_MERGE, MIN_DENSITY, triage=True)

    assert cluster.status == ClusterStatus.PENDING_API.value
    assert cluster.resolution_source is None
    assert cluster.auto_decision is None


def test_density_is_irrelevant_for_a_pair() -> None:
    """Two records cannot chain; the guard applies from three members."""
    assert route(_cluster("0.990", density=0.0), PAIR, AUTO_MERGE, MIN_DENSITY) == "auto_merged"


class _CountSession:
    """Answers each count query in turn: PENDING_REVIEW first, then PENDING_API."""

    def __init__(self, *counts: int) -> None:
        self.counts = list(counts)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        count = self.counts.pop(0) if self.counts else 0

        class _Result:
            def scalar_one(self) -> int:
                return count

        return _Result()


def test_outstanding_reviews_hold_the_job_in_awaiting_review() -> None:
    """Counted from the table, not this pass: a re-run that routed nothing new
    must not complete a job whose earlier clusters still await a steward."""
    assert statuses_after_gate(_CountSession(3), "job") == (JobStatus.AWAITING_REVIEW,)


def test_clusters_waiting_for_triage_need_no_one() -> None:
    """Only the triage worker is waited on: RESOLVING, not AWAITING_REVIEW."""
    assert statuses_after_gate(_CountSession(0, 5), "job") == (JobStatus.RESOLVING,)


def test_nothing_outstanding_completes_through_resolving() -> None:
    assert statuses_after_gate(_CountSession(0, 0), "job") == (
        JobStatus.RESOLVING,
        JobStatus.COMPLETED,
    )


# --- Survivorship --------------------------------------------------------------


def test_master_selection_is_deterministic() -> None:
    tied = [("rec-9", {"a": "1"}), ("rec-2", {"b": "2"})]

    assert select_master_record(tied) == select_master_record(list(reversed(tied))) == "rec-9"


def test_master_selection_on_an_empty_cluster() -> None:
    assert select_master_record([]) is None


# --- Stable entity ids ---------------------------------------------------------

E1, E2 = uuid.UUID(int=1), uuid.UUID(int=2)


def test_an_unchanged_cluster_keeps_its_entity_id() -> None:
    previous = {E1: frozenset({"a", "b", "c"})}

    assert assign_entity_ids([frozenset({"a", "b", "c"})], previous) == [E1]


def test_a_mostly_overlapping_cluster_inherits_the_id() -> None:
    previous = {E1: frozenset({"a", "b", "c"})}
    (assigned,) = assign_entity_ids([frozenset({"a", "b", "c", "d"})], previous)

    assert assigned == E1  # Jaccard 3/4


def test_a_weak_overlap_is_a_new_entity() -> None:
    previous = {E1: frozenset({"a", "b", "c", "d"})}
    (assigned,) = assign_entity_ids([frozenset({"a", "x", "y"})], previous)

    assert assigned != E1


def test_an_old_id_is_inherited_by_at_most_one_cluster() -> None:
    """A split of one old cluster into two halves: the larger half wins."""
    previous = {E1: frozenset({"a", "b", "c", "d", "e"})}
    ids = assign_entity_ids([frozenset({"a", "b", "c"}), frozenset({"d", "e"})], previous)

    assert ids[0] == E1 and ids[1] != E1


def test_the_best_overlap_wins_each_id() -> None:
    previous = {E1: frozenset({"a", "b"}), E2: frozenset({"c", "d"})}
    ids = assign_entity_ids([frozenset({"c", "d"}), frozenset({"a", "b"})], previous)

    assert ids == [E2, E1]


# --- Chain density -------------------------------------------------------------


def _edges(*pairs: tuple[str, str]) -> list[ScoredEdge]:
    return [ScoredEdge(a, b, 0.97) for a, b in pairs]


def test_a_four_chain_is_half_dense() -> None:
    members = ["a", "b", "c", "d"]

    assert cluster_density(members, _edges(("a", "b"), ("b", "c"), ("c", "d")), 0.5) == 0.5


def test_a_clique_is_fully_dense() -> None:
    members = ["a", "b", "c"]

    assert cluster_density(members, _edges(("a", "b"), ("b", "c"), ("a", "c")), 0.5) == 1.0


def test_edges_below_threshold_or_outside_the_cluster_do_not_count() -> None:
    edges = [ScoredEdge("a", "b", 0.97), ScoredEdge("a", "c", 0.2), ScoredEdge("a", "z", 0.99)]

    assert cluster_density(["a", "b", "c"], edges, 0.5) == pytest.approx(1 / 3)


# --- What a verdict asserts ------------------------------------------------------

R = [uuid.UUID(int=i) for i in range(1, 6)]


def test_merge_asserts_every_kept_pair_matches() -> None:
    matches, non = decision_pairs(FeedbackDecision.MERGE, set(R[:3]), set())

    assert len(matches) == 3 and not non


def test_split_never_claims_the_excluded_records_relate_to_each_other() -> None:
    kept, excluded = set(R[:2]), set(R[2:4])
    matches, non = decision_pairs(FeedbackDecision.SPLIT, kept, excluded)

    assert matches == {(R[0], R[1])}
    assert len(non) == 4
    assert (R[2], R[3]) not in matches | non, "excluded x excluded is unknown, not a label"


def test_rejecting_a_pair_is_a_non_match() -> None:
    matches, non = decision_pairs(FeedbackDecision.REJECT, set(R[:2]), set())

    assert not matches and non == {(R[0], R[1])}


def test_rejecting_a_larger_group_asserts_nothing_pairwise() -> None:
    """ "This grouping is wrong" is not "no two of these match"."""
    assert decision_pairs(FeedbackDecision.REJECT, set(R[:3]), set()) == (set(), set())


def test_pairs_are_canonically_ordered_for_the_check_constraint() -> None:
    matches, non = decision_pairs(FeedbackDecision.SPLIT, set(R[:3]), set(R[3:]))

    assert all(str(a) < str(b) for a, b in matches | non)
