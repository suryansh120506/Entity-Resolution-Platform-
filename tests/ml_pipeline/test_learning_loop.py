"""The Bayesian m-update and constraint-aware clustering. No database."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from app.ml_pipeline.matching.clustering import Constraints, ScoredEdge, build_clusters
from app.ml_pipeline.matching.feedback_model import (
    M_FLOOR,
    blend_m_probabilities,
    level_gammas,
)


def _settings() -> dict[str, Any]:
    """Splink's saved-model shape: null level first, most similar next."""
    return {
        "probability_two_random_records_match": 0.01,
        "comparisons": [
            {
                "output_column_name": "pincode",
                "comparison_levels": [
                    {"sql_condition": "null", "is_null_level": True},
                    {"sql_condition": "exact", "m_probability": 0.9, "u_probability": 0.01},
                    {"sql_condition": "prefix", "m_probability": 0.08, "u_probability": 0.1},
                    {"sql_condition": "ELSE", "m_probability": 0.02, "u_probability": 0.89},
                ],
            },
            {
                "output_column_name": "name",
                "comparison_levels": [
                    {"sql_condition": "exact", "m_probability": 0.7, "u_probability": 0.001},
                    # A level EM never observed: no m at all.
                    {"sql_condition": "ELSE", "u_probability": 0.999},
                ],
            },
        ],
    }


def _m(settings: dict[str, Any], comparison: int) -> list[float]:
    levels = settings["comparisons"][comparison]["comparison_levels"]
    return [level["m_probability"] for level in levels if not level.get("is_null_level")]


# --- Level numbering -----------------------------------------------------------------


def test_gammas_count_down_from_most_similar_with_null_minus_one() -> None:
    levels = _settings()["comparisons"][0]["comparison_levels"]

    assert level_gammas(levels) == [None, 2, 1, 0]


# --- The Bayesian update ---------------------------------------------------------------


def test_no_evidence_leaves_the_model_exactly_as_em_fitted_it() -> None:
    base = _settings()
    blended, evidence = blend_m_probabilities(base, {}, prior_strength=50)

    assert blended == base and evidence == 0


def test_the_input_settings_are_not_mutated() -> None:
    base = _settings()
    blend_m_probabilities(base, {(0, 0): 100.0}, prior_strength=50)

    assert base == _settings()


def test_evidence_moves_m_toward_what_stewards_saw() -> None:
    """100 labelled matches that disagreed on pincode: m(ELSE) must rise."""
    blended, evidence = blend_m_probabilities(_settings(), {(0, 0): 100.0}, prior_strength=50)
    exact, prefix, other = _m(blended, 0)

    assert evidence == 100
    assert other == pytest.approx((100 + 50 * 0.02) / 150)
    assert exact == pytest.approx(50 * 0.9 / 150)
    assert exact + prefix + other == pytest.approx(1.0)


def test_prior_strength_controls_how_far_few_labels_move_the_model() -> None:
    weak, _ = blend_m_probabilities(_settings(), {(0, 0): 5.0}, prior_strength=1000)
    strong, _ = blend_m_probabilities(_settings(), {(0, 0): 5.0}, prior_strength=1)

    assert _m(weak, 0)[2] < 0.03 < 0.5 < _m(strong, 0)[2]


def test_an_unobserved_level_gets_the_unclaimed_prior_mass_not_zero() -> None:
    # Gamma 0 is the ELSE level (the one EM never observed).
    blended, _ = blend_m_probabilities(_settings(), {(1, 0): 10.0}, prior_strength=50)
    exact, other = _m(blended, 1)

    assert other == pytest.approx((10 + 50 * 0.3) / 60)
    assert exact + other == pytest.approx(1.0)


def test_no_level_can_reach_zero() -> None:
    """m = 0 makes the Bayes factor 0: one level would veto every pair."""
    blended, _ = blend_m_probabilities(_settings(), {(0, 2): 1e9}, prior_strength=1)

    assert min(_m(blended, 0)) >= M_FLOOR


def test_u_and_the_prior_are_never_touched() -> None:
    """Labelled non-matches are a biased sample; u stays from random sampling."""
    base = _settings()
    blended, _ = blend_m_probabilities(base, {(0, 0): 100.0, (1, 1): 7.0}, prior_strength=50)

    for b, o in zip(blended["comparisons"], base["comparisons"], strict=True):
        assert [lv.get("u_probability") for lv in b["comparison_levels"]] == [
            lv.get("u_probability") for lv in o["comparison_levels"]
        ]
    assert blended["probability_two_random_records_match"] == 0.01


def test_a_nonpositive_prior_strength_is_refused() -> None:
    with pytest.raises(ValueError):
        blend_m_probabilities(_settings(), {}, prior_strength=0)


# --- Constrained clustering --------------------------------------------------------------


def E(a: str, b: str, s: float = 0.97) -> ScoredEdge:  # noqa: N802 - reads as notation
    return ScoredEdge(a, b, s)


def test_a_cannot_link_cuts_the_weakest_join_of_a_chain() -> None:
    edges = [E("a", "b", 0.99), E("b", "c", 0.98), E("c", "d", 0.97)]
    result = build_clusters(edges, constraints=Constraints.from_pairs([], [("a", "d")]))

    assert [c.members for c in result.clusters] == [["a", "b", "c"]]
    assert result.cut_links == 1


def test_the_strongest_edges_win_when_a_constraint_forces_a_choice() -> None:
    """b is closer to a (0.99) than to c (0.6); a and c may not share a cluster."""
    edges = [E("a", "b", 0.99), E("b", "c", 0.6)]
    result = build_clusters(edges, constraints=Constraints.from_pairs([], [("a", "c")]))

    assert [c.members for c in result.clusters] == [["a", "b"]]


def test_a_cut_edge_is_not_counted_inside_any_cluster() -> None:
    edges = [E("a", "b"), E("c", "d"), E("b", "c", 0.8)]
    result = build_clusters(edges, constraints=Constraints.from_pairs([], [("a", "d")]))

    assert sorted(c.members for c in result.clusters) == [["a", "b"], ["c", "d"]]
    assert all(c.confidence == pytest.approx(0.97) for c in result.clusters)


def test_must_link_forces_a_merge_the_model_never_proposed() -> None:
    result = build_clusters(
        [E("a", "b", 0.2)], constraints=Constraints.from_pairs([("a", "b")], [])
    )

    (cluster,) = result.clusters
    assert cluster.members == ["a", "b"] and cluster.confidence == 1.0
    assert result.forced_links == 1


def test_must_link_overrides_a_low_score_inside_a_larger_cluster() -> None:
    edges = [E("a", "b", 0.95), E("b", "c", 0.1)]
    result = build_clusters(edges, constraints=Constraints.from_pairs([("b", "c")], []))

    assert [c.members for c in result.clusters] == [["a", "b", "c"]]
    assert result.clusters[0].confidence == pytest.approx(0.95)


def test_contradictory_steward_constraints_are_flagged_not_resolved() -> None:
    result = build_clusters(
        [], constraints=Constraints.from_pairs([("a", "b"), ("b", "c")], [("a", "c")])
    )

    (cluster,) = result.clusters
    assert cluster.members == ["a", "b", "c"] and cluster.constraint_conflict
    assert result.constraint_conflicts == 1


def test_without_constraints_clustering_is_unchanged() -> None:
    edges = [E("a", "b"), E("b", "c"), E("x", "y", 0.4)]
    result = build_clusters(edges)

    assert [c.members for c in result.clusters] == [["a", "b", "c"]]
    assert result.cut_links == result.forced_links == result.constraint_conflicts == 0


def test_tightening_an_oversized_component_never_drops_a_forced_link() -> None:
    chain = [E(f"r{i}", f"r{i + 1}", 0.6) for i in range(5)]
    result = build_clusters(
        chain, max_size=3, constraints=Constraints.from_pairs([("r0", "r5")], [])
    )

    assert ["r0", "r5"] in [c.members for c in result.clusters]


# --- The gate honours the conflict ---------------------------------------------------------


@pytest.mark.parametrize("triage", [False, True])
def test_a_conflicted_cluster_is_kept_apart_whatever_its_score(triage: bool) -> None:
    """Steward rules that contradict each other cannot be settled by a score
    or by an LLM; the records stay separate (migration 0010)."""
    from app.models.cluster import Cluster
    from app.models.enums import ClusterStatus
    from app.workers.tasks.resolution import route

    cluster = Cluster()
    cluster.confidence_score = Decimal("1.000")
    cluster.density = 1.0
    cluster.review_reason = "constraint_conflict"
    members = [("a", {"n": "1"}), ("b", {"n": "2"})]

    assert route(cluster, members, Decimal("0.950"), 0.6, triage=triage) == "constraint_conflict"
    assert cluster.status == ClusterStatus.REJECTED.value
    assert cluster.master_record_id is None
    assert cluster.resolution_source == "threshold"
