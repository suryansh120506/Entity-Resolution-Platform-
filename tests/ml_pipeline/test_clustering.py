"""Unit tests for building clusters from scored pairs."""

from __future__ import annotations

import pytest

from app.ml_pipeline.matching.clustering import ScoredEdge, build_clusters


def _members(result) -> list[list[str]]:
    return sorted(sorted(c.members) for c in result.clusters)


def test_only_pairs_the_model_believes_link_records() -> None:
    result = build_clusters(
        [ScoredEdge("a", "b", 0.97), ScoredEdge("b", "c", 0.2), ScoredEdge("x", "y", 0.6)]
    )

    assert _members(result) == [["a", "b"], ["x", "y"]]
    assert result.links == 2


def test_links_are_transitive() -> None:
    result = build_clusters([ScoredEdge("a", "b", 0.9), ScoredEdge("b", "c", 0.8)])

    assert _members(result) == [["a", "b", "c"]]


def test_confidence_is_the_bottleneck_of_the_strongest_chain() -> None:
    """A redundant weak edge must not sink a cluster held together by strong ones."""
    result = build_clusters(
        [
            ScoredEdge("a", "b", 0.99),
            ScoredEdge("b", "c", 0.97),
            ScoredEdge("a", "c", 0.55),  # redundant: a-c is already joined via b
        ]
    )

    assert result.clusters[0].confidence == pytest.approx(0.97)


def test_a_cluster_is_only_as_strong_as_the_link_it_depends_on() -> None:
    result = build_clusters([ScoredEdge("a", "b", 0.99), ScoredEdge("b", "c", 0.6)])

    assert result.clusters[0].confidence == pytest.approx(0.6)


def test_member_scores_are_each_members_strongest_link() -> None:
    result = build_clusters([ScoredEdge("a", "b", 0.99), ScoredEdge("b", "c", 0.6)])

    scores = result.clusters[0].member_scores
    assert scores == {"a": 0.99, "b": 0.99, "c": 0.6}


def test_an_oversized_component_is_split_at_a_stricter_threshold() -> None:
    # Two tight groups of 3, joined by one weak link: 6 > max_size=4.
    edges = [
        ScoredEdge("a1", "a2", 0.99),
        ScoredEdge("a2", "a3", 0.99),
        ScoredEdge("b1", "b2", 0.99),
        ScoredEdge("b2", "b3", 0.99),
        ScoredEdge("a3", "b1", 0.6),
    ]

    result = build_clusters(edges, max_size=4)

    assert _members(result) == [["a1", "a2", "a3"], ["b1", "b2", "b3"]]
    assert result.tightened_components == 1
    assert result.refused_sizes == []


def test_a_component_that_cannot_be_split_is_refused_and_counted() -> None:
    edges = [ScoredEdge(str(i), str(i + 1), 0.999) for i in range(10)]

    result = build_clusters(edges, max_size=4)

    assert result.clusters == []
    assert result.refused_sizes == [11]


def test_records_shed_while_tightening_are_counted() -> None:
    edges = [
        ScoredEdge("a", "b", 0.99),
        ScoredEdge("b", "c", 0.6),
        ScoredEdge("c", "d", 0.6),
    ]

    result = build_clusters(edges, max_size=3)

    assert _members(result) == [["a", "b"]]
    assert result.released_records == 2


def test_no_edges_no_clusters() -> None:
    result = build_clusters([])

    assert result.clusters == []
    assert result.as_dict()["clusters"] == 0


def test_a_long_chain_does_not_recurse() -> None:
    edges = [ScoredEdge(str(i), str(i + 1), 0.99) for i in range(5_000)]

    result = build_clusters(edges, max_size=10_000)

    assert len(result.clusters) == 1
