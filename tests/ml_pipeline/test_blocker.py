"""Unit tests for candidate generation — FR-03 / SAD 3.4.

No Postgres. The session is a stand-in that records the statements it was
handed and replays canned neighbour rows, which is enough to test everything
this class actually decides: which queries it issues, and how it folds the
results into pairs. What it cannot test is whether pgvector returns sensible
neighbours — that is pgvector's job and belongs in the integration suite.

The statements are still compiled against the real PostgreSQL dialect, so a
query that could not run is caught here rather than in staging.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from sqlalchemy import Select
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql.elements import TextClause

from app.ml_pipeline.blocking.blocker import (
    CandidateBlocker,
    CandidatePair,
    ProbeRecord,
    _canonical_pair,
)

DIM = 8


def _embedding(seed: float = 0.1) -> list[float]:
    return [seed] * DIM


def _id(suffix: int) -> uuid.UUID:
    """Deterministic, ordered UUIDs so pair ordering is predictable."""
    return uuid.UUID(f"00000000-0000-0000-0000-{suffix:012d}")


A, B, C, D = _id(1), _id(2), _id(3), _id(4)


class FakeSession:
    """Records statements; returns queued neighbour rows for each SELECT."""

    def __init__(self, results: list[list[tuple[uuid.UUID, float]]] | None = None) -> None:
        self.results = list(results or [])
        self.selects: list[Select[Any]] = []
        self.text_statements: list[str] = []

    def execute(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(statement, TextClause):
            self.text_statements.append(str(statement))
            return None
        self.selects.append(statement)
        return iter(self.results.pop(0) if self.results else [])

    def compiled_selects(self) -> list[str]:
        return [str(s.compile(dialect=postgresql.dialect())) for s in self.selects]


# --- Query construction --------------------------------------------------------


def test_query_uses_pgvector_cosine_distance_operator() -> None:
    """The `<=>` operator is what makes the HNSW cosine index usable.

    An l2_distance query against a `vector_cosine_ops` index does not use the
    index at all — it silently falls back to a sequential scan, which is
    correct and unusably slow.
    """
    session = FakeSession([[]])
    blocker = CandidateBlocker(session)  # type: ignore[arg-type]

    list(blocker.find_candidates([ProbeRecord(A, _embedding())]))

    sql = session.compiled_selects()[0]
    assert "<=>" in sql, sql
    assert "ORDER BY" in sql.upper()


def test_query_overfetches_to_survive_rls_post_filtering() -> None:
    """LIMIT must be top_k * oversample, not top_k.

    pgvector applies the approximation before RLS filters by tenant, so a
    LIMIT of exactly top_k can come back short with no error anywhere.
    """
    session = FakeSession([[]])
    blocker = CandidateBlocker(session, top_k=10, oversample=3)  # type: ignore[arg-type]

    list(blocker.find_candidates([ProbeRecord(A, _embedding())]))

    assert " LIMIT " in session.compiled_selects()[0].upper()
    assert session.selects[0]._limit == 30


def test_probe_never_matches_itself() -> None:
    session = FakeSession([[]])
    blocker = CandidateBlocker(session)  # type: ignore[arg-type]

    list(blocker.find_candidates([ProbeRecord(A, _embedding())]))

    # Excluded in SQL rather than in Python, so the row never occupies one of
    # the LIMIT slots.
    assert "record_id !=" in session.compiled_selects()[0].replace('"', "")


def test_job_scope_joins_school_records() -> None:
    session = FakeSession([[]])
    job_id = _id(99)
    blocker = CandidateBlocker(session, job_id=job_id)  # type: ignore[arg-type]

    list(blocker.find_candidates([ProbeRecord(A, _embedding())]))

    sql = session.compiled_selects()[0]
    assert "school_records" in sql
    assert "JOIN" in sql.upper()


def test_ef_search_is_set_local_only_when_requested() -> None:
    """LOCAL so the setting cannot follow a pooled connection to another job."""
    session = FakeSession([[]])
    list(
        CandidateBlocker(session, ef_search=200).find_candidates(  # type: ignore[arg-type]
            [ProbeRecord(A, _embedding())]
        )
    )
    assert session.text_statements == ["SET LOCAL hnsw.ef_search = 200"]

    quiet = FakeSession([[]])
    list(CandidateBlocker(quiet).find_candidates([ProbeRecord(A, _embedding())]))  # type: ignore[arg-type]
    assert quiet.text_statements == []


# --- Constructor validation ----------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"top_k": 0}, "top_k"),
        ({"oversample": 0}, "oversample"),
        ({"max_distance": -0.1}, "max_distance"),
        # A cosine *similarity* passed where a distance was meant. Bounded
        # [0, 2], so 5 is a unit mix-up that would otherwise accept everything.
        ({"max_distance": 5.0}, "max_distance"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CandidateBlocker(FakeSession(), **kwargs)  # type: ignore[arg-type]


# --- Pair folding --------------------------------------------------------------


def test_no_probes_issues_no_queries() -> None:
    session = FakeSession()
    assert list(CandidateBlocker(session).find_candidates([])) == []  # type: ignore[arg-type]
    assert session.selects == []


def test_distance_ceiling_truncates_the_sorted_result() -> None:
    """Rows arrive sorted, so the first row over the ceiling ends the scan."""
    session = FakeSession([[(B, 0.10), (C, 0.20), (D, 0.90)]])
    blocker = CandidateBlocker(session, max_distance=0.35)  # type: ignore[arg-type]

    pairs = list(blocker.find_candidates([ProbeRecord(A, _embedding())]))

    assert {p.record_id_b for p in pairs} == {B, C}


def test_top_k_caps_results_even_when_the_database_returns_more() -> None:
    """The over-fetch is for recall under RLS, not a bigger candidate set."""
    session = FakeSession([[(B, 0.01), (C, 0.02), (D, 0.03)]])
    blocker = CandidateBlocker(session, top_k=2, oversample=3)  # type: ignore[arg-type]

    pairs = list(blocker.find_candidates([ProbeRecord(A, _embedding())]))

    assert len(pairs) == 2


def test_pairs_are_canonically_ordered() -> None:
    """(A, B) and (B, A) must be one key, or nothing downstream deduplicates."""
    assert _canonical_pair(A, B) == (A, B)
    assert _canonical_pair(B, A) == (A, B)

    session = FakeSession([[(A, 0.05)]])
    pairs = list(
        CandidateBlocker(session).find_candidates([ProbeRecord(B, _embedding())])  # type: ignore[arg-type]
    )

    assert pairs[0].record_id_a == A
    assert pairs[0].record_id_b == B


def test_reciprocal_discovery_yields_one_pair() -> None:
    """A finds B, then B finds A. That is one candidate, not two."""
    session = FakeSession([[(B, 0.10)], [(A, 0.10)]])
    blocker = CandidateBlocker(session, search_address=False)  # type: ignore[arg-type]

    pairs = list(
        blocker.find_candidates([ProbeRecord(A, _embedding()), ProbeRecord(B, _embedding())])
    )

    assert len(pairs) == 1


def test_agreement_across_axes_is_recorded_and_keeps_the_best_distance() -> None:
    """FR-03: name and address are searched independently.

    A pair found on both axes is stronger evidence than one found on either,
    so the union must not throw that away — the matcher is entitled to know.
    """
    session = FakeSession([[(B, 0.20)], [(B, 0.05)]])  # name hit, then address hit
    # Explicit max_distance: the test's fixture distances are independent of
    # wherever the production default happens to sit.
    blocker = CandidateBlocker(session, max_distance=0.35)  # type: ignore[arg-type]

    pairs = list(blocker.find_candidates([ProbeRecord(A, _embedding(), _embedding(0.2))]))

    assert len(pairs) == 1
    assert pairs[0].source == "both"
    assert pairs[0].distance == pytest.approx(0.05)


def test_single_axis_hit_names_its_axis() -> None:
    session = FakeSession([[(B, 0.10)], []])
    pairs = list(
        CandidateBlocker(session).find_candidates(  # type: ignore[arg-type]
            [ProbeRecord(A, _embedding(), _embedding(0.2))]
        )
    )

    assert pairs[0].source == "name"


def test_missing_address_embedding_still_blocks_on_name() -> None:
    """A record whose address never normalised must not drop out entirely."""
    session = FakeSession([[(B, 0.10)]])
    blocker = CandidateBlocker(session)  # type: ignore[arg-type]

    pairs = list(blocker.find_candidates([ProbeRecord(A, _embedding(), None)]))

    assert len(session.selects) == 1, "should not have queried the address index"
    assert len(pairs) == 1


def test_search_address_false_queries_only_the_name_index() -> None:
    session = FakeSession([[(B, 0.10)]])
    blocker = CandidateBlocker(session, search_address=False)  # type: ignore[arg-type]

    list(blocker.find_candidates([ProbeRecord(A, _embedding(), _embedding(0.2))]))

    assert len(session.selects) == 1


def test_null_distance_row_is_skipped() -> None:
    """Defensive: a NULL distance must not become a 0.0 candidate."""
    session = FakeSession([[(B, None)]])  # type: ignore[list-item]
    pairs = list(
        CandidateBlocker(session, search_address=False).find_candidates(  # type: ignore[arg-type]
            [ProbeRecord(A, _embedding())]
        )
    )

    assert pairs == []


# --- Frame output --------------------------------------------------------------


def test_candidate_frame_matches_the_matcher_contract() -> None:
    """These column names are what RecordMatcher.score_pairs expects."""
    session = FakeSession([[(B, 0.10)], []])
    frame = CandidateBlocker(session).candidate_frame(  # type: ignore[arg-type]
        [ProbeRecord(A, _embedding(), _embedding(0.2))]
    )

    assert list(frame.columns) == ["record_id_a", "record_id_b", "distance", "source"]
    assert len(frame) == 1
    # Stringified: the matcher keys on str ids, and UUID objects round-trip
    # through DuckDB unpredictably.
    assert frame.iloc[0]["record_id_a"] == str(A)


def test_candidate_frame_is_empty_but_shaped_when_nothing_matches() -> None:
    session = FakeSession([[]])
    frame = CandidateBlocker(session, search_address=False).candidate_frame(  # type: ignore[arg-type]
        [ProbeRecord(A, _embedding())]
    )

    assert len(frame) == 0
    assert list(frame.columns) == ["record_id_a", "record_id_b", "distance", "source"]


def test_candidate_pair_is_hashable_and_frozen() -> None:
    pair = CandidatePair(A, B, 0.1, "name")
    with pytest.raises(AttributeError):
        pair.distance = 0.2  # type: ignore[misc]
