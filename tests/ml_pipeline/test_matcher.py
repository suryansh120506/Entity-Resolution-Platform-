"""Unit tests for probabilistic matching — FR-04 / TRD 2.2.

No Postgres: Splink runs on DuckDB in-process, which is the backend the
matcher uses in production anyway, so these exercise the real engine rather
than a stand-in. Training is genuinely run, so the module-scoped fixture
matters — refitting per test would make this suite minutes long.

The assertions are about *properties*, not exact scores. A specific
probability is a function of the synthetic fixture and of Splink's version,
and pinning one would produce a test that fails on every dependency bump
without anything being wrong. What must hold is that duplicates outrank
non-duplicates, that scores are writable to `clusters.confidence_score`, and
that the pair set in equals the pair set out.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.ml_pipeline.matching.matcher import (
    CONFIDENCE_DECIMALS,
    RecordMatcher,
    labels_from_feedback,
)

N_GROUPS = 20


def _records() -> pd.DataFrame:
    """Twenty triples: a/b are the same school written twice, c is not."""
    rows = []
    for i in range(N_GROUPS):
        pin = f"56{i:04d}"
        rows += [
            {
                "record_id": f"r{i}a",
                "name_normalized": f"government higher primary school unit {i}",
                "address_normalized": f"main road block {i} bangalore",
                "pincode": pin,
            },
            {
                "record_id": f"r{i}b",
                "name_normalized": f"government higher primary school unit {i}",
                "address_normalized": f"main rd block {i} bangalore",
                "pincode": pin,
            },
            {
                "record_id": f"r{i}c",
                "name_normalized": f"saint xaviers academy {i}",
                "address_normalized": f"hill view layout {i}",
                "pincode": f"11{i:04d}",
            },
        ]
    return pd.DataFrame(rows)


def _candidates() -> pd.DataFrame:
    """Blocker-shaped input: three true pairs, two false ones."""
    return pd.DataFrame(
        {
            "record_id_a": ["r0a", "r1a", "r0a", "r5a", "r9a"],
            "record_id_b": ["r0b", "r1b", "r7c", "r5c", "r9b"],
            "distance": [0.02, 0.03, 0.31, 0.29, 0.01],
            "source": ["both", "name", "name", "address", "both"],
        }
    )


@pytest.fixture(scope="module")
def records() -> pd.DataFrame:
    return _records()


@pytest.fixture(scope="module")
def trained(records: pd.DataFrame) -> RecordMatcher:
    matcher = RecordMatcher(records, max_pairs_for_u=20_000)
    matcher.train()
    return matcher


# --- Construction --------------------------------------------------------------


def test_missing_record_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing required column"):
        RecordMatcher(pd.DataFrame({"record_id": ["a"], "name_normalized": ["x"]}))


def test_invalid_recall_is_rejected(records: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="deterministic_recall"):
        RecordMatcher(records, deterministic_recall=0.0)


def test_construction_does_not_train(records: pd.DataFrame) -> None:
    matcher = RecordMatcher(records)

    assert not matcher.is_trained
    with pytest.raises(RuntimeError, match="not been trained"):
        _ = matcher.model_dict


# --- Training ------------------------------------------------------------------


def test_train_summary_is_legible_in_model_versions(records: pd.DataFrame) -> None:
    """The summary lands in `model_versions.metrics`, so it must be legible.

    `str()` on a Splink rule object yields `<... object at 0x7f...>` — a
    memory address that differs every run and tells a future reader nothing
    about what was trained.
    """
    matcher = RecordMatcher(records, max_pairs_for_u=5_000)
    summary = matcher.train()

    assert summary["n_records"] == N_GROUPS * 3
    assert summary["em_rules_failed"] == 0
    assert len(summary["em_rules_trained"]) == 2
    for rule in summary["em_rules_trained"]:
        assert "0x" not in rule, f"rule description is an object repr: {rule}"
        assert "=" in rule
    assert matcher.is_trained


# --- Scoring -------------------------------------------------------------------


def test_scoring_before_training_raises(records: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError, match="not been trained"):
        RecordMatcher(records).score_pairs(_candidates())


def test_missing_pair_columns_are_rejected(trained: RecordMatcher) -> None:
    with pytest.raises(ValueError, match="missing required column"):
        trained.score_pairs(pd.DataFrame({"record_id_a": ["r0a"]}))


def test_every_input_pair_is_scored_exactly_once(trained: RecordMatcher) -> None:
    """Splink must score our pairs — not its own, and not a subset.

    The whole reason `score_pairs` pivots onto a pair key is that Splink
    otherwise picks pairs via blocking rules. If that ever regresses, the row
    count is the first thing that changes.
    """
    candidates = _candidates()
    scored = trained.score_pairs(candidates)

    assert len(scored) == len(candidates)
    assert list(scored["record_id_a"]) == list(candidates["record_id_a"])
    assert list(scored["record_id_b"]) == list(candidates["record_id_b"])


def test_duplicates_outrank_non_duplicates(trained: RecordMatcher) -> None:
    """The property that matters. Absolute values are free to move."""
    scored = trained.score_pairs(_candidates())

    duplicates = scored[scored["record_id_b"].str.endswith("b")]["confidence_score"]
    non_duplicates = scored[scored["record_id_b"].str.endswith("c")]["confidence_score"]

    assert duplicates.min() > non_duplicates.max()


def test_confidence_score_is_writable_to_the_clusters_column(
    trained: RecordMatcher,
) -> None:
    """`clusters.confidence_score` is NUMERIC(4,3) CHECKed to [0, 1].

    Rounding at insert time instead would let a value pass every check in
    Python and then fail in the database, one stage too late.
    """
    scored = trained.score_pairs(_candidates())
    scores = scored["confidence_score"]

    assert scores.between(0.0, 1.0).all()
    assert (scores == scores.round(CONFIDENCE_DECIMALS)).all()
    assert scores.notna().all()


def test_blocker_columns_pass_through_untouched(trained: RecordMatcher) -> None:
    """Vector evidence and probabilistic score stay side by side for review."""
    candidates = _candidates()
    scored = trained.score_pairs(candidates)

    assert list(scored["distance"]) == list(candidates["distance"])
    assert list(scored["source"]) == list(candidates["source"])
    assert "match_weight" in scored.columns


def test_pair_orientation_is_preserved(trained: RecordMatcher) -> None:
    """Splink orders its output by its own synthetic ids, not ours.

    The caller gets back the orientation it supplied, so a downstream join on
    (record_id_a, record_id_b) cannot silently miss.
    """
    flipped = pd.DataFrame({"record_id_a": ["r0b"], "record_id_b": ["r0a"]})
    scored = trained.score_pairs(flipped)

    assert scored.iloc[0]["record_id_a"] == "r0b"
    assert scored.iloc[0]["record_id_b"] == "r0a"
    assert scored.iloc[0]["confidence_score"] > 0.0


def test_empty_candidate_frame_returns_an_empty_shaped_frame(
    trained: RecordMatcher,
) -> None:
    scored = trained.score_pairs(pd.DataFrame(columns=["record_id_a", "record_id_b"]))

    assert len(scored) == 0
    assert "confidence_score" in scored.columns


def test_unknown_record_id_is_an_error_not_a_silent_drop(
    trained: RecordMatcher,
) -> None:
    """A pair naming a record we have no attributes for cannot be scored.

    Dropping it silently would mean the blocker proposed a pair that simply
    vanished, and the missing cluster would be blamed on the matcher's
    threshold rather than on the two stages disagreeing about the record set.
    """
    with pytest.raises(ValueError, match="absent from the record frame"):
        trained.score_pairs(
            pd.DataFrame({"record_id_a": ["r0a"], "record_id_b": ["does-not-exist"]})
        )


# --- Active learning -----------------------------------------------------------


def test_labels_from_feedback_maps_decisions_to_scores() -> None:
    feedback = pd.DataFrame(
        {
            "record_id_a": ["a", "b", "c"],
            "record_id_b": ["x", "y", "z"],
            "decision": ["MERGE", "SPLIT", "REJECT"],
        }
    )

    labels = labels_from_feedback(feedback)

    assert len(labels) == 2, "SPLIT is not a pairwise verdict and must be dropped"
    assert set(labels["record_id_b"]) == {"x", "z"}
    assert labels["clerical_match_score"].tolist() == [1.0, 0.0]


def test_update_from_labels_requires_a_trained_model(records: pd.DataFrame) -> None:
    labels = pd.DataFrame(
        {"record_id_a": ["r0a"], "record_id_b": ["r0b"], "clerical_match_score": [1.0]}
    )

    with pytest.raises(RuntimeError, match="Train the matcher before"):
        RecordMatcher(records).update_model_from_labels(labels)


def test_update_from_labels_validates_its_input(trained: RecordMatcher) -> None:
    with pytest.raises(ValueError, match="missing required column"):
        trained.update_model_from_labels(pd.DataFrame({"record_id_a": ["r0a"]}))

    with pytest.raises(ValueError, match="empty"):
        trained.update_model_from_labels(
            pd.DataFrame(columns=["record_id_a", "record_id_b", "clerical_match_score"])
        )

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        trained.update_model_from_labels(
            pd.DataFrame(
                {
                    "record_id_a": ["r0a"],
                    "record_id_b": ["r0b"],
                    "clerical_match_score": [1.5],
                }
            )
        )


def test_update_from_labels_changes_the_model(records: pd.DataFrame) -> None:
    """The overnight fine-tune has to actually move the parameters.

    A wrapper that ran without error and changed nothing would be the worst
    outcome here: the retrain job would report success every night while the
    stewards' labels went nowhere.
    """
    matcher = RecordMatcher(records, max_pairs_for_u=5_000)
    matcher.train()
    before = matcher.score_pairs(_candidates())["confidence_score"].tolist()

    feedback = pd.DataFrame(
        {
            "record_id_a": [f"r{i}a" for i in range(10)] * 2,
            "record_id_b": [f"r{i}b" for i in range(10)] + [f"r{i}c" for i in range(10)],
            "decision": ["MERGE"] * 10 + ["REJECT"] * 10,
        }
    )
    summary = matcher.update_model_from_labels(labels_from_feedback(feedback))

    assert summary == {"n_labels": 20, "n_positive": 10, "n_negative": 10}
    after = matcher.score_pairs(_candidates())["confidence_score"].tolist()
    assert before != after, "labels did not influence the model"


# --- Persistence ---------------------------------------------------------------


def test_save_load_round_trip_reproduces_scores(
    trained: RecordMatcher, records: pd.DataFrame, tmp_path: object
) -> None:
    path = str(tmp_path / "matcher.json")  # type: ignore[operator]
    trained.save(path)

    reloaded = RecordMatcher(records)
    assert not reloaded.is_trained
    reloaded.load(path)
    assert reloaded.is_trained

    expected = trained.score_pairs(_candidates())["confidence_score"].tolist()
    assert reloaded.score_pairs(_candidates())["confidence_score"].tolist() == expected


def test_saving_an_untrained_matcher_raises(records: pd.DataFrame, tmp_path: object) -> None:
    with pytest.raises(RuntimeError, match="not been trained"):
        RecordMatcher(records).save(str(tmp_path / "nope.json"))  # type: ignore[operator]
