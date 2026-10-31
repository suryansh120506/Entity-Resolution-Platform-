"""Unit tests for the ingestion classifier — TRD 2.2 / FR-02.

No database. Every function under test takes `db=None` and degrades to the
string-only pincode check, which is exactly the path that must keep working
when a worker has no session to spare. The DB-backed half of the pincode
feature is covered by the integration suite, not here.

These are behaviour tests, not snapshot tests: they assert the properties the
rest of the pipeline relies on (probability columns that sum to 1 and stay
bound to CLASS_ORDER, pseudo-labels that abstain instead of guessing, an
artefact that refuses to load against a changed feature contract) rather than
specific model outputs, which are free to move with any hyperparameter change.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.ml_pipeline.classification.classifier import (
    ABSTAIN,
    CLASS_ORDER,
    FEATURE_NAMES,
    JUNK,
    MISSING_CONTEXT,
    VALID,
    DataQualityClassifier,
    apply_labeling_functions,
    build_feature_frame,
    extract_features,
    generate_pseudo_labels,
)
from app.models.enums import RecordClassification

# --- Fixtures data -------------------------------------------------------------

VALID_ROWS: list[dict[str, Any]] = [
    {
        "name": "Government Higher Primary School, Hosur",
        "address": "Main Road, Hosur",
        "district": "Krishnagiri",
        "state": "Tamil Nadu",
        "pincode": "635109",
    },
    # Deliberately different header casing/wording — raw_row is keyed by
    # whatever the tenant's CSV used.
    {
        "School Name": "St Josephs Convent High School",
        "Address": "12 Church Street",
        "District": "Bangalore",
        "State": "Karnataka",
        "Pincode": "560001",
    },
    {
        "name": "Kendriya Vidyalaya No 2 Jalahalli",
        "address": "Jalahalli West",
        "district": "Bangalore",
        "state": "Karnataka",
        "pin": "560015",
    },
    {
        "name": "Delhi Public School Rohini",
        "address": "Sector 24 Rohini",
        "district": "North West Delhi",
        "state": "Delhi",
        "postal_code": "110085",
    },
]

JUNK_ROWS: list[dict[str, Any]] = [
    {"name": "N/A", "address": "", "district": "", "state": "", "pincode": "000000"},
    {"name": "TEST", "address": "test", "district": "test", "state": "test", "pincode": "111111"},
    {"name": "XXXX", "address": "-", "district": "-", "state": "-", "pincode": "999999"},
    {"name": "12345", "address": "", "district": "", "state": "", "pincode": ""},
    # The all-empty row: no name at all. Junk, not Missing_Context — a row with
    # nothing in it is not "a real school described thinly".
    {"name": "", "address": "", "district": "", "state": "", "pincode": ""},
]

MISSING_CONTEXT_ROWS: list[dict[str, Any]] = [
    {
        "name": "Govt Primary School",
        "address": "",
        "district": "",
        "state": "Bihar",
        "pincode": "84120",  # five digits — malformed
    },
    {
        "name": "Sacred Heart Convent School",
        "address": "",
        "district": "",
        "state": "",
        "pincode": "",
    },
    {
        "name": "Adarsh Vidya Mandir",
        "address": "Village Road",
        "district": "",
        "state": "",
        "pincode": "0",
    },
]

ALL_ROWS = VALID_ROWS + JUNK_ROWS + MISSING_CONTEXT_ROWS


@pytest.fixture(scope="module")
def trained_classifier() -> DataQualityClassifier:
    """One small model shared across the tests that only read from it."""
    clf = DataQualityClassifier(n_estimators=30, max_depth=3)
    clf.train_on_pseudo_labels(ALL_ROWS * 12)
    return clf


# --- Feature extraction --------------------------------------------------------


def test_extract_features_returns_plain_float_list() -> None:
    """The signature TRD 2.2 specifies: list[float], in FEATURE_NAMES order."""
    features = extract_features(VALID_ROWS[0])

    assert isinstance(features, list)
    assert len(features) == len(FEATURE_NAMES)
    assert all(type(f) is float for f in features)


def test_extract_features_scores_a_clean_row_as_clean() -> None:
    features = dict(zip(FEATURE_NAMES, extract_features(VALID_ROWS[0]), strict=True))

    assert features["missing_required_field_count"] == 0.0
    assert features["name_is_placeholder"] == 0.0
    assert features["pincode_is_placeholder"] == 0.0
    assert features["name_token_count"] == 5.0


def test_extract_features_flags_a_junk_row() -> None:
    features = dict(zip(FEATURE_NAMES, extract_features(JUNK_ROWS[0]), strict=True))

    assert features["name_is_placeholder"] == 1.0
    assert features["pincode_is_placeholder"] == 1.0
    assert features["missing_required_field_count"] == 3.0


@pytest.mark.parametrize("row", VALID_ROWS)
def test_header_aliases_resolve_to_the_same_canonical_fields(row: dict[str, Any]) -> None:
    """`School Name`, `Pincode`, `pin`, `postal_code` must all be found.

    A missed alias reads as a missing required field, which pushes a perfectly
    good row toward Missing_Context — a silent quality regression driven by
    nothing but the tenant's choice of CSV header.
    """
    features = dict(zip(FEATURE_NAMES, extract_features(row), strict=True))
    assert features["missing_required_field_count"] == 0.0


def test_a_real_name_containing_a_sentinel_word_is_not_flagged() -> None:
    """The placeholder regex matches whole values, never substrings."""
    row = {
        "name": "Test Valley Senior Secondary School",
        "address": "Ridge Road",
        "district": "Shimla",
        "state": "Himachal Pradesh",
        "pincode": "171001",
    }
    features = dict(zip(FEATURE_NAMES, extract_features(row), strict=True))
    assert features["name_is_placeholder"] == 0.0


def test_build_feature_frame_handles_an_empty_batch() -> None:
    """An empty job must produce an empty frame, not an exception."""
    frame = build_feature_frame([])

    assert len(frame) == 0
    assert list(frame.columns)[: len(FEATURE_NAMES)] == list(FEATURE_NAMES)


def test_single_row_and_batch_paths_agree() -> None:
    """`extract_features` delegates to the batch path; prove it stays true.

    Training runs through `build_feature_frame` and single-row inference may
    run through `extract_features`. If these two ever diverge, the model is
    scored on features it was not trained on and nothing announces it.
    """
    frame = build_feature_frame(ALL_ROWS)

    for index, row in enumerate(ALL_ROWS):
        single = extract_features(row)
        batched = [float(v) for v in frame.iloc[index][list(FEATURE_NAMES)]]
        assert single == batched, f"row {index} disagrees: {single} != {batched}"


# --- Weak supervision ----------------------------------------------------------


def test_labeling_functions_return_a_well_formed_label_matrix() -> None:
    frame = build_feature_frame(ALL_ROWS)
    votes = apply_labeling_functions(frame)

    assert votes.shape[0] == len(ALL_ROWS)
    assert set(np.unique(votes)) <= {ABSTAIN, VALID, JUNK, MISSING_CONTEXT}


def test_pseudo_labels_are_bounded_and_shaped() -> None:
    pseudo = generate_pseudo_labels(build_feature_frame(ALL_ROWS))

    assert pseudo.labels.shape == (len(ALL_ROWS),)
    assert pseudo.confidence.shape == (len(ALL_ROWS),)
    assert ((pseudo.confidence >= 0.0) & (pseudo.confidence <= 1.0)).all()
    assert 0.0 <= pseudo.coverage <= 1.0


def test_obvious_rows_get_the_obvious_pseudo_label() -> None:
    pseudo = generate_pseudo_labels(build_feature_frame(ALL_ROWS))
    labels = pseudo.labels

    valid_slice = labels[: len(VALID_ROWS)]
    junk_slice = labels[len(VALID_ROWS) : len(VALID_ROWS) + len(JUNK_ROWS)]
    thin_slice = labels[len(VALID_ROWS) + len(JUNK_ROWS) :]

    assert (valid_slice == VALID).all(), f"clean rows mislabelled: {valid_slice}"
    assert (junk_slice == JUNK).all(), f"junk rows mislabelled: {junk_slice}"
    assert (thin_slice == MISSING_CONTEXT).all(), f"thin rows mislabelled: {thin_slice}"


def test_a_row_empty_in_every_field_is_junk_not_missing_context() -> None:
    """Regression: the missing-field LFs must not claim a nameless row.

    They gate on `_has_usable_name`. Without that gate this row scores
    Missing_Context 3.5 to Junk 2.0 purely on how many fields are absent, and
    the class that means "a real school described too thinly" ends up holding
    rows that describe nothing at all.
    """
    empty_row = {"name": "", "address": "", "district": "", "state": "", "pincode": ""}
    pseudo = generate_pseudo_labels(build_feature_frame([empty_row]))

    assert pseudo.labels[0] == JUNK


def test_a_placeholder_pin_with_a_district_is_still_valid() -> None:
    """Regression: 29,794 complete rows of a real export were set aside for
    carrying PIN "000000". A district is enough to block and match on."""
    row = {
        "instituteName": "BABUJI CONVENT SCHOOL",
        "addressLine1": "ALIGARH, ALIGARH, UTTAR PRADESH",
        "localityDistrictState": "ALIGARH, ALIGARH, UTTAR PRADESH",
        "pinCode": "000000",
    }
    pseudo = generate_pseudo_labels(build_feature_frame([row]))

    assert pseudo.labels[0] == VALID


@pytest.mark.parametrize("name", ["D P S", "K.V NO1", "HMY HSS", "MGM", "J N V"])
def test_initials_are_missing_context_not_junk(name: str) -> None:
    """Regression: real schools named only by initials were voted Junk."""
    row = {
        "name": name,
        "address": "Main Road",
        "district": "Sagar",
        "state": "MP",
        "pincode": "470228",
    }
    pseudo = generate_pseudo_labels(build_feature_frame([row]))

    assert pseudo.labels[0] == MISSING_CONTEXT


@pytest.mark.parametrize("name", ["OASIS", "GURUKUL", "KIDZEE"])
def test_a_real_single_word_name_with_a_location_is_valid(name: str) -> None:
    row = {
        "name": name,
        "address": "Station Road",
        "district": "Ganjam",
        "state": "Odisha",
        "pincode": "000000",
    }
    pseudo = generate_pseudo_labels(build_feature_frame([row]))

    assert pseudo.labels[0] == VALID


def test_punctuation_is_still_junk() -> None:
    row = {"name": "*", "address": "", "district": "Kendrapara", "state": "", "pincode": "000000"}
    pseudo = generate_pseudo_labels(build_feature_frame([row]))

    assert pseudo.labels[0] == JUNK


def test_a_placeholder_pin_with_no_district_is_missing_context() -> None:
    row = {"name": "Babuji Convent School", "address": "Main Road", "pincode": "000000"}
    pseudo = generate_pseudo_labels(build_feature_frame([row]))

    assert pseudo.labels[0] == MISSING_CONTEXT


def test_contested_rows_fall_below_the_training_threshold() -> None:
    """Disagreement must produce a low-confidence row, not a coin flip.

    Those rows are the review queue. A weighted vote that split evenly cannot
    be allowed to land in the training set on argmax tie-breaking alone.
    """
    pseudo = generate_pseudo_labels(build_feature_frame(ALL_ROWS))

    for confidence in pseudo.confidence:
        assert confidence == 0.0 or confidence >= 0.5

    mask = pseudo.training_mask(min_confidence=0.99)
    assert mask.sum() <= len(ALL_ROWS)
    assert not mask[pseudo.confidence < 0.99].any()


def test_label_counts_cover_every_class_and_abstain() -> None:
    counts = generate_pseudo_labels(build_feature_frame(ALL_ROWS)).label_counts()

    assert set(counts) == {cls.value for cls in CLASS_ORDER} | {"ABSTAIN"}
    assert sum(counts.values()) == len(ALL_ROWS)


# --- Training ------------------------------------------------------------------


def test_train_on_pseudo_labels_reports_what_it_did(
    trained_classifier: DataQualityClassifier,
) -> None:
    summary = DataQualityClassifier(n_estimators=10, max_depth=2).train_on_pseudo_labels(
        ALL_ROWS * 12
    )

    assert summary["n_rows"] == len(ALL_ROWS) * 12
    assert summary["n_trained_on"] + summary["n_excluded"] == summary["n_rows"]
    assert 0.0 <= summary["lf_coverage"] <= 1.0
    assert summary["classes_fitted"] == [cls.value for cls in CLASS_ORDER]
    assert trained_classifier.is_trained


def test_training_on_no_rows_is_an_error() -> None:
    with pytest.raises(ValueError, match="no rows"):
        DataQualityClassifier().train_on_pseudo_labels([])


def test_training_on_a_single_class_is_an_error() -> None:
    """A constant classifier is not a classifier. Fail loudly instead."""
    with pytest.raises(ValueError, match="single class"):
        DataQualityClassifier(n_estimators=5).train_on_pseudo_labels(JUNK_ROWS * 10)


def test_unreachable_confidence_threshold_is_an_error() -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        DataQualityClassifier(n_estimators=5).train_on_pseudo_labels(
            ALL_ROWS * 4, min_confidence=1.01
        )


# --- Prediction ----------------------------------------------------------------


def test_predict_before_training_raises() -> None:
    with pytest.raises(RuntimeError, match="has not been trained"):
        DataQualityClassifier().predict_batch(ALL_ROWS)


def test_predict_batch_shape_and_labels(trained_classifier: DataQualityClassifier) -> None:
    predictions = trained_classifier.predict_batch(ALL_ROWS)

    assert len(predictions) == len(ALL_ROWS)
    assert set(predictions["classification"]) <= {cls.value for cls in CLASS_ORDER}


def test_predicted_labels_are_writable_to_school_records(
    trained_classifier: DataQualityClassifier,
) -> None:
    """Output strings must satisfy the CHECK constraint in migration 0001.

    `classification IN ('Valid', 'Junk', 'Missing_Context')` — if this drifts,
    every insert from the classify stage fails at the database, one stage too
    late to be obvious.
    """
    predictions = trained_classifier.predict_batch(ALL_ROWS)
    allowed = {cls.value for cls in RecordClassification}

    assert set(predictions["classification"]) <= allowed


def test_probabilities_sum_to_one_across_fixed_columns(
    trained_classifier: DataQualityClassifier,
) -> None:
    predictions = trained_classifier.predict_batch(ALL_ROWS)
    columns = [f"p_{cls.value.lower()}" for cls in CLASS_ORDER]
    probabilities = predictions[columns].to_numpy()

    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(predictions["confidence"].to_numpy(), probabilities.max(axis=1))


def test_predict_batch_handles_an_empty_batch(
    trained_classifier: DataQualityClassifier,
) -> None:
    assert len(trained_classifier.predict_batch([])) == 0


def test_two_class_cold_start_keeps_columns_aligned() -> None:
    """Regression: a tenant whose first upload has no Missing_Context rows.

    XGBoost's sklearn wrapper only sets `num_class` when it infers more than
    two classes, so `multi:softprob` on a two-class batch dies inside the C
    library. Beyond training at all, the column layout must not shift: the
    unseen class reads 0.0 and stays in its CLASS_ORDER position rather than
    the remaining two silently sliding left.
    """
    clf = DataQualityClassifier(n_estimators=10, max_depth=2)
    summary = clf.train_on_pseudo_labels((VALID_ROWS + JUNK_ROWS) * 10)

    assert summary["classes_fitted"] == [
        RecordClassification.VALID.value,
        RecordClassification.JUNK.value,
    ]

    predictions = clf.predict_batch(ALL_ROWS)
    assert (predictions["p_missing_context"].to_numpy() == 0.0).all()

    columns = [f"p_{cls.value.lower()}" for cls in CLASS_ORDER]
    assert np.allclose(predictions[columns].to_numpy().sum(axis=1), 1.0, atol=1e-5)


# --- Persistence ---------------------------------------------------------------


def test_save_load_round_trip_preserves_predictions(
    trained_classifier: DataQualityClassifier, tmp_path: Any
) -> None:
    path = str(tmp_path / "classifier.json")
    trained_classifier.save(path)

    reloaded = DataQualityClassifier.load(path)
    assert reloaded.is_trained

    before = trained_classifier.predict_batch(ALL_ROWS)
    after = reloaded.predict_batch(ALL_ROWS)

    assert list(before["classification"]) == list(after["classification"])
    np.testing.assert_allclose(
        before[[f"p_{cls.value.lower()}" for cls in CLASS_ORDER]].to_numpy(),
        after[[f"p_{cls.value.lower()}" for cls in CLASS_ORDER]].to_numpy(),
        atol=1e-6,
    )


def test_saving_an_untrained_model_raises(tmp_path: Any) -> None:
    with pytest.raises(RuntimeError, match="untrained"):
        DataQualityClassifier().save(str(tmp_path / "nope.json"))


def test_two_class_model_survives_a_round_trip(tmp_path: Any) -> None:
    """The fitted-class list is what keeps columns aligned; it must persist."""
    clf = DataQualityClassifier(n_estimators=10, max_depth=2)
    clf.train_on_pseudo_labels((VALID_ROWS + JUNK_ROWS) * 10)

    path = str(tmp_path / "two_class.json")
    clf.save(path)
    reloaded = DataQualityClassifier.load(path)

    predictions = reloaded.predict_batch(ALL_ROWS)
    assert (predictions["p_missing_context"].to_numpy() == 0.0).all()
    columns = [f"p_{cls.value.lower()}" for cls in CLASS_ORDER]
    assert np.allclose(predictions[columns].to_numpy().sum(axis=1), 1.0, atol=1e-5)


def test_loading_a_foreign_artefact_raises(tmp_path: Any) -> None:
    """A booster we did not write has no class/feature metadata to trust."""
    from xgboost import XGBClassifier

    foreign = XGBClassifier(n_estimators=3, max_depth=2)
    foreign.fit(np.zeros((6, len(FEATURE_NAMES))), np.array([0, 1, 0, 1, 0, 1]))
    path = str(tmp_path / "foreign.json")
    foreign.save_model(path)

    with pytest.raises(ValueError, match="not written by DataQualityClassifier"):
        DataQualityClassifier.load(path)


def test_loading_against_a_changed_feature_contract_raises(
    trained_classifier: DataQualityClassifier, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reordering FEATURE_NAMES must invalidate every existing artefact.

    The shapes still match after a reorder, so nothing crashes — the model
    just reads each feature out of the wrong column and is confidently wrong.
    This is the check that turns that into a startup failure.
    """
    path = str(tmp_path / "classifier.json")
    trained_classifier.save(path)

    from app.ml_pipeline.classification import classifier as classifier_module

    monkeypatch.setattr(
        classifier_module, "FEATURE_NAMES", tuple(reversed(FEATURE_NAMES)), raising=True
    )

    with pytest.raises(ValueError, match="different feature contract"):
        DataQualityClassifier.load(path)
