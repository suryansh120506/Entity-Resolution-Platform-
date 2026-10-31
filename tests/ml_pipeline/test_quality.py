"""Unit tests for plain-language data-quality issues."""

from __future__ import annotations

from app.ml_pipeline.quality import ISSUE_LABELS, issue_counts, record_issues

COMPLETE = {
    "name": "Emerald Valley Public School",
    "address": "Yercaud Foot Hills",
    "district": "Salem",
    "state": "Tamil Nadu",
    "pincode": "636601",
}


def test_a_complete_record_has_no_issues() -> None:
    assert record_issues(COMPLETE, pincode_known=True) == []


def test_missing_fields_are_named() -> None:
    fields = {**COMPLETE, "district": "", "state": " "}

    assert record_issues(fields, pincode_known=True) == ["missing_district", "missing_state"]


def test_placeholder_pincodes_are_malformed_not_unknown() -> None:
    assert record_issues({**COMPLETE, "pincode": "000000"}, pincode_known=False) == [
        "malformed_pincode"
    ]


def test_a_well_formed_pin_absent_from_the_reference_is_unknown() -> None:
    assert record_issues(COMPLETE, pincode_known=False) == ["unknown_pincode"]


def test_an_unchecked_pin_is_not_reported_as_unknown() -> None:
    assert record_issues(COMPLETE, pincode_known=None) == []


def test_initials_are_flagged_as_abbreviated() -> None:
    assert record_issues({**COMPLETE, "name": "D.P.S"}, pincode_known=True) == ["abbreviated_name"]


def test_a_single_real_word_is_not_abbreviated() -> None:
    assert record_issues({**COMPLETE, "name": "GURUKUL"}, pincode_known=True) == []


def test_placeholder_names_are_flagged() -> None:
    assert "placeholder_name" in record_issues({**COMPLETE, "name": "N/A"}, pincode_known=True)


def test_every_issue_code_has_a_label() -> None:
    fields = {"name": "", "address": "", "district": "", "state": "", "pincode": ""}
    for issue in record_issues(fields, pincode_known=None):
        assert issue in ISSUE_LABELS


def test_counts_skip_issues_nobody_has() -> None:
    counts = issue_counts([["missing_district"], ["missing_district", "missing_state"], []])

    assert counts == {"missing_district": 2, "missing_state": 1}
