"""Unit tests for `clean_value` — display cleaning, not embedding normalisation.

Every non-trivial case here is a real value observed in a live tenant export,
not a synthesised edge case.
"""

from __future__ import annotations

import pytest

from app.ml_pipeline.cleaning import clean_fields, clean_value


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Leading comma from a blank first CSV segment.
        (
            ", GILLY MUNDI MEMORIAL COMMUNITY SCHOOL, LEHRIAN",
            "GILLY MUNDI MEMORIAL COMMUNITY SCHOOL, LEHRIAN",
        ),
        # Comma with no following space.
        ("MINI. SECTT,HISAR, HISAR, HARYANA", "MINI. SECTT, HISAR, HISAR, HARYANA"),
        # Trailing dash left by a missing pincode suffix.
        ("MUMBAI, MUMBAI, MAHARASHTRA -", "MUMBAI, MUMBAI, MAHARASHTRA"),
        # A genuinely blank middle segment.
        ("A,, B", "A, B"),
        ("A, , B", "A, B"),
        # Already clean — must be a no-op.
        ("VADDEM, NEW VADDEM, SOUTH GOA, GOA", "VADDEM, NEW VADDEM, SOUTH GOA, GOA"),
        ("Test Valley High School", "Test Valley High School"),
        # Internal whitespace runs collapse.
        ("Too    Many   Spaces", "Too Many Spaces"),
    ],
)
def test_clean_value_real_cases(raw: str, expected: str) -> None:
    assert clean_value(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "N/A", "n/a", "NA", "None", "null", "NULL", "nil", "NaN", "-", "--", "unknown"],
)
def test_clean_value_null_sentinels_become_empty(raw: str) -> None:
    assert clean_value(raw) == ""


def test_clean_value_none_becomes_empty() -> None:
    assert clean_value(None) == ""


def test_clean_value_whole_value_of_only_punctuation_does_not_survive() -> None:
    """A value that is *only* junk must not come out as a lone `,` or `-`."""
    assert clean_value("-") == ""
    assert clean_value(",") == ""
    assert clean_value(" - ") == ""


def test_clean_value_does_not_touch_a_real_name_containing_na() -> None:
    """The sentinel match is whole-value, not substring — "NA" inside a
    longer name must survive."""
    assert clean_value("NATIONAL PUBLIC SCHOOL") == "NATIONAL PUBLIC SCHOOL"


def test_clean_fields_maps_over_a_dict() -> None:
    assert clean_fields({"name": ", A School", "address": "N/A"}) == {
        "name": "A School",
        "address": "",
    }
