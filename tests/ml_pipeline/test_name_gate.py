"""Unit tests for the distinctive-name veto.

Every case is a pair observed in a real 21,736-record upload, labelled by
what a person reading it would conclude.
"""

from __future__ import annotations

import pytest

from app.ml_pipeline.matching.name_gate import RecordText, agreement, compatible


def _r(name: str, address: str = "", district: str = "", state: str = "") -> RecordText:
    return RecordText.build(name, address, district, state)


SAME = [
    (
        _r("niraja sahay dayanand anglo vedic public school", "kankey ranchi", "ranchi"),
        _r("nirja sahay dayanand anglo vedic public school", "kanke ranchi", "ranchi"),
    ),
    (
        _r("government senior secondary school sainj", "sainj ani banjar kullu", "kullu"),
        _r("government senior secondary school sainj", "sainj ani banjar kullu", "kullu"),
    ),
    (
        _r("government senior secondary school nagwain", "nagwain kullu ho", "kullu"),
        _r("government senior secondary school nangwain", "nangwain kullu ho", "kullu"),
    ),
    (
        _r("satyam international", "abdullah chik bairia patna 800007", "patna"),
        _r("satyam international school", "abdullah chak bairia patna 800007", "patna"),
    ),
]

DIFFERENT = [
    # Share only boilerplate and the town.
    (_r("tagore sen senior school", "pilani"), _r("lohiya sen senior school", "pilani")),
    (
        _r("government college kullu", "dhalpur kullu", "kullu"),
        _r("government college banjjar", "dhalpur kullu", "kullu"),
    ),
    # District name inside the school name.
    (
        _r("government senior secondary school dhendri hisar", "", "hisar"),
        _r("sk senior secondary school hisar", "", "hisar"),
    ),
    # State abbreviation appended.
    (
        _r("government senior secondary school bahadurpur sangrur pb", "", "sangrur"),
        _r("government senior secondary school ubewal sangrur pb", "", "sangrur"),
    ),
    # Gender, one-sided and two-sided.
    (_r("marwari boys college", "upper bazar"), _r("marwari women college", "upper bazar")),
    (
        _r("government senior secondary school kullu", "kullu", "kullu"),
        _r("government senior secondary school girls kullu", "kullu", "kullu"),
    ),
    # A college and a school.
    (
        _r("government girls college", "jhunjhunu"),
        _r("government girls sen senior school", "jhunjhunu"),
    ),
    # Numbered branches.
    (
        _r("atomic energy central school number 2", "anushakti nagar"),
        _r("atomic energy central school number 3", "anushakti nagar"),
    ),
    # Chain name, different localities.
    (
        _r(
            "dayanand anglo vedic public school",
            "sector 6 dwarka south west delhi",
            "south west delhi",
            "delhi",
        ),
        _r(
            "dayanand anglo vedic public school",
            "sector b pocket 1 vasant kunj south west delhi",
            "south west delhi",
            "delhi",
        ),
    ),
    # Pure boilerplate, told apart only by sector number and pincode.
    (
        _r("government model school", "sector 8 chandigarh 160009", "chandigarh"),
        _r("government model school", "sector 18 chandigarh 160018", "chandigarh"),
    ),
    # A generic name against a specific one.
    (_r("government sen senior school", "jhunjhunu"), _r("rainbow sen senior school", "jhunjhunu")),
    # Half the distinctive words is the town, not the school.
    (
        _r("raja ram senior secondary school mandi dabwali", "mandi dabwali sirsa", "sirsa"),
        _r("arya senior secondary school mandi dabwali", "mandi dabwali sirsa", "sirsa"),
    ),
    # Booksellers share their trade, not their identity.
    (_r("sheetla book depot", "aminabad lucknow"), _r("zeeshan book depot", "telibag lucknow")),
]


@pytest.mark.parametrize(("a", "b"), SAME)
def test_genuine_duplicates_pass(a: RecordText, b: RecordText) -> None:
    assert compatible(a, b), agreement(a, b)


@pytest.mark.parametrize(("a", "b"), DIFFERENT)
def test_different_schools_are_vetoed(a: RecordText, b: RecordText) -> None:
    assert not compatible(a, b), agreement(a, b)


def test_identical_boilerplate_with_nothing_more_specific_defers_to_the_model() -> None:
    a = _r(
        "government senior secondary school",
        "kullu ho kullu himachal pradesh",
        "kullu",
        "himachal pradesh",
    )
    assert agreement(a, a) == 1.0


def test_the_gate_is_symmetric() -> None:
    for a, b in SAME + DIFFERENT:
        assert agreement(a, b) == agreement(b, a)
