"""Unit tests for location-partitioned blocking — no database, no model.

Vectors are constructed directly so each assertion is about *which records
may be compared*, not about what a sentence-transformer happens to embed.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from app.ml_pipeline.blocking.location import (
    BlockingRecord,
    find_location_candidates,
    location_keys,
    normalise_place,
    usable_pincode,
)

DIM = 8


def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _record(
    vector: np.ndarray,
    *,
    districts: tuple[str, ...] = (),
    pincode: str | None = None,
    state: str | None = "karnataka",
    has_address: bool = True,
) -> BlockingRecord:
    return BlockingRecord(
        record_id=uuid.uuid4(),
        name_vector=vector,
        address_vector=vector,
        has_name=True,
        has_address=has_address,
        district_keys=frozenset(districts),
        pincode=pincode,
        state=state,
    )


def _pair_ids(pairs: list) -> set[frozenset[uuid.UUID]]:
    return {frozenset((p.record_id_a, p.record_id_b)) for p in pairs}


def test_identical_schools_in_the_same_district_are_candidates() -> None:
    same = _vec(1)
    a = _record(same, districts=("karnataka|bangalore",))
    b = _record(same, districts=("karnataka|bangalore",))

    pairs, stats = find_location_candidates([a, b])

    assert _pair_ids(pairs) == {frozenset((a.record_id, b.record_id))}
    assert pairs[0].source == "both"
    assert stats.blocks == 1


def test_identical_schools_in_different_districts_are_never_compared() -> None:
    """The whole point: a generic name in two districts is two schools."""
    same = _vec(1)
    a = _record(same, districts=("haryana|hisar",))
    b = _record(same, districts=("kerala|kollam",), state="kerala")

    pairs, _ = find_location_candidates([a, b])

    assert pairs == []


def test_a_shared_pincode_rescues_a_misspelled_district() -> None:
    same = _vec(2)
    a = _record(same, districts=("haryana|gurugram",), pincode="122001")
    b = _record(same, districts=("haryana|gurgaon",), pincode="122001")

    pairs, _ = find_location_candidates([a, b])

    assert len(pairs) == 1


def test_far_apart_vectors_in_one_block_are_not_candidates() -> None:
    a = _record(_vec(3), districts=("x|y",))
    b = _record(-_vec(3), districts=("x|y",))

    pairs, _ = find_location_candidates([a, b], max_distance=0.3)

    assert pairs == []


def test_unlocated_records_probe_their_state_but_not_each_other_across_it() -> None:
    same = _vec(4)
    located = _record(same, districts=("karnataka|mysore",))
    unlocated = _record(same)  # no district, no pincode, but a state

    pairs, stats = find_location_candidates([located, unlocated])

    assert len(pairs) == 1
    assert stats.state_fallback_records == 1


def test_records_with_no_location_at_all_are_counted_not_guessed() -> None:
    a = _record(_vec(5), state=None)
    b = _record(_vec(5), state=None)

    pairs, stats = find_location_candidates([a, b])

    assert pairs == []
    assert stats.unlocated_records == 2


def test_blank_addresses_do_not_look_like_a_perfect_address_match() -> None:
    """An empty string embeds to the same vector every time."""
    blank = _vec(6)
    a = BlockingRecord(uuid.uuid4(), _vec(7), blank, True, False, frozenset({"k|d"}), None, "k")
    b = BlockingRecord(uuid.uuid4(), _vec(8), blank, True, False, frozenset({"k|d"}), None, "k")

    pairs, _ = find_location_candidates([a, b], max_distance=0.3)

    assert pairs == []


def test_top_k_bounds_candidates_per_record() -> None:
    base = _vec(9)
    records = [_record(base, districts=("s|d",)) for _ in range(30)]

    pairs, _ = find_location_candidates(records, top_k=3)

    # Each record proposes at most 3 (pairs dedupe across both directions).
    assert 0 < len(pairs) <= 30 * 3


def test_pairs_are_canonically_ordered() -> None:
    same = _vec(10)
    records = [_record(same, districts=("s|d",)) for _ in range(5)]

    pairs, _ = find_location_candidates(records)

    assert all(str(p.record_id_a) < str(p.record_id_b) for p in pairs)


def test_reference_pincode_supplies_the_government_district_spelling() -> None:
    keys, pincode, state = location_keys(
        {"district": "GURGAON", "state": "Haryana", "pincode": "122001"},
        {"122001": ("GURUGRAM", "HARYANA")},
    )

    assert keys == {"haryana|gurugram", "haryana|gurgaon"}
    assert pincode == "122001"
    assert state == "haryana"


def test_state_is_taken_from_the_reference_when_the_row_has_none() -> None:
    _, _, state = location_keys(
        {"district": "", "state": "", "pincode": "560001"}, {"560001": ("BANGALORE", "KARNATAKA")}
    )

    assert state == "karnataka"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("560001", "560001"),
        (" 560001 ", "560001"),
        ("000000", None),
        ("111111", None),
        ("12345", None),
        ("056001", None),
        ("", None),
    ],
)
def test_usable_pincode(value: str, expected: str | None) -> None:
    assert usable_pincode(value) == expected


def test_place_names_compare_across_spacing_and_case() -> None:
    assert normalise_place("Tamil Nadu") == normalise_place("TAMILNADU")
