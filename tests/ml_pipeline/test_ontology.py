"""Ontology engine: loading guarantees and context-aware standardisation."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ontology import OntologyEngine, OntologyError, available_ontologies


@pytest.fixture(scope="module")
def edu() -> OntologyEngine:
    return OntologyEngine.load("education")


@pytest.fixture(scope="module")
def health() -> OntologyEngine:
    return OntologyEngine.load("healthcare")


def _write(directory: Path, name: str, body: str) -> None:
    (directory / f"{name}.yaml").write_text(body, encoding="utf-8")


MINIMAL = """
name: {name}
fields:
  name: [name]
  address: [address]
  district: [district]
  state: [state]
  pincode: [pincode]
pincode:
  pattern: '([1-9]\\d{{5}})'
"""


# --- Loading -----------------------------------------------------------------------


def test_shared_fragments_are_not_listed_or_loadable() -> None:
    assert available_ontologies() == ["corporate", "education", "generic", "healthcare"]
    with pytest.raises(OntologyError, match="shared fragment"):
        OntologyEngine.load("_india")


def test_a_typo_in_a_key_fails_the_load(tmp_path: Path) -> None:
    _write(tmp_path, "t", MINIMAL.format(name="t") + "abbreviaitons: {}\n")

    with pytest.raises(OntologyError, match="abbreviaitons"):
        OntologyEngine.load("t", tmp_path)


def test_an_expansion_cycle_fails_the_load(tmp_path: Path) -> None:
    body = MINIMAL.format(name="t") + "text:\n  abbreviations:\n    aa: bb x\n    bb: aa y\n"
    _write(tmp_path, "t", body)

    with pytest.raises(OntologyError, match="cycle"):
        OntologyEngine.load("t", tmp_path)


def test_an_extends_cycle_fails_the_load(tmp_path: Path) -> None:
    _write(tmp_path, "a", MINIMAL.format(name="a") + "extends: b\n")
    _write(tmp_path, "b", MINIMAL.format(name="b") + "extends: a\n")

    with pytest.raises(OntologyError, match="cycle"):
        OntologyEngine.load("a", tmp_path)


def test_yes_no_on_off_are_words_not_booleans(tmp_path: Path) -> None:
    """YAML 1.1 would load `no: number` as `False: number`."""
    body = MINIMAL.format(name="t") + "text:\n  abbreviations:\n    no: number\n    on: upon\n"
    _write(tmp_path, "t", body)

    assert OntologyEngine.load("t", tmp_path).normalize_name("no 5 on") == "number 5 upon"


def test_a_registry_must_key_on_a_declared_natural_key(tmp_path: Path) -> None:
    body = MINIMAL.format(name="t") + (
        "enrichment:\n  registries:\n    - name: r\n      key: missing\n"
        "      url: 'https://x/{key}'\n      fields: {email: a}\n"
    )
    _write(tmp_path, "t", body)

    with pytest.raises(OntologyError, match="not a declared natural key"):
        OntologyEngine.load("t", tmp_path)


def test_the_version_covers_the_extended_base(tmp_path: Path) -> None:
    _write(tmp_path, "_base", "state_aliases: {hp: himachal pradesh}\n")
    _write(tmp_path, "t", MINIMAL.format(name="t") + "extends: _base\n")
    before = OntologyEngine.load("t", tmp_path).version

    _write(tmp_path, "_base", "state_aliases: {hp: himachal}\n")

    assert OntologyEngine.load("t", tmp_path).version != before


# --- Context-aware expansion ----------------------------------------------------------


def test_the_same_abbreviation_means_different_things_per_domain(
    edu: OntologyEngine, health: OntologyEngine
) -> None:
    assert edu.normalize_name("Eng Med School") == "english medium school"
    assert health.normalize_name("Govt Med Coll") == "government medical college"


def test_network_names_only_expand_in_their_domain(
    edu: OntologyEngine, health: OntologyEngine
) -> None:
    assert edu.normalize_name("K.V. Jalahalli").startswith("kendriya")
    assert health.normalize_name("K.V. Jalahalli") == "kv jalahalli"


def test_st_is_saint_in_a_name_and_street_in_an_address(edu: OntologyEngine) -> None:
    assert edu.normalize_name("St. Joseph's") == "saint josephs"
    assert edu.normalize_address("MG St.") == "mg street"


def test_expansion_chains_to_a_fixed_point(edu: OntologyEngine) -> None:
    assert edu.normalize_name("KV") == "kendriya school"


def test_indic_words_are_not_split_or_joined(edu: OntologyEngine) -> None:
    """`\\w` excludes Indic vowel signs; whitespace is the only safe boundary."""
    assert edu.normalize_name("राजकीय विद्यालय, शिमला") == "राजकीय विद्यालय शिमला"


# --- Standardisation ------------------------------------------------------------------


def test_a_combined_location_column_is_split(edu: OntologyEngine) -> None:
    silver = edu.standardize(
        {"School Name": "GPS Tohana", "localityDistrictState": "TOHANA MANDI, FATEHABAD, HR"}
    )

    assert (silver.locality, silver.district, silver.state) == (
        "tohana mandi",
        "fatehabad",
        "haryana",
    )


def test_renamed_districts_and_dotted_states_canonicalise(edu: OntologyEngine) -> None:
    silver = edu.standardize({"name": "x", "district": "Gurgaon", "state": "H.P."})

    assert (silver.district, silver.state) == ("gurugram", "himachal pradesh")


def test_a_placeholder_pin_is_not_a_pin(edu: OntologyEngine) -> None:
    assert edu.standardize({"name": "x", "pincode": "000000"}).pincode is None


def test_the_pin_falls_back_to_the_end_of_the_address(edu: OntologyEngine) -> None:
    silver = edu.standardize({"name": "x", "address": "Plot 110 Sector 5, Rohini 110085"})

    assert silver.pincode == "110085"


def test_attributes_are_extracted_with_conflict_policy(edu: OntologyEngine) -> None:
    girls = edu.standardize({"name": "Govt Girls Sr Sec School No. 3"})
    both = edu.standardize({"name": "Boys and Girls High School"})

    assert girls.attributes == {"gender": "female", "kind": "school", "branch_number": "3"}
    assert both.attributes["gender"] == "mixed"


def test_distinctive_name_drops_generic_and_place_words(edu: OntologyEngine) -> None:
    silver = edu.standardize(
        {"name": "Govt Sr Sec School Ramnagar Kullu", "district": "Kullu", "state": "HP"}
    )

    assert silver.name_distinctive == "ramnagar"


# --- Natural keys ----------------------------------------------------------------------


def test_a_key_column_is_validated(edu: OntologyEngine) -> None:
    good = edu.standardize({"name": "x", "UDISE Code": "0202 0300 101"})
    bad_state = edu.standardize({"name": "x", "udise": "99020300101"})

    assert good.natural_keys == {"udise_code": "02020300101"}
    assert bad_state.natural_keys == {}


def test_a_labelled_key_is_found_in_free_text(edu: OntologyEngine) -> None:
    silver = edu.standardize({"name": "x", "remarks": "CBSE Affiliation No. 2130176"})

    assert silver.natural_keys == {"cbse_affiliation_no": "2130176"}


def test_an_unlabelled_number_is_not_mistaken_for_a_key(edu: OntologyEngine) -> None:
    """A bare 6-7 digit number is as likely a PIN or a phone fragment."""
    silver = edu.standardize({"name": "x", "remarks": "Call 2130176 or visit 110085"})

    assert silver.natural_keys == {}


def test_a_display_exception_must_name_a_real_abbreviation() -> None:
    from app.ontology.engine import OntologySpec

    raw = OntologyEngine.load("education").spec.model_dump(by_alias=True)
    raw["display"] = {"keep_as_written": ["vidyalaya", "not an abbreviation"]}

    with pytest.raises(ValueError, match="not an abbreviation"):
        OntologySpec.model_validate(raw)


def test_display_rules_change_the_display_name_but_not_the_matching_key() -> None:
    school = OntologyEngine.load("education")
    spec = school.spec.model_copy(
        update={"display": school.spec.display.model_copy(update={"keep_as_written": []})}
    )
    collapsing = OntologyEngine(spec)

    assert school.display_name("Vidya Mandir") == "vidya mandir"
    assert collapsing.display_name("Vidya Mandir") == "school"
    assert (
        school.normalize_name("Vidya Mandir")
        == collapsing.normalize_name("Vidya Mandir")
        == "school"
    )
    assert school.version == collapsing.version, "display rules do not make Silver stale"
