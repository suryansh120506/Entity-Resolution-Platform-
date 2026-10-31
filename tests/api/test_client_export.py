"""The client-ready export's formatting: headers, values, and passthrough."""

from __future__ import annotations

import pytest

from app.exports.client import (
    build_layout,
    client_pincode,
    labelize,
    merge_natural_keys,
    title_case,
)
from app.ml_pipeline.column_mapping import ColumnMapping, alias_tables, apply_mapping
from app.ontology import OntologyEngine
from app.ontology.generator import build_spec, sanitize_draft

METADATA = {
    "entity_id",
    "stable_id",
    "record_id",
    "source_id",
    "kind",
    "decided_by",
    "confidence_score",
    "cluster_status",
    "member_count",
    "enrichment_status",
    "enrichment_sources",
}

#: The real SNOP export's header row, in its own order.
SNOP = [
    "id",
    "instituteName",
    "affiliationNumber",
    "udiseCode",
    "addressLine1",
    "addressLine2",
    "localityDistrictState",
    "pinCode",
    "phone1",
    "website",
    "medium",
    "district",
    "state",
]


def _education():
    return OntologyEngine.load("education").spec


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ARYA SENIOR SECONDARY SCHOOL", "Arya Senior Secondary School"),
        ("SGRR INTER COLLEGE", "SGRR Inter College"),
        ("DAV PUBLIC SCHOOL", "DAV Public School"),
        ("V.P.O. KHERA, NH-48", "V.P.O. Khera, NH-48"),
        ("ST. MARY'S CONVENT", "St. Mary's Convent"),
        ("15TH CROSS, B-12", "15th Cross, B-12"),
        ("KENDRIYA VIDYALAYA NO. II", "Kendriya Vidyalaya No. II"),
        ("DADRA AND NAGAR HAVELI AND DAMAN AND DIU", "Dadra and Nagar Haveli and Daman and Diu"),
        ("MR SHARMA PVT LTD", "Mr Sharma Pvt Ltd"),
        ("  extra   spaces  ", "Extra Spaces"),
    ],
)
def test_values_are_title_cased_without_mangling_acronyms(raw: str, expected: str) -> None:
    assert title_case(raw) == expected


def test_a_domains_own_acronyms_stay_in_capitals() -> None:
    # Vowel-less words are acronyms anyway; the list is for ones with vowels.
    assert title_case("KV ABACUS CAMPUS") == "KV Abacus Campus"
    assert title_case("KV ABACUS CAMPUS", ["abacus"]) == "KV ABACUS Campus"


@pytest.mark.parametrize(
    ("key", "label"),
    [
        ("principal_name", "Principal Name"),
        ("ceo_name", "CEO Name"),
        ("website", "Website"),
        ("established_year", "Established Year"),
    ],
)
def test_enrichment_headers_lose_their_prefix_and_read_as_words(key: str, label: str) -> None:
    assert labelize(key) == label


def test_placeholder_pins_are_blank() -> None:
    assert [client_pincode(p) for p in ("110065", "000000", "NA", " 560001 ")] == [
        "110065",
        "",
        "",
        "560001",
    ]


def test_the_header_has_no_pipeline_metadata_and_keeps_every_other_column_in_order() -> None:
    layout = build_layout(_education(), mapping=None, raw_keys=sorted(SNOP), file_headers=SNOP)
    header = layout.header()

    assert header == [
        "School Name",
        "Address",
        "District",
        "State",
        "PIN Code",
        "UDISE Code",
        "Affiliation Number",
        "Principal Name",
        "Website",
        "Phone",
        "Email",
        "Board",
        "Established Year",
        # The client's own columns that no standard column holds, in file order.
        # The file's affiliation number is not trusted (trust_file: false), so it
        # passes through under its own name instead of feeding "Affiliation Number":
        "id",
        "affiliationNumber (original)",
        "addressLine2",
        "localityDistrictState",
        "phone1",
        "website (original)",  # same name as the web-found Website: kept, renamed
        "medium",
    ]
    assert not METADATA & set(header)


def test_a_second_alias_column_is_passed_through_not_lost() -> None:
    """`name` feeds School Name; `school` (also an alias) is still the client's data."""
    layout = build_layout(
        _education(), mapping=None, raw_keys=["name", "school"], file_headers=None
    )

    assert [h for _, h in layout.passthrough] == ["school"]


def test_a_client_column_that_reads_like_an_output_header_is_renamed_not_dropped() -> None:
    layout = build_layout(
        _education(), mapping=None, raw_keys=["name", "schoolname"], file_headers=None
    )

    assert [h for _, h in layout.passthrough] == ["schoolname (original)"]


def test_a_mapped_upload_passes_its_columns_through_under_their_original_names() -> None:
    spec = _education()
    mapping = ColumnMapping(fields={"name": "schl_nm"}, record_id="uid")
    fields, keys = alias_tables(spec.fields, {k: r.headers for k, r in spec.natural_keys.items()})
    raw = apply_mapping(
        {"schl_nm": "DPS", "uid": "A1", "name": "Mrs Rao", "fee": "9000"},
        mapping,
        field_aliases=fields,
        natural_key_headers=keys,
    )

    layout = build_layout(
        spec,
        mapping=mapping.model_dump(exclude_none=True),
        raw_keys=list(raw),
        file_headers=["schl_nm", "uid", "name", "fee"],
    )

    assert [h for _, h in layout.passthrough] == ["uid", "name", "fee"]
    row = dict(
        zip(
            layout.header(),
            layout.row(
                name="delhi public school", fields={}, natural_keys={}, enriched={}, raw=raw
            ),
            strict=True,
        )
    )
    assert row["School Name"] == "Delhi Public School"
    assert (row["uid"], row["name"], row["fee"]) == ("A1", "Mrs Rao", "9000")


def test_a_row_reads_like_a_report() -> None:
    layout = build_layout(_education(), mapping=None, raw_keys=sorted(SNOP), file_headers=SNOP)
    raw = {k: "" for k in SNOP} | {"id": "cms1", "medium": "N/A", "website": "old.example"}

    row = dict(
        zip(
            layout.header(),
            layout.row(
                name="arya senior secondary school",
                fields={
                    "address": "LOHARU ROAD, BHIWANI",
                    "district": "BHIWANI",
                    "state": "HARYANA",
                    "pincode": "000000",
                },
                natural_keys={"cbse_affiliation_no": "531234", "udise_code": "06010101001"},
                enriched={
                    "principal_name": "Dr. Asha Rao",
                    "website": "arya.example",
                    "cbse_affiliation_no": "530001",
                },
                raw=raw,
            ),
            strict=True,
        )
    )

    assert row["School Name"] == "Arya Senior Secondary School"
    assert (row["Address"], row["District"], row["State"]) == (
        "Loharu Road, Bhiwani",
        "Bhiwani",
        "Haryana",
    )
    assert row["PIN Code"] == ""
    # The uploaded affiliation number is not exported; the web-verified one is.
    # UDISE, which the file is trusted for, still comes from the file.
    assert (row["Affiliation Number"], row["UDISE Code"]) == ("530001", "06010101001")
    assert (row["Principal Name"], row["Website"]) == ("Dr. Asha Rao", "arya.example")
    assert row["website (original)"] == "old.example"
    assert row["medium"] == "N/A", "the client's own values are passed through untouched"
    assert row["id"] == "cms1"


def test_a_merged_entity_takes_identifiers_from_any_member_master_first() -> None:
    master = {"cbse_affiliation_no": "531234"}
    others = [{"udise_code": "06010101001", "cbse_affiliation_no": "999999"}, None]

    assert merge_natural_keys([master, *others]) == {
        "cbse_affiliation_no": "531234",
        "udise_code": "06010101001",
    }


def test_a_custom_domain_names_its_own_columns() -> None:
    draft, _ = sanitize_draft(
        {
            "label": "Car dealerships",
            "entity_noun": "car dealership",
            "natural_keys": [
                {
                    "key": "gstin",
                    "label": "GSTIN",
                    "pattern": r"\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]",
                }
            ],
            "enrichment_targets": [{"key": "brands_sold", "description": "x"}],
        },
        description="Car dealerships",
    )
    spec = OntologyEngine(
        OntologyEngine.load("generic").spec.model_validate(build_spec(draft, "custom_cars_abc123"))
    ).spec

    layout = build_layout(
        spec, mapping=None, raw_keys=["name", "gstin", "showroom_size"], file_headers=None
    )

    assert layout.header()[:7] == [
        "Car Dealership Name",
        "Address",
        "District",
        "State",
        "PIN Code",
        "GSTIN",
        "Brands Sold",
    ]
    assert [h for _, h in layout.passthrough] == ["showroom_size"]


def test_an_unverified_affiliation_number_is_blank_not_the_files_guess() -> None:
    layout = build_layout(_education(), mapping=None, raw_keys=sorted(SNOP), file_headers=SNOP)
    row = dict(
        zip(
            layout.header(),
            layout.row(
                name="x school",
                fields={},
                natural_keys={"cbse_affiliation_no": "200427"},
                enriched={},
                raw={},
            ),
            strict=True,
        )
    )

    assert row["Affiliation Number"] == ""


def test_a_missing_pin_is_filled_only_from_a_source_that_can_vouch_for_it() -> None:
    from app.enrichment.pincode_infer import PincodeDirectory
    from app.exports.client import resolve_pincode

    directory = PincodeDirectory([("303313", "Bandikui S.O", "Dausa", "Rajasthan")])
    where = {"district": "dausa", "state": "rajasthan"}

    assert resolve_pincode("110001", address="Bandikui", directory=directory, **where) == "110001"
    assert (
        resolve_pincode("000000", address="Ward 8, Bandikui", directory=directory, **where)
        == "303313"
    )
    assert (
        resolve_pincode("", address="Ward 8", web="303001", directory=directory, **where)
        == "303001"
    )
    assert resolve_pincode("", address="Ward 8", siblings=["", "303002"], **where) == "303002"
    assert resolve_pincode("", address="Ward 8", **where) == ""
