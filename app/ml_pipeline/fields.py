"""Canonical field extraction from a raw CSV row — no pandas, no xgboost.

Deliberately dependency-light: SAD 3.2 requires that FastAPI "never imports
pandas, torch, or xgboost in a request path", enforced by the API image not
installing the `[ml]` extra at all. `canonical_fields` is needed on both
sides of that line — the classifier uses it to build training features, and
the cluster-detail API route uses it to serve display values — so it lives
in its own module that only the classifier (heavy) sits on top of, rather
than in `classification/classifier.py` itself, which an API-side import
would have pulled `xgboost` in through.

Two alias tables for the same `raw_row` drift apart silently — a header one
recognises but the other does not produces an empty string with no error.
This is the one table; nothing else should keep its own copy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: What a school row must carry to be resolvable at all. Absence of any of
#: these is what separates Missing_Context from Junk: Missing_Context is a
#: real school described too thinly to match, Junk is not a school.
REQUIRED_FIELDS: tuple[str, ...] = ("name", "address", "district", "state", "pincode")

#: `raw_row` is keyed by whatever headers the tenant's CSV happened to use, so
#: the same tolerant normalisation the pincode seeder applies to government
#: exports applies here. Keys are normalised (lower-cased, non-alphanumerics
#: stripped) before lookup.
_FIELD_ALIASES: dict[str, str] = {
    "name": "name",
    "schoolname": "name",
    "institutionname": "name",
    "institutename": "name",
    "school": "name",
    "address": "address",
    "addressline1": "address",
    "streetaddress": "address",
    "location": "address",
    "district": "district",
    "districtname": "district",
    "state": "state",
    "statename": "state",
    "pincode": "pincode",
    "pin": "pincode",
    "postalcode": "pincode",
    "zip": "pincode",
    "zipcode": "pincode",
}

#: Some exports carry one combined "office, district, state" column instead
#: of separate `district`/`state` columns — SNOP's own export does exactly
#: this (`localityDistrictState`, e.g. "TOHANA MANDI, FATEHABAD, HARYANA").
#: Confirmed live: without this fallback, every row from that export reads
#: `district=""` and `state=""`, `missing_required_field_count` is >= 2 for
#: the entire file, and `lf_several_missing_fields` sends 100% of it to
#: Missing_Context — a real upload that "succeeds" having resolved nothing,
#: not a hypothetical edge case.
_COMBINED_LOCATION_KEYS = frozenset({"localitydistrictstate", "districtstate", "citydistrictstate"})


def aliases_for(field: str) -> list[str]:
    """Every normalised header this table reads as `field`."""
    return [alias for alias, canonical in _FIELD_ALIASES.items() if canonical == field]


def normalize_key(key: str) -> str:
    return "".join(ch for ch in key.strip().lower() if ch.isalnum())


def canonical_fields(row: Mapping[str, Any]) -> dict[str, str]:
    """Pull the required fields out of a raw CSV row, whatever it called them.

    First alias wins, so a row carrying both `name` and `school_name` does not
    depend on dict ordering for which one is read.
    """
    out = dict.fromkeys(REQUIRED_FIELDS, "")
    for key, value in row.items():
        canonical = _FIELD_ALIASES.get(normalize_key(str(key)))
        if canonical is None or out[canonical]:
            continue
        out[canonical] = "" if value is None else str(value).strip()

    if not out["district"] or not out["state"]:
        _fill_district_state_from_combined_column(out, row)

    return out


def _fill_district_state_from_combined_column(out: dict[str, str], row: Mapping[str, Any]) -> None:
    """Split a combined "office, district, state" column as a last resort.

    Only ever fills a field that is still empty after the alias pass, so an
    explicit `district`/`state` column always wins over this. State is the
    last comma-separated segment, district the one before it — checked
    against the real export this exists for, including its ragged rows
    ("MINI. SECTT,HISAR, HISAR, HARYANA" — extra leading segments, same last
    two).
    """
    for key, value in row.items():
        if normalize_key(str(key)) not in _COMBINED_LOCATION_KEYS:
            continue
        parts = [p.strip() for p in str(value or "").split(",") if p.strip()]
        if not out["state"] and len(parts) >= 1:
            out["state"] = parts[-1]
        if not out["district"] and len(parts) >= 2:
            out["district"] = parts[-2]
        return
