"""Plain-language data-quality issues for one record — no pandas, no xgboost.

The classifier's verdict (Valid / Junk / Missing_Context) says *what* was
decided about a row; this says *why*, in terms a data owner can act on —
"no district", "pincode is not a real Indian PIN". It is deliberately a
separate, rule-based description rather than an explanation of the model:
the same rules the labelling functions are built on, stated per field.

API-safe for the same reason as `fields.py`: the records drill-down route
calls it per row, and the API image does not install the `[ml]` extra.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

#: Stable issue codes. The frontend maps these to sentences; the codes are
#: what the ingest stage counts into `stage_progress`, so renaming one is a
#: change to both.
ISSUE_LABELS: dict[str, str] = {
    "missing_name": "No school name",
    "placeholder_name": "Name is a placeholder (e.g. N/A, TEST)",
    "abbreviated_name": "Name is only initials (e.g. D P S) — give the full name",
    "missing_address": "No address",
    "missing_district": "No district",
    "missing_state": "No state",
    "missing_pincode": "No pincode",
    "malformed_pincode": "Pincode is not six digits",
    "unknown_pincode": "Pincode is not a real Indian PIN",
}

_PINCODE_SHAPE = re.compile(r"[1-9]\d{5}")

#: Mirrors the classifier's placeholder-name pattern closely enough to explain
#: its verdicts; kept here rather than imported because the classifier module
#: pulls in xgboost.
_PLACEHOLDER_NAME = re.compile(
    r"""\W*(?:
        n[./\s]?a | nil | none | null | nan | tbd
      | test\w* | dummy | sample | unknown | unnamed | no[\s_-]?name
      | x+ | school | schools | \d+ | -+ | \.+ | _+
    )\W*""",
    re.IGNORECASE | re.VERBOSE,
)


def record_issues(fields: Mapping[str, str], *, pincode_known: bool | None) -> list[str]:
    """Issue codes for one record's canonical fields, most serious first.

    `pincode_known` is whether the PIN is present in `reference_pincodes`;
    pass None when that has not been checked, and the unknown-PIN issue is
    simply not reported rather than guessed.
    """
    issues: list[str] = []

    name = (fields.get("name") or "").strip()
    words = re.sub(r"[^\w\s]", " ", name).split()
    if not name:
        issues.append("missing_name")
    elif _PLACEHOLDER_NAME.fullmatch(name):
        issues.append("placeholder_name")
    elif any(ch.isalpha() for ch in name) and max(len(w) for w in words) <= 4:
        issues.append("abbreviated_name")

    for key in ("address", "district", "state"):
        if not (fields.get(key) or "").strip():
            issues.append(f"missing_{key}")

    pincode = (fields.get("pincode") or "").strip()
    if not pincode:
        issues.append("missing_pincode")
    elif not _PINCODE_SHAPE.fullmatch(pincode) or len(set(pincode)) == 1:
        issues.append("malformed_pincode")
    elif pincode_known is False:
        issues.append("unknown_pincode")

    return issues


def issue_counts(all_issues: list[list[str]]) -> dict[str, int]:
    """How many records carry each issue — the results page's breakdown."""
    counts = dict.fromkeys(ISSUE_LABELS, 0)
    for issues in all_issues:
        for issue in issues:
            counts[issue] = counts.get(issue, 0) + 1
    return {code: n for code, n in counts.items() if n}
