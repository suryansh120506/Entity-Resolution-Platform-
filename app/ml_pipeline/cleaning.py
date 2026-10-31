"""Display-value cleaning: strip the punctuation and null-sentinel noise a raw
CSV export carries, without touching the row it came from.

Not a replacement for `encoder.normalize_text` — that function exists to make
two *different* spellings compare equal for embedding, and lowercases and
expands abbreviations to do it. This one exists to make a *single* value
presentable: `raw_row` stays byte-for-byte what the tenant uploaded (TRD 2.3),
and this is what the review screen and, later, the Google Places query
construction read instead of the raw string directly.

Confirmed live against a real export: leading commas from a blank first
segment (", GILLY MUNDI MEMORIAL COMMUNITY SCHOOL, LEHRIAN"), doubled commas
from a blank middle segment ("MINI. SECTT,HISAR, HISAR, HARYANA"), and a
trailing dash left by a missing pincode suffix ("MUMBAI, MUMBAI, MAHARASHTRA
-") are not edge cases in this data — they are routine.
"""

from __future__ import annotations

import re

#: Whole-value sentinels a source system uses in place of "nothing here".
#: Matched only against the *entire* trimmed value — "N/A School" is a name,
#: not a blank.
_NULL_SENTINELS = frozenset(
    {"", "n/a", "na", "n.a.", "none", "null", "nil", "nan", "-", "--", "unknown"}
)

#: Two or more commas, optionally separated by whitespace — a blank segment
#: between two real ones ("A,, B" or "A, , B").
_REPEATED_COMMA = re.compile(r",\s*(?:,\s*)+")

#: A comma with no space after it ("SECTT,HISAR") — not a redundant value,
#: just inconsistent spacing, but common enough in this data to fix here
#: rather than leave for every downstream reader to notice separately.
_COMMA_NO_SPACE = re.compile(r",(?=\S)")

#: Comma/dash/whitespace clinging to either end of the value — a blank
#: leading or trailing segment.
_LEADING_JUNK = re.compile(r"^[,\-\s]+")
_TRAILING_JUNK = re.compile(r"[,\-\s]+$")

_WHITESPACE = re.compile(r"\s+")


def clean_value(value: str | None) -> str:
    """One field value, presentable. Empty string for anything unusable.

    Order matters: null-sentinel check first (on the untouched input — a
    value that is *only* punctuation, like "-", must not survive the
    stripping passes below and come out as a non-empty string of nothing).
    """
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in _NULL_SENTINELS:
        return ""

    text = _REPEATED_COMMA.sub(", ", text)
    text = _COMMA_NO_SPACE.sub(", ", text)
    text = _LEADING_JUNK.sub("", text)
    text = _TRAILING_JUNK.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def clean_fields(fields: dict[str, str]) -> dict[str, str]:
    """`clean_value` over every value in a canonical-fields dict."""
    return {key: clean_value(value) for key, value in fields.items()}
