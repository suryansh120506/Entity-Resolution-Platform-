"""The client-ready export: the final database as a business user reads it.

One row per real-world entity, and nothing a client has to ask about:

* **No pipeline metadata.** No entity ids, scores, cluster statuses or
  provenance columns; those stay in the raw dump (`/export/gold`).
* **Readable values.** The entity's fully expanded name ("Arya Senior
  Secondary School"), title-cased address and place names, a real PIN or
  nothing. Acronyms survive title-casing: vowel-less words ("RPS", "SGRR",
  "NH") and the domain's own list (`display.keep_upper`: CBSE, DAV, ...).
* **Identifiers in their own columns,** labelled by the domain
  (`display.key_labels`: "UDISE Code", "Affiliation Number"). They come from
  the Silver layer's extracted keys, not from `stable_id`: a stable id holds
  only the strongest key, so a school with both a UDISE code and an
  affiliation number would lose one.
* **Web-found details** under plain headers ("Principal Name").
* **Every other column the client uploaded,** passed through unchanged in the
  file's own order. For a merged entity these are the master record's
  values. A column that would collide with an output header is suffixed
  " (original)" rather than dropped or merged.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.enrichment.pincode_infer import PincodeDirectory
from app.ml_pipeline.column_mapping import RECORD_ID_HEADER, RENAMED_PREFIX, alias_tables
from app.ml_pipeline.fields import normalize_key
from app.ontology.engine import REQUIRED_FIELDS, OntologySpec

#: Kept in capitals in values whatever the domain.
_VALUE_ACRONYMS = frozenset(
    {"cbse", "icse", "isc", "dav", "udise", "gst", "gstin", "ngo", "llp", "iti", "aiims", "nabh"}
)
#: Kept in capitals in column headers built from keys ("ceo_name" -> "CEO Name").
_LABEL_ACRONYMS = _VALUE_ACRONYMS | {
    "id",
    "url",
    "ceo",
    "cfo",
    "cto",
    "pan",
    "cin",
    "pin",
    "phc",
    "chc",
    "icu",
    "sms",
    "it",
    "hr",
}
#: Vowel-less words that are ordinary abbreviations, not acronyms: "Mr", "Pvt".
_TITLE_NOT_UPPER = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "st",
        "sr",
        "jr",
        "smt",
        "rd",
        "pvt",
        "ltd",
        "nr",
        "distt",
        "dstt",
        "sq",
        "blvd",
        "hq",
        "mkt",
        "bldg",
        "flr",
        "sctr",
        "th",
        "nd",
        "w",
        "s",
        "n",
        "e",
    }
)
_SMALL_WORDS = frozenset(
    {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to"}
)
_ROMAN = re.compile(r"^(?=[ivx]+$)x{0,3}(ix|iv|v?i{0,3})$")
_LETTERS = re.compile(r"[^\W\d_]+")
_VOWELS = frozenset("aeiouy")
_INDIAN_PIN = re.compile(r"[1-9]\d{5}")

STANDARD_HEADERS = ("Address", "District", "State", "PIN Code")


def title_case(text: str, keep_upper: Collection[str] = ()) -> str:
    """Title case that does not mangle acronyms, ordinals or possessives."""
    upper = _VALUE_ACRONYMS | {w.lower() for w in keep_upper}
    words = " ".join(str(text).split()).split(" ")
    return " ".join(_title_word(w, upper, first=(i == 0)) for i, w in enumerate(words) if w)


def _title_word(word: str, upper: Collection[str], *, first: bool) -> str:
    bare = "".join(ch for ch in word if ch.isalnum()).lower()
    if bare in upper or _ROMAN.match(bare):
        return word.upper()
    if not first and bare in _SMALL_WORDS:
        return word.lower()

    # Per run of letters, so "NH-48", "B-12" and "V.P.O." are judged by
    # their letters, not by the digits and dots around them.
    def capitalise(match: re.Match[str]) -> str:
        run, start = match.group(), match.start()
        previous = word[start - 1] if start else ""
        low = run.lower()
        if previous.isdigit():
            return low  # 15th, 2nd
        if previous in ("'", "’") and len(run) <= 2:
            return low  # Mary's, Don't
        if low in upper or _is_acronym(low):
            return run.upper()
        return run[:1].upper() + run[1:].lower()

    return _LETTERS.sub(capitalise, word)


def _is_acronym(letters: str) -> bool:
    """Short, vowel-less and not a known abbreviation: RPS, SGRR, NH, KVS."""
    return (
        letters.isascii()
        and len(letters) <= 5
        and not _VOWELS & set(letters)
        and letters not in _TITLE_NOT_UPPER
    )


def labelize(key: str) -> str:
    """A column header from a snake_case key: "ceo_name" -> "CEO Name"."""
    words = [w for w in key.split("_") if w]
    return " ".join(w.upper() if w in _LABEL_ACRONYMS else w.capitalize() for w in words)


def client_pincode(value: str) -> str:
    """A real six-digit Indian PIN, or nothing ("000000" is a placeholder)."""
    digits = str(value or "").strip()
    return digits if _INDIAN_PIN.fullmatch(digits) else ""


def resolve_pincode(
    own: str,
    *,
    address: str = "",
    district: str = "",
    state: str = "",
    web: str = "",
    siblings: Iterable[str] = (),
    directory: PincodeDirectory | None = None,
) -> str:
    """The PIN for an entity's row, from the best source that has one.

    1. **Its own**, as uploaded (or corrected by a steward). Never overridden:
       the export fills blanks, it does not second-guess what the client sent.
    2. **The post office directory**: the address names a locality that belongs
       to exactly one PIN in this district (pincode_infer.py).
    3. **The web**: a PIN found beside the school's name and listed by the
       directory for its district (patterns.find_pincode).
    4. **A duplicate's**: another record of the same merged school had one.

    Blank only when none of these can say. A guess is never made.
    """
    pin = client_pincode(own)
    if pin:
        return pin
    if directory is not None:
        pin = directory.infer(address=address, district=district, state=state) or ""
        if pin:
            return pin
    pin = client_pincode(web)
    if pin:
        return pin
    return next((p for p in map(client_pincode, siblings) if p), "")


@dataclass(frozen=True, slots=True)
class ClientLayout:
    name_header: str
    #: (natural key, header)
    key_columns: tuple[tuple[str, str], ...]
    #: (enrichment target, header)
    target_columns: tuple[tuple[str, str], ...]
    #: (raw_row key, header) — the client's own columns, in their file's order.
    passthrough: tuple[tuple[str, str], ...]
    keep_upper: frozenset[str]
    #: Natural keys whose uploaded values are not exported (`trust_file: false`);
    #: the column shows the web-verified value of the same key instead.
    untrusted_keys: frozenset[str] = frozenset()

    def header(self) -> list[str]:
        return [
            self.name_header,
            *STANDARD_HEADERS,
            *(h for _, h in self.key_columns),
            *(h for _, h in self.target_columns),
            *(h for _, h in self.passthrough),
        ]

    def row(
        self,
        *,
        name: str,
        fields: Mapping[str, str],
        natural_keys: Mapping[str, Any],
        enriched: Mapping[str, Any],
        raw: Mapping[str, Any],
    ) -> list[str]:
        """`name` is the standardised name; `fields` the cleaned identity
        fields; `enriched` the golden fields; `raw` the master's raw_row."""
        return [
            title_case(name, self.keep_upper),
            title_case(fields.get("address", ""), self.keep_upper),
            title_case(fields.get("district", ""), self.keep_upper),
            title_case(fields.get("state", ""), self.keep_upper),
            client_pincode(fields.get("pincode", "")),
            *(self._identifier(key, natural_keys, enriched) for key, _ in self.key_columns),
            *(str(enriched.get(key) or "") for key, _ in self.target_columns),
            *("" if raw.get(key) is None else str(raw.get(key)) for key, _ in self.passthrough),
        ]

    def _identifier(
        self, key: str, natural_keys: Mapping[str, Any], enriched: Mapping[str, Any]
    ) -> str:
        """One identifier column. A trusted key shows the uploaded value, else
        what the web verified; an untrusted key shows only the verified one."""
        uploaded = "" if key in self.untrusted_keys else natural_keys.get(key)
        return str(uploaded or enriched.get(key) or "")


def consumed_keys(
    spec: OntologySpec, mapping: Mapping[str, Any] | None, raw_keys: Iterable[str]
) -> set[str]:
    """The raw_row keys whose values are already in the standard columns.

    With a column mapping, the mapped fields were stored under canonical
    names; anything left unmapped was read by alias. Either way the rule for
    an alias-read field is the engine's: the first alias present wins, so
    only that column is consumed and a second, unused alias column is
    passed through rather than lost.
    """
    keys = list(raw_keys)
    by_normal: dict[str, str] = {}
    for key in keys:
        by_normal.setdefault(normalize_key(key), key)

    field_aliases, key_headers = alias_tables(
        spec.fields, {k: rule.headers for k, rule in spec.natural_keys.items()}
    )
    ontology_order = {
        f: [normalize_key(a) for a in spec.fields.get(f, [])] for f in REQUIRED_FIELDS
    }
    mapped_fields = set((mapping or {}).get("fields", {}))
    mapped_keys = set((mapping or {}).get("natural_keys", {}))

    consumed: set[str] = set()
    for field in REQUIRED_FIELDS:
        if field in mapped_fields:
            consumed.add(field)
            continue
        order = ontology_order[field] + [
            a for a in field_aliases[field] if a not in ontology_order[field]
        ]
        hit = next((by_normal[a] for a in order if a in by_normal), None)
        if hit:
            consumed.add(hit)
    for key, headers in key_headers.items():
        if key in mapped_keys and headers:
            consumed.add(headers[0])
            continue
        if not spec.natural_keys[key].trust_file:
            # Not exported as this identifier, so not "used up" either: the
            # column passes through under its own name, untouched.
            continue
        hit = next((by_normal[h] for h in headers if h in by_normal), None)
        if hit:
            consumed.add(hit)
    return consumed


def build_layout(
    spec: OntologySpec,
    *,
    mapping: Mapping[str, Any] | None,
    raw_keys: Sequence[str],
    file_headers: Sequence[str] | None,
) -> ClientLayout:
    display = spec.display
    keep_upper = frozenset(w.lower() for w in display.keep_upper)
    name_header = f"{display.entity_label or 'Entity'} Name"
    key_columns = tuple(
        (key, display.key_labels.get(key) or labelize(key)) for key in spec.natural_keys
    )
    # A target that shares its key with a natural key feeds that key's column
    # ("Affiliation Number"), not a second column of its own.
    target_columns = tuple(
        (t.key, labelize(t.key)) for t in spec.enrichment_targets if t.key not in spec.natural_keys
    )
    mapped_keys = set((mapping or {}).get("natural_keys", {}))
    untrusted_keys = frozenset(
        key
        for key, rule in spec.natural_keys.items()
        if not rule.trust_file and key not in mapped_keys
    )

    record_id_header = (mapping or {}).get("record_id")
    consumed = consumed_keys(spec, mapping, raw_keys)

    def original_header(key: str) -> str:
        if key.startswith(RENAMED_PREFIX):
            return key[len(RENAMED_PREFIX) :]
        if key == RECORD_ID_HEADER and record_id_header:
            return str(record_id_header)
        return key

    passthrough_keys = [k for k in raw_keys if k not in consumed]
    position = {h: i for i, h in enumerate(file_headers or [])}
    passthrough_keys.sort(
        key=lambda k: (position.get(original_header(k), len(position)), original_header(k))
    )

    taken = {normalize_key(h) for h in (name_header, *STANDARD_HEADERS)}
    taken |= {normalize_key(h) for _, h in (*key_columns, *target_columns)}
    passthrough: list[tuple[str, str]] = []
    for key in passthrough_keys:
        header = original_header(key)
        if normalize_key(header) in taken:
            header = f"{header} (original)"
        taken.add(normalize_key(header))
        passthrough.append((key, header))

    return ClientLayout(
        name_header=name_header,
        key_columns=key_columns,
        target_columns=target_columns,
        passthrough=tuple(passthrough),
        keep_upper=keep_upper,
        untrusted_keys=untrusted_keys,
    )


def merge_natural_keys(ordered: Iterable[Mapping[str, Any] | None]) -> dict[str, Any]:
    """A merged entity's identifiers: the master's first, then any key only
    another member carried (one record had the UDISE code, another the
    affiliation number)."""
    merged: dict[str, Any] = {}
    for keys in ordered:
        for key, value in (keys or {}).items():
            if value and key not in merged:
                merged[key] = value
    return merged
