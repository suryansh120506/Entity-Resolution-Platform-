"""Ontology engine: YAML-defined, domain-scoped standardisation (the Silver layer).

An ontology (`ontologies/<name>.yaml`) declares everything that used to be
hard-coded across `fields.py`, `encoder.py` and `name_gate.py`: which CSV
headers mean what, which abbreviations expand to what *in this domain*, which
words are generic, which attributes to extract, and which natural keys to
pull out and how to validate them. The engine compiles that once and applies
it per row with no domain knowledge of its own.

Guarantees, enforced at load rather than discovered mid-job:

* **Schema-checked.** Unknown keys are errors (`extra="forbid"`), so a typo
  like `abbreviaitons:` fails the load instead of silently doing nothing.
* **Terminating.** Abbreviation tables are checked for expansion cycles
  ("a" -> "b x", "b" -> "a"), which would otherwise only be stopped by the
  pass cap, leaving half-expanded text that matches nothing.
* **Versioned.** `version` is the SHA-256 of the fully resolved spec
  (including anything pulled in via `extends`), stamped on every Silver
  row. Editing `_india.yaml` changes the version of every ontology that
  extends it — which is exactly when their Silver rows became stale.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.ontology.identity import Identity, derive_identity

REQUIRED_FIELDS: tuple[str, ...] = ("name", "address", "district", "state", "pincode")

_ONTOLOGY_NAME = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
_POST_RESOLUTION_BLOCKS = {
    "enrichment",
    "enrichment_targets",
    "golden_enrichment",
    "triage",
    "display",
}
_APOSTROPHES = re.compile(r"[’'`ʼ]")
_WHITESPACE = re.compile(r"\s+")


class OntologyError(ValueError):
    """The ontology file is missing, malformed, or internally inconsistent."""


# --- Schema ----------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class TextRules(_Strict):
    abbreviations: dict[str, str] = Field(default_factory=dict)
    name_abbreviations: dict[str, str] = Field(default_factory=dict)
    address_abbreviations: dict[str, str] = Field(default_factory=dict)
    collapse_spaced_acronyms: bool = True
    max_expansion_passes: int = Field(default=5, ge=1, le=20)


class PincodeRules(_Strict):
    pattern: str
    search_address: bool = False


class AttributeRule(_Strict):
    source: Literal["name", "address"] = Field(alias="from")
    values: dict[str, str] = Field(default_factory=dict)
    pattern: str | None = None
    on_conflict: Literal["first", "mixed", "list"] = "first"

    @model_validator(mode="after")
    def _one_extractor(self) -> AttributeRule:
        if bool(self.values) == bool(self.pattern):
            raise ValueError("an attribute needs exactly one of `values` or `pattern`")
        return self


class KeyValidation(_Strict):
    length: int | None = Field(default=None, ge=1)
    prefix_range: tuple[int, int] | None = None
    prefix_digits: int = Field(default=2, ge=1)


class NaturalKeyRule(_Strict):
    headers: list[str] = Field(default_factory=list)
    pattern: str
    context: str | None = None
    validate_: KeyValidation | None = Field(default=None, alias="validate")
    #: False = a value read from the uploaded file is not trusted as this
    #: identifier. It still matches and identifies records, but the client
    #: export shows only a value verified on the web (an enrichment target
    #: with the same key), and the uploaded column passes through under its own
    #: name instead of being relabelled. For registry numbers whose source
    #: column turns out to be a row id or a guess: a wrong "Affiliation
    #: Number" is worse than a blank one.
    trust_file: bool = True


class RegistryAdapter(_Strict):
    """A public registry lookup, entirely declared in YAML."""

    name: str
    key: str
    url: str
    method: Literal["GET", "POST"] = "GET"
    params: dict[str, str] = Field(default_factory=dict)
    fields: dict[str, str]
    not_found_selector: str | None = None
    enabled: bool = False


class SerpRules(_Strict):
    enabled: bool = True
    query: str
    fields: list[str]
    max_results: int = Field(default=3, ge=1, le=10)


class IdentityRules(_Strict):
    """How a record's stable identity is derived (app/ontology/identity.py)."""

    #: Natural keys to root the identity on, strongest first. Empty = use the
    #: natural_keys declaration order.
    keys: list[str] = Field(default_factory=list)
    #: Records sharing any natural-key value are the same entity: merged
    #: within an upload and linked to the gold layer whatever the score.
    force_merge_on_key: bool = True


class EnrichmentRules(_Strict):
    registries: list[RegistryAdapter] = Field(default_factory=list)
    serp: SerpRules | None = None
    ttl_days: int = Field(default=30, ge=1)
    negative_ttl_days: int = Field(default=7, ge=1)


#: Placeholders a search template may use: a golden record's canonical fields.
_TEMPLATE_FIELDS = frozenset(REQUIRED_FIELDS)
_TARGET_KEY = re.compile(r"^[a-z][a-z0-9_]{1,40}$")

TargetType = Literal["text", "person", "phone", "email", "url", "number"]


def infer_target_type(key: str) -> TargetType:
    """A target's type from its name, when the YAML does not say.

    The type picks the shape check a verified value must also pass, so
    `ceo_name` is held to a person's shape and `contact_email` to an email's.
    """
    if key == "email" or key.endswith("_email"):
        return "email"
    if key in {"phone", "mobile", "telephone"} or key.endswith(("_phone", "_mobile")):
        return "phone"
    if key in {"website", "url"} or key.endswith(("_url", "_website")):
        return "url"
    if key.endswith("_name"):
        return "person"
    return "text"


class EnrichmentTarget(_Strict):
    """One attribute Tier 3 looks for on the web for each golden record.

    Written in YAML as a bare key (`principal_name`) or as a mapping when the
    LLM needs telling what the key means (`{key: revenue, type: number,
    description: "annual revenue, with its currency"}`).
    """

    key: str
    type: TargetType = "text"
    description: str = ""
    #: A regex with at most one capture group. A target that has one is read
    #: straight from the search snippets, nearest to the entity's own name, and
    #: never asked of the LLM: closed-form values ("CBSE", "estd. 1992", an
    #: affiliation number) need no model, and every model call is rate-limited
    #: capacity. The match still passes every grounding check.
    pattern: str | None = None

    @field_validator("pattern")
    @classmethod
    def _valid_pattern(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            compiled = re.compile(value)
        except re.error as exc:
            raise ValueError(f"enrichment target pattern {value!r} is not a regex: {exc}") from exc
        if compiled.groups > 1:
            raise ValueError("an enrichment target pattern may have at most one capture group")
        return value

    @model_validator(mode="before")
    @classmethod
    def _from_bare_key(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = {"key": value}
        if isinstance(value, dict) and "type" not in value and isinstance(value.get("key"), str):
            value = {**value, "type": infer_target_type(value["key"])}
        return value

    @field_validator("key")
    @classmethod
    def _valid_key(cls, value: str) -> str:
        if not _TARGET_KEY.match(value):
            raise ValueError(f"enrichment target {value!r} must be snake_case")
        if value in _TEMPLATE_FIELDS:
            # Identity fields come from the source data and the stewards. The
            # open web may add to a golden record; it may not rename one.
            raise ValueError(f"enrichment target {value!r} would overwrite an identity field")
        return value


class SearchTemplate(_Strict):
    query: str
    max_results: int = Field(default=3, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def _known_placeholders(cls, value: str) -> str:
        used = {name for _, name, _, _ in string.Formatter().parse(value) if name}
        unknown = sorted(used - _TEMPLATE_FIELDS)
        if unknown:
            raise ValueError(f"query uses {unknown}; available: {sorted(_TEMPLATE_FIELDS)}")
        return value


class DisplayRules(_Strict):
    """How a name is shown on a golden record, as opposed to matched.

    `text.abbreviations` serves matching, so it also collapses synonyms
    ("vidyalaya" -> "school") that make two spellings of one school compare
    equal. That is right for a match key and wrong for a name a person reads:
    "Maharishi Vidya Mandir" is not "Maharishi School". Abbreviations listed
    here are still used to match, but are left as written in the display name.
    """

    keep_as_written: list[str] = Field(default_factory=list)
    #: What one record is, for client-facing headers: "School" -> "School Name".
    entity_label: str = Field(default="Entity", max_length=40)
    #: Client-facing column names for natural keys ("UDISE Code"). A key
    #: without one is labelled from its name.
    key_labels: dict[str, str] = Field(default_factory=dict)
    #: Words kept in capitals when values are title-cased for a report
    #: ("CBSE", "DAV"). Vowel-less words ("RPS", "NH") are kept anyway.
    keep_upper: list[str] = Field(default_factory=list)


class TriageRules(SearchTemplate):
    """Tier 2: how to look up two records the matcher was unsure about."""

    query: str = '"{name}" {address} {district} {state}'
    #: Members verified against the master per cluster. Members past the cap
    #: are separated, not merged unverified.
    max_members_checked: int = Field(default=4, ge=1, le=20)


class GoldenEnrichmentRules(SearchTemplate):
    """Tier 3: how to look up a resolved golden record."""

    query: str = '"{name}" {district} {state}'
    max_results: int = Field(default=5, ge=1, le=10)


def render_query(template: str, fields: Mapping[str, str | None]) -> str:
    """Fill a search template; missing fields render empty, not as 'None'."""
    values = {name: (fields.get(name) or "").strip() for name in _TEMPLATE_FIELDS}
    query = template.format(**values).replace('""', " ")
    return " ".join(query.split())


class OntologySpec(_Strict):
    name: str
    #: Shown in the upload picker ("Education"). Defaults to the name.
    label: str = ""
    description: str = ""
    fields: dict[str, list[str]]
    combined_location_fields: list[str] = Field(default_factory=list)
    state_aliases: dict[str, str] = Field(default_factory=dict)
    district_aliases: dict[str, str] = Field(default_factory=dict)
    pincode: PincodeRules
    text: TextRules = Field(default_factory=TextRules)
    generic_name_tokens: list[str] = Field(default_factory=list)
    attributes: dict[str, AttributeRule] = Field(default_factory=dict)
    natural_keys: dict[str, NaturalKeyRule] = Field(default_factory=dict)
    identity: IdentityRules = Field(default_factory=IdentityRules)
    enrichment: EnrichmentRules = Field(default_factory=EnrichmentRules)
    #: Tier 3: what to find on the web for every resolved golden record.
    #: Empty = golden records are not enriched in this domain.
    enrichment_targets: list[EnrichmentTarget] = Field(default_factory=list)
    golden_enrichment: GoldenEnrichmentRules = Field(default_factory=GoldenEnrichmentRules)
    triage: TriageRules = Field(default_factory=TriageRules)
    display: DisplayRules = Field(default_factory=DisplayRules)

    @field_validator("enrichment_targets")
    @classmethod
    def _unique_targets(cls, value: list[EnrichmentTarget]) -> list[EnrichmentTarget]:
        keys = [t.key for t in value]
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        if duplicates:
            raise ValueError(f"enrichment_targets lists {duplicates} more than once")
        return value

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _ONTOLOGY_NAME.match(value):
            raise ValueError(f"invalid ontology name {value!r}")
        return value

    @model_validator(mode="after")
    def _cross_checks(self) -> OntologySpec:
        missing = [f for f in REQUIRED_FIELDS if not self.fields.get(f)]
        if missing:
            raise ValueError(f"fields has no aliases for: {', '.join(missing)}")
        for adapter in self.enrichment.registries:
            if adapter.key not in self.natural_keys:
                raise ValueError(
                    f"registry {adapter.name!r} keys on {adapter.key!r}, "
                    "which is not a declared natural key"
                )
        unknown = [k for k in self.identity.keys if k not in self.natural_keys]
        if unknown:
            raise ValueError(f"identity.keys names undeclared natural keys: {unknown}")
        known = {
            _basic_normalize(k) for k in {**self.text.abbreviations, **self.text.name_abbreviations}
        }
        stray = [k for k in self.display.keep_as_written if _basic_normalize(k) not in known]
        if stray:
            raise ValueError(f"display.keep_as_written names no name abbreviation: {stray}")
        unlabelled = sorted(set(self.display.key_labels) - set(self.natural_keys))
        if unlabelled:
            raise ValueError(f"display.key_labels names undeclared natural keys: {unlabelled}")
        return self


# --- Output ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SilverFields:
    name: str | None
    name_distinctive: str | None
    address: str | None
    locality: str | None
    district: str | None
    state: str | None
    pincode: str | None
    attributes: dict[str, Any] = field(default_factory=dict)
    natural_keys: dict[str, str] = field(default_factory=dict)


# --- Loading ---------------------------------------------------------------------


def default_ontology_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "ontologies"


def available_ontologies(directory: Path | None = None) -> list[str]:
    root = directory or default_ontology_dir()
    return sorted(p.stem for p in root.glob("*.yaml") if not p.stem.startswith("_"))


def _deep_merge(base: dict[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    """Mappings merge recursively, child wins; lists and scalars replace."""
    merged = dict(base)
    for key, value in child.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


class _Yaml12BoolLoader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 booleans: only true/false.

    YAML 1.1 (PyYAML's default) also reads yes/no/on/off/y/n as booleans,
    so the abbreviation `no: number` would load as `False: number`. In an
    abbreviation table, `no`, `on` and `y` are ordinary words.
    """


_Yaml12BoolLoader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Yaml12BoolLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def _read_resolved(name: str, directory: Path, chain: tuple[str, ...] = ()) -> dict[str, Any]:
    if name in chain:
        raise OntologyError(f"`extends` cycle: {' -> '.join((*chain, name))}")
    if not _ONTOLOGY_NAME.match(name.lstrip("_")):
        raise OntologyError(f"invalid ontology reference {name!r}")
    path = directory / f"{name}.yaml"
    if not path.is_file():
        raise OntologyError(f"ontology file not found: {path}")
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_Yaml12BoolLoader) or {}  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError as exc:
        raise OntologyError(f"{path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise OntologyError(f"{path}: top level must be a mapping")

    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _deep_merge(_read_resolved(str(parent), directory, (*chain, name)), raw)


# --- Engine ----------------------------------------------------------------------


def normalize_key(key: str) -> str:
    return "".join(ch for ch in key.strip().lower() if ch.isalnum())


def _phrase_regex(phrases: list[str]) -> re.Pattern[str] | None:
    """Whole-token phrase alternation, longest first.

    Token boundaries are whitespace, not `\\b`/`\\w`: Python's `\\w` excludes
    Indic combining marks, so `\\b` falls *inside* a Devanagari word after
    every vowel sign. Input is always normalised first, so whitespace is
    exactly the token boundary.
    """
    if not phrases:
        return None
    ordered = sorted(set(phrases), key=lambda p: (-len(p), p))
    return re.compile(r"(?<!\S)(?:" + "|".join(re.escape(p) for p in ordered) + r")(?!\S)")


def _collapse_spaced_acronyms(text: str) -> str:
    """Join runs of 2+ single-letter tokens: "k v no 2" -> "kv no 2"."""
    out: list[str] = []
    run: list[str] = []
    for token in [*text.split(" "), ""]:
        if len(token) == 1 and token.isalpha():
            run.append(token)
            continue
        if len(run) >= 2:
            out.append("".join(run))
        else:
            out.extend(run)
        run = []
        if token:
            out.append(token)
    return " ".join(out)


def _compile(pattern: str, where: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise OntologyError(f"{where}: invalid regex {pattern!r}: {exc}") from exc


class _Expander:
    """Whole-word phrase expansion to a fixed point, cycle-checked at build."""

    def __init__(self, table: dict[str, str], max_passes: int, where: str) -> None:
        self.table = {_basic_normalize(k): _basic_normalize(v) for k, v in table.items()}
        self.max_passes = max_passes
        self.regex = _phrase_regex(list(self.table))
        self._check_cycles(where)

    def _check_cycles(self, where: str) -> None:
        if self.regex is None:
            return
        edges = {k: set(self.regex.findall(v)) - {k} for k, v in self.table.items()}
        for key, value in self.table.items():
            if key in self.regex.findall(value):
                raise OntologyError(f"{where}: {key!r} expands to text containing itself")
        visiting: set[str] = set()
        done: set[str] = set()

        def visit(node: str, path: list[str]) -> None:
            if node in done:
                return
            if node in visiting:
                raise OntologyError(f"{where}: expansion cycle {' -> '.join([*path, node])}")
            visiting.add(node)
            for nxt in edges.get(node, ()):
                visit(nxt, [*path, node])
            visiting.discard(node)
            done.add(node)

        for node in edges:
            visit(node, [])

    def __call__(self, text: str) -> str:
        if self.regex is None or not text:
            return text
        for _ in range(self.max_passes):
            expanded = self.regex.sub(lambda m: self.table[m.group(0)], text)
            if expanded == text:
                break
            text = expanded
        return _WHITESPACE.sub(" ", text).strip()


def _basic_normalize(text: str, *, collapse_acronyms: bool = False) -> str:
    """NFKC, casefold, punctuation/symbols to spaces, whitespace collapsed.

    Punctuation is removed by Unicode *category* (P*, S*), not by keeping
    `\\w`: Indic vowel signs and viramas are combining marks (Mn/Mc), and a
    `\\w`-based strip would delete them and corrupt every Devanagari name.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _APOSTROPHES.sub("", text)
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    text = _WHITESPACE.sub(" ", text).strip()
    if collapse_acronyms:
        # "k v" (from "K.V.") -> "kv", so the abbreviation table sees it.
        text = _collapse_spaced_acronyms(text)
    return text


class OntologyEngine:
    def __init__(self, spec: OntologySpec) -> None:
        self.spec = spec
        # The version marks Silver rows stale, so it covers only what shapes
        # Silver output. What happens after resolution (triage, enrichment)
        # can be edited without making every standardised row look outdated.
        shaping = spec.model_dump(mode="json", by_alias=True, exclude=_POST_RESOLUTION_BLOCKS)
        for rule in shaping.get("natural_keys", {}).values():
            # How a key is *exported* does not change a single Silver value.
            rule.pop("trust_file", None)
        canonical = json.dumps(shaping, sort_keys=True)
        self.version = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        rules = spec.text
        passes = rules.max_expansion_passes
        self._collapse = rules.collapse_spaced_acronyms
        self._name_expand = _Expander(
            {**rules.abbreviations, **rules.name_abbreviations}, passes, "text.name"
        )
        self._address_expand = _Expander(
            {**rules.abbreviations, **rules.address_abbreviations}, passes, "text.address"
        )
        kept = {_basic_normalize(k) for k in spec.display.keep_as_written}
        self._display_expand = _Expander(
            {
                k: v
                for k, v in {**rules.abbreviations, **rules.name_abbreviations}.items()
                if _basic_normalize(k) not in kept
            },
            passes,
            "display.name",
        )

        self._aliases = {
            canonical_field: [normalize_key(a) for a in aliases]
            for canonical_field, aliases in spec.fields.items()
        }
        self._combined = {normalize_key(k) for k in spec.combined_location_fields}
        self._states = {
            _basic_normalize(k): _basic_normalize(v) for k, v in spec.state_aliases.items()
        }
        self._districts = {
            _basic_normalize(k): _basic_normalize(v) for k, v in spec.district_aliases.items()
        }
        self._generic = {_basic_normalize(t) for t in spec.generic_name_tokens}
        self._pincode = _compile(spec.pincode.pattern, "pincode.pattern")

        self._attributes: list[tuple[str, AttributeRule, re.Pattern[str], dict[str, str]]] = []
        for attr, rule in spec.attributes.items():
            if rule.pattern:
                self._attributes.append(
                    (attr, rule, _compile(rule.pattern, f"attributes.{attr}.pattern"), {})
                )
                continue
            values = {_basic_normalize(k): v for k, v in rule.values.items()}
            phrases = _phrase_regex(list(values))
            if phrases is None:  # unreachable: the schema requires non-empty values
                raise OntologyError(f"attributes.{attr}: no values")
            self._attributes.append((attr, rule, phrases, values))

        self._keys: list[tuple[str, NaturalKeyRule, re.Pattern[str], re.Pattern[str] | None]] = []
        for key, key_rule in spec.natural_keys.items():
            self._keys.append(
                (
                    key,
                    key_rule,
                    _compile(key_rule.pattern, f"natural_keys.{key}.pattern"),
                    _compile(key_rule.context, f"natural_keys.{key}.context")
                    if key_rule.context
                    else None,
                )
            )

    @classmethod
    def load(cls, name: str, directory: Path | None = None) -> OntologyEngine:
        if name.startswith("_"):
            raise OntologyError(f"{name!r} is a shared fragment, not a loadable ontology")
        raw = _read_resolved(name, directory or default_ontology_dir())
        try:
            spec = OntologySpec.model_validate(raw)
        except ValueError as exc:
            raise OntologyError(f"ontology {name!r}: {exc}") from exc
        if spec.name != name:
            raise OntologyError(f"ontologies/{name}.yaml declares name {spec.name!r}")
        return cls(spec)

    # --- public API -------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def identity_keys(self) -> list[str]:
        """Natural keys an identity may be rooted on, strongest first."""
        return self.spec.identity.keys or list(self.spec.natural_keys)

    def identity(self, record_id: str, silver: SilverFields) -> Identity:
        return derive_identity(
            ontology=self.name,
            record_id=record_id,
            natural_keys=silver.natural_keys,
            key_priority=self.identity_keys,
            name=silver.name,
            name_distinctive=silver.name_distinctive,
            address=silver.address,
            pincode=silver.pincode,
            district=silver.district,
            state=silver.state,
        )

    def normalize_name(self, text: str) -> str:
        return self._name_expand(_basic_normalize(text, collapse_acronyms=self._collapse))

    def display_name(self, text: str) -> str:
        """The name with abbreviations expanded, for people to read.

        Like `normalize_name` but without the matching-only synonym
        collapses named in `display.keep_as_written`.
        """
        return self._display_expand(_basic_normalize(text, collapse_acronyms=self._collapse))

    def normalize_address(self, text: str) -> str:
        return self._address_expand(_basic_normalize(text, collapse_acronyms=self._collapse))

    def standardize(self, raw_row: Mapping[str, Any]) -> SilverFields:
        by_header: dict[str, str] = {}
        for header, value in raw_row.items():
            key = normalize_key(str(header))
            if key and key not in by_header:
                by_header[key] = "" if value is None else str(value).strip()

        raw = {f: self._first(by_header, f) for f in REQUIRED_FIELDS}
        locality = None
        if not raw["district"] or not raw["state"]:
            locality = self._fill_from_combined(by_header, raw)

        name = self.normalize_name(raw["name"]) or None
        address = self.normalize_address(raw["address"]) or None
        state = self._canonical_place(raw["state"], self._states)
        district = self._canonical_place(raw["district"], self._districts)

        return SilverFields(
            name=name,
            name_distinctive=self._distinctive(name, district, state),
            address=address,
            locality=locality,
            district=district,
            state=state,
            pincode=self._pincode_of(raw["pincode"], raw["address"]),
            attributes=self._extract_attributes(name or "", address or ""),
            natural_keys=self._extract_keys(by_header, raw_row),
        )

    # --- internals --------------------------------------------------------------

    def _first(self, by_header: dict[str, str], canonical_field: str) -> str:
        for alias in self._aliases.get(canonical_field, ()):
            value = by_header.get(alias)
            if value:
                return value
        return ""

    def _fill_from_combined(self, by_header: dict[str, str], raw: dict[str, str]) -> str | None:
        for header in self._combined:
            value = by_header.get(header)
            if not value:
                continue
            parts = [p.strip() for p in value.split(",") if p.strip()]
            if not raw["state"] and parts:
                raw["state"] = parts[-1]
            if not raw["district"] and len(parts) >= 2:
                raw["district"] = parts[-2]
            rest = parts[:-2]
            return _basic_normalize(", ".join(rest)) or None if rest else None
        return None

    def _canonical_place(self, value: str, aliases: dict[str, str]) -> str | None:
        place = _basic_normalize(value, collapse_acronyms=self._collapse)
        return aliases.get(place, place) or None

    def _distinctive(self, name: str | None, district: str | None, state: str | None) -> str | None:
        if not name:
            return None
        place = set((district or "").split()) | set((state or "").split())
        kept = [t for t in name.split() if t not in self._generic and t not in place]
        return " ".join(kept) or None

    def _pincode_of(self, pincode_field: str, raw_address: str) -> str | None:
        match = self._pincode.search(pincode_field)
        if match is None and self.spec.pincode.search_address:
            found = self._pincode.findall(raw_address)
            # The PIN conventionally ends an Indian address; the last match
            # is the PIN rather than a house or plot number.
            return found[-1] if found else None
        return match.group(match.lastindex or 0) if match else None

    def _extract_attributes(self, name: str, address: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for attr, rule, regex, values in self._attributes:
            text = name if rule.source == "name" else address
            if rule.pattern:
                match = regex.search(text)
                if match:
                    out[attr] = match.group(match.lastindex or 0)
                continue
            found = list(dict.fromkeys(values[m.group(0)] for m in regex.finditer(text)))
            if not found:
                continue
            if len(found) == 1 or rule.on_conflict == "first":
                out[attr] = found[0]
            elif rule.on_conflict == "mixed":
                out[attr] = "mixed"
            else:
                out[attr] = found
        return out

    def _extract_keys(
        self, by_header: dict[str, str], raw_row: Mapping[str, Any]
    ) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, rule, pattern, context in self._keys:
            value = self._key_from_headers(rule, pattern, by_header)
            if value is None and context is not None:
                value = self._key_from_context(rule, context, raw_row)
            if value is not None:
                out[key] = value
        return out

    def _key_from_headers(
        self, rule: NaturalKeyRule, pattern: re.Pattern[str], by_header: dict[str, str]
    ) -> str | None:
        for header in rule.headers:
            candidate = re.sub(r"[\s\-/]", "", by_header.get(normalize_key(header), ""))
            if candidate and pattern.fullmatch(candidate) and _valid_key(candidate, rule):
                return candidate
        return None

    @staticmethod
    def _key_from_context(
        rule: NaturalKeyRule, context: re.Pattern[str], raw_row: Mapping[str, Any]
    ) -> str | None:
        for value in raw_row.values():
            if not isinstance(value, str):
                continue
            for match in context.finditer(value):
                candidate = match.group(match.lastindex or 0)
                if _valid_key(candidate, rule):
                    return candidate
        return None


def _valid_key(value: str, rule: NaturalKeyRule) -> bool:
    check = rule.validate_
    if check is None:
        return True
    if check.length is not None and len(value) != check.length:
        return False
    if check.prefix_range is not None:
        prefix = value[: check.prefix_digits]
        if not prefix.isdigit():
            return False
        low, high = check.prefix_range
        if not low <= int(prefix) <= high:
            return False
    return True
