"""Explicit CSV column -> field mapping, chosen by the user on the upload screen.

Header aliases (`fields.py`, each ontology's `fields:`) recognise the usual
spellings, but real exports say `schl_nm`, `inst_title` or `col_7`. When the
user maps columns explicitly, the mapping is authoritative:

* each mapped column is stored under its canonical name (`name`, `address`,
  `district`, `state`, `pincode`, `id`, or the natural key's first header);
* any *other* column whose header an alias table would also claim for a
  mapped field is renamed `original <header>`, so it cannot outrank the
  user's choice. (Postgres JSONB does not keep key order, so "the mapped
  key comes first" cannot be relied on; renaming can.)
* every other column is kept exactly as uploaded.

It is applied once, at ingestion. Every later reader of `raw_row` (the
classifier, the Silver layer, survivorship, the exports) then works
unchanged. The original file stays in storage and the mapping is recorded on
the job, so each stored row can be reproduced from the two (NFR-04).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ml_pipeline.fields import REQUIRED_FIELDS, aliases_for, normalize_key

FieldName = Literal["name", "address", "district", "state", "pincode"]

#: Headers `_source_id` reads as the tenant's own row id.
RECORD_ID_HEADER = "id"
_RECORD_ID_ALIASES = frozenset({"id", "recordid", "rowid"})
RENAMED_PREFIX = "original "
_MAX_HEADER = 200


class ColumnMapping(BaseModel):
    """{"fields": {"name": "schl_nm", ...}, "record_id": "uid",
    "natural_keys": {"udise_code": "udise_no"}}. Values are CSV headers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fields: dict[FieldName, str]
    record_id: str | None = Field(default=None, max_length=_MAX_HEADER)
    natural_keys: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _well_formed(self) -> ColumnMapping:
        if not self.fields.get("name"):
            raise ValueError("the name column must be mapped")
        used = [*self.fields.values(), *self.natural_keys.values()]
        if self.record_id:
            used.append(self.record_id)
        for header in used:
            if not header.strip() or len(header) > _MAX_HEADER:
                raise ValueError(f"invalid column name {header!r}")
        repeated = sorted({h for h in used if used.count(h) > 1})
        if repeated:
            raise ValueError(f"a column can be mapped to one field only: {repeated}")
        return self

    def headers(self) -> set[str]:
        out = {*self.fields.values(), *self.natural_keys.values()}
        if self.record_id:
            out.add(self.record_id)
        return out


def missing_headers(mapping: ColumnMapping, fieldnames: Collection[str]) -> list[str]:
    """Mapped columns the file does not have (e.g. the wrong file was chosen)."""
    present = set(fieldnames)
    return sorted(h for h in mapping.headers() if h not in present)


def apply_mapping(
    row: Mapping[str, Any],
    mapping: ColumnMapping,
    *,
    field_aliases: Mapping[str, Collection[str]],
    natural_key_headers: Mapping[str, Collection[str]],
) -> dict[str, Any]:
    """One CSV row, re-keyed by the mapping (see the module docstring).

    `field_aliases`: canonical field -> every header spelling that would be
    read as it (fields.py and the ontology, normalised). `natural_key_headers`:
    key name -> its ontology headers (normalised, first is the one written).
    """
    targets: dict[str, str] = {
        field: header for field, header in mapping.fields.items() if header
    }  # canonical name -> source header
    contested: set[str] = set()
    for field in targets:
        contested |= {normalize_key(a) for a in field_aliases.get(field, ())}
        contested.add(field)
    for key in mapping.natural_keys:
        contested |= {normalize_key(h) for h in natural_key_headers.get(key, ())}
    if mapping.record_id:
        contested |= _RECORD_ID_ALIASES

    mapped_sources = mapping.headers()
    out: dict[str, Any] = {}
    for header, value in row.items():
        if header in mapped_sources:
            continue
        name = f"{RENAMED_PREFIX}{header}" if normalize_key(header) in contested else header
        out[name] = value

    for field, header in targets.items():
        out[field] = row.get(header)
    for key, header in mapping.natural_keys.items():
        written = next(iter(natural_key_headers.get(key, ())), None) or key
        out[written] = row.get(header)
    if mapping.record_id:
        out[RECORD_ID_HEADER] = row.get(mapping.record_id)
    return out


def alias_tables(
    ontology_fields: Mapping[str, Collection[str]],
    natural_keys: Mapping[str, Collection[str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """(field -> header spellings, natural key -> headers) for `apply_mapping`:
    the ontology's own aliases plus the platform-wide table in fields.py."""
    fields = {
        field: sorted(
            {normalize_key(a) for a in ontology_fields.get(field, ())} | set(aliases_for(field))
        )
        for field in REQUIRED_FIELDS
    }
    keys = {key: [normalize_key(h) for h in headers] for key, headers in natural_keys.items()}
    return fields, keys
