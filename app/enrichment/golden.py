"""Tier 3: universal, domain-agnostic enrichment of a resolved golden record.

Nothing here knows what a school is. The ontology supplies everything
domain-specific: the search template (`golden_enrichment.query`), what to
look for (`enrichment_targets`, each with a type and an optional meaning for
the LLM), and which name words are generic (for attribution). A hospital
ontology asking for `bed_count` and a corporate one asking for `ceo_name`
and `revenue` run through this same function.

One search; closed-form targets read by pattern (patterns.py), the rest by one
strict-JSON extraction over the snippets that are about this entity; then
`verify_extraction`: only values found verbatim in a snippet that names this
entity survive.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from app.enrichment.grounding import (
    RejectedValue,
    VerifiedValue,
    mentions_entity,
    normalize,
    verify_extraction,
)
from app.enrichment.patterns import extract_by_pattern, find_pincode
from app.enrichment.sources import EntityFields, SearchFn, SearchHit
from app.ontology.engine import EnrichmentTarget, GoldenEnrichmentRules, render_query

Status = Literal["found", "not_found"]

#: `LlmExtractor.extract`'s shape: (keys, subject, hits, descriptions) -> {key: raw}.
ExtractFn = Callable[
    [Sequence[str], str, Sequence[SearchHit], Mapping[str, str]], Mapping[str, Any]
]


@dataclass(frozen=True, slots=True)
class GoldenEnrichment:
    status: Status
    query: str
    fields: dict[str, VerifiedValue] = field(default_factory=dict)
    rejected: tuple[RejectedValue, ...] = ()
    hits: int = 0

    def as_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "query": self.query,
            "hits": self.hits,
            "fields": {key: value.as_json() for key, value in self.fields.items()},
            "rejected": [r.as_json() for r in self.rejected],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> GoldenEnrichment:
        return cls(
            status=data["status"],
            query=data.get("query", ""),
            hits=int(data.get("hits", 0)),
            fields={
                key: VerifiedValue(v["value"], v["url"], v["evidence"])
                for key, v in (data.get("fields") or {}).items()
            },
            rejected=tuple(
                RejectedValue(r["key"], r["value"], r["reason"]) for r in data.get("rejected") or []
            ),
        )


#: Where a PIN the web found is kept: beside, never inside, the golden record's
#: own `pincode`, which holds a steward's correction. The export reads it only
#: when the record has no PIN of its own.
WEB_PINCODE_KEY = "web_pincode"

_PHONE_LIKE = re.compile(r"\d[\d\s().+-]{6,}\d")


def _worth_asking(target: EnrichmentTarget, about: Sequence[SearchHit]) -> bool:
    """Could these snippets plausibly hold this target? A cheap check that
    saves a model call (and its tokens) when they cannot: an email needs an
    "@", a phone needs a run of digits, a person named by role ("principal")
    needs the role word. Anything else is always worth asking."""
    text = " ".join(f"{h.title} {h.snippet}" for h in about).casefold()
    if target.type == "email":
        return "@" in text
    if target.type == "phone":
        return _PHONE_LIKE.search(text) is not None
    if target.type == "person":
        role = [w for w in target.key.split("_") if w not in {"name", "person"}]
        return not role or any(w in text for w in role)
    return True


def find_hits(
    entity: EntityFields, rules: GoldenEnrichmentRules, search: SearchFn
) -> tuple[str, list[SearchHit]]:
    """`(query, hits)`: the entity's search, retried once without quotes when
    the exact-phrase version finds nothing (a spelling the engine does not
    index verbatim). Pure web I/O, so it can run on a worker thread ahead of
    the LLM and database work of the entity before it."""
    query = render_query(rules.query, entity.template_fields())
    hits = search(query, rules.max_results)
    if not hits and '"' in query:
        hits = search(query.replace('"', " "), rules.max_results)
    return query, hits


def enrich_entity(
    entity: EntityFields,
    *,
    targets: Sequence[EnrichmentTarget],
    rules: GoldenEnrichmentRules,
    generic_tokens: Collection[str],
    search: SearchFn | None = None,
    extract: ExtractFn,
    is_valid_pincode: Callable[[str], bool] | None = None,
    found: tuple[str, list[SearchHit]] | None = None,
) -> GoldenEnrichment:
    """One search, then: closed-form targets by pattern, the rest by one LLM
    call over only the snippets that are about this entity, then the same
    strict `verify_extraction` for everything.

    `found` is the `find_hits` result when the search was already run ahead of
    time; otherwise `search` is used. `is_valid_pincode` (a PIN that exists in
    this entity's district and state) switches on PIN recovery for an entity
    that has none.
    """
    if found is None:
        if search is None:
            raise ValueError("enrich_entity needs `search` or `found`")
        found = find_hits(entity, rules, search)
    query, hits = found
    if not hits:
        return GoldenEnrichment("not_found", query)

    about = [
        h
        for h in hits
        if mentions_entity(normalize(f"{h.title} {h.snippet}"), entity, generic_tokens)
    ]
    extracted: dict[str, Any] = extract_by_pattern(targets, hits, entity, generic_tokens)
    asked = [t for t in targets if not t.pattern and about and _worth_asking(t, about)]
    if asked:
        extracted.update(
            extract(
                [t.key for t in asked],
                entity.describe(),
                about,
                {t.key: t.description for t in asked},
            )
        )
    verified, rejected = verify_extraction(targets, extracted, hits, entity, generic_tokens)
    if is_valid_pincode is not None and not entity.pincode:
        pin = find_pincode(hits, entity, generic_tokens, is_valid=is_valid_pincode)
        if pin is not None:
            verified[WEB_PINCODE_KEY] = pin
    return GoldenEnrichment(
        status="found" if verified else "not_found",
        query=query,
        fields=verified,
        rejected=tuple(rejected),
        hits=len(hits),
    )


def merge_into_golden(
    golden_fields: dict[str, str] | None, found: dict[str, VerifiedValue]
) -> dict[str, str] | None:
    """Add web values to a golden record without overwriting anything.

    A key already present (a steward's correction, or an earlier value)
    wins over the web. Targets can never be identity fields (the ontology
    refuses them), so name, address and location are never touched here.
    """
    merged = dict(golden_fields or {})
    for key, value in found.items():
        if not merged.get(key):
            merged[key] = value.value
    return merged or golden_fields
