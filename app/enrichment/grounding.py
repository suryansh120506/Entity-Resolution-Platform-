"""Strict hallucination check for Tier 3: a value the LLM extracted is kept
only if it is literally in a search snippet *about this entity*.

The LLM is a copying tool here, never a source. Every value it returns must
pass, in order:

1. **Present.** null, "", "N/A", "unknown" and friends mean "not found",
   which is fine and is not counted as a rejection.
2. **Shaped.** The target's type (email, phone, url, person, number, text)
   decides what a value may look like. A `principal_name` of "12345" is
   dropped before anyone checks where it came from.
3. **Not an echo.** A value that merely repeats the entity's own name
   ("principal: Delhi Public School") is in the snippets, but says nothing.
4. **Verbatim.** The value must occur as a contiguous substring of one
   snippet's title or body, after only case folding, Unicode NFKC and
   whitespace collapsing, and on word boundaries: "Ram" is not found inside
   "Ramesh". A reformatted phone, a "corrected" spelling or a value stitched
   across two snippets all fail. This is the check that makes an invented
   value impossible to store. (A `url` target may also match the result's
   own link, which is part of the raw DDGS result too.)
5. **Attributed.** The snippet holding the value must itself name the
   entity: at least half of the distinctive (non-generic) words of its name,
   or, for an all-generic name like "Government Senior Secondary School",
   its PIN code or district. Search results mix entities; a principal copied
   verbatim from a *different* school's snippet is not this school's.

What survives carries its evidence (URL and snippet) into `clusters.enrichment`.
Verbatim does not mean true: a planted or out-of-date snippet passes. It
means traceable, which is what makes a wrong value fixable.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from app.enrichment.sources import EntityFields, SearchHit
from app.ontology.engine import EnrichmentTarget

_NULLISH = frozenset(
    {
        "",
        "null",
        "none",
        "nil",
        "n/a",
        "na",
        "-",
        "unknown",
        "not found",
        "not available",
        "not mentioned",
        "not specified",
        "not given",
    }
)
_PUNCTUATION_FOLD = str.maketrans(
    {"’": "'", "‘": "'", "`": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "}
)
_EMAIL = re.compile(r"^[a-z0-9._%+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}$")
_PHONE_CHARS = re.compile(r"^[\d\s+()./-]+$")
_PERSON = re.compile(r"^[^\W\d_][^\W\d_ .'-]*(?:[ .'-]+[^\W\d_][^\W\d_ .'-]*){0,6}\.?$")
_WORD = re.compile(r"[^\W_]+")

MAX_LENGTH = 200


@dataclass(frozen=True, slots=True)
class VerifiedValue:
    value: str
    url: str
    evidence: str

    def as_json(self) -> dict[str, str]:
        return {"value": self.value, "url": self.url, "evidence": self.evidence}


@dataclass(frozen=True, slots=True)
class RejectedValue:
    key: str
    value: str
    reason: str

    def as_json(self) -> dict[str, str]:
        return {"key": self.key, "value": self.value, "reason": self.reason}


def normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).translate(_PUNCTUATION_FOLD)
    return " ".join(folded.casefold().split())


def occurs_verbatim(needle: str, haystack: str) -> bool:
    """`needle` inside `haystack` (both already normalised), on word edges.

    A boundary is only demanded where the needle itself starts or ends with a
    word character: "+91 98765 43210" and "kvs.gov.in/" may sit against
    anything, but "ram" must not match inside "ramesh".
    """
    if not needle:
        return False
    start = r"(?<!\w)" if re.match(r"\w", needle[0]) else ""
    end = r"(?!\w)" if re.match(r"\w", needle[-1]) else ""
    return re.search(start + re.escape(needle) + end, haystack) is not None


def _extracted_text(raw: Any) -> str | None:
    """The value out of `{"value": ..., "snippet_index": ...}` or a bare value."""
    value = raw.get("value") if isinstance(raw, Mapping) else raw
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        value = str(value)
    if not isinstance(value, str):
        return None
    return " ".join(value.split()).strip(" ,;:|")


def shaped(target: EnrichmentTarget, value: str) -> str | None:
    """The value in its stored form if it has the target's shape, else None."""
    if len(value) > MAX_LENGTH or len(value) < 2:
        return None
    kind = target.type
    if kind == "email":
        email = value.lower().removeprefix("mailto:")
        return email if _EMAIL.match(email) else None
    if kind == "phone":
        digits = re.sub(r"\D", "", value)
        return value if _PHONE_CHARS.match(value) and 8 <= len(digits) <= 13 else None
    if kind == "url":
        candidate = value if "://" in value else f"https://{value}"
        host = urlsplit(candidate).hostname or ""
        return value if "." in host and " " not in value else None
    if kind == "person":
        letters = sum(ch.isalpha() for ch in value)
        return value if _PERSON.match(value) and letters >= 3 else None
    if kind == "number":
        return value if any(ch.isdigit() for ch in value) and len(value) <= 60 else None
    return value if len(value) <= 120 else None


def distinctive_tokens(name: str, generic: Collection[str]) -> list[str]:
    """The words of a name that say *which* entity, not what kind."""
    generic_set = {normalize(g) for g in generic}
    return [
        token
        for token in dict.fromkeys(_WORD.findall(normalize(name)))
        if len(token) >= 3 and token not in generic_set and not token.isdigit()
    ]


def mentions_entity(text: str, entity: EntityFields, generic: Collection[str]) -> bool:
    """Does this (normalised) snippet text talk about this entity?"""
    tokens = distinctive_tokens(entity.name, generic)
    if tokens:
        needed = max(1, math.ceil(len(tokens) / 2))
        return sum(occurs_verbatim(t, text) for t in tokens) >= needed
    # An all-generic name names a kind of place, not a place: demand where.
    anchors = [normalize(a) for a in (entity.pincode, entity.district) if a]
    return any(occurs_verbatim(anchor, text) for anchor in anchors)


def _echoes_entity(target: EnrichmentTarget, value: str, entity: EntityFields) -> bool:
    if target.type not in {"person", "text"}:
        return False
    own = normalize(value)
    return any(own and own in normalize(field) for field in (entity.name, entity.address))


def verify_extraction(
    targets: Sequence[EnrichmentTarget],
    extracted: Mapping[str, Any],
    hits: Sequence[SearchHit],
    entity: EntityFields,
    generic_tokens: Collection[str] = (),
) -> tuple[dict[str, VerifiedValue], list[RejectedValue]]:
    """Split the LLM's answer into values that passed every check and values
    that did not (with the reason). Keys the LLM added are never read."""
    passages = [
        (hit, normalize(hit.title), normalize(hit.snippet), normalize(f"{hit.title} {hit.snippet}"))
        for hit in hits
    ]
    verified: dict[str, VerifiedValue] = {}
    rejected: list[RejectedValue] = []

    for target in targets:
        text = _extracted_text(extracted.get(target.key))
        if text is None or normalize(text) in _NULLISH:
            continue

        value = shaped(target, text)
        if value is None:
            rejected.append(RejectedValue(target.key, text, "wrong_shape"))
            continue
        if _echoes_entity(target, value, entity):
            rejected.append(RejectedValue(target.key, text, "echoes_entity"))
            continue

        needle = normalize(value)
        containing = [
            (hit, whole)
            for hit, title, body, whole in passages
            if occurs_verbatim(needle, title)
            or occurs_verbatim(needle, body)
            # A site's address is often only in the result's own link, which
            # is as much a part of the raw DDGS result as its text.
            or (target.type == "url" and occurs_verbatim(needle, normalize(hit.url)))
        ]
        if not containing:
            rejected.append(RejectedValue(target.key, text, "not_in_snippets"))
            continue

        attributed = next(
            (
                (hit, whole)
                for hit, whole in containing
                if mentions_entity(whole, entity, generic_tokens)
            ),
            None,
        )
        if attributed is None:
            rejected.append(RejectedValue(target.key, text, "not_attributed"))
            continue

        hit, _ = attributed
        verified[target.key] = VerifiedValue(
            value=value, url=hit.url, evidence=f"{hit.title} | {hit.snippet}"[:400]
        )
    return verified, rejected
