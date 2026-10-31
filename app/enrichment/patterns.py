"""Tier 3 values that need no LLM: read by pattern, from the right snippet.

An LLM is the right tool for "who is the principal?" and the wrong one for
"which exam board?", "since when?" or "what is its CBSE number?": those are
closed-form, a regex finds them exactly, and every model call spends rate-
limited capacity (a free tier allows a few hundred schools a day). The ontology
declares such a target with a `pattern`; this module reads it.

A match is only taken from a snippet that is about this entity *and* names its
place (district, state or PIN), and from the match nearest the entity's own
name, so a results page listing twenty schools does not hand this one a
neighbour's number. What comes out has the same shape as an LLM's answer and
goes through the same `verify_extraction` (shape, verbatim, attributed), so a
pattern cannot store anything a model could not.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Sequence
from typing import Any

from app.enrichment.grounding import (
    VerifiedValue,
    distinctive_tokens,
    mentions_entity,
    normalize,
    occurs_verbatim,
)
from app.enrichment.sources import EntityFields, SearchHit
from app.ontology.engine import EnrichmentTarget

#: How far, in characters, a value may sit from the entity's name and still be
#: read as being about it. A result's title and snippet are a few hundred
#: characters; beyond this the value belongs to whatever else is listed.
MAX_DISTANCE = 300

_PIN = re.compile(r"(?<!\d)([1-9]\d{5})(?!\d)")


def _positions(text: str, tokens: Sequence[str]) -> list[int]:
    """Where, in `text`, any of the entity's distinctive name words occurs."""
    found: list[int] = []
    for token in tokens:
        found.extend(m.start() for m in re.finditer(rf"(?<!\w){re.escape(token)}(?!\w)", text))
    return found


def _names_place(normalized: str, entity: EntityFields) -> bool:
    """Does the passage mention the entity's district, state or PIN?

    Without this a namesake in another state passes: "Don Bosco School" is a
    different school in every district. An entity with no place at all (an
    unusually thin record) cannot be held to it.
    """
    anchors = [normalize(a) for a in (entity.district, entity.state, entity.pincode) if a]
    return not anchors or any(occurs_verbatim(a, normalized) for a in anchors)


def _passages(
    hits: Sequence[SearchHit], entity: EntityFields, generic: Collection[str]
) -> list[tuple[int, str, list[int]]]:
    """(hit index, text, name positions) for each snippet about this entity in this place."""
    tokens = distinctive_tokens(entity.name, generic)
    out: list[tuple[int, str, list[int]]] = []
    for index, hit in enumerate(hits):
        text = f"{hit.title} {hit.snippet}"
        folded = normalize(text)
        if mentions_entity(folded, entity, generic) and _names_place(folded, entity):
            out.append((index, text, _positions(text.casefold(), tokens)))
    return out


def _nearest(position: int, anchors: Sequence[int]) -> int | None:
    """Distance to the closest anchor; None when there are no anchors to measure from."""
    return min((abs(position - a) for a in anchors), default=None)


def extract_by_pattern(
    targets: Sequence[EnrichmentTarget],
    hits: Sequence[SearchHit],
    entity: EntityFields,
    generic_tokens: Collection[str] = (),
) -> dict[str, Any]:
    """`{key: {"value", "snippet_index"}}` for every pattern target that matched.

    The closest match to the entity's name wins; ties go to the earlier (higher
    ranked) result. A target with no match is simply absent: "not found".
    """
    passages = _passages(hits, entity, generic_tokens)
    out: dict[str, Any] = {}
    for target in targets:
        if not target.pattern:
            continue
        regex = re.compile(target.pattern)
        best: tuple[int, int, str] | None = None  # (distance, hit index, value)
        for index, text, anchors in passages:
            for match in regex.finditer(text):
                value = (match.group(1) if regex.groups else match.group(0)).strip()
                if not value:
                    continue
                distance = _nearest(match.start(), anchors)
                if distance is not None and distance > MAX_DISTANCE:
                    continue
                rank = (distance if distance is not None else 0, index)
                if best is None or rank < best[:2]:
                    best = (*rank, value)
        if best is not None:
            out[target.key] = {"value": best[2], "snippet_index": best[1]}
    return out


def find_pincode(
    hits: Sequence[SearchHit],
    entity: EntityFields,
    generic_tokens: Collection[str],
    *,
    is_valid: Callable[[str], bool],
) -> VerifiedValue | None:
    """A PIN code for an entity that has none, from the web, or None.

    Stricter than a target, because a PIN is an identity field and a wrong one
    reads as a right one. It must be: six digits written in a snippet about
    this entity, within `MAX_DISTANCE` of its name, **and a PIN that the post
    office directory lists in this entity's own district and state** (`is_valid`).
    Several different PINs near the name is no answer.
    """
    candidates: dict[str, tuple[SearchHit, int]] = {}
    for index, text, anchors in _passages(hits, entity, generic_tokens):
        if not anchors:
            continue
        for match in _PIN.finditer(text):
            pin = match.group(1)
            distance = _nearest(match.start(), anchors)
            if distance is None or distance > MAX_DISTANCE or not is_valid(pin):
                continue
            if pin not in candidates or distance < candidates[pin][1]:
                candidates[pin] = (hits[index], distance)
    if len(candidates) != 1:
        return None
    pin, (hit, _) = next(iter(candidates.items()))
    return VerifiedValue(pin, hit.url, f"{hit.title} | {hit.snippet}"[:400])
