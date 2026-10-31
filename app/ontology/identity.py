"""Stable, upload-independent identity for a Silver record.

A record's database id is minted per upload, so anything keyed on it — a
steward's never-merge rule, a learned constraint — dies with the job. The
stable id is derived from *what the record says*, so the same school yields
the same id in every upload, and decisions keyed on it carry forward.

Three tiers, strongest first:

``nk:<key_type>:<value>``
    An ontology natural key (UDISE code, CBSE affiliation number), taken in
    the ontology's declared priority order. Externally assigned and unique
    by construction: the strongest identity there is.

``fp:<ontology>:<sha256/32>``
    A fingerprint of stable Silver fields: the standardised name plus a
    location anchor (PIN, else district + state). Silver has already folded
    "K.V." and "Kendriya Vidyalaya", "Gurgaon" and "Gurugram" together, so
    spelling variants of one school share a fingerprint.

    The hazard is the opposite one — distinct schools sharing a fingerprint.
    "Government Primary School" in one PIN code is several schools, and a
    rule bound to their shared fingerprint would silently bind all of them.
    So when the name is *entirely* generic (no distinctive token), the
    address joins the fingerprint; and with no address either, there is no
    safe fingerprint at all.

``rec:<record_id>``
    No stable basis. Correct but local: it never matches another upload, so
    nothing keyed on it carries over. Counted, so how much of an upload
    could not be identified is visible rather than silent.

Fingerprints are built from Silver text, so editing an ontology's
expansions can change them; natural keys are immune. That is the price of
normalising before hashing, and the reason natural keys rank first.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

IdentityKind = Literal["natural_key", "fingerprint", "record"]


@dataclass(frozen=True, slots=True)
class Identity:
    stable_id: str
    kind: IdentityKind


def derive_identity(
    *,
    ontology: str,
    record_id: str,
    natural_keys: Mapping[str, str],
    key_priority: Sequence[str],
    name: str | None,
    name_distinctive: str | None,
    address: str | None,
    pincode: str | None,
    district: str | None,
    state: str | None,
) -> Identity:
    for key_type in key_priority:
        value = natural_keys.get(key_type)
        if value:
            return Identity(f"nk:{key_type}:{value}", "natural_key")

    fingerprint = _fingerprint(
        name=name,
        name_distinctive=name_distinctive,
        address=address,
        pincode=pincode,
        district=district,
        state=state,
    )
    if fingerprint is not None:
        return Identity(f"fp:{ontology}:{fingerprint}", "fingerprint")
    return Identity(f"rec:{record_id}", "record")


def _fingerprint(
    *,
    name: str | None,
    name_distinctive: str | None,
    address: str | None,
    pincode: str | None,
    district: str | None,
    state: str | None,
) -> str | None:
    if not name:
        return None
    if pincode:
        anchor = [f"p={pincode}"]
    elif district and state:
        anchor = [f"d={district}", f"s={state}"]
    else:
        return None
    parts = [f"n={name}", *anchor]
    if not name_distinctive:
        # An all-generic name names a category, not a school.
        if not address:
            return None
        parts.append(f"a={address}")
    # Labelled, unit-separated parts: ("ab", "c") and ("a", "bc") must not
    # hash alike.
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]
