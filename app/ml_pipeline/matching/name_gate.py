"""Distinctive-name agreement — a veto layered on the probabilistic score.

Indian school names are mostly institutional boilerplate: "Government
Senior Secondary School", "Public School", "International Academy". Two
schools in the same town share that boilerplate, their town, their pincode
and often most of their address — and on a real 21,736-record upload the
Fellegi-Sunter model, trained by EM on same-pincode pairs, learned exactly
that pattern as "a match". It linked "government college kullu" to
"government college banjjar" at 0.875, and 26 different schools in one
Jhunjhunu pincode into one cluster.

What identifies a school is the part of its name that is *not* boilerplate
— Tagore vs Lohiya, Sainj vs Dhameta. This module strips the boilerplate,
and the record's own district and state (a district name inside a school
name — "SK Senior Secondary School Hisar" — is shared by every school in
Hisar, not evidence that two of them are one), and measures how much of
what is left the two names share, tolerating spelling variation ("niraja"
/ "nirja", "bharti" / "bharati").

Where both names are *pure* boilerplate ("Government Sen Secondary School",
seventeen of them in one district), the name says nothing, and the address
has to supply the distinctive part instead: the village or locality.

A few conflicts are decisive on their own: a girls' school and a boys' (or
unmarked) school, a college and a school.

It is a veto, not a score: it never *raises* a pair the model doubted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

#: Institution-type and boilerplate words that say what kind of school a
#: record is, not which one. Normalised forms (lowercase, abbreviations
#: already expanded by `encoder.normalize_text`), plus the common
#: unexpanded leftovers seen in real exports.
# fmt: off
GENERIC_TOKENS = frozenset(
    {
        # institution type
        "school", "schools", "vidyalaya", "vidyalay", "vidhyalaya", "mahavidyalaya",
        "college", "academy", "institute", "institution", "university", "gurukul",
        "pathshala", "shala", "madrasa", "madarsa", "kendra", "niketan", "sadan",
        "mandir", "ashram", "bhavan", "bhawan", "centre", "center",
        # level
        "senior", "sen", "sr", "secondary", "sec", "higher", "hr", "high", "hs",
        "primary", "middle", "upper", "junior", "jr", "jun", "elementary", "inter",
        "intermediate", "pre", "nursery", "kindergarten", "basic",
        # management / kind
        "government", "govt", "public", "private", "model", "convent", "national",
        "central", "international", "global", "english", "hindi", "medium", "residential",
        "boarding", "day", "co", "ed", "coed", "composite", "adarsh", "kendriya",
        "navodaya", "jawahar", "sarvodaya", "rajkiya", "uchch", "uchcha",
        "madhyamik", "prathmik", "shiksha", "education", "educational", "society",
        "trust", "mission", "foundation",
        # booksellers and coaching centres, which real exports include too
        "book", "books", "depot", "store", "stores", "agency", "agencies",
        "publishers", "publication", "publications", "distributors", "stationers",
        "stationery", "traders", "enterprises", "classes", "class", "coaching",
        "tutorials", "tuition", "point", "hub", "zone",
        "chemistry", "physics", "maths", "mathematics", "biology", "science",
        "commerce",
        # connectives and filler
        "the", "of", "and", "for", "at", "in", "no", "new", "shri", "sri", "smt",
        "st", "saint",
    }
)
# fmt: on

#: Address words that locate *anything*, not a particular school.
# fmt: off
ADDRESS_GENERIC_TOKENS = frozenset(
    {
        "road", "rd", "near", "nr", "opp", "opposite", "behind", "post", "po", "ho",
        "bo", "so", "gpo", "village", "vill", "vpo", "tehsil", "teh", "district",
        "dist", "distt", "block", "ward", "main", "city", "town", "india",
        "country", "state", "pin", "pincode", "street", "st", "lane", "marg",
        "the", "of", "and", "at", "via", "area", "office", "national", "highway",
        "nh", "sh", "school", "campus", "building", "floor", "house", "no",
        "sector", "sec", "pocket", "phase", "plot", "extension", "ext", "colony",
    }
)
# fmt: on

#: State / union-territory abbreviations as they appear appended to names
#: and addresses ("... Sangrur PB", "... Kullu HP"). Shared by every record
#: in the state, so never evidence two records are one school.
# fmt: off
STATE_ABBREVIATIONS = frozenset(
    {
        "ap", "ar", "as", "br", "cg", "ct", "ga", "gj", "hr", "hp", "jh", "jk", "ka",
        "kl", "mp", "mh", "mn", "ml", "mz", "nl", "od", "or", "pb", "rj", "sk", "tn",
        "tg", "ts", "tr", "up", "uk", "ut", "wb", "an", "ch", "dn", "dd", "dl", "ld",
        "py", "la",
    }
)
# fmt: on

#: Who a school is for. Two names that say differently — or one that says
#: and one that does not — are two schools: "GSSS Kullu" and "GSSS Girls
#: Kullu" share an address and every other word.
# fmt: off
_GENDER = {
    "girls": "f", "girl": "f", "women": "f", "womens": "f", "ladies": "f",
    "mahila": "f", "kanya": "f", "balika": "f",
    "boys": "m", "boy": "m", "men": "m", "mens": "m", "balak": "m",
}
# fmt: on

#: What kind of institution a name says it is. A college and a school
#: sharing a town and a gender word ("Government Girls College" /
#: "Government Girls Sen Senior School") are two institutions.
# fmt: off
_KIND = {
    "school": "school", "schools": "school", "vidyalaya": "school",
    "vidyalay": "school", "vidhyalaya": "school", "pathshala": "school",
    "shala": "school", "madrasa": "school", "madarsa": "school",
    "college": "college", "mahavidyalaya": "college", "university": "college",
    "classes": "coaching", "coaching": "coaching", "tuition": "coaching",
}
# fmt: on

#: Two records must agree on *more than* this share of distinctive words.
#: Strictly more than half: at exactly half, the shared word was measured to
#: be the locality ("Raja Ram ... Mandi Dabwali" / "Arya ... Mandi Dabwali"),
#: not the school.
DEFAULT_MIN_AGREEMENT = 0.5

#: Token similarity at or above which two words count as the same word.
_TOKEN_SIMILARITY = 0.8

_DIGITS = re.compile(r"^\d+$")


@dataclass(frozen=True)
class RecordText:
    """What the gate needs about one record."""

    name: str
    address: str = ""
    #: Words of the record's district and state — ignored when comparing,
    #: because every school in the district shares them.
    place_words: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def build(cls, name: str, address: str, district: str, state: str) -> RecordText:
        place = {w for w in f"{district} {state}".lower().split() if w}
        return cls(name=name or "", address=address or "", place_words=frozenset(place))


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^\w]+", text.lower()) if t]


def distinctive_tokens(name: str, ignore: frozenset[str] = frozenset()) -> list[str]:
    """The words of a normalised name that are not boilerplate or place names."""
    return [t for t in _tokens(name) if t not in GENERIC_TOKENS and t not in ignore]


def _same_word(a: str, b: str) -> bool:
    if a == b:
        return True
    # Short tokens (numbers, initials) must match exactly: "1" vs "2" is the
    # whole difference between Kendriya Vidyalaya No. 1 and No. 2.
    if len(a) < 4 or len(b) < 4:
        return False
    return SequenceMatcher(None, a, b).ratio() >= _TOKEN_SIMILARITY


def token_overlap(a: list[str], b: list[str]) -> float:
    """Share of the longer list matched one-to-one by the shorter, fuzzily."""
    if not a or not b:
        return 0.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    unused = list(longer)
    matched = 0
    for token in shorter:
        for index, candidate in enumerate(unused):
            if _same_word(token, candidate):
                matched += 1
                del unused[index]
                break
    return matched / len(longer)


def _genders(tokens: list[str]) -> set[str]:
    return {_GENDER[t] for t in tokens if t in _GENDER}


def _kinds(name: str) -> set[str]:
    return {_KIND[t] for t in _tokens(name) if t in _KIND}


def _address_tokens(address: str, ignore: frozenset[str]) -> list[str]:
    # Digits are kept on purpose: "Sector 8" / "Sector 18" and their
    # pincodes are the *only* thing distinguishing nine Government Model
    # Schools in Chandigarh, which merged automatically when they were dropped.
    return [t for t in _tokens(address) if t not in ADDRESS_GENERIC_TOKENS and t not in ignore]


def _numbers(tokens: list[str]) -> set[str]:
    return {t for t in tokens if _DIGITS.match(t)}


def agreement(a: RecordText, b: RecordText) -> float:
    """How strongly two records' names say they are the same school, [0, 1]."""
    ignore = a.place_words | b.place_words | STATE_ABBREVIATIONS
    name_a = distinctive_tokens(a.name, ignore)
    name_b = distinctive_tokens(b.name, ignore)
    address_a = _address_tokens(a.address, ignore)
    address_b = _address_tokens(b.address, ignore)

    if _genders(name_a) != _genders(name_b):
        return 0.0
    kind_a, kind_b = _kinds(a.name), _kinds(b.name)
    if kind_a and kind_b and not kind_a & kind_b:
        return 0.0

    # Numbered branches: "Kendriya Vidyalaya No. 1" and "No. 2" are two schools.
    numbers_a, numbers_b = _numbers(name_a), _numbers(name_b)
    if numbers_a and numbers_b and numbers_a != numbers_b:
        return 0.0

    name_a = [t for t in name_a if t not in _GENDER]
    name_b = [t for t in name_b if t not in _GENDER]

    if name_a and name_b:
        # Chain schools share a whole name across branches ("Dayanand Anglo
        # Vedic Public School" in Dwarka and in Vasant Kunj). Where both
        # addresses name a locality and they share none, the name is the
        # chain's, not evidence of one school.
        if address_a and address_b and token_overlap(address_a, address_b) == 0.0:
            return 0.0
        return token_overlap(name_a, name_b)
    if name_a or name_b:
        # One names a particular school, the other only its kind.
        return 0.0

    # Both pure boilerplate: the name is no evidence either way unless it is
    # the *same* boilerplate, and then the address has to be distinctive.
    if _tokens(a.name) != _tokens(b.name):
        return 0.0
    if not address_a and not address_b:
        # Same boilerplate name, and neither address says anything more
        # specific than the district ("GSSS Kullu", "Kullu HO"): nothing
        # distinguishes them, so the probabilistic score decides.
        return 1.0
    return token_overlap(address_a, address_b)


def compatible(a: RecordText, b: RecordText, min_agreement: float = DEFAULT_MIN_AGREEMENT) -> bool:
    return agreement(a, b) > min_agreement
