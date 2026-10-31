"""Fill a missing PIN code from the India Post directory, offline.

Half the schools in a real export arrive with a placeholder PIN ("000000") and
an address that ends in a bare dash. The directory (`reference_pincodes`: one
row per post office, with its district and state) is already loaded, so the
PIN can be found without the web or an LLM: the address names a locality, the
locality is a post office in that district, and the post office has a PIN.

The rule is deliberately conservative, because a wrong PIN looks exactly like
a right one on a client's spreadsheet:

* Only post offices **in the record's own district and state** are candidates.
* The district and state names are stripped from the address first: "DAUSA,
  DAUSA, RAJASTHAN" must not match the district's head office and hide the
  locality before it.
* A match is a whole phrase of the address ("bandikui"), not a substring.
  Single generic words ("nagar", "colony", "road") never count on their own.
* The answer must be **unambiguous**: every surviving match points at one PIN.
  Two localities with different PINs, or one locality name shared by two PINs,
  give no answer at all.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping

#: Post-office type suffixes in the directory: "Bandikui S.O", "Kothimir B.O".
_OFFICE_SUFFIX = re.compile(r"(?:\s+(?:b\s?o|s\s?o|h\s?o|g\s?p\s?o|p\s?o|bo|so|ho|gpo))+$")
_PARENTHETICAL = re.compile(r"\([^)]*\)")
_PIN = re.compile(r"(?<!\d)[1-9]\d{5}(?!\d)")
_MIN_PHRASE_CHARS = 4
_MAX_PHRASE_WORDS = 5

#: One word that names a kind of place, not a place. Whole-address words like
#: "road" are also office names ("Sadar Bazar", "Gandhi Nagar" are fine, as two
#: words), but alone they would match any street in the district.
GENERIC_PLACE_WORDS = frozenset(
    {
        "nagar", "colony", "road", "marg", "street", "market", "bazar", "bazaar", "chowk",
        "village", "vill", "town", "city", "extension", "sector", "phase", "block", "area",
        "near", "opposite", "behind", "post", "office", "district", "tehsil", "taluka",
        "taluk", "mandal", "ward", "main", "old", "new", "east", "west", "north", "south",
        "central", "complex", "campus", "school", "college", "plot", "gali", "lane",
        "layout", "park", "garden", "gardens", "enclave", "vihar", "puram", "pur", "abad",
        "ganj", "pada", "wadi", "wada", "palli", "halli", "peth", "gate", "stand", "station",
        "temple", "mandir", "masjid", "church", "hospital", "bypass", "highway", "cross",
        "circle", "square", "number", "house", "floor", "building", "tower", "apartment",
    }
)  # fmt: skip


def normalize(text: str) -> str:
    """Casefolded words only: punctuation and symbols become spaces."""
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    spaced = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in folded)
    return " ".join(spaced.split())


def office_phrase(office_name: str) -> str:
    """The locality in a post-office name: "Bandikui S.O" -> "bandikui"."""
    text = normalize(_PARENTHETICAL.sub(" ", office_name or ""))
    return _OFFICE_SUFFIX.sub("", text).strip()


class PincodeDirectory:
    """(state, district) -> {locality phrase: {PINs}}, built once per export."""

    def __init__(
        self,
        rows: Iterable[tuple[str, str, str, str]],
        *,
        state_aliases: Mapping[str, str] | None = None,
        district_aliases: Mapping[str, str] | None = None,
    ) -> None:
        self._states = {normalize(k): normalize(v) for k, v in (state_aliases or {}).items()}
        self._districts = {normalize(k): normalize(v) for k, v in (district_aliases or {}).items()}
        self._index: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        #: Every PIN the directory lists for a (state, district), for `has_pin`.
        self._pins: dict[tuple[str, str], set[str]] = defaultdict(set)
        for pincode, office_name, district_name, state_name in rows:
            if not _PIN.fullmatch(pincode or ""):
                continue
            key = (self.state_key(state_name), self.district_key(district_name))
            self._pins[key].add(pincode)
            phrase = office_phrase(office_name)
            if phrase and len(phrase) >= _MIN_PHRASE_CHARS:
                self._index[key][phrase].add(pincode)

    def state_key(self, state: str) -> str:
        key = normalize(state)
        return self._states.get(key, key)

    def district_key(self, district: str) -> str:
        key = normalize(district)
        return self._districts.get(key, key)

    def __len__(self) -> int:
        return sum(len(places) for places in self._index.values())

    def has_pin(self, pincode: str, *, state: str, district: str) -> bool:
        """Does the directory list this PIN in this state and district? The check
        that keeps a PIN read off the web honest: a real PIN from the wrong city
        (an aggregator's address, a namesake school) is not accepted."""
        return pincode in self._pins.get((self.state_key(state), self.district_key(district)), ())

    def infer(self, *, address: str, district: str, state: str) -> str | None:
        """The one PIN the address points at, or None if it points at none or many."""
        state_key, district_key = self.state_key(state), self.district_key(district)
        places = self._index.get((state_key, district_key))
        if not places or not address:
            return None

        text = _PIN.sub(" ", address)
        words = normalize(text).split()
        # The district and state are named in the address as often as not; they
        # would only ever match the district's head office.
        for place in (district_key, state_key, normalize(district), normalize(state)):
            words = _without_phrase(words, place.split())
        if not words:
            return None

        matches: list[tuple[int, int, str]] = []  # (start, length in words, phrase)
        for length in range(1, _MAX_PHRASE_WORDS + 1):
            for start in range(len(words) - length + 1):
                phrase = " ".join(words[start : start + length])
                if phrase in places and not (length == 1 and phrase in GENERIC_PLACE_WORDS):
                    matches.append((start, length, phrase))
        if not matches:
            return None

        # A phrase inside a longer matched phrase says nothing extra.
        kept = [
            m
            for m in matches
            if not any(
                o is not m and o[0] <= m[0] and m[0] + m[1] <= o[0] + o[1] and o[1] > m[1]
                for o in matches
            )
        ]
        pins = {pin for _, _, phrase in kept for pin in places[phrase]}
        if len(pins) == 1:
            return next(iter(pins))

        # Several localities named: trust only the most specific ones, and only
        # if they agree. "Civil Lines" (two words) outranks a one-word match.
        longest = max(length for _, length, _ in kept)
        if longest >= 2:
            specific = {
                pin for _, length, phrase in kept if length == longest for pin in places[phrase]
            }
            if len(specific) == 1:
                return next(iter(specific))
        return None


def _without_phrase(words: list[str], phrase: list[str]) -> list[str]:
    """`words` with every occurrence of the word sequence `phrase` removed."""
    if not phrase:
        return words
    out: list[str] = []
    i = 0
    while i < len(words):
        if words[i : i + len(phrase)] == phrase:
            i += len(phrase)
        else:
            out.append(words[i])
            i += 1
    return out
