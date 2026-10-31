#!/usr/bin/env python
"""Field-by-field verification and offline enrichment of a school export.

    python scripts/verify_enrich_schools.py \
        --input  snop-schools-export.csv \
        --pincodes all_india_pincodes.csv \
        --outdir verified/

Every field of every row gets a status and, where one is justified, a cleaned
value. Nothing is invented and no network or paid API is called (zero-cost
mandate): the only reference data is the India Post directory, the UDISE+ state
codes and the rules below. The source file is never modified.

Cross-checks:

* PIN        format, present in the directory, and its state/district agree with
             the record's own; a missing PIN is recovered from the address only
             when `PincodeDirectory.infer` finds exactly one answer.
* State/district  canonicalised to the directory's spelling (official renames).
* Coordinates  inside India, not swapped, and near the PIN's post offices.
             Where absent, the PIN centroid is offered in separate columns,
             labelled approximate, never written over a real value.
* UDISE      11 digits and its state prefix matches the record's state.
* Phone/email/website  syntax plus placeholder and synthetic-value detection.
* Name       leading punctuation, "(MA001)" codes, 30-character truncation.
* Category   the export carries the *country* there; moved to `country`.
* Affiliation number  kept as `source_ref_no`: on a real export it was a row id,
             not a CBSE number (see ontologies/education.yaml).
"""

from __future__ import annotations

import argparse
import difflib
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.enrichment.pincode_infer import PincodeDirectory, normalize  # noqa: E402

# --- header resolution ------------------------------------------------------------

FIELD_ALIASES: dict[str, list[str]] = {
    "id": ["id", "schoolid"],
    "name": ["institutename", "schoolname", "name", "institute"],
    "affiliation": ["affiliationnumber", "affiliationno", "affiliation"],
    "udise": ["udisecode", "udise"],
    "addr1": ["addressline1", "address1", "address"],
    "addr2": ["addressline2", "address2"],
    "addr3": ["addressline3", "address3"],
    "locality": ["localitydistrictstate", "locality"],
    "pin": ["pincode", "pin"],
    "district": ["district"],
    "state": ["state"],
    "phone1": ["phone1", "phone"],
    "phone2": ["phone2"],
    "fax": ["fax", "faxnumber"],
    "email1": ["emailprimary", "email"],
    "email2": ["emailsecondary"],
    "website": ["website"],
    "highest_class": ["highestclass"],
    "medium": ["medium"],
    "curriculum": ["curriculum", "board"],
    "grade": ["grade"],
    "category": ["category"],
    "streams": ["streams", "stream"],
    "lat": ["lat", "latitude"],
    "lng": ["lng", "longitude", "long"],
    "geo_source": ["geocodesource", "locationsource"],
    "place_id": ["googleplaceid", "placeid"],
    "status": ["status"],
    "created": ["createdat", "created"],
    "updated": ["updatedat", "lastupdated", "updated"],
}


def resolve_columns(columns: list[str]) -> tuple[dict[str, str], list[str]]:
    """canonical field -> actual header. Exact normalised match, then prefix
    (a screenshot-style header such as "Email (primary)" or a truncated one)."""
    norm = {c: re.sub(r"[^a-z0-9]", "", c.lower()) for c in columns}
    found: dict[str, str] = {}
    for field, aliases in FIELD_ALIASES.items():
        hit = next((c for c, n in norm.items() if n in aliases and c not in found.values()), None)
        if hit is None:
            hit = next(
                (
                    c
                    for c, n in norm.items()
                    if c not in found.values() and any(len(a) >= 5 and n.startswith(a) for a in aliases)
                ),
                None,
            )
        if hit:
            found[field] = hit
    unmapped = [c for c in columns if c not in found.values()]
    return found, unmapped


# --- reference data -----------------------------------------------------------------

#: UDISE+ state codes (the LGD numbering). Prefix of an 11-digit UDISE code.
UDISE_STATE = {
    "01": "jammu and kashmir", "02": "himachal pradesh", "03": "punjab", "04": "chandigarh",
    "05": "uttarakhand", "06": "haryana", "07": "delhi", "08": "rajasthan",
    "09": "uttar pradesh", "10": "bihar", "11": "sikkim", "12": "arunachal pradesh",
    "13": "nagaland", "14": "manipur", "15": "mizoram", "16": "tripura", "17": "meghalaya",
    "18": "assam", "19": "west bengal", "20": "jharkhand", "21": "odisha",
    "22": "chhattisgarh", "23": "madhya pradesh", "24": "gujarat", "25": "daman and diu",
    "26": "dadra and nagar haveli", "27": "maharashtra", "28": "andhra pradesh",
    "29": "karnataka", "30": "goa", "31": "lakshadweep", "32": "kerala", "33": "tamil nadu",
    "34": "puducherry", "35": "andaman and nicobar islands", "36": "telangana", "37": "ladakh",
}  # fmt: skip
#: States that were one at the time a code or PIN was issued.
STATE_EQUIV = [
    {"daman and diu", "dadra and nagar haveli", "the dadra and nagar haveli and daman and diu"},
    {"jammu and kashmir", "ladakh"},
    {"andhra pradesh", "telangana"},
]

#: Spellings seen in real exports that _india.yaml does not carry. Kept here, not in the
#: YAML: editing an ontology while the queue worker runs strands its jobs.
EXTRA_STATE_ALIASES = {
    "andaman and nicobar": "andaman and nicobar islands",
    "andaman nicobar": "andaman and nicobar islands",
    "dadra and nagar haveli": "the dadra and nagar haveli and daman and diu",
    "daman and diu": "the dadra and nagar haveli and daman and diu",
    "dadra nagar haveli": "the dadra and nagar haveli and daman and diu",
    "jammu and kashmir ut": "jammu and kashmir",
    "j and k": "jammu and kashmir",
}
EXTRA_DISTRICT_ALIASES = {
    "bangalore urban": "bengaluru urban",
    "bangalore rural": "bengaluru rural",
    "bangalore": "bengaluru urban",
    "rupnagar": "ropar",
    "mohali": "sahibzada ajit singh nagar",
    "sas nagar": "sahibzada ajit singh nagar",
}

INDIA_BBOX = (6.0, 38.0, 68.0, 98.0)  # lat min/max, lng min/max
#: Last words that close a school's name: a 30-character name ending on one is whole.
NAME_ENDINGS = frozenset(
    {
        "school", "schools", "college", "vidyalaya", "vidyalay", "vidhyalaya", "academy", "institute",
        "university", "centre", "center", "mandir", "niketan", "bhawan", "bhavan", "sansthan",
        "kendra", "convent", "gurukul", "ashram", "pathshala", "campus", "institution", "foundation",
        "mahavidyalaya", "mahavidyalay", "sadan", "cbse", "icse", "isc",
    }
)  # fmt: skip
COUNTRY_RE = re.compile(r"^(?:country\s*:\s*)?([A-Za-z.\s]+?)\s*(?:\|.*)?$", re.I)
COORD_IN_ADDR = re.compile(r"Lat:\s*(-?\d+(?:\.\d+)?)\s*\|\s*Lng:\s*(-?\d+(?:\.\d+)?)", re.I)
COUNTRY_ALIASES = {"uae": "United Arab Emirates", "u a e": "United Arab Emirates", "foreign": "Foreign"}
BOARDS = {
    "cbse": "CBSE", "icse": "ICSE", "isc": "ISC", "cisce": "CISCE", "igcse": "IGCSE",
    "ib": "IB", "nios": "NIOS", "state": "State Board", "state board": "State Board",
}  # fmt: skip
GOVT_RE = re.compile(
    r"\b(govt|government|gov|kendriya|kv|jnv|navodaya|municipal|mcd|nagar nigam|zilla parishad|zp|"
    r"rajkiya|sarkari|army public|air force)\b",
    re.I,
)

# --- helpers ------------------------------------------------------------------------


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lng2 - lng1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def to_float(value: str) -> float | None:
    try:
        f = float(str(value).strip())
    except ValueError:
        return None
    return f if math.isfinite(f) else None


def title_case(text: str, keep_upper: set[str]) -> str:
    out = []
    for word in text.split():
        bare = re.sub(r"[^a-z]", "", word.lower())
        if bare in keep_upper:
            out.append(word.upper())
        elif word.isupper() or word.islower():
            out.append(word.capitalize() if len(word) > 3 or not word.isupper() or bare not in {"of", "and"} else word.lower())
        else:
            out.append(word)
    return " ".join(out)


class Reference:
    """India Post directory with PIN -> places, centroid, and name keying."""

    def __init__(self, path: Path, state_aliases: dict, district_aliases: dict) -> None:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df.columns = [c.strip().lower() for c in df.columns]
        rows = list(zip(df["pincode"], df["officename"], df["district"], df["statename"], strict=True))
        self.directory = PincodeDirectory(rows, state_aliases=state_aliases, district_aliases=district_aliases)
        self.pin_places: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for pin, _, district, state in rows:
            self.pin_places[pin].add((self.directory.state_key(state), self.directory.district_key(district)))
        coords = df[(df.latitude != "NA") & (df.longitude != "NA")].copy()
        # the directory has a few hand-typed values ("17°57'17.7"): unparseable means unknown
        coords["lat"] = pd.to_numeric(coords.latitude, errors="coerce")
        coords["lng"] = pd.to_numeric(coords.longitude, errors="coerce")
        coords = coords[coords.lat.between(INDIA_BBOX[0], INDIA_BBOX[1]) & coords.lng.between(INDIA_BBOX[2], INDIA_BBOX[3])]
        med = coords.groupby("pincode")[["lat", "lng"]].median()
        self.centroid = {p: (r.lat, r.lng) for p, r in med.iterrows()}
        # canonical display spelling per key
        self.state_display = {self.directory.state_key(s): s.title() for s in df["statename"].unique()}
        self.district_display = {self.directory.district_key(d): d.title() for d in df["district"].unique()}
        self.state_keys = set(self.state_display)
        self.districts_of = defaultdict(set)
        for sk, dk in {(s, d) for v in self.pin_places.values() for s, d in v}:
            self.districts_of[sk].add(dk)
        #: (state, district as the export spells it) -> the directory's spelling,
        #: learned from the export's own verified PINs by `learn_districts`.
        self.learned: dict[tuple[str, str], str] = {}

    def learn_districts(self, observed: list[tuple[str, str, str]], min_n: int = 3, min_share: float = 0.8) -> int:
        """observed: (state key, district key, PIN) for rows whose PIN the directory
        knows. A spelling the directory lacks, always seen with PINs of one directory
        district, is that district's other name ("rupnagar" -> "ropar"). Needs
        `min_n` rows and `min_share` agreement, so one typo cannot create an alias."""
        votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
        for sk, dk, pin in observed:
            if dk in self.districts_of.get(sk, ()):
                continue
            for s2, d2 in self.pin_places.get(pin, ()):
                if states_agree(sk, s2):
                    votes[(sk, dk)][d2] += 1
        for key, counter in votes.items():
            top, n = counter.most_common(1)[0]
            total = sum(counter.values())
            if total >= min_n and n / total >= min_share:
                self.learned[key] = top
        return len(self.learned)

    def same_district(self, a: str, b: str) -> bool:
        if not a or not b:
            return False
        if a == b or a in b or b in a:
            return True
        return difflib.SequenceMatcher(None, a, b).ratio() >= 0.85

    def best_district(self, state_key: str, district_key: str) -> str | None:
        if (state_key, district_key) in self.learned:
            return self.learned[(state_key, district_key)]
        if district_key in self.districts_of.get(state_key, ()):
            return district_key
        # "West Delhi" / "East Sikkim" where the directory says "WEST" / "EAST": drop the
        # state's own name (and the word "district") and look again.
        drop = set(state_key.split()) | {"district", "dist"}
        bare = " ".join(w for w in district_key.split() if w not in drop)
        if bare and bare != district_key and bare in self.districts_of.get(state_key, ()):
            return bare
        match = difflib.get_close_matches(district_key, self.districts_of.get(state_key, ()), n=1, cutoff=0.85)
        return match[0] if match else None


def states_agree(a: str, b: str) -> bool:
    return a == b or any(a in grp and b in grp for grp in STATE_EQUIV)


# --- per-row verification ------------------------------------------------------------


class Verifier:
    def __init__(self, ref: Reference, keep_upper: set[str]) -> None:
        self.ref, self.keep_upper = ref, keep_upper

    # name ---------------------------------------------------------------------------
    def name(self, raw: str) -> dict:
        flags: list[str] = []
        text = re.sub(r"\s+", " ", raw or "").strip()
        if not text:
            return {"name_clean": "", "name_status": "missing", "name_code": "", "_flags": ["name_missing"]}
        # "(MA001) ACTIVITY HIGH SCHOOL": the code goes first, or the bracket is stripped as punctuation
        code = ""
        stripped = text
        m = re.match(r"^\(([A-Z]{1,4}\d{2,5})\)\s*", stripped)
        if m:
            code, stripped = m.group(1), stripped[m.end():]
            flags.append("name_code_prefix")
        tidy = re.sub(r"^[\W_]+", "", stripped).strip()
        if tidy != stripped:
            flags.append("name_leading_punctuation")
        stripped = tidy
        # The export cut names at 30 characters. A 30-character name is suspect only if it
        # does not end on a word that closes a school's name ("... School" is complete).
        last = re.sub(r"[^a-z]", "", text.split()[-1].lower()) if text.split() else ""
        if len(text) == 30 and last not in NAME_ENDINGS and not text.endswith((".", ")")):
            flags.append("name_possibly_truncated")
        status = "fixed" if flags and flags != ["name_possibly_truncated"] else "ok"
        if "name_possibly_truncated" in flags:
            status = "suspect"
        return {"name_clean": title_case(stripped, self.keep_upper), "name_status": status, "name_code": code, "_flags": flags}

    def name_derived(self, clean: str) -> dict:
        low = normalize(clean)
        gender = ""
        if re.search(r"\b(girls?|kanya|balika|mahila|women|ladies)\b", low):
            gender = "mixed" if re.search(r"\b(boys?|balak)\b", low) else "female"
        elif re.search(r"\b(boys?|balak)\b", low):
            gender = "male"
        kind = "college" if re.search(r"\b(college|mahavidyalaya|university)\b", low) else (
            "coaching" if re.search(r"\b(coaching|classes|tuition)\b", low) else ("school" if re.search(r"\b(school|vidyalaya|vidyalay|academy|vidya)\b", low) else "")
        )
        board = next((v for k, v in BOARDS.items() if " " not in k and re.search(rf"\b{k}\b", low)), "")
        mgmt = "government" if GOVT_RE.search(low) else ""
        return {"gender_from_name": gender, "kind_from_name": kind, "board_from_name": board, "management_from_name": mgmt}

    # location -----------------------------------------------------------------------
    def country(self, raw_category: str, addr3: str) -> dict:
        src = addr3 if "country" in (addr3 or "").lower() else raw_category
        m = COUNTRY_RE.match((src or "").strip())
        text = (m.group(1) if m else (src or "")).strip().strip(".")
        key = re.sub(r"[^a-z]+", " ", text.lower()).strip()
        if not key:
            return {"country": "", "country_status": "missing", "_flags": ["country_missing"]}
        country = COUNTRY_ALIASES.get(key, text.title())
        if key == "india":
            return {"country": "India", "country_status": "ok", "_flags": []}
        return {"country": country, "country_status": "non_india", "_flags": ["non_india_record"]}

    def state_district(self, row: dict, non_india: bool = False) -> dict:
        flags: list[str] = []
        if non_india:
            parts = [p.strip() for p in (row.get("locality") or "").split(",") if p.strip()]
            return {"state_clean": "", "state_status": "non_india", "district_clean": "", "district_status": "non_india",
                    "_sk": "", "_dk": "", "_flags": [], "_raw_city": parts[0] if parts else ""}  # fmt: skip
        parts = [p.strip() for p in (row.get("locality") or "").split(",") if p.strip()]
        raw_state = row.get("state") or (parts[-1] if parts else "")
        raw_district = row.get("district") or (parts[-2] if len(parts) >= 2 else "")
        d = self.ref.directory
        sk = d.state_key(raw_state)
        if sk not in self.ref.state_keys:
            close = difflib.get_close_matches(sk, self.ref.state_keys, n=1, cutoff=0.85)
            if close:
                sk, flags = close[0], flags + ["state_spelling_fixed"]
        state_ok = sk in self.ref.state_keys
        dk = d.district_key(raw_district)
        best = self.ref.best_district(sk, dk) if state_ok else None
        if best and best != dk:
            flags.append("district_spelling_fixed")
        out = {
            "state_clean": self.ref.state_display.get(sk, raw_state.title()),
            "state_status": "ok" if state_ok else "unknown",
            # "West Delhi" matched the directory's "WEST": keep the fuller spelling the export gave
            "district_clean": raw_district.title()
            if best and set(best.split()) < set(dk.split())
            else self.ref.district_display.get(best or dk, raw_district.title()),
            "district_status": "ok" if best else ("missing" if not raw_district else "not_in_directory"),
            "_sk": sk, "_dk": best or dk, "_flags": flags,
        }  # fmt: skip
        if not state_ok:
            out["_flags"].append("state_unrecognised")
        if raw_district and not best:
            out["_flags"].append("district_not_in_directory")
        return out

    def pin(self, row: dict, loc: dict, non_india: bool) -> dict:
        flags: list[str] = []
        raw = (row.get("pin") or "").strip()
        in_addr = re.findall(r"(?<!\d)([1-9]\d{5})(?!\d)", " ".join([row.get("addr1", ""), row.get("addr2", "")]))
        sk, dk = loc["_sk"], loc["_dk"]
        pin, status, source = "", "missing", ""
        if non_india:
            return {"pin_clean": "", "pin_raw": "", "pin_suggested": "", "pin_status": "not_applicable", "pin_source": "", "_flags": []}
        if re.fullmatch(r"[1-9]\d{5}", raw):
            pin, source = raw, "column"
        elif raw and raw not in {"000000", "0"}:
            flags.append("pin_malformed")
            status = "invalid"
        if not pin and in_addr:
            pin, source = in_addr[-1], "address"
            flags.append("pin_taken_from_address")
        elif pin and in_addr and in_addr[-1] != pin:
            flags.append("pin_conflicts_with_address")
        suggested = ""
        if pin:
            places = self.ref.pin_places.get(pin)
            if not places:
                status, flags = "unknown_pin", flags + ["pin_not_in_directory"]
            elif loc["state_status"] == "ok" and not any(states_agree(sk, s) for s, _ in places):
                status, flags = "state_mismatch", flags + ["pin_state_mismatch"]
            elif dk and not any(states_agree(sk, s) and self.ref.same_district(dk, d_) for s, d_ in places):
                status, flags = "district_mismatch", flags + ["pin_district_mismatch"]
            else:
                status = "verified"
        elif loc["state_status"] == "ok":
            address = " ".join([row.get("addr1", ""), row.get("addr2", ""), row.get("locality", "")])
            suggested = self.ref.directory.infer(address=address, district=dk, state=loc["state_clean"]) or ""
            if suggested:
                status, source = "suggested_unverified", "directory_inference"
                flags.append("pin_suggested_offline")
        # `pin_clean` holds only a PIN the directory confirms for this state and district. Anything
        # else is kept apart (`pin_raw`, `pin_suggested`) so it cannot pass for a verified value.
        return {
            "pin_clean": pin if status == "verified" else "",
            "pin_raw": pin if status != "verified" else "",
            "pin_suggested": suggested,
            "pin_status": status,
            "pin_source": source,
            "_flags": flags,
        }

    def coords(self, row: dict, pin: str, non_india: bool) -> dict:
        flags: list[str] = []
        lat, lng = to_float(row.get("lat", "")), to_float(row.get("lng", ""))
        source = (row.get("geo_source") or "").strip().upper()
        if lat is None or lng is None:
            m = COORD_IN_ADDR.search(row.get("addr3", ""))
            if m:
                lat, lng, source = float(m.group(1)), float(m.group(2)), source or "ADDRESS_FIELD"
                flags.append("coords_parsed_from_address")
        out = {"lat_clean": "", "lng_clean": "", "geo_source_clean": source, "geo_status": "missing",
               "geo_km_from_pin": "", "lat_pin_centroid": "", "lng_pin_centroid": "", "_flags": flags}  # fmt: skip
        cen = self.ref.centroid.get(pin) if pin else None
        if lat is not None and lng is not None:
            inb = INDIA_BBOX[0] <= lat <= INDIA_BBOX[1] and INDIA_BBOX[2] <= lng <= INDIA_BBOX[3]
            swapped = INDIA_BBOX[0] <= lng <= INDIA_BBOX[1] and INDIA_BBOX[2] <= lat <= INDIA_BBOX[3]
            if not inb and swapped:
                lat, lng, inb = lng, lat, True
                flags.append("lat_lng_swapped")
            if not inb:
                out["geo_status"] = "foreign_or_invalid" if not non_india else "non_india_ok"
                flags.append("coords_outside_india" if not non_india else "coords_non_india")
                return out
            out.update(lat_clean=round(lat, 6), lng_clean=round(lng, 6))
            if cen and not non_india:
                km = haversine_km(lat, lng, *cen)
                out["geo_km_from_pin"] = round(km, 1)
                out["geo_status"] = "verified" if km <= 25 else ("plausible" if km <= 100 else "far_from_pin")
                if km > 100:
                    flags.append("coords_far_from_pin")
            else:
                out["geo_status"] = "unchecked"
            return out
        if cen and not non_india:
            out.update(lat_pin_centroid=round(cen[0], 5), lng_pin_centroid=round(cen[1], 5))
            flags.append("coords_missing_pin_centroid_offered")
        return out

    # contact / identifiers -------------------------------------------------------------
    @staticmethod
    def phone(raw: str) -> dict:
        text = (raw or "").strip()
        if not text:
            return {"clean": "", "status": "missing", "type": ""}
        if re.search(r"\d[eE][+-]?\d", text):  # "1.1E+08": spreadsheet coercion destroyed it
            return {"clean": "", "status": "corrupted_scientific", "type": ""}
        digits = re.sub(r"\D", "", text)
        if digits.startswith("91") and len(digits) in (12, 13):
            digits = digits[2:]
        digits = digits.lstrip("0") if len(digits) == 11 and digits.startswith("0") else digits
        if len(set(digits)) <= 2 and len(digits) >= 6 or digits in {"1234567890", "1234657895", "0123456789", "9876543210"}:
            return {"clean": "", "status": "placeholder", "type": ""}
        if re.fullmatch(r"[6-9]\d{9}", digits):
            return {"clean": f"+91{digits}", "status": "valid", "type": "mobile"}
        if re.fullmatch(r"[2-5]\d{9}", digits) or re.fullmatch(r"[1-9]\d{7,9}", digits):
            return {"clean": digits, "status": "landline_unverified", "type": "landline"}
        return {"clean": "", "status": "invalid", "type": ""}

    @staticmethod
    def email(raw: str) -> dict:
        text = (raw or "").strip().lower()
        if not text:
            return {"clean": "", "status": "missing"}
        if not re.fullmatch(r"[a-z0-9._%+\-]+@[a-z0-9\-]+(\.[a-z0-9\-]+)*\.[a-z]{2,}", text):
            return {"clean": "", "status": "invalid"} if not text.endswith(".local") else {"clean": "", "status": "synthetic"}
        if text.endswith("@imported.local") or text.endswith(".local") or text.split("@")[1] in {"example.com", "test.com", "xyz.com"}:
            return {"clean": "", "status": "synthetic"}
        return {"clean": text, "status": "syntax_ok"}

    @staticmethod
    def website(raw: str) -> dict:
        text = (raw or "").strip()
        if not text:
            return {"clean": "", "status": "missing"}
        url = re.sub(r"^https?://", "", text, flags=re.I).rstrip("/").lower()
        if re.fullmatch(r"(?:www\.)?[a-z0-9\-]+(?:\.[a-z0-9\-]+)*\.[a-z]{2,}(?:/\S*)?", url):
            return {"clean": "https://" + url, "status": "syntax_ok"}
        return {"clean": "", "status": "invalid"}

    def udise(self, raw: str, sk: str, non_india: bool) -> dict:
        digits = re.sub(r"\D", "", raw or "")
        if not digits:
            return {"udise_clean": "", "udise_status": "missing", "_flags": []}
        if len(digits) != 11:
            return {"udise_clean": "", "udise_status": "invalid_length", "_flags": ["udise_invalid_length"]}
        owner = UDISE_STATE.get(digits[:2])
        if not owner:
            return {"udise_clean": "", "udise_status": "invalid_state_code", "_flags": ["udise_invalid_state_code"]}
        if sk and not non_india and not states_agree(owner, sk):
            return {"udise_clean": digits, "udise_status": "state_mismatch", "_flags": ["udise_state_mismatch"]}
        return {"udise_clean": digits, "udise_status": "ok", "_flags": []}

    @staticmethod
    def academic(row: dict) -> dict:
        cur = (row.get("curriculum") or "").strip()
        board = BOARDS.get(re.sub(r"[^a-z ]", "", cur.lower()).strip(), "")
        hc = re.search(r"\d{1,2}", row.get("highest_class") or "")
        return {
            "board_clean": board,
            "board_status": "missing" if not cur else ("ok" if board else "unrecognised"),
            "highest_class_clean": int(hc.group()) if hc and 1 <= int(hc.group()) <= 12 else "",
            "medium_clean": (row.get("medium") or "").strip().title(),
            "streams_clean": "; ".join(s.strip().title() for s in re.split(r"[,/;|]", row.get("streams") or "") if s.strip()),
        }

    @staticmethod
    def timestamps(row: dict) -> dict:
        created, updated = (pd.to_datetime(row.get(k) or None, errors="coerce", utc=True) for k in ("created", "updated"))
        flags = []
        if pd.notna(created) and pd.notna(updated) and updated < created:
            flags.append("updated_before_created")
        return {"created_clean": created.isoformat() if pd.notna(created) else "", "updated_clean": updated.isoformat() if pd.notna(updated) else "", "_flags": flags}


# --- main ----------------------------------------------------------------------------

#: How much each field's verification counts toward the 0-100 quality score.
WEIGHTS = {"name": 2, "location": 2, "pin": 3, "geo": 2, "phone": 2, "email": 1, "udise": 2, "website": 1, "board": 1}


def run(args: argparse.Namespace) -> None:
    india = yaml.safe_load((ROOT / "ontologies" / "_india.yaml").read_text(encoding="utf8"))
    edu = yaml.safe_load((ROOT / "ontologies" / "education.yaml").read_text(encoding="utf8"))
    ref = Reference(
        Path(args.pincodes),
        {**india.get("state_aliases", {}), **EXTRA_STATE_ALIASES},
        {**india.get("district_aliases", {}), **EXTRA_DISTRICT_ALIASES},
    )
    ver = Verifier(ref, {k.lower() for k in edu["display"]["keep_upper"]})

    src = pd.read_csv(args.input, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    mapping, unmapped = resolve_columns(list(src.columns))
    print(f"{len(src):,} rows; mapped {len(mapping)} fields; unmapped columns: {unmapped or 'none'}")
    rows = [{f: str(r.get(c, "")) for f, c in mapping.items()} for r in src.to_dict("records")]

    # Pass 1: learn the export's district spellings from rows whose PIN the directory knows.
    observed = []
    for row in rows:
        if re.fullmatch(r"[1-9]\d{5}", (row.get("pin") or "").strip()) and not ver.country(row.get("category", ""), row.get("addr3", ""))["country_status"] == "non_india":
            loc0 = ver.state_district(row)
            if loc0["state_status"] == "ok":
                observed.append((loc0["_sk"], ref.directory.district_key(loc0["_dk"]), row["pin"].strip()))
    print(f"learned {ref.learn_districts(observed)} district aliases from {len(observed):,} rows with a PIN")

    out: list[dict] = []
    for row in rows:
        flags: list[str] = []
        rec: dict = {"id": row.get("id", ""), "source_ref_no": row.get("affiliation", "")}
        for part in (
            n := ver.name(row.get("name", "")),
            c := ver.country(row.get("category", ""), row.get("addr3", "")),
        ):
            flags += part.pop("_flags", [])
            rec.update({k: v for k, v in part.items() if not k.startswith("_")})
        non_india = c["country_status"] == "non_india"
        loc = ver.state_district(row, non_india)
        flags += loc.pop("_flags")
        loc.pop("_raw_city", None)
        rec.update({k: v for k, v in loc.items() if not k.startswith("_")})
        rec.update(ver.name_derived(n["name_clean"]))
        p = ver.pin(row, loc, non_india)
        flags += p.pop("_flags")
        rec.update(p)
        g = ver.coords(row, p["pin_clean"], non_india)
        flags += g.pop("_flags")
        rec.update(g)
        for tag, key in (("phone1", "phone1"), ("phone2", "phone2"), ("fax", "fax")):
            r = ver.phone(row.get(key, ""))
            rec.update({f"{tag}_clean": r["clean"], f"{tag}_status": r["status"], f"{tag}_type": r["type"]})
            if r["status"] in {"placeholder", "invalid", "corrupted_scientific"}:
                flags.append(f"{tag}_{r['status']}")
        for tag in ("email1", "email2"):
            r = ver.email(row.get(tag, ""))
            rec.update({f"{tag}_clean": r["clean"], f"{tag}_status": r["status"]})
            if r["status"] in {"synthetic", "invalid"}:
                flags.append(f"{tag}_{r['status']}")
        w = ver.website(row.get("website", ""))
        rec.update(website_clean=w["clean"], website_status=w["status"])
        if w["status"] == "invalid":
            flags.append("website_invalid")
        u = ver.udise(row.get("udise", ""), loc["_sk"], non_india)
        flags += u.pop("_flags")
        rec.update(u)
        rec.update(ver.academic(row))
        if rec["board_clean"] == "" and rec["board_from_name"]:
            rec["board_clean"], rec["board_status"] = rec["board_from_name"], "from_name"
        t = ver.timestamps(row)
        flags += t.pop("_flags")
        rec.update(t)
        rec["place_id"] = row.get("place_id", "")
        if rec["place_id"] and rec["geo_status"] == "missing":
            flags.append("place_id_without_coords")
        # status of fields the source never carries is "n/a", not a failure
        ok = {
            "name": rec["name_status"] in {"ok", "fixed"},
            "location": rec["state_status"] == "ok" and rec["district_status"] == "ok",
            "pin": rec["pin_status"] in {"verified", "not_applicable"},
            "geo": rec["geo_status"] in {"verified", "plausible", "non_india_ok", "unchecked"},
            "phone": rec["phone1_status"] == "valid" or rec["phone2_status"] == "valid",
            "email": rec["email1_status"] == "syntax_ok" or rec["email2_status"] == "syntax_ok",
            "udise": rec["udise_status"] == "ok",
            "website": rec["website_status"] == "syntax_ok",
            "board": rec["board_status"] in {"ok", "from_name"},
        }
        rec["quality_score"] = round(100 * sum(WEIGHTS[k] for k, v in ok.items() if v) / sum(WEIGHTS.values()))
        rec["flags"] = "|".join(dict.fromkeys(flags))
        out.append(rec)

    df = pd.DataFrame(out)
    # cross-record checks
    key = df["name_clean"].str.lower() + "|" + df["pin_clean"]
    df["dup_group_size"] = key.map(key.value_counts()).where(df["pin_clean"] != "", 1)
    coord_key = df["lat_clean"].astype(str) + "," + df["lng_clean"].astype(str)
    shared = coord_key.map(coord_key.value_counts()).where(df["lat_clean"] != "", 0)
    df["shared_coords_count"] = shared
    df.loc[shared >= 5, "flags"] = df.loc[shared >= 5, "flags"].str.cat(pd.Series("shared_coordinates", index=df.index[shared >= 5]), sep="|").str.lstrip("|")
    for col in ("email1_clean", "phone1_clean"):
        v = df[col].where(df[col] != "")
        multi = v.map(v.value_counts()) >= 3
        df.loc[multi, "flags"] = df.loc[multi, "flags"].str.cat(pd.Series(f"{col.split('_')[0]}_shared_by_3plus", index=df.index[multi]), sep="|").str.lstrip("|")
    junk = df["name_clean"].str.lower().isin(df.loc[df["udise_status"] == "invalid_length", "name_clean"].str.lower())
    df.loc[junk, "flags"] = df.loc[junk, "flags"].str.cat(pd.Series("looks_like_test_data", index=df.index[junk]), sep="|").str.lstrip("|")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "schools_verified.csv", index=False, encoding="utf-8-sig")

    # field report: one row per (field, status)
    report = []
    for col in [c for c in df.columns if c.endswith("_status")]:
        for status, n in df[col].value_counts().items():
            report.append({"field": col.removesuffix("_status"), "status": status, "rows": int(n), "pct": round(100 * n / len(df), 2)})
    pd.DataFrame(report).to_csv(outdir / "field_report.csv", index=False)
    flag_counts = Counter(f for s in df["flags"] for f in s.split("|") if f)
    pd.DataFrame(flag_counts.most_common(), columns=["flag", "rows"]).to_csv(outdir / "flag_counts.csv", index=False)

    if args.backtest_pin:
        # Hide the PIN of rows the directory confirmed and see whether inference gets it back:
        # the measured reliability of `pin_suggested` on this export.
        hit = wrong = abstain = 0
        for row, rec in zip(rows, out, strict=True):
            if rec["pin_status"] != "verified" or rec["pin_source"] != "column":
                continue
            where = " ".join([row.get("addr1", ""), row.get("addr2", ""), row.get("locality", "")])
            guess = ref.directory.infer(
                address=re.sub(r"(?<!\d)[1-9]\d{5}(?!\d)", " ", where),
                district=ref.best_district(sk := ref.directory.state_key(rec["state_clean"]), dk := ref.directory.district_key(rec["district_clean"])) or dk,
                state=rec["state_clean"],
            )
            hit += guess == rec["pin_clean"]
            wrong += guess not in (None, rec["pin_clean"])
            abstain += guess is None
        n = hit + wrong + abstain
        line = (
            f"PIN inference back-test on {n:,} verified rows: correct {hit / n:.1%}, "
            f"wrong {wrong / n:.1%}, abstained {abstain / n:.1%}; "
            f"precision when it answers {hit / max(hit + wrong, 1):.1%}"
        )
        (outdir / "pin_backtest.txt").write_text(line + "\n", encoding="utf8")
        print(line)

    print(f"\nwrote {outdir}/schools_verified.csv ({len(df):,} rows, {len(df.columns)} columns)")
    print(pd.DataFrame(report).to_string(index=False))
    print("\nTop flags:")
    for f, n in flag_counts.most_common(25):
        print(f"  {n:7,}  {f}")
    print(f"\nquality_score: mean {df.quality_score.mean():.1f}, median {df.quality_score.median():.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--pincodes", required=True, help="India Post directory CSV (pincode, officename, district, statename, latitude, longitude)")
    ap.add_argument("--outdir", default="verified")
    ap.add_argument("--backtest-pin", action="store_true", help="measure how often offline PIN inference is right")
    run(ap.parse_args())
