#!/usr/bin/env python
"""Check a school export against Google Places, by the place ID each row carries.

    set GOOGLE_PLACES_API_KEY=...            (PowerShell: $env:GOOGLE_PLACES_API_KEY = "...")
    python scripts/verify_places.py \
        --input  var/uploads/<tenant>/<job>.csv \
        --outdir verified_places/ \
        --max-calls 20000

One Place Details call per *distinct* place ID (rows that share an ID share the
call). For each row it compares Google's name, address, PIN and business status
with the row's own and writes a verdict. Nothing is written back to the source.

This spends money the moment it runs, so it is guarded:

* `--max-calls` is required and has no default. It bounds the calls of one run.
  Also set a daily quota and a budget alert on the key in the Google Cloud
  console; that is the limit Google enforces, this one is only ours.
* The key comes from the environment, never an argument (no shell history).
* A rejected key (401/403: wrong key, API not enabled, billing off) stops the
  run on the first call; a streak of failures stops it too.
* Every answer is appended to `--cache` as it arrives, so a stopped or crashed
  run resumes without paying twice. Google's terms limit how long place data
  other than the ID may be kept: delete the cache after handover.
* `--fields basic` asks only for name, address, location and status. Phone and
  website (`--fields contact`) sit in a dearer Google tier: opt in on purpose.

Note: the Latitude/Longitude in a Google-sourced export were themselves taken
from Google, so they agree with it by construction. The checks that can
disagree are the name, the PIN in Google's address, and the business status.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import os
import random
import re
import socket
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.enrichment.pincode_infer import normalize  # noqa: E402

PLACES_URL = "https://places.googleapis.com/v1/places/{place_id}"
MASKS = {
    "basic": "id,displayName,formattedAddress,location,businessStatus",
    "contact": "id,displayName,formattedAddress,location,businessStatus,nationalPhoneNumber,websiteUri",
}
CLOSED = {"CLOSED_PERMANENTLY", "CLOSED_TEMPORARILY"}

#: Name-match thresholds. Heuristics: the score distribution is in the summary.
NAME_MATCH = 0.85
NAME_PARTIAL = 0.60

_PIN = re.compile(r"(?<!\d)[1-9]\d{5}(?!\d)")
_CODE = re.compile(r"^[a-z]{1,3}\d{2,5}$")  # "(MA001)" style codes in names
_ABORT_AFTER_FAILURES = 20

HEADERS = {
    "id": ("schooliddonotchange", "schoolid", "id"),
    "name": ("institutename", "schoolname", "name"),
    "place_id": ("googleplaceid", "placeid"),
    "pin": ("pincode", "pin"),
    "lat": ("latitude", "lat"),
    "lng": ("longitude", "lng"),
}


class PlacesAbort(Exception):
    """The run cannot usefully continue (bad key, quota, a streak of failures).

    Not a RuntimeError: `fetch_with_retry` retries those, and retrying a
    rejected key is a billable call that cannot succeed."""


class PlacesNetworkDown(PlacesAbort):
    """The calls failed to connect (DNS, no route). They never reached Google, so
    nothing was billed, and waiting for the network is the right response."""


# --- comparison -----------------------------------------------------------------------


#: Abbreviations seen in the export ("GOVT. BOYS SR. SEC. SCHOOL") that Google spells out.
_ABBREVIATIONS = {
    "govt": "government", "gov": "government", "sr": "senior", "sec": "secondary", "jr": "junior",
    "pub": "public", "sch": "school", "hr": "higher", "mdl": "model", "inst": "institute",
}  # fmt: skip


#: Words that most schools share; they say what kind of school, not which one.
_GENERIC = frozenset(
    "school public senior secondary government junior higher high model the of and".split()
)


def _tokens(name: str) -> list[str]:
    words = [_ABBREVIATIONS.get(t, t) for t in normalize(name).split() if not _CODE.match(t)]
    merged: list[str] = []
    initials: list[str] = []
    for word in [*words, ""]:  # "D.A.V" is d a v after normalize: join a run of initials
        if len(word) == 1 and word.isalpha():
            initials.append(word)
            continue
        if initials:
            merged.append("".join(initials))
            initials = []
        if word:
            merged.append(word)
    return merged


def name_score(ours: str, theirs: str) -> float:
    """0..1. A prefix or subset counts as a match, because exports truncate names
    at 30 characters and Google appends the locality ("... Rohini")."""
    a, b = _tokens(ours), _tokens(theirs)
    if not a or not b:
        return 0.0
    ratio = _similarity(a, b, min_subset=2)
    # "DAV Public School" vs "DPS Public School" share every word but the one that
    # names the school: judge the distinguishing words on their own and take the lower.
    da, db = [t for t in a if t not in _GENERIC], [t for t in b if t not in _GENERIC]
    if da and db:
        ratio = min(ratio, _similarity(da, db, min_subset=1))
    return round(ratio, 3)


def _similarity(a: list[str], b: list[str], min_subset: int) -> float:
    """Edit similarity, or (when both sides have `min_subset` words) the share of the
    shorter side found in the longer; a lone generic word must not match everything."""
    ratio = difflib.SequenceMatcher(None, " ".join(a), " ".join(b)).ratio()
    if min(len(a), len(b)) >= min_subset:
        sa, sb = set(a), set(b)
        ratio = max(ratio, len(sa & sb) / min(len(sa), len(sb)))
    return ratio


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lng2 - lng1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def _float(value: str) -> float | None:
    try:
        f = float(str(value).strip())
    except ValueError:
        return None
    return f if math.isfinite(f) else None


def judge(row: dict[str, str], place: dict | None, shared: int) -> dict[str, object]:
    """One row's verdict from Google's answer (`place`; None = not fetched)."""
    out: dict[str, object] = {"place_shared_n": shared}
    if not row["place_id"]:
        return {**out, "places_verdict": "no_place_id"}
    if place is None:
        return {**out, "places_verdict": "not_fetched"}
    if place.get("_error"):
        return {**out, "places_verdict": place["_error"]}  # not_found / invalid_place_id

    google_name = (place.get("displayName") or {}).get("text", "")
    address = place.get("formattedAddress", "")
    status = place.get("businessStatus", "")
    where = place.get("location") or {}
    score = name_score(row["name"], google_name)
    google_pin = (_PIN.findall(address) or [""])[-1]
    our_pin = row["pin"].strip() if _PIN.fullmatch(row["pin"].strip()) else ""

    if not google_pin or not our_pin:
        pin_check = "uncorroborated"
    else:
        pin_check = "match" if google_pin == our_pin else "differs"

    dist = None
    lat, lng = _float(row["lat"]), _float(row["lng"])
    if None not in (lat, lng, where.get("latitude"), where.get("longitude")):
        dist = round(haversine_km(lat, lng, where["latitude"], where["longitude"]), 3)

    if status in CLOSED:
        verdict = "closed"
    elif score < NAME_PARTIAL:
        verdict = "name_mismatch"
    elif score >= NAME_MATCH and pin_check == "match":
        verdict = "confirmed"
    elif score >= NAME_MATCH and pin_check == "uncorroborated":
        verdict = "name_only"
    else:
        verdict = "review"

    out.update(
        places_verdict=verdict,
        name_score=score,
        google_name=google_name,
        google_address=address,
        google_status=status,
        google_pin=google_pin,
        pin_check=pin_check,
        pin_from_google=google_pin if (not our_pin and google_pin) else "",
        dist_km=dist if dist is not None else "",
        google_phone=place.get("nationalPhoneNumber", ""),
        google_website=place.get("websiteUri", ""),
    )
    return out


# --- fetching -------------------------------------------------------------------------


class RateLimiter:
    """At most `qps` calls a second across all threads."""

    def __init__(self, qps: float) -> None:
        self._gap = 1.0 / qps
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self._gap
        if slot > now:
            time.sleep(slot - now)


def fetch_one(client: httpx.Client, place_id: str, key: str, mask: str, limiter: RateLimiter) -> dict:
    """Place Details for one ID. Returns the place JSON, or `{"_error": ...}` for a
    definite "no such place". Raises PlacesAbort for a key/quota problem and
    httpx.HTTPError / RuntimeError for transient ones (the caller retries)."""
    limiter.wait()
    response = client.get(
        PLACES_URL.format(place_id=place_id),
        headers={"X-Goog-Api-Key": key, "X-Goog-FieldMask": mask},
        params={"languageCode": "en", "regionCode": "IN"},
    )
    if response.status_code == 200:
        return response.json()
    detail = _error_text(response)
    if response.status_code == 404:
        return {"_error": "not_found"}
    if response.status_code == 400:
        return {"_error": "invalid_place_id"}
    if response.status_code in (401, 403):
        raise PlacesAbort(f"Google rejected the key (HTTP {response.status_code}): {detail}")
    raise RuntimeError(f"HTTP {response.status_code}: {detail}")


def _error_text(response: httpx.Response) -> str:
    try:
        return str(response.json()["error"]["message"])[:300]
    except (ValueError, KeyError, TypeError):
        return response.text[:300]


def fetch_with_retry(client, place_id, key, mask, limiter, tries: int = 4) -> dict:
    for attempt in range(tries):
        try:
            return fetch_one(client, place_id, key, mask, limiter)
        except (httpx.HTTPError, RuntimeError):
            if attempt == tries - 1:
                raise
            time.sleep(2**attempt + random.random())
    raise AssertionError("unreachable")


def load_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                cache[entry["place_id"]] = entry["place"]
    return cache


def fetch_all(
    place_ids: list[str],
    cache: dict[str, dict],
    cache_path: Path,
    key: str,
    mask: str,
    max_calls: int,
    qps: float,
    workers: int,
) -> Counter:
    todo = [p for p in place_ids if p not in cache][:max_calls]
    stats: Counter = Counter()
    if not todo:
        return stats
    limiter = RateLimiter(qps)
    lock = threading.Lock()
    streak = 0
    started = time.monotonic()
    with httpx.Client(timeout=20.0) as client, cache_path.open("a", encoding="utf-8") as sink:

        def work(place_id: str) -> tuple[str, dict]:
            return place_id, fetch_with_retry(client, place_id, key, mask, limiter)

        # The first call alone: a wrong key or a disabled API fails here, once,
        # not on `workers` calls at the same time.
        try:
            first_id, first = work(todo[0])
        except PlacesAbort:
            raise
        except Exception as exc:
            stop = PlacesNetworkDown if isinstance(exc, httpx.TransportError) else PlacesAbort
            raise stop(f"the first call failed ({exc}); no network, or Google is down") from exc
        _record(first_id, first, cache, sink, stats)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(work, p): p for p in todo[1:]}
            try:
                for done in as_completed(futures):
                    try:
                        place_id, place = done.result()
                    except PlacesAbort:
                        raise
                    except Exception as exc:  # transient, retries exhausted
                        stats["failed"] += 1
                        streak += 1
                        if streak >= _ABORT_AFTER_FAILURES:
                            stop = PlacesNetworkDown if isinstance(exc, httpx.TransportError) else PlacesAbort
                            raise stop(f"{streak} calls in a row failed; last: {exc}") from exc
                        continue
                    streak = 0
                    with lock:
                        _record(place_id, place, cache, sink, stats)
                    if stats["fetched"] % 500 == 0:
                        rate = stats["fetched"] / max(time.monotonic() - started, 1e-9)
                        print(f"  {stats['fetched']}/{len(todo)} fetched ({rate:.1f}/s)", flush=True)
            except PlacesAbort:
                for future in futures:
                    future.cancel()
                raise
    return stats


def _record(place_id: str, place: dict, cache: dict, sink, stats: Counter) -> None:
    cache[place_id] = place
    stats["fetched"] += 1
    stats[place.get("_error") or "ok"] += 1
    entry = {"place_id": place_id, "place": place, "at": datetime.now(UTC).isoformat()}
    sink.write(json.dumps(entry, ensure_ascii=False) + "\n")
    sink.flush()


# --- run ------------------------------------------------------------------------------


def wait_for_network(
    timeout_s: float,
    host: str = "places.googleapis.com",
    probe=socket.getaddrinfo,
    sleep=time.sleep,
    clock=time.monotonic,
) -> bool:
    """Block until `host` resolves, up to `timeout_s`. True if the network came back."""
    deadline = clock() + timeout_s
    while clock() < deadline:
        try:
            probe(host, 443)
        except OSError:
            sleep(10)
            continue
        sleep(3)  # a link that has just returned is often not yet steady
        return True
    return False


def read_key() -> str:
    """The environment first, then the project's `.env`. Only this one line of
    `.env` is read, and the key is never printed."""
    key = os.environ.get("GOOGLE_PLACES_API_KEY", "").strip()
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "GOOGLE_PLACES_API_KEY":
                key = value.strip().strip("\"'")
    return key


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        index = {re.sub(r"[^a-z0-9]", "", h.lower()): h for h in reader.fieldnames or []}
        pick = {}
        for field, aliases in HEADERS.items():
            pick[field] = next((index[a] for a in aliases if a in index), None)
        missing = [f for f in ("name", "place_id") if not pick[f]]
        if missing:
            raise SystemExit(f"{path.name}: no column found for {missing}; headers: {reader.fieldnames}")
        return [{f: (row.get(h) or "").strip() if h else "" for f, h in pick.items()} for row in reader]


def run(args: argparse.Namespace) -> None:
    rows = read_rows(Path(args.input))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache or outdir / "places_cache.jsonl")
    cache = load_cache(cache_path)

    shared: Counter = Counter(r["place_id"] for r in rows if r["place_id"])
    place_ids = sorted(shared)
    pending = [p for p in place_ids if p not in cache]
    will_call = min(len(pending), args.max_calls)
    print(
        f"{len(rows)} rows; {sum(shared.values())} with a place ID; {len(place_ids)} distinct IDs; "
        f"{len(cache)} cached; {len(pending)} to fetch; this run will make {will_call} call(s) "
        f"(cap {args.max_calls})."
    )
    if args.dry_run:
        return

    key = read_key()
    if not key and will_call:
        raise SystemExit("Set GOOGLE_PLACES_API_KEY in the environment or .env (it is not read from arguments).")
    if args.limit:
        place_ids = place_ids[: args.limit]

    aborted = ""
    cached_at_start = len(cache)
    while True:
        # The cap is for the whole run, across the waits for the network below.
        left = max(args.max_calls - (len(cache) - cached_at_start), 0)
        try:
            stats = fetch_all(place_ids, cache, cache_path, key, MASKS[args.fields], left, args.qps, args.workers)
            print("fetch:", dict(stats) or "nothing new")
            break
        except PlacesNetworkDown as exc:
            print(f"Network down ({exc}); waiting up to {args.network_wait:g} min for it to return.", flush=True)
            if not wait_for_network(args.network_wait * 60):
                aborted = "the network did not come back in time"
                print(f"STOPPED: {aborted}\nWhat was fetched is cached; run again to resume.")
                break
            print("Network is back; resuming.", flush=True)
        except PlacesAbort as exc:
            aborted = str(exc)
            print(f"STOPPED: {aborted}\nWhat was fetched is cached; fix the cause and run again to resume.")
            break

    results = [judge(r, cache.get(r["place_id"]), shared[r["place_id"]]) for r in rows]
    _write_outputs(outdir, rows, results, args.fields == "contact")
    if aborted:
        raise SystemExit(1)


COLUMNS = [
    "school_id", "name", "place_id", "places_verdict", "name_score", "google_name", "google_address",
    "google_status", "google_pin", "pin_check", "pin_from_google", "dist_km", "place_shared_n",
]  # fmt: skip


def _write_outputs(outdir: Path, rows, results, contact: bool) -> None:
    columns = COLUMNS + (["google_phone", "google_website"] if contact else [])
    with (outdir / "places_verified.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row, result in zip(rows, results, strict=True):
            writer.writerow({"school_id": row["id"], "name": row["name"], "place_id": row["place_id"], **result})

    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if row["place_id"]:
            groups[row["place_id"]].append(i)
    with (outdir / "shared_place_groups.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["place_id", "rows", "rows_matching_google_name", "google_name", "school_ids", "names"])
        for place_id, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if len(members) < 2:
                continue
            matching = sum(1 for i in members if (results[i].get("name_score") or 0) >= NAME_MATCH)
            writer.writerow([
                place_id, len(members), matching, results[members[0]].get("google_name", ""),
                " | ".join(rows[i]["id"] for i in members), " | ".join(rows[i]["name"] for i in members),
            ])  # fmt: skip

    verdicts = Counter(str(r["places_verdict"]) for r in results)
    lines = [f"{v:>16} {n:>7} {100 * n / len(rows):6.2f}%" for v, n in verdicts.most_common()]
    scores = [r["name_score"] for r in results if isinstance(r.get("name_score"), float)]
    if scores:
        buckets = Counter(min(int(s * 10), 9) / 10 for s in scores)
        lines += ["", "name_score distribution (lower edge of each 0.1 bucket):"]
        lines += [f"  {b:.1f}  {buckets[b]}" for b in sorted(buckets)]
    (outdir / "places_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--max-calls", type=int, required=True, help="most Google calls this run may make")
    p.add_argument("--cache", help="response cache (default: <outdir>/places_cache.jsonl)")
    p.add_argument("--fields", choices=sorted(MASKS), default="basic")
    p.add_argument("--qps", type=float, default=8.0, help="calls per second (default 8)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--network-wait", type=float, default=30.0, help="minutes to wait out a network outage")
    p.add_argument("--limit", type=int, help="only the first N distinct IDs (a smoke test)")
    p.add_argument("--dry-run", action="store_true", help="count the calls; contact nobody")
    args = p.parse_args()
    if args.max_calls < 0:
        p.error("--max-calls must be >= 0")
    run(args)


if __name__ == "__main__":
    main()
