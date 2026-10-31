"""scripts/verify_places.py: the verdicts, and the guards on spending.

No network: Google is a `httpx.MockTransport`. What matters most is negative:
a rejected key stops the run on one call, the cap is never exceeded, and a
second run pays for nothing it already has.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_places.py"
_spec = importlib.util.spec_from_file_location("verify_places", _SCRIPT)
assert _spec and _spec.loader
vp = importlib.util.module_from_spec(_spec)
sys.modules["verify_places"] = vp
_spec.loader.exec_module(vp)  # scripts/ is not a package

_REAL_CLIENT = httpx.Client


def row(**over: str) -> dict[str, str]:
    base = {"id": "s1", "name": "Delhi Public School", "place_id": "ChIJa", "pin": "110085", "lat": "", "lng": ""}
    return {**base, **over}


def place(name: str = "Delhi Public School, Rohini", pin: str = "110085", status: str = "OPERATIONAL") -> dict:
    return {
        "id": "ChIJa",
        "displayName": {"text": name},
        "formattedAddress": f"Sector 24, Rohini, New Delhi, Delhi {pin}, India",
        "location": {"latitude": 28.7, "longitude": 77.1},
        "businessStatus": status,
    }


# --- verdicts ---------------------------------------------------------------------------


def test_truncated_and_localised_names_still_match():
    assert vp.name_score("Delhi Public School", "Delhi Public School, Rohini") >= vp.NAME_MATCH
    assert vp.name_score("(MA001) Kendriya Vidyalaya No. 4", "Kendriya Vidyalaya No 4") >= vp.NAME_MATCH
    assert vp.name_score("Delhi Public School", "Sunrise Montessori Academy") < vp.NAME_PARTIAL


def test_abbreviations_and_initials_are_the_same_school():
    assert vp.name_score("GOVT. BOYS SR. SEC. SCHOOL", "Government Boys Senior Secondary School") >= vp.NAME_MATCH
    assert vp.name_score("DAV SR SEC SCHOOL", "D.A.V Senior Secondary School") >= vp.NAME_MATCH
    assert vp.name_score("DAV Public School", "DPS Public School") < vp.NAME_MATCH  # not a loose initial match


def test_a_single_generic_word_is_not_a_subset_match():
    assert vp.name_score("School", "Anything Else School") < vp.NAME_MATCH


@pytest.mark.parametrize(
    ("our", "google", "shared", "verdict"),
    [
        (row(), place(), 1, "confirmed"),
        (row(pin=""), place(), 1, "name_only"),
        (row(), place(pin="834001"), 1, "review"),  # same name, another city's PIN
        (row(), place("Sunrise Montessori Academy"), 1, "name_mismatch"),
        (row(), place(status="CLOSED_PERMANENTLY"), 1, "closed"),
        (row(), {"_error": "not_found"}, 1, "not_found"),
        (row(place_id=""), None, 0, "no_place_id"),
        (row(), None, 1, "not_fetched"),
    ],
)
def test_verdicts(our, google, shared, verdict):
    assert vp.judge(our, google, shared)["places_verdict"] == verdict


def test_pin_is_offered_from_google_only_when_the_row_has_none():
    assert vp.judge(row(pin=""), place(), 1)["pin_from_google"] == "110085"
    assert vp.judge(row(), place(), 1)["pin_from_google"] == ""


def test_a_placeholder_pin_is_not_corroboration():
    assert vp.judge(row(pin="000000"), place(), 1)["pin_check"] == "uncorroborated"


# --- the call ---------------------------------------------------------------------------


def client(handler) -> httpx.Client:
    return _REAL_CLIENT(transport=httpx.MockTransport(handler))


def test_the_key_travels_in_a_header_never_the_url():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(url=str(request.url), key=request.headers["x-goog-api-key"], mask=request.headers["x-goog-fieldmask"])
        return httpx.Response(200, json=place())

    got = vp.fetch_one(client(handler), "ChIJa", "SECRET", vp.MASKS["basic"], vp.RateLimiter(1000))
    assert got["displayName"]["text"].startswith("Delhi")
    assert seen["key"] == "SECRET" and "SECRET" not in seen["url"]
    assert "websiteUri" not in seen["mask"] and "nationalPhoneNumber" not in seen["mask"]


def test_status_codes():
    def respond(code):
        return client(lambda r: httpx.Response(code, json={"error": {"message": "nope"}}))

    limiter = vp.RateLimiter(1000)
    assert vp.fetch_one(respond(404), "x", "k", "id", limiter) == {"_error": "not_found"}
    assert vp.fetch_one(respond(400), "x", "k", "id", limiter) == {"_error": "invalid_place_id"}
    with pytest.raises(vp.PlacesAbort, match="nope"):
        vp.fetch_one(respond(403), "x", "k", "id", limiter)
    with pytest.raises(RuntimeError):
        vp.fetch_one(respond(500), "x", "k", "id", limiter)


def test_key_is_read_from_env_then_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("OTHER=1\nGOOGLE_PLACES_API_KEY=\"from-file\"\n", encoding="utf-8")
    monkeypatch.setattr(vp, "ROOT", tmp_path)
    monkeypatch.delenv("GOOGLE_PLACES_API_KEY", raising=False)
    assert vp.read_key() == "from-file"
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "from-env")
    assert vp.read_key() == "from-env"


def _patch_client(monkeypatch, handler):
    monkeypatch.setattr(vp.httpx, "Client", lambda **kw: client(handler))
    monkeypatch.setattr(vp.time, "sleep", lambda s: None)


def test_cap_is_never_exceeded_and_a_rerun_is_free(tmp_path, monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json=place())

    _patch_client(monkeypatch, handler)
    ids, cache, path = [f"id{i}" for i in range(10)], {}, tmp_path / "c.jsonl"

    vp.fetch_all(ids, cache, path, "k", "id", max_calls=4, qps=1000, workers=3)
    assert len(calls) == 4 and len(cache) == 4

    again = vp.load_cache(path)
    assert set(again) == set(cache)  # survived to disk, resumable
    vp.fetch_all(ids, again, path, "k", "id", max_calls=100, qps=1000, workers=3)
    assert len(calls) == 10  # only the six it did not have


def test_a_rejected_key_costs_one_call(tmp_path, monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(403, json={"error": {"message": "API not enabled"}})

    _patch_client(monkeypatch, handler)
    with pytest.raises(vp.PlacesAbort, match="API not enabled"):
        vp.fetch_all([f"id{i}" for i in range(50)], {}, tmp_path / "c.jsonl", "k", "id", 50, 1000, 8)
    assert len(calls) == 1


def test_a_streak_of_failures_stops_the_run(tmp_path, monkeypatch):
    first = True

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal first
        if first:
            first = False
            return httpx.Response(200, json=place())
        return httpx.Response(503, text="down")

    _patch_client(monkeypatch, handler)
    with pytest.raises(vp.PlacesAbort, match="in a row failed"):
        vp.fetch_all([f"id{i}" for i in range(200)], {}, tmp_path / "c.jsonl", "k", "id", 200, 1000, 2)


def test_connection_failures_wait_for_the_network_not_abort(tmp_path, monkeypatch):
    """A dropped network (DNS) never reaches Google, so it is waited out, and the call
    cap still holds across the wait."""
    seen = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        if seen["n"] <= 8:  # the first call, retried 4 times, twice over
            raise httpx.ConnectError("[Errno 11001] getaddrinfo failed")
        return httpx.Response(200, json=place())

    _patch_client(monkeypatch, handler)
    waits = []
    monkeypatch.setattr(vp, "wait_for_network", lambda minutes_s: waits.append(minutes_s) or True)
    src = tmp_path / "in.csv"
    src.write_text(
        "School ID (do not change),Institute name,PIN code,Latitude,Longitude,Google place ID\n"
        + "".join(f"s{i},Delhi Public School,110085,,,P{i}\n" for i in range(6)),
        encoding="utf-8",
    )
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "k")
    out = tmp_path / "out"
    vp.run(
        vp.argparse.Namespace(
            input=str(src), outdir=str(out), max_calls=4, cache=None, fields="basic",
            qps=1000, workers=2, limit=None, dry_run=False, network_wait=1.0,
        )
    )  # fmt: skip
    assert len(waits) == 2  # waited out two outages, then carried on
    cached = vp.load_cache(out / "places_cache.jsonl")
    assert len(cached) == 4  # the cap of the whole run, not of each attempt


def test_a_streak_of_connection_failures_is_a_network_outage(tmp_path, monkeypatch):
    first = True

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal first
        if first:
            first = False
            return httpx.Response(200, json=place())
        raise httpx.ConnectError("getaddrinfo failed")

    _patch_client(monkeypatch, handler)
    with pytest.raises(vp.PlacesNetworkDown):
        vp.fetch_all([f"id{i}" for i in range(200)], {}, tmp_path / "c.jsonl", "k", "id", 200, 1000, 2)


def test_wait_for_network_returns_once_dns_answers_and_gives_up_at_the_deadline():
    clock = iter(range(0, 1000, 10))
    answers = iter([OSError(), OSError(), None])

    def probe(host, port):
        answer = next(answers)
        if answer is not None:
            raise answer

    assert vp.wait_for_network(600, probe=probe, sleep=lambda s: None, clock=lambda: next(clock)) is True

    def never(host, port):
        raise OSError

    clock2 = iter(range(0, 1000, 10))
    assert vp.wait_for_network(30, probe=never, sleep=lambda s: None, clock=lambda: next(clock2)) is False


def test_end_to_end_writes_the_report(tmp_path, monkeypatch):
    src = tmp_path / "in.csv"
    src.write_text(
        "School ID (do not change),Institute name,PIN code,Latitude,Longitude,Google place ID\n"
        "a,Delhi Public School,110085,28.7,77.1,P1\n"
        "b,Delhi Public School,110085,28.7,77.1,P1\n"
        "c,No Place School,110001,,,\n",
        encoding="utf-8",
    )
    _patch_client(monkeypatch, lambda r: httpx.Response(200, json=place()))
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "k")
    out = tmp_path / "out"
    vp.run(
        vp.argparse.Namespace(
            input=str(src), outdir=str(out), max_calls=10, cache=None, fields="basic",
            qps=1000, workers=2, limit=None, dry_run=False,
        )
    )  # fmt: skip
    text = (out / "places_verified.csv").read_text(encoding="utf-8-sig")
    assert text.count("confirmed") == 2 and "no_place_id" in text
    assert len(json.loads((out / "places_cache.jsonl").read_text().splitlines()[0])["place"]) > 1  # one call, shared ID
    groups = (out / "shared_place_groups.csv").read_text(encoding="utf-8-sig")
    assert "P1" in groups and "a | b" in groups
