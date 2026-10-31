#!/usr/bin/env python
"""Ask the tenant's Premium AI whether Google's listing is the same school as ours.

    python scripts/adjudicate_places.py \
        --places  verified_places/places_verified.csv \
        --source  var/uploads/<tenant>/<job>.csv \
        --outdir  verified_places/ \
        --max-calls 12000

`verify_places.py` confirms most schools by name and PIN; the rest it leaves as
`review` (partial name, or a PIN that differs) or `name_mismatch`. This sends
each of those, once per distinct pair, to the tenant's own Anthropic key with
our record and Google's listing side by side, and gets YES / NO / UNSURE and a
short reason. UNSURE stays `review`: nothing is decided by a guess.

Spends the tenant's own key (the Premium AI saved in Settings), so like the
Google pass it has a required `--max-calls`, a resumable cache, and refuses to
run on the shared Standard AI: that is a decision the user made for this key.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.enrichment.sources import LlmClient, LlmUnusableError  # noqa: E402
from app.queue.registry import RetryLaterError  # noqa: E402

TENANT = "00000000-0000-0000-0000-000000000001"
SELECT = ("review", "name_mismatch")
_ABORT_AFTER_FAILURES = 15
_MAX_RATE_WAITS = 6

SYSTEM = (
    "You verify school records against Google Maps listings. Judge only from the text "
    "given; do not guess from what you remember about a school."
)
_ANSWER = re.compile(r"\s*\W*(YES|NO|UNSURE)\b\W*(.*)", re.IGNORECASE | re.DOTALL)
_UNPARSEABLE = "unparseable reply"


class AdjudicateAbort(Exception):
    """Stop the run: the key is unusable, or calls keep failing."""


def prompt_for(ours: dict[str, str], row: dict[str, str]) -> str:
    where = ", ".join(p for p in (ours["addr1"], ours["addr2"], ours["locality"]) if p)
    return (
        "School in our database:\n"
        f"  name: {row['name']}\n  address: {where}\n  PIN: {ours['pin']}\n\n"
        "Google Maps listing:\n"
        f"  name: {row['google_name']}\n  address: {row['google_address']}\n"
        f"  status: {row['google_status']}\n\n"
        "Is the Google listing the same school as ours? Names may differ by abbreviation, "
        "spelling or an added locality. The same name in a different town is NOT the same "
        "school, and neither is a different school at the same address.\n"
        'Reply with one line: YES, NO or UNSURE, then " - " and a reason of at most 15 words.'
    )


def parse_answer(text: str) -> tuple[str, str]:
    """(YES|NO|UNSURE, reason). Anything that is not clearly one of the three is UNSURE."""
    match = _ANSWER.fullmatch(text or "")
    if not match:
        return "UNSURE", _UNPARSEABLE
    return match.group(1).upper(), match.group(2).strip(" -:\n")[:200]


def final_verdict(answer: str) -> str:
    return {"YES": "confirmed_by_ai", "NO": "different_school"}.get(answer, "review")


def ask(llm, prompt: str) -> dict[str, str]:
    """One question; waits out a rate limit, raises for a permanent failure."""
    for waits in range(_MAX_RATE_WAITS + 1):
        try:
            reply = llm.chat(SYSTEM, prompt, json_mode=False, max_tokens=60)
        except RetryLaterError as exc:
            if waits == _MAX_RATE_WAITS:
                raise
            time.sleep(min(exc.countdown, 60.0))
            continue
        answer, reason = parse_answer(reply)
        if reason == _UNPARSEABLE:
            # Not cached and not a verdict: a model whose replies cannot be read
            # (all 25 in the first trial, when it spent its tokens thinking) must
            # trip the failure streak, not quietly store "UNSURE" for every school.
            raise ValueError(f"unreadable reply: {reply[:80]!r}")
        return {"answer": answer, "reason": reason}
    raise AssertionError("unreachable")


def key_of(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, dict[str, str]]:
    cache: dict[str, dict[str, str]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                cache[entry["key"]] = entry["result"]
    return cache


def adjudicate(llm, prompts: dict[str, str], cache: dict, cache_path: Path, max_calls: int, workers: int) -> Counter:
    todo = [k for k in prompts if k not in cache][:max_calls]
    stats: Counter = Counter()
    lock = threading.Lock()
    streak = 0

    def work(key: str) -> tuple[str, dict[str, str]]:
        return key, ask(llm, prompts[key])

    with cache_path.open("a", encoding="utf-8") as sink, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, k) for k in todo]
        try:
            for done in as_completed(futures):
                try:
                    key, result = done.result()
                except LlmUnusableError as exc:
                    raise AdjudicateAbort(f"the model is unusable: {exc}") from exc
                except Exception as exc:  # transient; retries exhausted
                    stats["failed"] += 1
                    streak += 1
                    if streak >= _ABORT_AFTER_FAILURES:
                        raise AdjudicateAbort(f"{streak} calls in a row failed; last: {exc}") from exc
                    continue
                streak = 0
                with lock:
                    cache[key] = result
                    stats[result["answer"]] += 1
                    sink.write(json.dumps({"key": key, "result": result}, ensure_ascii=False) + "\n")
                    sink.flush()
                if sum(stats.values()) % 500 == 0:
                    print(f"  {sum(stats.values())}/{len(todo)} answered", flush=True)
        except AdjudicateAbort:
            for future in futures:
                future.cancel()
            raise
    return stats


def open_llm(client: httpx.Client):
    from app.db.session import tenant_sync_session
    from app.enrichment.llm_config import resolve_llm_pool

    with tenant_sync_session(TENANT) as session:
        pool = resolve_llm_pool(session, TENANT)
    if not pool or not pool[0].premium:
        raise SystemExit("The tenant has no Premium AI key saved; this pass is for that key only.")
    return LlmClient(client, **pool[0].client_kwargs()), pool[0].audit_label


def read_source(path: Path) -> dict[str, dict[str, str]]:
    wanted = {
        "id": ("schooliddonotchange", "schoolid", "id"),
        "addr1": ("addressline1", "address1"),
        "addr2": ("addressline2", "address2"),
        "locality": ("localitydistrictstateasentered", "locality"),
        "pin": ("pincode", "pin"),
    }
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        index = {re.sub(r"[^a-z0-9]", "", h.lower()): h for h in reader.fieldnames or []}
        pick = {f: next((index[a] for a in names if a in index), None) for f, names in wanted.items()}
        return {
            (row.get(pick["id"]) or "").strip(): {f: (row.get(h) or "").strip() if h else "" for f, h in pick.items()}
            for row in reader
        }


def run(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = outdir / "adjudication_cache.jsonl"
    cache = load_cache(cache_path)
    source = read_source(Path(args.source))
    with Path(args.places).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = list(reader.fieldnames or [])

    prompts: dict[str, str] = {}
    row_key: dict[int, str] = {}
    for i, row in enumerate(rows):
        if row["places_verdict"] in SELECT:
            prompt = prompt_for(source.get(row["school_id"], {"addr1": "", "addr2": "", "locality": "", "pin": ""}), row)
            row_key[i] = key_of(prompt)
            prompts[row_key[i]] = prompt
    todo = [k for k in prompts if k not in cache]
    print(
        f"{len(row_key)} rows to judge; {len(prompts)} distinct questions; {len(cache)} cached; "
        f"{len(todo)} to ask; this run will ask {min(len(todo), args.max_calls)} (cap {args.max_calls})."
    )
    if args.dry_run:
        return

    label = ""
    aborted = ""
    if todo and args.max_calls:
        with httpx.Client(timeout=60.0) as client:
            llm, label = open_llm(client)
            llm.check()  # one known-answer question before real work
            try:
                print("asked:", dict(adjudicate(llm, prompts, cache, cache_path, args.max_calls, args.workers)))
            except AdjudicateAbort as exc:
                aborted = str(exc)
                print(f"STOPPED: {aborted}\nWhat was answered is cached; fix the cause and run again to resume.")

    out_columns = [*columns, "llm_answer", "llm_reason", "final_verdict"]
    counts: Counter = Counter()
    with (outdir / "places_adjudicated.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_columns)
        writer.writeheader()
        for i, row in enumerate(rows):
            result = cache.get(row_key.get(i, ""))
            row["llm_answer"] = result["answer"] if result else ""
            row["llm_reason"] = result["reason"] if result else ""
            row["final_verdict"] = final_verdict(result["answer"]) if result else row["places_verdict"]
            counts[row["final_verdict"]] += 1
            writer.writerow(row)
    summary = [f"{v:>18} {n:>7} {100 * n / len(rows):6.2f}%" for v, n in counts.most_common()]
    summary.append(f"model: {label or '(none called this run)'}")
    (outdir / "adjudication_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    if aborted:
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--places", required=True, help="places_verified.csv from verify_places.py")
    p.add_argument("--source", required=True, help="the job's original CSV (for our addresses)")
    p.add_argument("--outdir", required=True)
    p.add_argument("--max-calls", type=int, required=True, help="most model calls this run may make")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--dry-run", action="store_true", help="count the questions; call nobody")
    args = p.parse_args()
    if args.max_calls < 0:
        p.error("--max-calls must be >= 0")
    run(args)


if __name__ == "__main__":
    main()
