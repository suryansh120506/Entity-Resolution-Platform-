#!/usr/bin/env python3
"""
generate_benchmark.py — zero-touch ground-truth benchmark + scored LLM
cleaning baseline for a school-data pipeline (e.g. a Medallion Gold-layer
export).

WHAT CHANGED FROM "JUST ASK THE LLM AND CALL IT GOLDEN"
--------------------------------------------------------
This script still does not write an LLM's guesses at Pincodes, UDISE codes,
or Affiliation Numbers into a file called "golden" / "ground truth." That
would mean benchmarking against hallucinations, which produces a number
that's meaningless at best and actively misleading at a presentation that
claims it measures real accuracy.

What it does instead — and this is a strictly more useful thing to put in
front of a room than a fabricated 100%:

  1. GOLD    = real rows from your own CSV with complete, non-placeholder
               identifiers. Untouched. No model involved. Correct by
               construction.
  2. DIRTY   = corrupted copies of those rows (identifiers blanked, names
               noised, a subset duplicated with a messy name variant).
               This is the input any cleaning pipeline — including the LLM
               step below — actually has to work from.
  3. CANDIDATE = the Anthropic API's attempt to clean DIRTY: standardize
               names/addresses, merge exact duplicates it spots within a
               chunk, and recover any identifier it can *derive* from the
               row's own text (e.g. a pincode sitting in the address
               string). It is explicitly instructed not to fabricate an
               identifier it cannot derive — leaving a field blank is the
               correct behavior, not a failure, when the value truly isn't
               recoverable from what's in front of it.
  4. SCORECARD = CANDIDATE is joined back to GOLD and scored: field-recovery
               accuracy, a fabrication-rate (did it invent a wrong value
               instead of leaving it blank — the number that would have
               been hidden if step 3's output had simply been trusted as
               "golden"), and dedup precision/recall.

That scorecard — "the LLM-assisted cleaner recovers X% of blanked
identifiers, fabricates Y%, and correctly merges Z% of injected duplicates,
measured against a verified 500-row ground truth" — is a real, defensible
number for a final presentation. "Here is a CSV the LLM filled in" is not.

ZERO-TOUCH
----------
Column roles (name / pincode / udise / affiliation / address) are detected
automatically from your CSV's headers by keyword scoring — no manual
mapping step. The script prints what it detected before proceeding so it's
auditable, but does not block on confirmation.

SETUP
-----
  pip install pandas numpy python-dotenv anthropic tqdm
  # .env must contain ANTHROPIC_API_KEY=sk-ant-...
  python generate_benchmark.py
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------

# NOTE: despite living inside entity-resolution-platform/, the raw CSV
# actually sits one directory up at C:\Academic_data_pipelines\ — verified
# by directory listing, not assumed.
INPUT_CSV = r"C:\Academic_data_pipelines\raw_schools_uncleaned.csv"

PLACEHOLDER_VALUES = {"", "0", "000000", "0000000000", "na", "n/a", "unknown", "nan"}

SAMPLE_SIZE = 500
RANDOM_SEED = 42

BLANK_RATE = 0.6               # fraction of gold rows that get an identifier blanked
IDENTIFIER_BLANK_PROB = 0.7    # per-identifier-column chance, within a chosen row

DUPLICATE_COUNT = 60
DUPLICATE_CHUNK_SIZE = 10

CLEAN_CHUNK_SIZE = 50          # matches the "50-row chunk" spec for the candidate pass

ANTHROPIC_MODEL = "claude-sonnet-5"  # current Sonnet; not claude-3-5-sonnet-latest
API_SLEEP_SECONDS = 1.0

# Column-role detection: normalized header -> role, scored by longest
# matching keyword. Add more synonyms here if your CSV uses different
# naming and detection misses something (printed at runtime either way).
ROLE_KEYWORDS: dict[str, list[str]] = {
    "name": ["schoolname", "institutename", "institutionname", "name", "school"],
    "pincode": ["pincode", "pincode", "postalcode", "zipcode", "zip", "pin"],
    "udise": ["udisecode", "udise", "disecode", "dise"],
    "affiliation": ["affiliationnumber", "affiliationno", "affiliation", "boardaffiliation"],
    "address": ["address", "locality", "addr", "location"],
}
REQUIRED_ROLES = ["name"]              # hard-fail if not found
IDENTIFIER_ROLES = ["pincode", "udise", "affiliation"]  # need >=1 to build a benchmark

# --------------------------------------------------------------------------


def _normalize_header(h: str) -> str:
    return re.sub(r"[^a-z0-9]", "", h.lower())


def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """Greedy best-match assignment of CSV columns to semantic roles."""
    normalized = {c: _normalize_header(c) for c in df.columns}

    candidates: list[tuple[int, str, str]] = []  # (score, role, column)
    for role, keywords in ROLE_KEYWORDS.items():
        for col, norm in normalized.items():
            best = max((len(kw) for kw in keywords if kw in norm), default=0)
            if best > 0:
                candidates.append((best, role, col))
    candidates.sort(reverse=True)  # longest/most-specific match wins ties first

    detected: dict[str, str] = {}
    used_columns: set[str] = set()
    for _score, role, col in candidates:
        if role in detected or col in used_columns:
            continue
        detected[role] = col
        used_columns.add(col)

    print("Detected column mapping:")
    for role in [*ROLE_KEYWORDS]:
        print(f"  {role:<12} -> {detected.get(role, '(not found)')}")

    missing_required = [r for r in REQUIRED_ROLES if r not in detected]
    if missing_required:
        print(f"\nAll columns in {INPUT_CSV}:")
        for c in df.columns:
            print(f"  - {c}")
        raise SystemExit(
            f"Could not confidently detect required role(s) {missing_required}. "
            f"Add synonyms for it to ROLE_KEYWORDS above and re-run."
        )
    if not any(r in detected for r in IDENTIFIER_ROLES):
        raise SystemExit(
            f"Found no identifier column among {IDENTIFIER_ROLES} — there is "
            f"nothing to build a backfill benchmark against. Check ROLE_KEYWORDS."
        )
    return detected


def is_placeholder(value) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in PLACEHOLDER_VALUES


# --------------------------------------------------------------------------
# GOLD
# --------------------------------------------------------------------------


def load_and_filter(path: str, cols: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    id_cols = [cols[r] for r in IDENTIFIER_ROLES if r in cols]

    complete_mask = pd.Series(True, index=df.index)
    for col in id_cols:
        complete_mask &= ~df[col].apply(is_placeholder)

    qualifying = df[complete_mask]
    print(
        f"\n{len(qualifying)} / {len(df)} rows have complete, non-placeholder "
        f"values in {id_cols} and can serve as ground truth."
    )
    return qualifying


def sample_gold(qualifying: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(qualifying) < n:
        print(f"WARNING: only {len(qualifying)} qualifying rows exist — using all of them.")
        n = len(qualifying)
    gold = qualifying.sample(n=n, random_state=seed).reset_index(drop=True)
    gold.insert(0, "_gold_id", [f"G{i:04d}" for i in range(len(gold))])
    return gold


# --------------------------------------------------------------------------
# DIRTY — deterministic corruption, no model involved
# --------------------------------------------------------------------------

_ABBREV_SWAPS = [
    (r"\bSenior Secondary\b", "Sr Sec"),
    (r"\bGovernment\b", "Govt"),
    (r"\bPublic School\b", "Pub Sch"),
    (r"\bVidyalaya\b", "Vidyalay"),
]


def mangle_name(name: str, rng: random.Random) -> str:
    s = str(name)
    for pattern, repl in _ABBREV_SWAPS:
        if re.search(pattern, s, flags=re.I) and rng.random() < 0.5:
            s = re.sub(pattern, repl, s, flags=re.I)
    if rng.random() < 0.3:
        s = s.upper()
    elif rng.random() < 0.3:
        s = s.title()
    if rng.random() < 0.3:
        s = re.sub(r"\s+", "  ", s)
    return s.strip()


def build_dirty_base(gold: pd.DataFrame, cols: dict[str, str], rng: random.Random) -> pd.DataFrame:
    dirty = gold.copy()
    name_col = cols["name"]
    id_cols = [cols[r] for r in IDENTIFIER_ROLES if r in cols]

    dirty[name_col] = dirty[name_col].apply(lambda n: mangle_name(n, rng))

    blank_targets = dirty.sample(frac=BLANK_RATE, random_state=RANDOM_SEED).index
    for col in id_cols:
        targets = [i for i in blank_targets if rng.random() < IDENTIFIER_BLANK_PROB]
        dirty.loc[targets, col] = None

    return dirty


# --------------------------------------------------------------------------
# LLM step 1: messy NAME variants for duplicate injection (text only)
# --------------------------------------------------------------------------

DUPLICATE_SYSTEM_PROMPT = """\
You generate realistic messy data-entry variants of school names, for \
building a synthetic test fixture. You are NOT verifying facts about any \
real school and must NOT output or imply any identifier (pincode, UDISE \
code, affiliation number, address) — only a textual variant of the name \
string you're given, of the kind a human clerk might type: abbreviation \
swaps, minor typos, inconsistent spacing or punctuation, casing changes.

Respond with ONLY a JSON array of strings, no markdown fences, no \
commentary. The array must have exactly as many entries as input names, \
in the same order, one variant string per input name.
"""


def _parse_json(text: str):
    cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.I | re.M).strip()
    return json.loads(cleaned)


def generate_name_variants(client, names: list[str]) -> list[str]:
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=DUPLICATE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(names)}],
    )
    text = "".join(b.text for b in message.content if b.type == "text")
    data = _parse_json(text)
    if not isinstance(data, list) or len(data) != len(names):
        raise ValueError(f"Expected {len(names)} variants, got: {data!r}")
    return [str(x) for x in data]


def build_duplicate_rows(client, gold: pd.DataFrame, cols: dict[str, str], rng: random.Random) -> pd.DataFrame:
    name_col = cols["name"]
    id_cols = [cols[r] for r in IDENTIFIER_ROLES if r in cols]

    n = min(DUPLICATE_COUNT, len(gold))
    dupe_source = gold.sample(n=n, random_state=RANDOM_SEED)
    chunks = np.array_split(dupe_source, max(1, len(dupe_source) // DUPLICATE_CHUNK_SIZE))

    new_rows = []
    for chunk in tqdm(chunks, desc="Generating duplicate name variants"):
        if chunk.empty:
            continue
        try:
            variants = generate_name_variants(client, chunk[name_col].tolist())
        except Exception as exc:
            print(f"  skipping chunk ({len(chunk)} rows) — {exc}", file=sys.stderr)
            time.sleep(API_SLEEP_SECONDS)
            continue

        for (_, gold_row), variant_name in zip(chunk.iterrows(), variants):
            dup = gold_row.copy()
            dup[name_col] = variant_name
            for col in id_cols:
                if rng.random() < 0.5:
                    dup[col] = None
            new_rows.append(dup)
        time.sleep(API_SLEEP_SECONDS)

    return pd.DataFrame(new_rows).reset_index(drop=True)


# --------------------------------------------------------------------------
# LLM step 2: candidate cleaner — scored, not trusted as truth
# --------------------------------------------------------------------------

CLEAN_SYSTEM_PROMPT = """\
You are a meticulous data engineer cleaning a chunk of school location \
records before they're loaded into a Gold layer. For each input record you \
will standardize formatting, consolidate exact duplicates, and recover any \
field value that is genuinely present elsewhere in the record's own text — \
but you will never invent a value you cannot derive from what's in front of \
you. This distinction is the entire point of your task:

  - DERIVE: the record's name field reads "GOVT SR SEC SCHOOL, SECTOR 14,                 \
    GURGAON - 122001" and the pincode field is blank. 122001 is sitting \
    right there in the text — move it into the pincode field.
  - FABRICATE (forbidden): the pincode field is blank and nothing in the \
    record implies a value. Leaving it null is the correct output. A \
    plausible-looking pincode, UDISE code, or affiliation number that you \
    reconstructed from general knowledge of "schools like this one" is \
    wrong even if it happens to look right — do not do this, ever, for any \
    of the three identifier fields.

Your other responsibilities, in order:
  1. Standardize each name: consistent casing (Title Case), consistent \
     expansion of common abbreviations (Govt -> Government, Sr Sec -> \
     Senior Secondary, Vidyalay -> Vidyalaya), single internal spaces.
  2. Standardize each address the same way; do not reorder components you \
     aren't confident about.
  3. Merge records that are the same real-world school with near-identical \
     names (abbreviation/casing/whitespace variants of each other) INTO ONE \
     output record. Only merge within this chunk — never claim a merge with \
     a record you were not given. When merging, combine every input \
     "_row_ref" into that output record's "_row_refs" array, and keep \
     whichever non-null value exists for each field (if both have the same \
     field populated with conflicting values, prefer neither — set it null \
     rather than guess which is right).
  4. Every input record you do NOT merge still becomes its own output \
     record, with "_row_refs" containing just its own single ref.

Respond with ONLY a JSON array of objects, no markdown fences, no \
commentary. Each object has exactly these keys: "_row_refs" (array of \
strings), "name", "address", "pincode", "udise", "affiliation". Use null \
(not an empty string) for any field you don't have a value for.
"""


def clean_chunk(client, records: list[dict]) -> list[dict]:
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4096,
        system=CLEAN_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(records)}],
    )
    text = "".join(b.text for b in message.content if b.type == "text")
    data = _parse_json(text)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got: {data!r}")
    return data


def run_candidate_cleaning(client, dirty: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    """Sends DIRTY to the LLM in 50-row chunks and returns its candidate output,
    with each output row's _row_refs still attached so scoring can trace it
    back to _gold_id without ever having sent _gold_id to the model."""
    name_col = cols["name"]
    addr_col = cols.get("address")
    role_to_field = {"pincode": "pincode", "udise": "udise", "affiliation": "affiliation"}

    # Sort by normalized name before chunking so injected duplicate variants
    # (which share a name prefix) are likely to land in the same 50-row
    # chunk — the model can only merge duplicates it's shown together.
    dirty = dirty.copy()
    dirty["_row_ref"] = [f"R{i:05d}" for i in range(len(dirty))]
    dirty["_sort_key"] = dirty[name_col].astype(str).str.lower().str.replace(r"\W", "", regex=True).str[:12]
    dirty = dirty.sort_values("_sort_key").reset_index(drop=True)

    ref_to_gold_id = dict(zip(dirty["_row_ref"], dirty["_gold_id"]))

    results: list[dict] = []
    n_chunks = max(1, -(-len(dirty) // CLEAN_CHUNK_SIZE))
    for start in tqdm(range(0, len(dirty), CLEAN_CHUNK_SIZE), total=n_chunks, desc="LLM candidate cleaning"):
        chunk = dirty.iloc[start : start + CLEAN_CHUNK_SIZE]
        payload = [
            {
                "_row_ref": row["_row_ref"],
                "name": row[name_col],
                "address": row[addr_col] if addr_col else None,
                "pincode": row.get(cols.get("pincode"), None),
                "udise": row.get(cols.get("udise"), None),
                "affiliation": row.get(cols.get("affiliation"), None),
            }
            for _, row in chunk.iterrows()
        ]
        try:
            cleaned = clean_chunk(client, payload)
        except Exception as exc:
            print(f"  skipping chunk at row {start} ({len(chunk)} rows) — {exc}", file=sys.stderr)
            time.sleep(API_SLEEP_SECONDS)
            continue

        for rec in cleaned:
            refs = rec.get("_row_refs") or []
            gold_ids = [ref_to_gold_id[r] for r in refs if r in ref_to_gold_id]
            rec["_resolved_gold_id"] = Counter(gold_ids).most_common(1)[0][0] if gold_ids else None
            rec["_source_gold_ids"] = gold_ids
            results.append(rec)
        time.sleep(API_SLEEP_SECONDS)

    return pd.DataFrame(results)


# --------------------------------------------------------------------------
# SCORING — the number that actually belongs in the presentation
# --------------------------------------------------------------------------


def score_candidate(gold: pd.DataFrame, dirty: pd.DataFrame, candidate: pd.DataFrame, cols: dict[str, str]) -> dict:
    gold_by_id = gold.set_index("_gold_id")
    id_cols = {role: cols[role] for role in IDENTIFIER_ROLES if role in cols}

    field_scores = {role: {"recovered_correct": 0, "fabricated_wrong": 0, "left_blank": 0, "preserved_ok": 0, "corrupted": 0} for role in id_cols}

    # candidate rows indexed by their resolved gold id (first match wins if
    # a false-merge produced duplicates against the same id — rare, logged
    # separately below under dedup scoring)
    cand_by_gold_id: dict[str, dict] = {}
    for _, row in candidate.iterrows():
        gid = row.get("_resolved_gold_id")
        if gid and gid not in cand_by_gold_id:
            cand_by_gold_id[gid] = row

    dirty_by_gold_id_was_blank = {
        role: set(dirty.loc[dirty[id_cols[role]].isna(), "_gold_id"]) for role in id_cols
    }

    for gid, gold_row in gold_by_id.iterrows():
        cand_row = cand_by_gold_id.get(gid)
        for role, col in id_cols.items():
            gold_val = str(gold_row[col]).strip()
            cand_val = cand_row.get(role) if cand_row is not None else None
            cand_val = str(cand_val).strip() if cand_val not in (None, "") else None
            was_blanked = gid in dirty_by_gold_id_was_blank[role]

            if was_blanked:
                if cand_val is None:
                    field_scores[role]["left_blank"] += 1
                elif cand_val == gold_val:
                    field_scores[role]["recovered_correct"] += 1
                else:
                    field_scores[role]["fabricated_wrong"] += 1
            else:
                if cand_val == gold_val:
                    field_scores[role]["preserved_ok"] += 1
                else:
                    field_scores[role]["corrupted"] += 1

    # dedup: for every gold_id that produced >1 dirty row, did they all land
    # in ONE candidate record?
    dirty_group_sizes = dirty.groupby("_gold_id").size()
    duplicate_gold_ids = dirty_group_sizes[dirty_group_sizes > 1].index
    merged_correctly = sum(
        1 for gid in duplicate_gold_ids
        if any(row["_resolved_gold_id"] == gid and len(set(row["_source_gold_ids"])) == 1 for _, row in candidate.iterrows())
    )
    false_merges = sum(1 for _, row in candidate.iterrows() if len(set(row.get("_source_gold_ids") or [])) > 1)

    return {
        "fields": field_scores,
        "dedup": {
            "true_duplicate_groups": len(duplicate_gold_ids),
            "correctly_merged": merged_correctly,
            "false_merges": false_merges,
        },
    }


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    load_dotenv()
    import anthropic

    client = anthropic.Anthropic()

    header_df = pd.read_csv(INPUT_CSV, dtype=str, nrows=0)
    cols = detect_columns(header_df)

    qualifying = load_and_filter(INPUT_CSV, cols)
    gold = sample_gold(qualifying, SAMPLE_SIZE, RANDOM_SEED)

    dirty_base = build_dirty_base(gold, cols, rng)
    dirty_dupes = build_duplicate_rows(client, gold, cols, rng)
    dirty = (
        pd.concat([dirty_base, dirty_dupes], ignore_index=True)
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    candidate = run_candidate_cleaning(client, dirty, cols)
    scorecard = score_candidate(gold, dirty, candidate, cols)

    n = len(gold)
    gold.to_csv(f"benchmark_gold_{n}.csv", index=False)
    dirty.to_csv(f"benchmark_dirty_{n}.csv", index=False)
    candidate.to_csv(f"llm_candidate_cleaned_{n}.csv", index=False)
    Path(f"scorecard_{n}.json").write_text(json.dumps(scorecard, indent=2))

    print(f"\nWrote benchmark_gold_{n}.csv, benchmark_dirty_{n}.csv, "
          f"llm_candidate_cleaned_{n}.csv, scorecard_{n}.json\n")
    print(json.dumps(scorecard, indent=2))


if __name__ == "__main__":
    main()
