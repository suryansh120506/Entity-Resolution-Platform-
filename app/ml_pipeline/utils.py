"""Pincode-aware feature-extraction helpers, backed by `reference_pincodes`.

`reference_pincodes` (app/models/reference_pincode.py) is large and almost
static — seeded once by scripts/seed_pincodes.py, then read constantly during
batch feature extraction over a job's records. Both lookups below cache their
results in a process-local dict keyed on the pincode string rather than via
`functools.lru_cache` directly on the public functions: `db` is a live
Session, and a cache keyed partly on session identity would stop sharing hits
across the very batch job this exists to speed up. A worker process handles
one job at a time and exits, so there is no cross-job staleness to worry
about; `clear_pincode_cache()` exists for tests that need a clean slate.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.reference_pincode import ReferencePincode

logger = logging.getLogger(__name__)

#: Common placeholder values a source CSV uses in place of a real PIN — all
#: six digits, none of them a real Indian postal code. A row bearing one of
#: these should be treated as "no pincode", not "invalid pincode we looked up
#: and failed to find" (the two feed different classifier features).
_PLACEHOLDER_PINCODES = frozenset(
    {
        "000000",
        "111111",
        "222222",
        "333333",
        "444444",
        "555555",
        "666666",
        "777777",
        "888888",
        "999999",
    }
)

_valid_pincode_cache: dict[str, bool] = {}
_pincode_context_cache: dict[str, dict | None] = {}


def _is_well_formed(pincode: str) -> bool:
    return bool(pincode) and pincode.isdigit() and len(pincode) == 6


def is_placeholder_pincode(pincode: str) -> bool:
    """True if `pincode` is malformed or one of the known sentinel values.

    Pure string test, no database. `validate_pincode` applies this first and
    only queries `reference_pincodes` for PINs that survive it, which is also
    what lets the classifier compute its pincode feature with no session at
    all (app/ml_pipeline/classification/classifier.py) — degraded, but never wrong about the
    cases it can decide on the string alone.
    """
    return not _is_well_formed(pincode) or pincode in _PLACEHOLDER_PINCODES


def validate_pincode(pincode: str, db: Session) -> bool:
    """True if `pincode` is a well-formed, non-placeholder PIN present in
    `reference_pincodes`.

    Cached per pincode for the life of the process: a batch run over tens of
    thousands of records typically sees a few thousand distinct pincodes
    repeated many times over, so this turns "one query per record" into "one
    query per distinct pincode ever seen". `warm_pincode_cache` collapses even
    that into a single query when the whole batch is known up front.
    """
    if pincode in _valid_pincode_cache:
        return _valid_pincode_cache[pincode]

    if is_placeholder_pincode(pincode):
        _valid_pincode_cache[pincode] = False
        return False

    exists = (
        db.execute(
            select(ReferencePincode.id).where(ReferencePincode.pincode == pincode).limit(1)
        ).first()
        is not None
    )
    _valid_pincode_cache[pincode] = exists
    return exists


def get_pincode_context(pincode: str, db: Session) -> dict | None:
    """State, district(s) and known delivery offices for `pincode`.

    Returns None if the pincode does not validate. Otherwise a dict:

        {
            "pincode": "560001",
            "state_name": "Karnataka",
            "district_names": ["Bangalore"],
            "offices": ["Bangalore GPO", "Bangalore City", ...],
        }

    `district_names` and `offices` are lists, not single values, because one
    PIN routinely spans several delivery offices (and, across dataset
    revisions, occasionally more than one recorded district name).
    """
    if pincode in _pincode_context_cache:
        return _pincode_context_cache[pincode]

    if not validate_pincode(pincode, db):
        _pincode_context_cache[pincode] = None
        return None

    rows = db.execute(
        select(
            ReferencePincode.state_name,
            ReferencePincode.district_name,
            ReferencePincode.office_name,
        ).where(ReferencePincode.pincode == pincode)
    ).all()

    if not rows:
        # Cache said valid, but a concurrent delete raced us. Treat as a miss
        # rather than raising mid-batch.
        _pincode_context_cache[pincode] = None
        return None

    context = {
        "pincode": pincode,
        "state_name": rows[0].state_name,
        "district_names": sorted({r.district_name for r in rows}),
        "offices": sorted({r.office_name for r in rows}),
    }
    _pincode_context_cache[pincode] = context
    return context


#: Chunk size for the IN list in `warm_pincode_cache`. India has roughly
#: 19,000 distinct PINs, so an unchunked query is a single bind list of that
#: size — well within what Postgres accepts, but slow to plan and needlessly
#: large in the logs. Chunking keeps each statement ordinary.
_WARM_CHUNK_SIZE = 5_000


def warm_pincode_cache(pincodes: Iterable[str], db: Session) -> int:
    """Resolve a whole batch's pincodes in one query instead of one each.

    `validate_pincode` already collapses repeats, so the cost of a cold batch
    is one query per *distinct* pincode — a few thousand round trips for a
    large job, which is the difference between a feature-extraction pass that
    takes seconds and one that takes minutes. Calling this first makes every
    subsequent `validate_pincode` a dict hit.

    Returns the number of pincodes newly resolved (already-cached ones and
    strings that fail `is_placeholder_pincode` cost nothing and are excluded
    from the count).
    """
    candidates = {
        p for p in pincodes if p not in _valid_pincode_cache and not is_placeholder_pincode(p)
    }
    if not candidates:
        return 0

    ordered = sorted(candidates)
    found: set[str] = set()
    for start in range(0, len(ordered), _WARM_CHUNK_SIZE):
        chunk = ordered[start : start + _WARM_CHUNK_SIZE]
        found.update(
            db.execute(select(ReferencePincode.pincode).where(ReferencePincode.pincode.in_(chunk)))
            .scalars()
            .all()
        )

    for pincode in candidates:
        _valid_pincode_cache[pincode] = pincode in found

    if not found:
        _warn_if_reference_table_is_empty(db, len(candidates))

    return len(candidates)


#: Warn once per process, not once per batch.
_warned_about_empty_reference = False


def _warn_if_reference_table_is_empty(db: Session, attempted: int) -> None:
    """Say something when `reference_pincodes` has never been seeded.

    This is the quietest failure in the pipeline. With the table empty every
    PIN fails validation, so `pincode_is_placeholder` is 1 for every row,
    `lf_complete_record` abstains on every row, and the classifier labels an
    entire upload `Missing_Context`. Nothing is embedded, no clusters form,
    and the job completes successfully having resolved nothing — no error,
    no exception, just a tenant wondering why their 50,000 rows produced no
    matches.

    A deployment that skipped `scripts/seed_pincodes.py` looks exactly like
    a tenant whose data is genuinely all missing context, and the only way
    to tell them apart is to check whether the table has any rows at all.
    """
    global _warned_about_empty_reference
    if _warned_about_empty_reference:
        return

    total = db.execute(select(func.count()).select_from(ReferencePincode)).scalar_one()
    if total == 0:
        _warned_about_empty_reference = True
        logger.warning(
            "reference_pincodes is EMPTY — all %d pincodes in this batch failed "
            "validation for that reason alone. Every row will classify as "
            "Missing_Context and the pipeline will produce no clusters. Seed it: "
            "python scripts/seed_pincodes.py --csv-path <all-india-pincodes.csv>",
            attempted,
        )


def clear_pincode_cache() -> None:
    """Reset both in-process caches. For tests — a fresh worker process
    starts with empty caches anyway, so production code never needs this."""
    _valid_pincode_cache.clear()
    _pincode_context_cache.clear()
