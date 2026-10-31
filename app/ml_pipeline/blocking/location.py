"""Location-partitioned candidate generation — the pipeline's blocking path.

Two schools can only be the same school if they are in the same place, so
candidates are only ever proposed between records that share a **district**
or a **pincode**. Inside each such block the nearest neighbours are found
exactly, with one matrix product, rather than by an approximate index search.

Why this replaced per-record pgvector queries for per-job blocking:

* **Speed.** The HNSW path issued one query per record per axis — ~43,000
  round trips for a 21,700-record job — and took hours. Every embedding a
  job's blocking needs is already loaded into this process, so the search is
  a handful of in-memory matrix products instead: seconds.
* **Precision.** Nationally, generic names ("GOVT SR SEC SCHOOL") sit so
  close together in embedding space that any usable distance threshold
  chained unrelated schools across states into one component. Inside a
  district that chain has nowhere to go.
* **Exactness.** A block is small enough to search exhaustively, so there is
  no approximation to lose recall to — and no RLS-after-approximation trap
  (see `blocker.py`), because nothing here touches the index at all.

`CandidateBlocker` in `blocker.py` stays as the tenant-wide ANN search over
the HNSW indexes; this module is what the per-job pipeline uses.

**Multi-pass blocking.** A record joins one block per key it has — its
district (from the reference pincode table when its PIN is real, and from
its own district column too if that spells it differently) and its pincode.
A pair is a candidate if it shares *any* block, which is what covers a
misspelled district (the pincode still matches) and a mistyped pincode (the
district still matches).

**Records with no location** (no district, no usable pincode) are probed
against every record in their state — asymmetrically, so they cannot chain a
whole state together through each other. A record with no state either is
left out and counted, not guessed at.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from app.ml_pipeline.blocking.blocker import CandidatePair

logger = logging.getLogger(__name__)

#: Neighbours kept per record, per axis, per block. Lower than the national
#: search's 20 because a block is already a strong filter; the scorer sees at
#: most `2 * top_k` pairs per record per block.
DEFAULT_TOP_K = 10

#: Cosine distance ceiling within a block. Deliberately looser than the
#: national path's 0.15: the location match has already done the filtering
#: that threshold was compensating for, and blocking is the one stage whose
#: misses can never be recovered — a spurious candidate costs one Splink
#: comparison, a missed one is a duplicate nobody ever sees.
DEFAULT_MAX_DISTANCE = 0.30

#: Rows per matrix product. Bounds peak memory at `chunk * block_size`
#: floats, so a 5,000-record metro district costs ~20MB, not ~100MB.
DEFAULT_CHUNK_ROWS = 1024

_PINCODE = re.compile(r"[1-9]\d{5}")


def normalise_place(value: str) -> str:
    """Lowercase, letters and digits only — "Tamil Nadu" == "TAMILNADU"."""
    return "".join(ch for ch in value.lower() if ch.isalnum())


def usable_pincode(value: str) -> str | None:
    """The PIN if it is six digits and not an obvious sentinel, else None."""
    pin = value.strip()
    if not _PINCODE.fullmatch(pin) or len(set(pin)) == 1:
        return None
    return pin


@dataclass(frozen=True)
class BlockingRecord:
    """One embedded record plus the place it claims to be in.

    `has_name` / `has_address` exist because an empty string still embeds to
    a vector — the *same* vector for every empty string — so two records with
    blank addresses would otherwise look like a perfect address match.
    """

    record_id: uuid.UUID
    name_vector: np.ndarray
    address_vector: np.ndarray
    has_name: bool
    has_address: bool
    district_keys: frozenset[str]
    pincode: str | None
    state: str | None


def location_keys(
    fields: Mapping[str, str],
    reference: Mapping[str, tuple[str, str]],
) -> tuple[frozenset[str], str | None, str | None]:
    """(district keys, pincode, state) for one record's canonical fields.

    `reference` maps a real PIN to its (district, state) from
    `reference_pincodes`. Where a record's PIN is real, the government's
    spelling of its district is used, so "GURGAON" and "GURUGRAM" records
    sharing a PIN land in the same block; the record's own district column is
    kept as a second key when it spells the place differently, so a
    duplicate with no PIN can still meet it.
    """
    state = normalise_place(fields.get("state", ""))
    district = normalise_place(fields.get("district", ""))
    pincode = usable_pincode(fields.get("pincode", ""))

    keys: set[str] = set()
    if pincode and pincode in reference:
        ref_district, ref_state = reference[pincode]
        ref_state_n = normalise_place(ref_state)
        keys.add(f"{ref_state_n}|{normalise_place(ref_district)}")
        state = state or ref_state_n
    if district:
        keys.add(f"{state}|{district}")

    return frozenset(keys), pincode, state or None


@dataclass
class BlockingStats:
    """What blocking did, for `stage_progress` and the results page."""

    records: int = 0
    blocks: int = 0
    largest_block: int = 0
    state_fallback_records: int = 0
    unlocated_records: int = 0
    pairs: int = 0
    by_source: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "records": self.records,
            "blocks": self.blocks,
            "largest_block": self.largest_block,
            "state_fallback_records": self.state_fallback_records,
            "unlocated_records": self.unlocated_records,
            "pairs": self.pairs,
            "by_source": dict(self.by_source),
        }


def find_location_candidates(
    records: Sequence[BlockingRecord],
    *,
    top_k: int = DEFAULT_TOP_K,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> tuple[list[CandidatePair], BlockingStats]:
    """Candidate pairs between records that share a district or a pincode."""
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if not 0.0 <= max_distance <= 2.0:
        raise ValueError(f"max_distance must be within [0, 2], got {max_distance}")

    stats = BlockingStats(records=len(records))
    if len(records) < 2:
        return [], stats

    names = np.stack([r.name_vector for r in records]).astype(np.float32, copy=False)
    addresses = np.stack([r.address_vector for r in records]).astype(np.float32, copy=False)
    has_name = np.array([r.has_name for r in records], dtype=bool)
    has_address = np.array([r.has_address for r in records], dtype=bool)

    # Symmetric blocks: every member probes every other member.
    blocks: dict[str, list[int]] = {}
    by_state: dict[str, list[int]] = {}
    fallback_by_state: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        for key in record.district_keys:
            blocks.setdefault(f"d:{key}", []).append(index)
        if record.pincode:
            blocks.setdefault(f"p:{record.pincode}", []).append(index)
        if record.state:
            by_state.setdefault(record.state, []).append(index)
        if not record.district_keys and not record.pincode:
            if record.state:
                fallback_by_state.setdefault(record.state, []).append(index)
                stats.state_fallback_records += 1
            else:
                stats.unlocated_records += 1

    # (probe indices, candidate indices). Fallback blocks are asymmetric: only
    # the unlocated records probe, against everyone in their state.
    work: list[tuple[np.ndarray, np.ndarray]] = [
        (np.asarray(members), np.asarray(members))
        for members in blocks.values()
        if len(members) > 1
    ]
    for state, fallback in fallback_by_state.items():
        in_state = by_state.get(state, [])
        if len(in_state) > 1:
            work.append((np.asarray(fallback), np.asarray(in_state)))

    stats.blocks = len(work)
    stats.largest_block = max((len(c) for _, c in work), default=0)

    axis_matrices: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("name", names, has_name),
        ("address", addresses, has_address),
    ]

    # (i, j) with i < j -> (best distance, axes that proposed it)
    found: dict[tuple[int, int], tuple[float, set[str]]] = {}

    for probes, candidates in work:
        for axis, matrix, usable in axis_matrices:
            probe_idx = probes[usable[probes]]
            cand_idx = candidates[usable[candidates]]
            if len(probe_idx) == 0 or len(cand_idx) < 2:
                continue
            _search_block(matrix, probe_idx, cand_idx, axis, top_k, max_distance, chunk_rows, found)

    pairs: list[CandidatePair] = []
    for (i, j), (distance, axes) in found.items():
        a, b = records[i].record_id, records[j].record_id
        if str(a) > str(b):
            a, b = b, a
        source = "both" if len(axes) > 1 else next(iter(axes))
        stats.by_source[source] = stats.by_source.get(source, 0) + 1
        pairs.append(CandidatePair(record_id_a=a, record_id_b=b, distance=distance, source=source))

    stats.pairs = len(pairs)
    logger.info("Location blocking: %s", stats.as_dict())
    return pairs, stats


def _search_block(
    matrix: np.ndarray,
    probe_idx: np.ndarray,
    cand_idx: np.ndarray,
    axis: str,
    top_k: int,
    max_distance: float,
    chunk_rows: int,
    found: dict[tuple[int, int], tuple[float, set[str]]],
) -> None:
    """Exact top-k by cosine distance for one block on one axis."""
    candidates = matrix[cand_idx]
    k = min(top_k, len(cand_idx) - 1)

    for start in range(0, len(probe_idx), chunk_rows):
        chunk = probe_idx[start : start + chunk_rows]
        # Vectors are L2-normalised at encode time, so the dot product is the
        # cosine similarity and 1 - it is pgvector's `<=>` distance.
        distances = 1.0 - matrix[chunk] @ candidates.T
        # A record is never its own neighbour.
        distances[chunk[:, None] == cand_idx[None, :]] = np.inf

        nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k] if k > 0 else None
        if nearest is None:
            continue

        for row, probe in enumerate(chunk):
            for col in nearest[row]:
                distance = float(distances[row, col])
                if distance > max_distance:
                    continue
                other = int(cand_idx[col])
                key = (int(probe), other) if probe < other else (other, int(probe))
                existing = found.get(key)
                if existing is None:
                    found[key] = (distance, {axis})
                else:
                    best, axes = existing
                    axes.add(axis)
                    found[key] = (min(best, distance), axes)
