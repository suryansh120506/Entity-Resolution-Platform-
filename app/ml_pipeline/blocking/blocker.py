"""FR-03 / SAD 3.4 — candidate generation over the pgvector HNSW indexes.

Comparing every row against every other row is quadratic: a 50,000-row job is
1.25 billion pairs, and Splink scoring that many is not a slow pipeline, it is
a pipeline that never finishes. Blocking is the step that makes the problem
tractable — an approximate nearest-neighbour search returns the handful of
rows any given record could plausibly match, and only those pairs are scored.

**Name and address are searched independently (FR-03).** Two separate HNSW
indexes, two separate searches, and the results are unioned. Searching a
blended vector instead would let a generic name ("Government Primary School",
of which there are thousands) pull in every similarly-generic name in the
country, and let a shared address pull in unrelated schools sharing a
building. Independently, each search has to stand on its own, and a pair only
needs to be plausible on *one* axis to be worth scoring — the matcher decides
whether it is plausible on both.

**Recall under RLS: why this over-fetches.** `app/models/embedding.py` states
the trap and this is the code that pays for it — pgvector applies the HNSW
approximation *first* and the tenant RLS predicate *second*. Asking for the
top 10 neighbours returns 10 rows from the whole index, which are then
filtered down to the ones this tenant may see, and the result can be far
fewer than 10 (or zero) with no error and no warning anywhere. So every query
asks for `top_k * oversample` and truncates in Python. That converts a silent
recall hole into an over-fetch we control. The same applies to the optional
`job_id` filter, which is post-applied for the same reason.

Nothing here writes to the database, and nothing here decides whether a pair
is a match — this stage only proposes. `app/ml_pipeline/matching/matcher.py`
scores.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import Float, Select, select, text
from sqlalchemy.orm import Session

from app.models.embedding import RecordEmbedding
from app.models.school_record import SchoolRecord

logger = logging.getLogger(__name__)

#: Neighbours to keep per probe, per axis. 20 is deliberately generous: a
#: missed candidate here can never be recovered downstream — the matcher only
#: ever sees pairs this stage proposed — whereas a spurious one costs a single
#: Splink comparison and is then scored away.
DEFAULT_TOP_K = 20

#: Cosine distance ceiling. pgvector's `<=>` returns 1 - cosine_similarity.
#:
#: 0.35 was the original value and is too loose for this domain — measured
#: live against 300 real `Valid` school records (44,850 pairs), the 1st
#: percentile of *all* pairwise name-embedding distances, most of which are
#: unrelated schools, was already 0.131. A multilingual sentence-transformer
#: embedding short, template-heavy strings ("GOVT SR SEC SCHOOL, <village>")
#: compresses most of the corpus into a narrow distance band, so 0.35 was
#: accepting something like the closest 15-20% of *all* pairs as candidates
#: — confirmed to chain 100% of a 901-record batch into one connected
#: component (`blocking_stage`'s `MAX_CLUSTER_SIZE` refuses that outcome
#: rather than persist it, but a refused component is zero usable clusters,
#: not a working pipeline).
#:
#: 0.15 sits just below that noise floor. It is an empirical starting point
#: from one dataset's distance distribution, not a validated precision/recall
#: tradeoff — retune against labelled duplicate/non-duplicate pairs once some
#: exist.
DEFAULT_MAX_DISTANCE = 0.15

#: Multiplier on the database-side LIMIT. See the module docstring.
DEFAULT_OVERSAMPLE = 3


@dataclass(frozen=True)
class ProbeRecord:
    """One record to search neighbours for.

    `address_embedding` is optional so that a record whose address never
    normalised into anything usable can still be blocked on its name alone,
    rather than being dropped from candidate generation entirely.
    """

    record_id: uuid.UUID
    name_embedding: Sequence[float]
    address_embedding: Sequence[float] | None = None


@dataclass(frozen=True)
class CandidatePair:
    """An unordered pair worth scoring, with the evidence that proposed it.

    `record_id_a` is always the lexicographically smaller UUID, so a pair is
    the same object regardless of which side was the probe — that is what
    makes deduplication across the two searches possible at all.

    `distance` is the best (smallest) cosine distance seen for this pair, and
    `source` names the axis that produced it: "name", "address", or "both"
    when independent searches agreed. "both" is a genuinely stronger signal
    than either alone and is preserved for the matcher rather than collapsed.
    """

    record_id_a: uuid.UUID
    record_id_b: uuid.UUID
    distance: float
    source: str


class CandidateBlocker:
    """ANN candidate generation against `record_embeddings`.

    The session must already be tenant-bound (`app.db.session.bind_tenant`);
    this class never filters on `tenant_id` itself. That is not an oversight —
    a hand-written tenant predicate here would be a second, weaker copy of the
    RLS policy that FIG 2.8 already enforces, and the failure mode of the two
    disagreeing is worse than having only the one that the database applies
    unconditionally.
    """

    def __init__(
        self,
        db: Session,
        *,
        top_k: int = DEFAULT_TOP_K,
        max_distance: float = DEFAULT_MAX_DISTANCE,
        oversample: int = DEFAULT_OVERSAMPLE,
        ef_search: int | None = None,
        search_address: bool = True,
        job_id: uuid.UUID | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if oversample < 1:
            raise ValueError(f"oversample must be >= 1, got {oversample}")
        if not 0.0 <= max_distance <= 2.0:
            # Cosine distance is bounded [0, 2]; anything outside is a unit
            # mix-up (a similarity passed where a distance was wanted), which
            # would otherwise silently accept or reject everything.
            raise ValueError(f"max_distance must be within [0, 2], got {max_distance}")

        self.db = db
        self.top_k = top_k
        self.max_distance = max_distance
        self.oversample = oversample
        self.ef_search = ef_search
        self.search_address = search_address
        self.job_id = job_id

    # --- Query construction ----------------------------------------------------

    def _neighbour_query(
        self, column: Any, embedding: Sequence[float], exclude_id: uuid.UUID
    ) -> Select[Any]:
        """Top-N by cosine distance on one embedding column.

        `.cosine_distance()` is pgvector's `<=>`. The same expression is used
        in both the SELECT list and the ORDER BY so the planner sees one
        expression and can serve the sort from the HNSW index instead of
        re-computing the distance per row.
        """
        distance = column.cosine_distance(list(embedding)).cast(Float).label("distance")

        query = (
            select(RecordEmbedding.record_id, distance)
            .where(RecordEmbedding.record_id != exclude_id)
            .order_by(distance)
            .limit(self.top_k * self.oversample)
        )

        if self.job_id is not None:
            query = query.join(SchoolRecord, SchoolRecord.id == RecordEmbedding.record_id).where(
                SchoolRecord.job_id == self.job_id
            )

        return query

    def _apply_ef_search(self) -> None:
        """Raise HNSW's search breadth for this transaction only.

        pgvector's default `ef_search` of 40 is tuned for small k. Asking for
        `top_k * oversample` neighbours with a default ef_search quietly costs
        recall, and the whole point of the over-fetch is recall. LOCAL so the
        setting dies with the transaction rather than following the connection
        back into the pool and changing the behaviour of unrelated queries.
        """
        if self.ef_search is None:
            return
        # Not a bind parameter: SET LOCAL is utility syntax that cannot take
        # one. Coerced through int() so nothing but a number reaches the SQL.
        self.db.execute(text(f"SET LOCAL hnsw.ef_search = {int(self.ef_search)}"))

    # --- Candidate generation --------------------------------------------------

    def find_candidates(self, probes: Sequence[ProbeRecord]) -> Iterator[CandidatePair]:
        """Yield unique candidate pairs for a batch of probes.

        Deduplicated across both axes and both directions: if A finds B by
        name and B later finds A by address, one pair is emitted, carrying the
        smaller distance and `source="both"`.

        Memory is O(number of distinct pairs) because deduplication needs to
        remember what it has already emitted — roughly `len(probes) * top_k`
        entries at worst. At the 50k-row scale in the TRD that is a few
        hundred MB at the very worst and typically far less; a job an order of
        magnitude larger should block in chunks and deduplicate downstream.
        """
        if not probes:
            return

        self._apply_ef_search()

        # pair -> (best distance, set of axes that proposed it)
        found: dict[tuple[uuid.UUID, uuid.UUID], tuple[float, set[str]]] = {}

        for probe in probes:
            searches: list[tuple[str, Any, Sequence[float]]] = [
                ("name", RecordEmbedding.name_embedding, probe.name_embedding)
            ]
            if self.search_address and probe.address_embedding is not None:
                searches.append(
                    ("address", RecordEmbedding.address_embedding, probe.address_embedding)
                )

            for axis, column, embedding in searches:
                query = self._neighbour_query(column, embedding, probe.record_id)
                kept = 0
                for neighbour_id, distance in self.db.execute(query):
                    if kept >= self.top_k:
                        break
                    if distance is None or distance > self.max_distance:
                        # Rows come back sorted, so the first row past the
                        # ceiling means every row after it is too.
                        break
                    if neighbour_id == probe.record_id:
                        continue

                    kept += 1
                    key = _canonical_pair(probe.record_id, neighbour_id)
                    existing = found.get(key)
                    if existing is None:
                        found[key] = (float(distance), {axis})
                    else:
                        best, axes = existing
                        axes.add(axis)
                        found[key] = (min(best, float(distance)), axes)

        for (record_id_a, record_id_b), (distance, axes) in found.items():
            yield CandidatePair(
                record_id_a=record_id_a,
                record_id_b=record_id_b,
                distance=distance,
                source="both" if len(axes) > 1 else next(iter(axes)),
            )

    def candidate_frame(self, probes: Sequence[ProbeRecord]) -> pd.DataFrame:
        """`find_candidates` as the DataFrame the matcher consumes.

        Column names line up with `RecordMatcher.score_pairs`, so the two
        stages compose without an adapter in between.
        """
        pairs = list(self.find_candidates(probes))
        frame = pd.DataFrame(
            [
                {
                    "record_id_a": str(pair.record_id_a),
                    "record_id_b": str(pair.record_id_b),
                    "distance": pair.distance,
                    "source": pair.source,
                }
                for pair in pairs
            ],
            columns=["record_id_a", "record_id_b", "distance", "source"],
        )
        logger.info(
            "Blocked %d probes into %d candidate pairs (top_k=%d, max_distance=%.3f)",
            len(probes),
            len(frame),
            self.top_k,
            self.max_distance,
        )
        return frame


def _canonical_pair(left: uuid.UUID, right: uuid.UUID) -> tuple[uuid.UUID, uuid.UUID]:
    """Order a pair so (A, B) and (B, A) are the same key."""
    return (left, right) if str(left) <= str(right) else (right, left)
