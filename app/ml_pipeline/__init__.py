"""ML pipeline stages (TRD 2.2, SAD 3.3). No implementations in Step 1.

One subpackage per pipeline stage, in the order FIG 3.2 runs them:

  classification/  XGBoost three-class ingestion classifier — Valid / Junk /
                   Missing_Context. Engineered features plus one
                   embedding-derived feature (distance from the row's name
                   embedding to a per-tenant centroid of confirmed-valid rows).
                   Cold-starts on heuristic pseudo-labels, tracked separately
                   from real steward labels so the pseudo-labels can be
                   discarded without contaminating the ground-truth set.

  embedding/       paraphrase-multilingual-mpnet-base-v2, pinned by revision
                   hash. Two independent 768-dim vectors per record, computed
                   from the normalized strings.

  blocking/        pgvector HNSW candidate search, per tenant. Name and
                   address searched independently (FR-03), so a generic name
                   in two cities is never one candidate cluster.

  matching/        Splink probabilistic scoring. Produces the single [0,1]
                   confidence per cluster that FR-04 specifies — the score is
                   Splink's output, NOT the raw embedding distance.

  validation/      Google Places lookup for the mid confidence band, behind
                   the query_hash cache, the field mask, and the per-tenant
                   budget check (SAD 3.5).

Nothing under the five stage packages above is importable from the API
process: those modules pull in pandas / torch / xgboost, which the API
image does not install (see pyproject.toml's [ml] extra and NFR-05).

Two flat modules are the deliberate exception, and stay dependency-light on
purpose so they remain safe for the API to import:

  fields.py    Canonical field extraction (`canonical_fields`) — the one
               alias table for reading name/address/district/state/pincode
               out of a raw CSV row, whatever it called them. Both
               `classification/classifier.py` (features) and the cluster
               API route (display) read through this rather than keeping
               their own copies, which is what silently drifting alias
               tables used to allow.

  cleaning.py  Display-value cleanup (`clean_value`) — strips the
               redundant-comma / null-sentinel noise a raw export carries,
               without touching `raw_row` itself.
"""
