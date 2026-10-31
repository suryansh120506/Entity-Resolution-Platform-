"""Constants that more than one layer has to agree on.

Anything in here is referenced from at least two places — usually a Python
module and a SQL migration — so a change in one without the other is a bug.
"""

from __future__ import annotations

from typing import Final

# --- Tenancy ------------------------------------------------------------------

#: The Postgres session variable that every RLS policy reads (TRD FIG 2.8).
#: Set per transaction by the DB dependency, from the verified JWT's tenant_id
#: claim. The string literal is duplicated in migrations/sql/0002_rls_policies.sql
#: because a policy body cannot reference a Python constant — if you change it
#: here, change it there, and vice versa.
TENANT_GUC: Final[str] = "app.tenant_id"

#: Tables covered by the tenant_isolation policy. TRD 2.4 describes "nine"
#: tenant-scoped tables and this is that set. `tenants` itself is governed by a
#: separate self-scoping policy (it keys on `id`, not `tenant_id`).
#: Kept in sync with migrations/sql/0002_rls_policies.sql — tests/test_rls_isolation.py
#: asserts the two agree against the live database.
TENANT_SCOPED_TABLES: Final[tuple[str, ...]] = (
    "jobs",
    "school_records",
    "record_embeddings",
    "clusters",
    "cluster_members",
    "candidate_pairs",
    "google_places_validations",
    "human_feedback",
    "model_versions",
    "dead_letter_tasks",
    # migration 0006
    "task_queue",
    "silver_records",
    "enrichment_results",
    "splink_settings",
    "pair_scores",
    "feedback_events",
    "pair_labels",
    "splink_level_stats",
    "entity_constraints",
    # migration 0008
    "entity_group_constraints",
    "gold_links",
    # migration 0011
    "tenant_ai_settings",
    "tenant_ontologies",
)

# --- ML pinning (TRD 2.2) ------------------------------------------------------

#: Dimensionality of paraphrase-multilingual-mpnet-base-v2. This is baked into
#: the vector(768) column type in FIG 2.4, so it is NOT runtime-configurable —
#: changing it is a migration, not a setting.
EMBEDDING_DIMENSIONS: Final[int] = 768

# --- Confidence bands (TRD 2.2) ------------------------------------------------
# Platform defaults only. The operative values are per-tenant columns on
# `tenants` (FIG 2.1) — read them from the tenant row, never from here.

DEFAULT_CONFIDENCE_AUTO_MERGE: Final[str] = "0.950"
#: Tier 1 cut (migration 0010): below this a cluster is kept separate, no review.
DEFAULT_CONFIDENCE_API_FLOOR: Final[str] = "0.800"
