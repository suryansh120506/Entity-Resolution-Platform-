-- ============================================================================
-- ER-PLATFORM-001 · migration 0010 — automated triage and golden enrichment
--
-- The review queue is replaced by three automatic tiers:
--
--   Tier 1  the confidence gate decides alone: >= auto_merge merges,
--           < confidence_api_floor (now 0.80) is kept separate.
--   Tier 2  the band between waits in PENDING_API while a worker searches
--           the web and asks an LLM "same entity, YES or NO?".
--   Tier 3  every resolved golden record is enriched from web snippets,
--           keeping only values found verbatim in a snippet.
--
-- A machine decision must be as reconstructable as a steward's (NFR-04), so
-- each one is recorded on the cluster: who decided (`resolution_source`)
-- and on what evidence (`auto_decision`, `enrichment`). Steward verdicts
-- keep their own audit trail in human_feedback and leave these NULL.
-- ============================================================================

ALTER TABLE clusters
  ADD COLUMN resolution_source text,
  ADD COLUMN auto_decision jsonb,
  ADD COLUMN enrichment jsonb;

ALTER TABLE clusters
  ADD CONSTRAINT ck_clusters_resolution_source_valid
    CHECK (resolution_source IS NULL OR resolution_source IN ('threshold', 'llm_triage')),
  ADD CONSTRAINT ck_clusters_auto_decision_object
    CHECK (auto_decision IS NULL OR jsonb_typeof(auto_decision) = 'object'),
  ADD CONSTRAINT ck_clusters_enrichment_object
    CHECK (enrichment IS NULL OR jsonb_typeof(enrichment) = 'object');

-- Golden enrichment is cached per stable entity_id, so a re-run of the same
-- schools does not search the web again until the cache expires.
ALTER TABLE enrichment_results DROP CONSTRAINT ck_enrichment_results_strategy_valid;
ALTER TABLE enrichment_results
  ADD CONSTRAINT ck_enrichment_results_strategy_valid
    CHECK (strategy IN ('registry', 'serp_llm', 'golden'));

-- The floor is now the Tier 1 cut: below it a cluster is kept separate with
-- no review. Tenants still on the old default move to the new one; a tenant
-- that chose its own floor keeps it.
ALTER TABLE tenants ALTER COLUMN confidence_api_floor SET DEFAULT 0.800;
UPDATE tenants SET confidence_api_floor = 0.800 WHERE confidence_api_floor = 0.750;
