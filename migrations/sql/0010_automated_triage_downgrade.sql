-- Reverses 0010_automated_triage.sql.

UPDATE tenants SET confidence_api_floor = 0.750 WHERE confidence_api_floor = 0.800;
ALTER TABLE tenants ALTER COLUMN confidence_api_floor SET DEFAULT 0.750;

DELETE FROM enrichment_results WHERE strategy = 'golden';
ALTER TABLE enrichment_results DROP CONSTRAINT ck_enrichment_results_strategy_valid;
ALTER TABLE enrichment_results
  ADD CONSTRAINT ck_enrichment_results_strategy_valid
    CHECK (strategy IN ('registry', 'serp_llm'));

ALTER TABLE clusters
  DROP CONSTRAINT ck_clusters_enrichment_object,
  DROP CONSTRAINT ck_clusters_auto_decision_object,
  DROP CONSTRAINT ck_clusters_resolution_source_valid;

ALTER TABLE clusters
  DROP COLUMN enrichment,
  DROP COLUMN auto_decision,
  DROP COLUMN resolution_source;
