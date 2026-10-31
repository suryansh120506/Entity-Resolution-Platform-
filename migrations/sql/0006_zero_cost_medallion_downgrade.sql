-- Reverses 0006_zero_cost_medallion.sql. Destroys queue history, Silver,
-- enrichment cache and all Fellegi-Sunter feedback state.

-- A parallel HNSW build over a populated table allocates its graph in
-- dynamic shared memory, which overflows Docker's default 64MB /dev/shm.
-- Serial is slower and fits.
SET LOCAL max_parallel_maintenance_workers = 0;

CREATE INDEX IF NOT EXISTS ix_record_embeddings_name_embedding_hnsw
  ON record_embeddings USING hnsw (name_embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS ix_record_embeddings_address_embedding_hnsw
  ON record_embeddings USING hnsw (address_embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

ALTER TABLE clusters DROP CONSTRAINT IF EXISTS ck_clusters_density_unit;
ALTER TABLE clusters DROP COLUMN IF EXISTS density;
DROP INDEX IF EXISTS ix_clusters_tenant_id_entity_id;
ALTER TABLE clusters DROP COLUMN IF EXISTS entity_id;

DROP VIEW IF EXISTS splink_level_estimates;
DROP TABLE IF EXISTS entity_constraints;
DROP TABLE IF EXISTS splink_level_stats;
DROP TABLE IF EXISTS pair_labels;
DROP FUNCTION IF EXISTS pair_labels_apply_stats();
DROP TABLE IF EXISTS feedback_events;
DROP TABLE IF EXISTS pair_scores;
DROP TABLE IF EXISTS splink_settings;
DROP TABLE IF EXISTS enrichment_results;
DROP TABLE IF EXISTS silver_records;
ALTER TABLE jobs DROP CONSTRAINT IF EXISTS ck_jobs_ontology_name;
ALTER TABLE jobs DROP COLUMN IF EXISTS ontology;
DROP TABLE IF EXISTS task_queue;
DROP FUNCTION IF EXISTS task_queue_notify();
