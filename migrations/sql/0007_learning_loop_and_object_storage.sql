-- ============================================================================
-- ER-PLATFORM-001 · migration 0007 — learning loop + object-storage uploads
--
-- model_versions.parameters / layout_hash
--     The trained Splink settings (m, u, prior) of a tenant's base model, and
--     the hash of the comparison layout they are valid for. Matching reuses
--     the active base model and blends steward evidence from
--     splink_level_stats into it, instead of re-running EM on every job.
-- jobs.source_uri
--     Where the upload lives: s3://bucket/key for presigned direct uploads,
--     a local path for the multipart fallback.
-- clusters.review_reason
--     Why a cluster must go to a human regardless of score (a steward
--     constraint conflict). The gate reads it; NULL means "score decides".
-- ============================================================================

ALTER TABLE model_versions ADD COLUMN parameters jsonb;
ALTER TABLE model_versions ADD COLUMN layout_hash text;
ALTER TABLE model_versions ADD CONSTRAINT ck_model_versions_splink_has_parameters
  CHECK (model_type <> 'splink' OR (parameters IS NOT NULL AND layout_hash ~ '^[0-9a-f]{64}$'));
ALTER TABLE model_versions ADD CONSTRAINT ck_model_versions_parameters_object
  CHECK (parameters IS NULL OR jsonb_typeof(parameters) = 'object');

ALTER TABLE jobs ADD COLUMN source_uri text;
ALTER TABLE jobs ADD CONSTRAINT ck_jobs_source_uri_scheme
  CHECK (source_uri IS NULL OR source_uri ~ '^(s3://[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]/.+|/.+|[A-Za-z]:.+|\./.+)$');

ALTER TABLE clusters ADD COLUMN review_reason text;
ALTER TABLE clusters ADD CONSTRAINT ck_clusters_review_reason_valid
  CHECK (review_reason IS NULL OR review_reason IN ('constraint_conflict'));
