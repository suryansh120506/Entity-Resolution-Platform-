ALTER TABLE clusters DROP CONSTRAINT IF EXISTS ck_clusters_review_reason_valid;
ALTER TABLE clusters DROP COLUMN IF EXISTS review_reason;
ALTER TABLE jobs DROP CONSTRAINT IF EXISTS ck_jobs_source_uri_scheme;
ALTER TABLE jobs DROP COLUMN IF EXISTS source_uri;
ALTER TABLE model_versions DROP CONSTRAINT IF EXISTS ck_model_versions_parameters_object;
ALTER TABLE model_versions DROP CONSTRAINT IF EXISTS ck_model_versions_splink_has_parameters;
ALTER TABLE model_versions DROP COLUMN IF EXISTS layout_hash;
ALTER TABLE model_versions DROP COLUMN IF EXISTS parameters;
