-- Reverses 0011_byok_custom_domains_mapping.sql.

ALTER TABLE jobs DROP CONSTRAINT ck_jobs_column_mapping_object;
ALTER TABLE jobs DROP COLUMN column_mapping;
DROP TABLE tenant_ontologies;
DROP TABLE tenant_ai_settings;
