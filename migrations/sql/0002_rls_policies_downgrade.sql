-- ============================================================================
-- ER-PLATFORM-001 · migration 0002 — DOWNGRADE (removes RLS)
--
-- Running this leaves the schema with exactly one layer of tenant isolation:
-- whatever `WHERE tenant_id = ?` the application remembers to write. That is
-- the layer TRD NFR-02 explicitly declines to rely on.
--
-- This exists so that `alembic downgrade` is not a dead end during local
-- development. It should not be run against an environment holding more than
-- one tenant's data.
-- ============================================================================

DROP POLICY IF EXISTS tenant_isolation ON dead_letter_tasks;
DROP POLICY IF EXISTS tenant_isolation ON model_versions;
DROP POLICY IF EXISTS tenant_isolation ON human_feedback;
DROP POLICY IF EXISTS tenant_isolation ON google_places_validations;
DROP POLICY IF EXISTS tenant_isolation ON cluster_members;
DROP POLICY IF EXISTS tenant_isolation ON clusters;
DROP POLICY IF EXISTS tenant_isolation ON record_embeddings;
DROP POLICY IF EXISTS tenant_isolation ON school_records;
DROP POLICY IF EXISTS tenant_isolation ON jobs;
DROP POLICY IF EXISTS tenant_self_isolation ON tenants;

ALTER TABLE dead_letter_tasks NO FORCE ROW LEVEL SECURITY;
ALTER TABLE dead_letter_tasks DISABLE ROW LEVEL SECURITY;
ALTER TABLE model_versions NO FORCE ROW LEVEL SECURITY;
ALTER TABLE model_versions DISABLE ROW LEVEL SECURITY;
ALTER TABLE human_feedback NO FORCE ROW LEVEL SECURITY;
ALTER TABLE human_feedback DISABLE ROW LEVEL SECURITY;
ALTER TABLE google_places_validations NO FORCE ROW LEVEL SECURITY;
ALTER TABLE google_places_validations DISABLE ROW LEVEL SECURITY;
ALTER TABLE cluster_members NO FORCE ROW LEVEL SECURITY;
ALTER TABLE cluster_members DISABLE ROW LEVEL SECURITY;
ALTER TABLE clusters NO FORCE ROW LEVEL SECURITY;
ALTER TABLE clusters DISABLE ROW LEVEL SECURITY;
ALTER TABLE record_embeddings NO FORCE ROW LEVEL SECURITY;
ALTER TABLE record_embeddings DISABLE ROW LEVEL SECURITY;
ALTER TABLE school_records NO FORCE ROW LEVEL SECURITY;
ALTER TABLE school_records DISABLE ROW LEVEL SECURITY;
ALTER TABLE jobs NO FORCE ROW LEVEL SECURITY;
ALTER TABLE jobs DISABLE ROW LEVEL SECURITY;
ALTER TABLE tenants NO FORCE ROW LEVEL SECURITY;
ALTER TABLE tenants DISABLE ROW LEVEL SECURITY;
