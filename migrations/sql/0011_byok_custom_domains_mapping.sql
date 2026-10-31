-- ============================================================================
-- ER-PLATFORM-001 · migration 0011 — BYOK AI, custom data domains, column maps
--
-- tenant_ai_settings  Standard AI (the server's free LLM) or Premium AI (the
--                     tenant's own OpenAI/Anthropic key). The key is stored
--                     only as Fernet ciphertext (app/core/secrets.py) and is
--                     never returned by the API; `key_hint` is "…3f9a".
-- tenant_ontologies   Data domains a tenant generated from a description and
--                     confirmed. Immutable: editing one saves a new one, so
--                     a job's ontology never changes under it.
-- jobs.column_mapping The user's explicit CSV column -> field mapping, kept so
--                     the stored rows can be reconstructed from the original
--                     file (NFR-04).
-- ============================================================================

CREATE TABLE tenant_ai_settings (
  tenant_id         uuid PRIMARY KEY REFERENCES tenants(id) ON DELETE CASCADE,
  mode              text NOT NULL DEFAULT 'standard',
  provider          text,
  model             text,
  api_key_ciphertext text,
  key_hint          text,
  verified_at       timestamptz,
  updated_by        text,
  updated_at        timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT ck_tenant_ai_settings_mode CHECK (mode IN ('standard', 'premium')),
  CONSTRAINT ck_tenant_ai_settings_provider
    CHECK (provider IS NULL OR provider IN ('openai', 'anthropic')),
  -- Premium means a verified key exists; the API never stores an unverified one.
  CONSTRAINT ck_tenant_ai_settings_premium_has_key
    CHECK (mode <> 'premium'
           OR (provider IS NOT NULL AND model IS NOT NULL
               AND api_key_ciphertext IS NOT NULL AND verified_at IS NOT NULL))
);

CREATE TABLE tenant_ontologies (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id     uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  name          text NOT NULL,
  label         text NOT NULL,
  description   text NOT NULL DEFAULT '',
  -- The fully resolved spec (`extends` already merged), validated by
  -- OntologySpec before it is written.
  spec          jsonb NOT NULL,
  -- What the user typed, kept so a domain can be explained and regenerated.
  prompt        text NOT NULL DEFAULT '',
  created_by    text,
  created_at    timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT uq_tenant_ontologies_name UNIQUE (tenant_id, name),
  CONSTRAINT ck_tenant_ontologies_name CHECK (name ~ '^custom_[a-z0-9_]{1,55}$'),
  CONSTRAINT ck_tenant_ontologies_spec_object CHECK (jsonb_typeof(spec) = 'object')
);

ALTER TABLE jobs ADD COLUMN column_mapping jsonb;
ALTER TABLE jobs ADD CONSTRAINT ck_jobs_column_mapping_object
  CHECK (column_mapping IS NULL OR jsonb_typeof(column_mapping) = 'object');

DO $rls$
DECLARE
  t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['tenant_ai_settings', 'tenant_ontologies'] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
    EXECUTE format(
      'CREATE POLICY tenant_isolation ON %I '
      'USING (tenant_id = current_setting(''app.tenant_id'')::uuid) '
      'WITH CHECK (tenant_id = current_setting(''app.tenant_id'')::uuid)', t);
  END LOOP;
END
$rls$;
