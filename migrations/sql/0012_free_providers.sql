-- ============================================================================
-- ER-PLATFORM-001 · migration 0012 — Cerebras as a bring-your-own-key provider
--
-- Cerebras has a free tier, so a tenant's own key can cost nothing. The
-- provider list is a CHECK, not an enum type, so this only widens it.
-- ============================================================================

ALTER TABLE tenant_ai_settings DROP CONSTRAINT ck_tenant_ai_settings_provider;
ALTER TABLE tenant_ai_settings
  ADD CONSTRAINT ck_tenant_ai_settings_provider
    CHECK (provider IS NULL OR provider IN ('openai', 'anthropic', 'cerebras'));
