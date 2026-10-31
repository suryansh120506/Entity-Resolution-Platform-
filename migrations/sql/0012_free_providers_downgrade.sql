-- Reverses 0012_free_providers.sql. A tenant on Cerebras goes back to Standard
-- AI (the premium CHECK requires a provider, so the row cannot stay premium).

UPDATE tenant_ai_settings
   SET mode = 'standard', provider = NULL, model = NULL, api_key_ciphertext = NULL,
       key_hint = NULL, verified_at = NULL
 WHERE provider = 'cerebras';

ALTER TABLE tenant_ai_settings DROP CONSTRAINT ck_tenant_ai_settings_provider;
ALTER TABLE tenant_ai_settings
  ADD CONSTRAINT ck_tenant_ai_settings_provider
    CHECK (provider IS NULL OR provider IN ('openai', 'anthropic'));
