import { apiFetch } from "./client";
import type { AiSettings, AiSettingsUpdate } from "./types";

/** GET /v1/settings/ai — the tenant's AI tier. Never returns the key. */
export function getAiSettings(init?: RequestInit): Promise<AiSettings> {
  return apiFetch<AiSettings>("/settings/ai", init);
}

/**
 * PUT /v1/settings/ai — Standard, or Premium with a key. The server verifies
 * a Premium key with one call to the provider before storing it, so a 422
 * here carries the provider's own reason.
 */
export function updateAiSettings(update: AiSettingsUpdate): Promise<AiSettings> {
  return apiFetch<AiSettings>("/settings/ai", { method: "PUT", body: update });
}

/** DELETE /v1/settings/ai/key — forget the key, back to Standard. */
export function deleteAiKey(): Promise<AiSettings> {
  return apiFetch<AiSettings>("/settings/ai/key", { method: "DELETE" });
}
