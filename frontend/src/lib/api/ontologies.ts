import { apiFetch } from "./client";
import type { DomainDraftResponse, OntologyDraft, OntologyOption } from "./types";

/**
 * POST /v1/ontologies/generate — the active AI tier drafts a data domain.
 * `sampleRows` go to the AI provider: keep them few and short.
 */
export function generateDomain(
  description: string,
  headers: string[],
  sampleRows: Record<string, string>[],
  init?: RequestInit,
): Promise<DomainDraftResponse> {
  return apiFetch<DomainDraftResponse>("/ontologies/generate", {
    ...init,
    method: "POST",
    body: { description, headers, sample_rows: sampleRows },
  });
}

/** POST /v1/ontologies — save a reviewed draft. The server re-checks it. */
export function saveDomain(
  draft: OntologyDraft,
  prompt: string,
  sampleRows: Record<string, string>[],
): Promise<OntologyOption> {
  return apiFetch<OntologyOption>("/ontologies", {
    method: "POST",
    body: { draft, prompt, sample_rows: sampleRows },
  });
}
