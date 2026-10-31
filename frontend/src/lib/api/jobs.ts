import { API_BASE_URL, ApiError, apiFetch, authHeader } from "./client";
import type {
  ColumnMapping,
  JobAccepted,
  JobPage,
  JobStats,
  JobStatusResponse,
  JobSummary,
  PresignedUpload,
  RecordPage,
  RecordView,
  SystemStatus,
  UploadOptions,
} from "./types";

/** GET /v1/jobs/{job_id} — status straight from Postgres, not Celery. */
export function getJob(jobId: string, init?: RequestInit): Promise<JobStatusResponse> {
  return apiFetch<JobStatusResponse>(`/jobs/${jobId}`, init);
}

/**
 * GET /v1/jobs — this tenant's jobs, newest first.
 *
 * What the dashboard reads on mount so uploads survive a navigation or a
 * reload. Previously the dashboard only knew about jobs accepted in the
 * current tab's React state, which is why a job "disappeared" the moment a
 * steward followed the review-queue link and came back — the job was never
 * lost, there was just nowhere to ask the API for it again.
 */
export function listJobs(limit = 20, init?: RequestInit): Promise<JobPage> {
  return apiFetch<JobPage>(`/jobs?limit=${limit}`, init);
}

/** GET /v1/jobs/{job_id}/stats — counts aggregated in the database. */
export function getJobStats(jobId: string, init?: RequestInit): Promise<JobStats> {
  return apiFetch<JobStats>(`/jobs/${jobId}/stats`, init);
}

/** GET /v1/jobs/{job_id}/summary — the results page in one request. */
export function getJobSummary(jobId: string, init?: RequestInit): Promise<JobSummary> {
  return apiFetch<JobSummary>(`/jobs/${jobId}/summary`, init);
}

/** GET /v1/jobs/{job_id}/records — one page of rows, cleaned, with issues. */
export function listRecords(
  jobId: string,
  { view = "all", limit = 50, cursor }: { view?: RecordView; limit?: number; cursor?: string | null },
  init?: RequestInit,
): Promise<RecordPage> {
  const query = new URLSearchParams({ view, limit: String(limit) });
  if (cursor) query.set("cursor", cursor);
  return apiFetch<RecordPage>(`/jobs/${jobId}/records?${query}`, init);
}

/** DELETE /v1/jobs/{job_id} — refused (409) while the job is processing. */
export function deleteJob(jobId: string): Promise<void> {
  return apiFetch<void>(`/jobs/${jobId}`, { method: "DELETE" });
}

/**
 * GET /system/status via the `/backend-health` rewrite — deliberately not
 * under `/v1`, where every route requires a token, and not through
 * `apiFetch`, which would mint one it does not need.
 */
export async function getSystemStatus(): Promise<SystemStatus> {
  const response = await fetch("/backend-health/system/status", { cache: "no-store" });
  if (!response.ok) throw new ApiError(response.status, null, "Status check failed");
  return (await response.json()) as SystemStatus;
}

/**
 * Which CSV to download: the ID mapping (FR-09), the full cleaned file, or the
 * Gold database (one row per entity, with web-enriched fields and who decided).
 */
export type ExportKind = "mapping" | "cleaned" | "gold" | "client";

/**
 * Download URL for an export.
 *
 * Points at our own route handler, NOT at `/v1/jobs/{id}/export` directly.
 * A browser navigation carries no Authorization header and the backend route
 * requires one, so linking straight to the API returns 401 every time. The
 * handler attaches the token server-side and pipes the stream through.
 */
export function exportUrl(
  jobId: string,
  kind: ExportKind = "mapping",
  options: { partial?: boolean } = {},
): string {
  if (kind === "client") return `/api/export/${jobId}?kind=client`;
  if (kind === "gold") {
    return `/api/export/${jobId}?kind=gold${options.partial ? "&partial=1" : ""}`;
  }
  return kind === "cleaned" ? `/api/export/${jobId}?kind=cleaned` : `/api/export/${jobId}`;
}

/** GET /v1/jobs/upload-options — ontologies and which upload path to use. */
export function getUploadOptions(init?: RequestInit): Promise<UploadOptions> {
  return apiFetch<UploadOptions>("/jobs/upload-options", init);
}

export interface UploadHandle {
  promise: Promise<JobAccepted>;
  /** Abort the in-flight upload. */
  cancel: () => void;
}

/**
 * Upload a CSV by whichever path the backend offers.
 *
 * Direct (object storage configured): three steps, and the file's bytes go
 * only to the bucket —
 *   1. POST /v1/jobs/uploads      → job id + presigned PUT URL (JSON, tiny)
 *   2. PUT  <presigned URL>       → the file, browser → bucket, with progress
 *   3. POST /v1/jobs/{id}/uploaded → the API verifies the object and enqueues
 * The API and the Next.js proxy never carry the file, so no request-body
 * limit anywhere in front of the API applies to it.
 *
 * Fallback (no bucket, local dev): multipart POST /v1/jobs through the proxy.
 */
export function uploadCsv(
  file: File,
  ontology: string,
  direct: boolean,
  onProgress?: (fraction: number) => void,
  mapping?: ColumnMapping | null,
): UploadHandle {
  if (!direct) return uploadViaApi(file, ontology, onProgress, mapping);

  const state: { request: XMLHttpRequest | null; cancelled: boolean } = {
    request: null,
    cancelled: false,
  };

  const promise = (async (): Promise<JobAccepted> => {
    const presigned = await apiFetch<PresignedUpload>("/jobs/uploads", {
      method: "POST",
      body: {
        filename: file.name,
        size_bytes: file.size,
        ontology,
        ...(mapping ? { column_mapping: mapping } : {}),
      },
    });
    if (state.cancelled) throw new ApiError(0, null, "Upload cancelled.");

    await new Promise<void>((resolve, reject) => {
      const request = new XMLHttpRequest();
      state.request = request;
      request.open(presigned.method, presigned.upload_url);
      // No Authorization header: the signature in the URL is the credential,
      // and the bucket would reject an unexpected one.
      for (const [key, value] of Object.entries(presigned.headers)) {
        request.setRequestHeader(key, value);
      }
      request.upload.addEventListener("progress", (event) => {
        if (event.lengthComputable && onProgress) onProgress(event.loaded / event.total);
      });
      request.addEventListener("load", () =>
        request.status >= 200 && request.status < 300
          ? resolve()
          : reject(new ApiError(request.status, null, `Storage rejected the upload (${request.status}).`)),
      );
      request.addEventListener("error", () =>
        reject(
          new ApiError(
            0,
            null,
            "Could not reach object storage. Check the bucket's CORS rule for this origin.",
          ),
        ),
      );
      request.addEventListener("abort", () => reject(new ApiError(0, null, "Upload cancelled.")));
      request.send(file);
    });

    return apiFetch<JobAccepted>(`/jobs/${presigned.job_id}/uploaded`, { method: "POST" });
  })();

  return {
    promise,
    cancel: () => {
      state.cancelled = true;
      state.request?.abort();
    },
  };
}

/**
 * POST /v1/jobs — stream a CSV up and get a job id back (202).
 *
 * Uses XMLHttpRequest, not fetch, and that is the whole reason this function
 * exists rather than another `apiFetch` call: **fetch cannot report upload
 * progress.** `ReadableStream` request bodies are the nearest thing and are
 * neither widely supported nor simpler. For a file that can be hundreds of
 * megabytes, a UI that shows nothing between "clicked" and "done" is a UI
 * that looks broken for a minute, and the user's instinct is to click again —
 * which uploads it twice.
 *
 * The field name is `file`, matching the `UploadFile` parameter in
 * `app/api/v1/routes/jobs.py`. Content-Type is deliberately NOT set: the
 * browser has to generate the multipart boundary, and setting the header by
 * hand omits it and produces a 422 that reads as "field required".
 */
function uploadViaApi(
  file: File,
  ontology: string,
  onProgress?: (fraction: number) => void,
  mapping?: ColumnMapping | null,
): UploadHandle {
  const request = new XMLHttpRequest();

  const promise = new Promise<JobAccepted>((resolve, reject) => {
    void (async () => {
      let headers: Record<string, string>;
      try {
        headers = await authHeader();
      } catch (error) {
        reject(error);
        return;
      }

      request.open("POST", `${API_BASE_URL}/jobs`);
      for (const [key, value] of Object.entries(headers)) {
        request.setRequestHeader(key, value);
      }

      request.upload.addEventListener("progress", (event) => {
        if (event.lengthComputable && onProgress) {
          onProgress(event.loaded / event.total);
        }
      });

      request.addEventListener("load", () => {
        let parsed: unknown = null;
        try {
          parsed = JSON.parse(request.responseText);
        } catch {
          // Non-JSON body — a proxy error page, most likely.
        }

        if (request.status >= 200 && request.status < 300) {
          resolve(parsed as JobAccepted);
          return;
        }

        const detail =
          parsed && typeof parsed === "object" && "detail" in parsed
            ? (parsed as { detail: unknown }).detail
            : null;
        reject(new ApiError(request.status, detail));
      });

      request.addEventListener("error", () => {
        reject(
          new ApiError(
            0,
            null,
            "The upload could not reach the API. Is the backend running on port 8000?",
          ),
        );
      });

      request.addEventListener("abort", () => {
        reject(new ApiError(0, null, "Upload cancelled."));
      });

      const form = new FormData();
      form.append("file", file);
      form.append("ontology", ontology);
      if (mapping) form.append("column_mapping", JSON.stringify(mapping));
      request.send(form);
    })();
  });

  return { promise, cancel: () => request.abort() };
}
