/**
 * The API client: base URL, the auth interceptor, and error normalisation.
 *
 * Every call to the backend goes through here. That is the point — the
 * bearer token is attached in exactly one place, so a new endpoint cannot be
 * written without it, and swapping the dev token for a real IdP is a change
 * to `getToken` alone.
 *
 * Errors are normalised into `ApiError` so callers branch on `status` rather
 * than re-parsing FastAPI's `{detail: ...}` envelope at every call site —
 * including the 422 case, where `detail` is an array of validation objects
 * rather than a string.
 */

const DEFAULT_BASE_URL = "/v1";

/**
 * Same-origin by default, proxied to FastAPI by the rewrite in
 * `next.config.ts`. Absolute URLs work too, but bypass the proxy and
 * therefore need CORS on the backend, which it does not currently have.
 */
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? DEFAULT_BASE_URL;

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(status: number, detail: unknown, message?: string) {
    super(message ?? `Request failed with ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }

  /** A sentence worth putting in front of a user. */
  get userMessage(): string {
    switch (this.status) {
      case 401:
        return "Your session is not valid. Reload the page to get a fresh dev token.";
      case 403:
        return "Your role is not permitted to do that.";
      case 404:
        return "Not found — it may belong to another tenant, or not exist.";
      case 409:
        return "Already reviewed by someone else.";
      case 413:
        return "That file is larger than the upload limit.";
      case 422:
        return formatValidationDetail(this.detail);
      default:
        return this.status >= 500
          ? "The server failed to handle that. Check the API logs."
          : typeof this.detail === "string"
            ? this.detail
            : this.message;
    }
  }
}

function formatValidationDetail(detail: unknown): string {
  // FastAPI's 422 body is {detail: [{loc, msg, type}, ...]}, not a string.
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) =>
        item && typeof item === "object" && "msg" in item ? String(item.msg) : null,
      )
      .filter(Boolean);
    if (messages.length) return messages.join("; ");
  }
  return typeof detail === "string" ? detail : "The request was rejected as invalid.";
}

// --- The auth interceptor -----------------------------------------------------

let cachedToken: string | null = null;
let inFlight: Promise<string> | null = null;

/**
 * The dev token, fetched once and reused.
 *
 * `inFlight` deduplicates concurrent callers: the dashboard fires several
 * requests on mount, and without it each one would mint its own token. Not a
 * correctness problem, just N round trips where one will do.
 */
async function getToken(): Promise<string> {
  if (cachedToken) return cachedToken;
  if (inFlight) return inFlight;

  inFlight = (async () => {
    const response = await fetch("/api/dev-token", { cache: "no-store" });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new ApiError(
        response.status,
        body?.detail,
        body?.detail ?? "Could not obtain a development token",
      );
    }
    const { token } = (await response.json()) as { token: string };
    cachedToken = token;
    return token;
  })();

  try {
    return await inFlight;
  } finally {
    inFlight = null;
  }
}

/** Drop the cached token so the next call mints a fresh one. */
export function clearToken(): void {
  cachedToken = null;
}

export async function authHeader(): Promise<Record<string, string>> {
  return { Authorization: `Bearer ${await getToken()}` };
}

// --- The request wrapper ------------------------------------------------------

export interface RequestOptions extends Omit<RequestInit, "body"> {
  body?: unknown;
  /** Skip JSON parsing and hand back the raw Response (used by the export). */
  raw?: boolean;
}

export async function apiFetch<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { body, raw, headers, ...rest } = options;

  const send = async (): Promise<Response> =>
    fetch(`${API_BASE_URL}${path}`, {
      ...rest,
      headers: {
        ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
        ...(await authHeader()),
        ...headers,
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });

  let response = await send();

  if (response.status === 401) {
    // The dev token has a one-hour TTL, so a long review session WILL outlive
    // it. Clearing the cache alone only helps the *next* call — the one that
    // expired still fails, and the steward loses the verdict they just
    // submitted. Retry once with a fresh token instead; a second 401 is a
    // real authentication problem rather than an expiry.
    clearToken();
    response = await send();
  }

  if (!response.ok) {
    const parsed = await response.json().catch(() => null);
    throw new ApiError(response.status, parsed?.detail ?? null);
  }

  if (raw) return response as unknown as T;
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}
