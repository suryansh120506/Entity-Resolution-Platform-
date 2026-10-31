import type { NextConfig } from "next";

/**
 * The `/v1/*` rewrite is not a convenience — it is what makes the browser
 * able to talk to FastAPI at all.
 *
 * `app/main.py` installs no CORS middleware, so a request from
 * http://localhost:3000 to http://localhost:8000 is refused by the browser
 * before FastAPI ever sees it. Proxying through Next means every call is
 * same-origin and the question never arises, and it keeps the backend
 * unmodified — CORS config there would be a production concern (an allow-list
 * of real origins), not something to loosen for local development.
 *
 * Set NEXT_PUBLIC_API_BASE_URL to an absolute URL to bypass the proxy and hit
 * FastAPI directly; that path *does* require CORS on the backend.
 */
const BACKEND_ORIGIN = process.env.BACKEND_ORIGIN ?? "http://localhost:8000";

/**
 * `next build` writes to `.next-build`, `next dev` keeps `.next`.
 *
 * They share `.next` by default, and a production build run while the dev
 * server is up overwrites the chunk manifests the dev server is still
 * serving from. The result is genuinely confusing to diagnose: SSR keeps
 * returning HTTP 200 with valid HTML, so every "is it up?" check passes,
 * while the browser gets 404s for `main-app.js`, `polyfills.js` and the
 * stylesheet — a blank, unstyled page that never finishes loading, and a
 * dev-server log full of `Cannot find module './331.js'`.
 *
 * Separate directories make the collision impossible rather than
 * remembered.
 */
const buildConfig = (phase: string): NextConfig => ({
  distDir: phase === "phase-production-build" ? ".next-build" : ".next",
  experimental: {
    // The `/v1/:path*` rewrite below still goes through Next's own request
    // handling, which caps request bodies at 10MB by default — silently
    // truncating anything larger rather than rejecting it outright. A real
    // school-records CSV clears 10MB easily (the reference upload here is
    // 13.8MB for ~52k rows), so every upload past that size failed with a
    // truncated body and a downstream "socket hang up", not a clear error.
    // 200MB tracks the backend's own `max_upload_bytes` (256MB) — the
    // backend's limit is the one that should actually govern this.
    middlewareClientMaxBodySize: "200mb",
  },
  async rewrites() {
    return [
      {
        source: "/v1/:path*",
        destination: `${BACKEND_ORIGIN}/v1/:path*`,
      },
      // Health probes live outside /v1 on the backend.
      {
        source: "/backend-health/:path*",
        destination: `${BACKEND_ORIGIN}/:path*`,
      },
    ];
  },
});

const config = (phase: string) => buildConfig(phase);

export default config;
