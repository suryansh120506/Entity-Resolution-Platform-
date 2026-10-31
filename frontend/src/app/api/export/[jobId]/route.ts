import { NextResponse } from "next/server";

import { mintDevToken } from "@/lib/auth/dev-token";

/**
 * Streams the export CSV out of FastAPI with the bearer token attached.
 *
 * This route exists because of a bug worth remembering. The export was
 * originally a plain `<a href="/v1/jobs/{id}/export">`, with a comment
 * explaining — correctly — that letting the browser handle the download keeps
 * a 50,000-row CSV out of JS memory. What that reasoning missed is that a
 * top-level navigation sends **no Authorization header**, and the `/v1/jobs`
 * router requires one. There is no cookie session and no other auth path, so
 * the endpoint returned 401 every single time: the steward clicked "Export
 * results" and the review page was replaced by the raw JSON body
 * `{"detail":"Missing bearer token"}`, losing their place in the queue.
 *
 * Proxying through a route handler keeps both properties. The token is
 * attached server-side (so it still never reaches the browser), and the body
 * is piped straight through as a stream rather than buffered — which is the
 * thing the original comment was protecting.
 */
export const dynamic = "force-dynamic";

const BACKEND_ORIGIN = process.env.BACKEND_ORIGIN ?? "http://localhost:8000";

export async function GET(
  request: Request,
  context: { params: Promise<{ jobId: string }> },
) {
  const { jobId } = await context.params;
  // Allow-listed, never interpolated from the client as-is.
  const params = new URL(request.url).searchParams;
  const requested = params.get("kind");
  const kind =
    requested === "cleaned" || requested === "gold" || requested === "client" ? requested : "mapping";
  // `partial=1` lets the Gold export run before the job is COMPLETED; the
  // backend then flags undecided rows instead of refusing.
  const upstreamPath =
    kind === "client"
      ? `/v1/jobs/${jobId}/export/client`
      : kind === "gold"
      ? `/v1/jobs/${jobId}/export/gold${params.get("partial") === "1" ? "?allow_partial=true" : ""}`
      : kind === "cleaned"
        ? `/v1/jobs/${jobId}/export/cleaned`
        : `/v1/jobs/${jobId}/export`;

  // Reject anything that is not a UUID before it reaches the backend: this
  // path is interpolated into a URL, and the job id comes from the client.
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(jobId)) {
    return NextResponse.json({ detail: "Invalid job id" }, { status: 400 });
  }

  let token: string;
  try {
    token = await mintDevToken();
  } catch (error) {
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : "Could not obtain a token" },
      { status: 500 },
    );
  }

  const upstream = await fetch(`${BACKEND_ORIGIN}${upstreamPath}`, {
    headers: { Authorization: `Bearer ${token}` },
    cache: "no-store",
  });

  if (!upstream.ok || !upstream.body) {
    const detail = await upstream.text().catch(() => "");
    return NextResponse.json(
      { detail: detail || `Export failed with ${upstream.status}` },
      { status: upstream.status },
    );
  }

  // `upstream.body` is a ReadableStream handed straight to the response, so
  // the CSV is never materialised in this process either.
  return new NextResponse(upstream.body, {
    status: 200,
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition":
        upstream.headers.get("content-disposition") ??
        `attachment; filename="job_${jobId}_${kind}.csv"`,
      "Cache-Control": "no-store",
    },
  });
}
