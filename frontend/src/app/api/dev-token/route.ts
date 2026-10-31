import { NextResponse } from "next/server";

import { mintDevToken } from "@/lib/auth/dev-token";

/**
 * Hands the browser a short-lived dev token. DEVELOPMENT ONLY.
 *
 * This route is the seam that keeps the signing secret out of the client
 * bundle: the browser asks for a token, it never learns how one is made.
 * That is the whole difference between a mock worth developing against and a
 * mock that teaches the app a habit it cannot keep in production.
 *
 * It refuses to run outside development. An endpoint that mints a valid
 * token for an arbitrary tenant on request is a total authentication bypass,
 * and "we'll remember to remove it before deploying" is not a control.
 */
export const dynamic = "force-dynamic";

export async function GET() {
  if (process.env.NODE_ENV === "production") {
    return NextResponse.json({ detail: "Not found" }, { status: 404 });
  }

  try {
    const token = await mintDevToken();
    return NextResponse.json(
      { token },
      // Never cached: a token in a shared cache is a token handed to whoever
      // asks next.
      { headers: { "Cache-Control": "no-store" } },
    );
  } catch (error) {
    // Configuration errors surface here rather than as an opaque 401 from
    // FastAPI three layers away.
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : "Could not mint a dev token" },
      { status: 500 },
    );
  }
}
