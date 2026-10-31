import "server-only";

import { SignJWT } from "jose";

/**
 * Mints the development bearer token the FastAPI backend expects.
 *
 * THIS IS A STAND-IN FOR AN IDENTITY PROVIDER, NOT AN AUTH SYSTEM.
 * It exists so the UI can be developed against a backend that already
 * enforces TRD 2.5 properly. Replacing it means pointing the app at a real
 * IdP and deleting this file plus `app/api/dev-token/route.ts`; nothing else
 * imports it.
 *
 * Two things about it are deliberate:
 *
 * 1. **`server-only`.** The signing secret must never reach the browser
 *    bundle. A client-side `jose` call would put HS256's shared secret in
 *    JavaScript that anyone can read, and at that point any visitor can mint
 *    a token for any tenant they like — which is precisely the boundary
 *    `app/core/security.py` exists to hold. The import above makes that a
 *    build error rather than a code-review question.
 *
 * 2. **The claims match `verify_token` exactly.** That function decodes with
 *    `options={"require": ["exp", "sub"]}` and validates `audience` and
 *    `issuer`, then reads `tenant_id` and `role`. A token missing any of
 *    those is a 401 with a deliberately uninformative message, so getting
 *    this wrong produces an error that says nothing about what was wrong.
 */

/** The three personas from PRD 1.2, as the backend's `Role` enum spells them. */
export type Role = "data_engineer" | "data_steward" | "org_admin";

const secretKey = () => {
  const secret = process.env.JWT_SECRET_KEY;
  if (!secret) {
    throw new Error(
      "JWT_SECRET_KEY is not set. It must match the backend's value (see the " +
        "repo-root .env) or every request will 401 with 'Invalid or expired token'.",
    );
  }
  return new TextEncoder().encode(secret);
};

export interface DevTokenOptions {
  tenantId?: string;
  subject?: string;
  role?: Role;
  /** Lifetime in seconds. Short by default; this is a dev credential. */
  ttlSeconds?: number;
}

export async function mintDevToken(options: DevTokenOptions = {}): Promise<string> {
  const tenantId = options.tenantId ?? process.env.DEV_TENANT_ID;
  if (!tenantId) {
    throw new Error(
      "DEV_TENANT_ID is not set. It must be the UUID of a row that actually " +
        "exists in the `tenants` table — `jobs.tenant_id` is a foreign key, so " +
        "an invented UUID passes JWT verification and then fails on insert. " +
        "See frontend/README.md for the one-line psql to create it.",
    );
  }

  const audience = process.env.JWT_AUDIENCE ?? "er-platform";
  const issuer = process.env.JWT_ISSUER ?? "er-platform-auth";
  const ttl = options.ttlSeconds ?? 60 * 60;

  return new SignJWT({
    tenant_id: tenantId,
    role: options.role ?? (process.env.DEV_ROLE as Role | undefined) ?? "data_steward",
  })
    .setProtectedHeader({ alg: "HS256" })
    .setSubject(options.subject ?? process.env.DEV_SUBJECT ?? "dev@example.com")
    .setAudience(audience)
    .setIssuer(issuer)
    .setIssuedAt()
    .setExpirationTime(`${ttl}s`)
    .sign(secretKey());
}
