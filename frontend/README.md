# Entity Resolution — frontend

Next.js 15 (App Router) · React 19 · TypeScript · Tailwind v4.

Presentation layer for the ER-PLATFORM API: upload and ingestion status, the
cursor-paginated review queue, and the cluster resolution screen.

```
src/
  app/
    dashboard/              Upload + live ingestion status
    review/                 Review queue (needs ?job=<uuid>)
    review/[cluster_id]/    Cluster resolution
    api/dev-token/          Mints the mock JWT. Dev only.
  components/
    UploadDropzone.tsx      CSV upload with progress
    JobStatusPanel.tsx      Polls one job's status and counts
    ReviewQueueScreen.tsx   Job stats header + queue
    ClusterQueue.tsx        Keyset-paginated table
    ClusterResolution.tsx   Field-diff comparison + Merge/Reject
  lib/
    api/client.ts           Base URL, auth interceptor, error normalisation
    api/jobs.ts             Job endpoints (upload uses XHR — see below)
    api/clusters.ts         Queue, detail, feedback
    api/types.ts            Wire types mirroring the backend schemas
    format.ts               Score parsing, field ordering, master proposal
    auth/dev-token.ts       Server-side JWT minting. Replaced by a real IdP.
```

## Three things about the review screens

**`PENDING_REVIEW` is a cluster status; `AWAITING_REVIEW` is a job status.**
They describe the same moment from opposite sides and are easy to confuse.
Filtering the clusters endpoint by `AWAITING_REVIEW` returns an empty page,
not an error.

**Scores arrive as strings.** `confidence_score` and `pairwise_score` are
`NUMERIC(4,3)`, and Pydantic serialises Decimal as JSON `"0.842"`. Sorting
them unparsed gives lexicographic order, where `"0.9" > "0.85"` is wrong.
`lib/format.ts::toScore` is the only place that should parse them.

**There is no master record to "approve" on an unreviewed cluster.** The
backend sets `master_record_id` on resolution only, so every member of a
`PENDING_REVIEW` cluster arrives with `is_master: false` — while `MERGE`
*requires* a `master_record_id`. The UI therefore proposes one (strongest
pairwise score → most complete row → record id, mirroring
`select_master_record` in the worker) and lets the steward override it before
committing. That override is not decoration: a merge is irreversible from
this UI, and every other record maps to the survivor permanently.

`SPLIT` is not offered. It needs the steward to say *which* records leave the
cluster, and the API enforces a non-empty `excluded_record_ids` for it — a
button that cannot express that would only produce 422s.

## Setup

The backend must be running first, from the repo root:

```bash
docker compose up -d
alembic upgrade head
uvicorn app.main:app --port 8000 --loop asyncio:SelectorEventLoop
```

**That `--loop` flag is not optional on Windows.** Uvicorn hardcodes
`ProactorEventLoop` there, and psycopg's async mode cannot use it — without
the flag every database-backed route returns 500 with
"Psycopg cannot use the 'ProactorEventLoop' to run in async mode", and the
upload looks broken for reasons that have nothing to do with the frontend.
`asyncio:SelectorEventLoop` is already the default on Linux and macOS, so the
same command is correct everywhere. `make api` passes it.

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev            # http://localhost:3000
```

### You must create the dev tenant first

`DEV_TENANT_ID` has to be a row that actually exists. `jobs.tenant_id` is a
foreign key, so an invented UUID passes JWT verification and then fails at
insert — the request 401s nowhere and 500s instead, which reads like a bug in
the upload.

`tenants` is RLS-protected and has no provisioning endpoint (that is what the
`er_platform` control-plane role is for), so create it directly:

```bash
docker exec er-postgres psql -U er_owner -d entity_resolution -c \
  "INSERT INTO tenants (id, name, slug) VALUES
   ('00000000-0000-0000-0000-000000000001', 'Dev Tenant', 'dev')
   ON CONFLICT (id) DO NOTHING;"
```

### And seed the pincodes

Not optional. With `reference_pincodes` empty, every PIN fails validation, the
classifier labels an entire upload `Missing_Context`, nothing is embedded, and
the job completes having resolved nothing — successfully, and silently. From
the repo root:

```bash
python scripts/seed_pincodes.py --csv-path <all-india-pincodes.csv>
```

## Three things worth knowing

**There is no CORS on the backend.** `app/main.py` installs no CORS
middleware, so a browser at `:3000` cannot call `:8000` directly. `next.config.ts`
rewrites `/v1/*` to the backend, making every call same-origin. Set
`NEXT_PUBLIC_API_BASE_URL` to an absolute URL to bypass the proxy — that path
needs CORS added to FastAPI, which is a production decision (an allow-list of
real origins), not something to loosen locally.

**Auth is a mock, and it is server-side on purpose.** `lib/auth/dev-token.ts`
imports `server-only`. HS256 uses a shared secret, so signing in the browser
would put that secret in the JS bundle and let any visitor mint a token for
any tenant — exactly the boundary `app/core/security.py` exists to hold. The
browser fetches a token from `/api/dev-token`, which refuses to run outside
development.

The token's claims match `verify_token` exactly: `sub`, `exp`, `tenant_id`,
`role`, plus `aud` and `iss`. Get any of them wrong and the response is a 401
whose message deliberately tells you nothing.

**The upload uses XMLHttpRequest, not fetch.** `fetch` cannot report upload
progress. For a file that can be hundreds of megabytes, a UI that shows
nothing between "clicked" and "done" looks broken, and the user's instinct is
to click again — which uploads it twice.

## Known gaps

- **No job list endpoint.** The backend has `GET /v1/jobs/{job_id}` but no
  `GET /v1/jobs`, so the dashboard only tracks jobs uploaded in the current
  session and a reload loses them. The `ix_jobs_tenant_id_created_at` index
  already exists for that query.
- **Types are hand-written**, not generated from OpenAPI. Once the surface
  settles, `openapi-typescript` should replace `lib/api/types.ts` so drift is
  a build error rather than a runtime surprise.
- **No tests yet.**
