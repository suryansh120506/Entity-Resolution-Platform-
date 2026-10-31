# Entity Resolution Platform — ER-PLATFORM-001 Rev A

PostgreSQL 16 + pgvector, Redis, the schema from TRD FIG 2.1–2.7, the
row-level security from FIG 2.8, the ML pipeline stages from TRD 2.2, and the
Celery orchestration from TRD 2.6 / FIG 3.2, and the Next.js review app.

## Start everything (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start.ps1    # Docker, migrations, API, worker, web app
powershell -ExecutionPolicy Bypass -File scripts\stop.ps1     # stop API, worker, web app
powershell -ExecutionPolicy Bypass -File scripts\stop.ps1 -Containers   # ...and Postgres/Redis (data kept)
```

`start.ps1` starts Docker Desktop if its engine is down, brings up Postgres and
Redis, applies migrations, then opens the API (port 8000), the pipeline worker
and the web app (port 3000) each in its own window, skipping anything already
running, and opens http://localhost:3000/dashboard. The status pill in the
app's header shows if any of them stops — an upload with no worker running
otherwise just sits at "Queued".

## How a job flows

1. **Reading & checking** — every row is stored verbatim, then classified
   Valid / Missing_Context / Junk. A row is usable if it has a real name and a
   district *or* a real pincode; each set-aside row records why.
2. **Reading names** — usable rows are embedded (name and address separately).
3. **Finding candidates** — rows are only compared within the same district
   or pincode, exactly, in memory ([blocking/location.py](app/ml_pipeline/blocking/location.py)).
   Output: `candidate_pairs`.
4. **Matching** — Splink scores every pair; a distinctive-name check vetoes
   pairs that share only boilerplate, town names, or a chain name
   ([matching/name_gate.py](app/ml_pipeline/matching/name_gate.py)); only
   pairs scoring ≥ 0.5 link records into clusters
   ([matching/clustering.py](app/ml_pipeline/matching/clustering.py)).
5. **Confidence gate** — ≥ 0.95 merges automatically; 0.75–0.95 goes to Google
   Places validation if a key is configured, otherwise to review; below that,
   to review.
6. **Review and export** — stewards merge (optionally correcting the kept
   record) or reject; the cleaned CSV carries every row with its outcome.

## Running the API

```bash
uvicorn app.main:app --reload --port 8000 --loop asyncio:SelectorEventLoop   # or: make api
```

**The `--loop` flag is required on Windows.** Uvicorn hardcodes
`ProactorEventLoop` there and psycopg's async mode refuses to run on it, so a
bare `uvicorn app.main:app` serves 500s from every database-backed route. It
cannot be fixed from inside the app — uvicorn passes an explicit
`loop_factory` to `asyncio.run`, which bypasses the event-loop policy — so
the flag is the fix. It is a no-op on Linux and macOS, where
`SelectorEventLoop` is already the default.

## The frontend

Next.js 15 app in [frontend/](frontend/): upload with live progress, a
per-job results page (where every row went, why rows were set aside, the
cleaned download), and the review queue and resolution screen. See
[frontend/README.md](frontend/README.md), and note that it needs a `tenants`
row and a seeded `reference_pincodes` table to do anything useful.

## Running the tests

```bash
cp .env.example .env          # credentials must match the pgdata volume
docker compose up -d
alembic upgrade head
pytest                        # unit + integration
pytest -m "not integration"   # unit only, no containers needed
```

`tests/conftest.py` loads `.env` itself, so a missing DSN no longer silently
*skips* the integration suite — a green run that exercised nothing is worse
than a red one.

The `.env` values must match what the `er-postgres` volume was initialised
with. Passwords are baked in on first init and are immutable after that;
`docker compose down -v` is what it takes to change them. To check what the
container actually has:

```bash
docker inspect er-postgres --format '{{range .Config.Env}}{{println .}}{{end}}'
```

---

## Quick start

```bash
cp .env.example .env          # PowerShell: Copy-Item .env.example .env
docker compose up -d          # Postgres + pgvector, Redis
pip install -e ".[dev]"
alembic upgrade head          # schema (0001), then RLS (0002)
pytest -v                     # proves the tenancy boundary actually holds
```

`make help` lists the rest. On Windows without `make`, the targets in
[Makefile](Makefile) are all single commands you can run directly.

---

## Layout

```
app/
  api/            HTTP only. Auth, validation, enqueue-and-return (SAD 3.2).
    deps.py       The DB dependency FIG 2.8 refers to — binds app.tenant_id.
    pagination.py Keyset cursors for the review queue (TRD 2.5).
    v1/routes/    health, jobs (upload/poll/queue/export), clusters (feedback).
  core/
    config.py     Settings. Three DSNs, one per database role.
    constants.py  Values two layers must agree on (the GUC name, 768, …).
    security.py   JWT → Principal. The only source of a tenant id.
  db/
    base.py       DeclarativeBase, naming convention, TenantScopedMixin.
    session.py    Engines + the transaction-local tenant binding.
  models/         FIG 2.1–2.7, one module per figure, plus reference_pincode.py.
  ml_pipeline/    Stage packages (TRD 2.2). utils.py has the pincode helpers.
    classification/classifier.py  XGBoost Valid/Junk/Missing_Context + the
                                  Snorkel-style LFs that cold-start it.
    embedding/encoder.py          mpnet wrapper + normalize_text (the
                                  abbreviation/punctuation normaliser).
  workers/        celery_app.py: queue topology + retry policy (TRD 2.6).
    job_state.py  The jobs.status state machine; refuses illegal transitions.
    tasks/        One module per stage; orchestration.py chains them (FIG 3.2).
  schemas/        Pydantic request/response models — empty in Step 1.

migrations/
  versions/0001_initial_schema.py            Tables, indexes, HNSW.
  versions/0002_row_level_security.py        Thin wrapper; the SQL is below.
  versions/0003_school_records_label_source.py  PSEUDO/STEWARD provenance.
  versions/0004_reference_pincodes.py        Global lookup table, no RLS.
  sql/0002_rls_policies.sql             ← the RLS control, as readable SQL.
  sql/0002_rls_policies_downgrade.sql

docker/postgres/initdb/                 Extensions + the three DB roles.
scripts/seed_pincodes.py                Bulk-loads reference_pincodes via COPY.
tests/test_rls_isolation.py             Asserts NFR-02 against real Postgres.
```

---

## The part worth reading first: three database roles

PostgreSQL **silently exempts a table's owner from that table's RLS
policies.** A schema where every policy from FIG 2.8 is installed, correct,
and reported as enabled by `\d+` will still return every tenant's rows if the
application connected as the owner. That failure passes code review.

So there are three roles, one per plane
([docker/postgres/initdb/002_app_role.sh](docker/postgres/initdb/002_app_role.sh)):

| Role | DSN | RLS | Used by |
|---|---|---|---|
| `er_owner` | `MIGRATION_DATABASE_URL` | forced | Alembic only |
| `er_app` | `DATABASE_URL` | **enforced** | API + Celery workers |
| `er_platform` | `PLATFORM_DATABASE_URL` | `BYPASSRLS` | control plane only |

`er_app` owns nothing and is `NOBYPASSRLS`, so every policy binds to it.
Migration 0002 additionally applies `FORCE ROW LEVEL SECURITY`, which subjects
even the owner to policy — that is what makes tenant isolation survive someone
pasting the migration DSN into a worker config.

`er_platform` exists because three operations are legitimately cross-tenant and
cannot be anything else: provisioning a tenant (there is no tenant to scope to
yet, and `tenants` is itself protected), reading `dead_letter_tasks` rows whose
`tenant_id` is NULL (TRD 2.7), and enumerating tenants for the nightly retrain
(SAD 3.6). Giving those a named, auditable role is the difference between RLS
having *a documented exception in `pg_roles`* and RLS having *an undocumented
exception, which is whoever happens to be connected*. It is reached only
through `platform_session()` in [app/db/session.py](app/db/session.py).

---

## How the tenant binding works

FIG 2.8's policy reads `current_setting('app.tenant_id')`. Something has to put
a value there, and that is
[app/db/session.py](app/db/session.py) + [app/api/deps.py](app/api/deps.py):

1. `verify_token` extracts `tenant_id` from the **verified JWT** — never from a
   request body or query parameter (TRD 2.5).
2. `bind_tenant` stores it on the session.
3. An `after_begin` event listener issues
   `set_config('app.tenant_id', …, true)` at the start of **every** transaction
   that session opens.

Two details that are easy to get wrong:

- **`set_config(..., true)` rather than literal `SET LOCAL`.** They are the
  same thing, but `SET LOCAL` is utility syntax that cannot take a bind
  parameter — which would mean interpolating a tenant id into a SQL string, in
  the one place where a mistake is a cross-tenant leak.
- **Re-applied per transaction, not once per session.** The value is
  transaction-local, so the first `commit()` discards it. A long-running
  embedding task that commits per batch would otherwise lose its scope
  partway through. The listener makes that impossible to forget.

---

## Deviations from the documents

Everything below is a deliberate departure, flagged so it can be overruled.
Each is also commented at the point it occurs.

### Added to FIG 2.8 (the RLS policy)

| Addition | Why |
|---|---|
| `FORCE ROW LEVEL SECURITY` | `ENABLE` alone exempts the owner. See above. |
| `WITH CHECK` alongside `USING` | `USING` governs reads; it does not constrain `INSERT` at all. Without it, a tenant can write rows stamped with another tenant's id — which then become invisible to the writer, so the bug surfaces later as "my data vanished". |
| A policy on `tenants` itself | TRD 2.4 covers tenant-scoped tables and `tenants` is not one (it keys on `id`). Left alone, any tenant could enumerate every other tenant's name, slug, plan and spend cap. |

The policy body is otherwise FIG 2.8 verbatim, including
`current_setting('app.tenant_id')::uuid` with no `missing_ok`. That errors when
no tenant is bound. The safer-*looking* `NULLIF(current_setting(…, true), '')`
returns zero rows **silently**, which reads as "this tenant has no clusters".
Erroring is the right failure.

### Columns added to the figures

| Table | Column | Required by |
|---|---|---|
| `clusters` | `job_id` | FIG 2.9's route is `GET /v1/jobs/{job_id}/clusters` — unservable without it. FIG 2.5 omits it. |
| `clusters` | `created_at` | TRD 2.5 mandates keyset pagination; FIG 2.9's `next_cursor` decodes to a `created_at` payload. |
| `human_feedback` | `cluster_id` | FIG 2.10 posts to `/v1/clusters/{cluster_id}/feedback`; without it the row is unattributable. |
| `human_feedback` | `master_record_id` | In FIG 2.10's request body. |
| `human_feedback` | `reviewed_by`, `created_at` | NFR-04 ("which reviewer if any"); retrain's "labels since last run" window. |
| `model_versions` | `version_tag` | FIG 2.3 references `model_versions.version_tag` by name; FIG 2.7 omits the column. |
| `model_versions` | `artifact_uri`, `metrics`, `trained_at` | SAD 3.6's promotion gate compares held-out metrics. Without somewhere to record them, "it beat the current version" has nothing behind it. |
| `dead_letter_tasks` | whole table | Specified in prose in TRD 2.7, not given a figure. |
| `candidate_pairs` | whole table (0005) | Blocking's output, so matching scores pairs *before* clustering. Clustering straight from blocking pairs chained unrelated schools into one group. RLS as for every tenant table. |
| `jobs` | `original_filename` (0005) | A job list of bare UUIDs gives no way to tell uploads apart. |
| `clusters` | `golden_fields` (0005) | A steward's corrections to the kept record, stored beside it — `raw_row` stays immutable (TRD 2.3). |
| `human_feedback` | `field_corrections` (0005) | Audit half of the above (NFR-04). |

### Changed: per-job blocking no longer uses the HNSW index

FR-03 / SAD 3.4 specify pgvector HNSW candidate search. The pipeline now
blocks exactly, in memory, within each district and pincode instead
([blocking/location.py](app/ml_pipeline/blocking/location.py)): on a real
21,736-record job the per-record HNSW queries took hours and chained schools
across states; the location-partitioned search takes about two seconds.
`CandidateBlocker` and the HNSW indexes remain for tenant-wide search.

### Judgement calls

- **`tenants.confidence_auto_merge` / `confidence_api_floor` as two NUMERIC
  columns.** TRD 2.2 calls this `tenants.confidence_thresholds` (implying one
  JSONB blob); FIG 2.1 gives two typed columns. The figure wins — two columns
  are checkable by the database and comparable in a `WHERE`. If an open-ended
  settings blob was the real intent, that is a decision to make on purpose.
- **`CHECK` constraints on the TEXT status columns.** The TRD specifies `TEXT`,
  not Postgres `ENUM`, and that is what the schema does — a new value stays an
  `INSERT`-shaped problem rather than an `ALTER TYPE` that takes a lock. The
  CHECKs enforce the documented value sets against paths that bypass the ORM.
- **`dead_letter_tasks.tenant_id` is nullable.** TRD 2.4 says every
  tenant-scoped table has it `NOT NULL`; TRD 2.7 says this table records
  "tenant_id where applicable". The more specific statement wins. Consequence:
  rows with no tenant are invisible to every tenant-scoped connection, so
  dead-letter alerting must run as `er_platform`.
- **`ON DELETE CASCADE` from `tenants`.** Correct for a tenancy root — a
  half-deleted tenant leaving orphaned clusters is worse than either outcome —
  but it means tenant deletion should stay an operator action, never an API
  route.
- **Keyset indexes declared ascending** even though the review queue reads
  newest-first. Every column sorts in the same direction, so Postgres scans the
  index backwards at identical cost; declaring `DESC` would make it an
  expression index that Alembic cannot reliably diff.
- **Nullability of derived columns.** Populated-by-a-later-stage columns
  (`name_normalized`, `classification`, `confidence_score`, `total_rows`, …)
  are nullable. `total_rows` in particular: `0` before the CSV is parsed is a
  lie the progress bar would render as "0 rows" instead of "counting".

### Resolved: `school_records.label_source`

TRD 2.2 requires that heuristic pseudo-labels and real steward labels be
"tracked separately so pseudo-labels can be discarded without contaminating
the real training set". No figure gave a column for this; migration 0003
adds one — `label_source TEXT NOT NULL DEFAULT 'UNLABELED'`, CHECKed to
`UNLABELED` / `PSEUDO` / `STEWARD` (`app/models/enums.py::LabelSource`).
Distinct from `classifier_version`, which records *which model build*
decided, not *who* decided.

### Added: `reference_pincodes` (global, un-tenanted)

Migration 0004 adds `reference_pincodes` — the All-India Pincode dataset,
seeded in bulk by [scripts/seed_pincodes.py](scripts/seed_pincodes.py) and
read by [app/ml_pipeline/utils.py](app/ml_pipeline/utils.py) during feature
extraction (`validate_pincode`, `get_pincode_context`, both process-cached).
It is the one table in this schema with no `tenant_id` and no RLS policy —
every tenant validates addresses against the same government PIN list, so
there is no tenant to scope a policy to. That exception is named explicitly
in `migrations/sql/0002_rls_policies.sql`'s "every table must be protected"
guard (`excluded_tables`), not silently allowed to slip past it — see
[app/models/reference_pincode.py](app/models/reference_pincode.py).

---

## Open decisions from the documents

Both are called out in the source docs as "your call", and both affect this
layer:

1. **Google Places cache scope** (SAD 3.5). Currently tenant-scoped per
   PRD 1.4 / FIG 2.6 — `UNIQUE (tenant_id, query_hash)`. Making it
   platform-wide would compound savings across tenants, since place data for a
   public school is not tenant-secret. It is exactly two changes: that unique
   constraint, and the policy on `google_places_validations`.
2. **`google_places` token bucket** (TRD 2.6). One platform-wide budget, or
   per tenant? PRD 1.4's cost-independence requirement argues per-tenant, sized
   against each tenant's monthly cap. Needs deciding before the
   `google_places` task is written — the bucket key is part of its contract.

---

## Verifying the foundation

[tests/test_rls_isolation.py](tests/test_rls_isolation.py) is the check that
matters. It runs against real Postgres, because the claim under test is about
Postgres — there is no way to test RLS against a mock, and a mocked test would
pass against a schema with no policies at all.

The queries in it deliberately omit `WHERE tenant_id = ?`. That is the point:
NFR-02 says a missed filter must not be a breach, so the tests issue the missed
filter and assert the rows still do not cross.

It also covers the drift cases that would otherwise go unnoticed: a table with
`ENABLE` but not `FORCE`, a table with no policy at all, `TENANT_SCOPED_TABLES`
disagreeing with the schema, the app role having gained `BYPASSRLS` or come to
own a table, and the `vector(768)` columns having lost their dimension.

```bash
docker compose up -d && alembic upgrade head && pytest -v
```

`make rls-verify` re-runs the policy script and its two guard blocks against a
live database, which is the fastest way to check an environment you did not
migrate yourself.
