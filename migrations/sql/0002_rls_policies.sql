-- ============================================================================
-- ER-PLATFORM-001 · migration 0002 — Row-Level Security
--
-- Implements TRD FIG 2.8. Satisfies NFR-02: "Tenant isolation is enforced at
-- two independent layers — application query scoping AND Postgres Row-Level
-- Security — so a single missed `WHERE tenant_id = ?` in application code is
-- not a data breach."
--
-- Executed by migrations/versions/0002_row_level_security.py. Also runnable
-- directly:  psql -1 -v ON_ERROR_STOP=1 -f migrations/sql/0002_rls_policies.sql
-- It contains no BEGIN/COMMIT so that it composes with Alembic's own
-- transaction; use psql's -1 to get atomicity when running it by hand.
--
-- ---------------------------------------------------------------------------
-- THREE THINGS THIS FILE DOES THAT FIG 2.8 DOES NOT SHOW, EACH DELIBERATE
-- ---------------------------------------------------------------------------
--
-- 1. FORCE ROW LEVEL SECURITY, in addition to ENABLE.
--
--    ENABLE alone exempts the table's OWNER from its own policies. A schema
--    where ENABLE is set, every policy is present and correct, and psql's \d+
--    dutifully reports them — but where the application connected with the
--    migration credentials and therefore sees every tenant's rows — is a
--    schema that passes review and leaks anyway. FORCE closes that. The
--    runtime role is a non-owner already
--    (docker/postgres/initdb/002_app_role.sh), so it would be governed either
--    way; FORCE is what makes the isolation survive the day someone reuses
--    the owner DSN in a worker config.
--
--    Operational consequence, stated up front: a data backfill run as the
--    owner is now subject to policy too. Cross-tenant maintenance needs a
--    role explicitly granted BYPASSRLS, provisioned out of band. That is the
--    intended shape — the exception becomes visible in pg_roles instead of
--    being implicit in whoever happens to be connected.
--
-- 2. WITH CHECK, in addition to USING.
--
--    FIG 2.8 shows only USING. USING governs what a statement can SEE: SELECT,
--    and the read half of UPDATE and DELETE. It does not constrain INSERT at
--    all, and does not constrain the row an UPDATE produces. With USING alone,
--    a tenant-scoped connection can insert rows stamped with another tenant's
--    id, or update a row to move it into another tenant. Those rows then
--    become invisible to the tenant that wrote them, so the bug surfaces as
--    "my data vanished" a long way from the write that caused it. WITH CHECK
--    makes the policy symmetric: you may only write rows you could read.
--
-- 3. A policy on `tenants` itself.
--
--    TRD 2.4 specifies the policy for tenant-scoped tables, and `tenants` is
--    not one — it keys on `id`, not `tenant_id`. Left alone, any tenant could
--    enumerate every other tenant's name, slug, plan and spend cap. The
--    policy below is the same idea applied to the root of the tree.
--
-- ---------------------------------------------------------------------------
-- ON THE UNSET-GUC FAILURE MODE
-- ---------------------------------------------------------------------------
-- The policy body is FIG 2.8 verbatim: current_setting('app.tenant_id')::uuid,
-- with no missing_ok argument and no NULLIF. That is a choice, not an
-- oversight. If the GUC was never set this errors — either "unrecognized
-- configuration parameter", or, once the session has set it at least once,
-- "invalid input syntax for type uuid" on the empty string. The
-- safer-looking alternative,
--
--     NULLIF(current_setting('app.tenant_id', true), '')::uuid
--
-- returns NULL, which matches nothing, and so returns zero rows SILENTLY.
-- A query that forgot to bind a tenant should abort loudly, not hand back an
-- empty result set that reads as "this tenant has no clusters".
-- ============================================================================


-- ---------------------------------------------------------------------------
-- 1. Privileges for the non-owner roles
-- ---------------------------------------------------------------------------
-- Role names are hard-coded rather than parameterised so that they are
-- greppable and so this file runs identically under Alembic and psql. If you
-- rename a role, it changes here and in .env / docker-compose.yml.
--
--   er_app      data plane    — NOBYPASSRLS, governed by every policy below
--   er_platform control plane — BYPASSRLS, the documented exception
--
-- Both get the same table privileges. The difference between them is not what
-- they may touch, it is whether RLS applies — which is a role attribute, set
-- in docker/postgres/initdb/002_app_role.sh, not a grant.

DO $grants$
DECLARE
  target_role text;
  roles CONSTANT text[] := ARRAY['er_app', 'er_platform'];
BEGIN
  FOREACH target_role IN ARRAY roles LOOP
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = target_role) THEN
      -- Not fatal: a CI database that runs its tests as the owner has neither
      -- role, and the policies below are still worth installing. But say so
      -- loudly — a production deployment that hits this warning has an
      -- application that cannot connect, and the cause should not have to be
      -- guessed at.
      RAISE WARNING
        'Role % does not exist; skipping its GRANTs. Create it before '
        'deploying (see docker/postgres/initdb/002_app_role.sh).', target_role;
      CONTINUE;
    END IF;

    EXECUTE format('GRANT USAGE ON SCHEMA public TO %I', target_role);
    EXECUTE format(
      'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO %I',
      target_role
    );
    -- Covers tables added by later migrations without a manual GRANT each time.
    EXECUTE format(
      'ALTER DEFAULT PRIVILEGES IN SCHEMA public '
      'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO %I',
      target_role
    );
    -- Neither role may own an object, because an owner escapes RLS.
    EXECUTE format('REVOKE CREATE ON SCHEMA public FROM %I', target_role);

    -- alembic_version is migration bookkeeping. Neither role has business
    -- reading it and neither has business writing it.
    EXECUTE format('REVOKE ALL ON TABLE alembic_version FROM %I', target_role);
  END LOOP;
END
$grants$;


-- ---------------------------------------------------------------------------
-- 2. The tenancy root
-- ---------------------------------------------------------------------------
-- Same shape as the policies below, keyed on `id` because this IS the tenant.

ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenants FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_self_isolation ON tenants;
CREATE POLICY tenant_self_isolation ON tenants
  USING (id = current_setting('app.tenant_id')::uuid)
  WITH CHECK (id = current_setting('app.tenant_id')::uuid);


-- ---------------------------------------------------------------------------
-- 3. Tenant-scoped tables — TRD FIG 2.8, applied table by table
-- ---------------------------------------------------------------------------
-- TRD 2.4 notes that repeating this policy nine times "adds length, not
-- information". True of a design document. Not true of the migration that
-- installs it: a loop over an array of table names is a few lines shorter and
-- turns "which tables are actually covered?" into a question you answer by
-- reasoning about control flow rather than by reading. For the one migration
-- in this schema whose failure mode is a cross-tenant data leak, the explicit
-- form earns its repetition — every table is greppable, and a table missing
-- from this list reads as an absence instead of hiding inside a loop bound.
--
-- The guard in section 4 catches that absence either way.

-- FIG 2.2
ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE jobs FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON jobs;
CREATE POLICY tenant_isolation ON jobs
  USING (tenant_id = current_setting('app.tenant_id')::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id')::uuid);

-- FIG 2.3
ALTER TABLE school_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE school_records FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON school_records;
CREATE POLICY tenant_isolation ON school_records
  USING (tenant_id = current_setting('app.tenant_id')::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id')::uuid);

-- FIG 2.4 — tenant_id is denormalised onto this table specifically so the
-- policy can be applied here without a join (SAD 3.4).
ALTER TABLE record_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE record_embeddings FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON record_embeddings;
CREATE POLICY tenant_isolation ON record_embeddings
  USING (tenant_id = current_setting('app.tenant_id')::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id')::uuid);

-- FIG 2.5
ALTER TABLE clusters ENABLE ROW LEVEL SECURITY;
ALTER TABLE clusters FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON clusters;
CREATE POLICY tenant_isolation ON clusters
  USING (tenant_id = current_setting('app.tenant_id')::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id')::uuid);

ALTER TABLE cluster_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE cluster_members FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON cluster_members;
CREATE POLICY tenant_isolation ON cluster_members
  USING (tenant_id = current_setting('app.tenant_id')::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id')::uuid);

-- FIG 2.6 — this policy is what makes the Places cache tenant-scoped, which
-- FR-10 requires explicitly ("...including through the Google Places response
-- cache"). If the shared-cache option in SAD 3.5 is ever taken, this is one
-- of exactly two places that change (the other is the UNIQUE constraint on
-- (tenant_id, query_hash) in migration 0001).
ALTER TABLE google_places_validations ENABLE ROW LEVEL SECURITY;
ALTER TABLE google_places_validations FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON google_places_validations;
CREATE POLICY tenant_isolation ON google_places_validations
  USING (tenant_id = current_setting('app.tenant_id')::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id')::uuid);

-- FIG 2.7
ALTER TABLE human_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE human_feedback FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON human_feedback;
CREATE POLICY tenant_isolation ON human_feedback
  USING (tenant_id = current_setting('app.tenant_id')::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id')::uuid);

ALTER TABLE model_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_versions FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON model_versions;
CREATE POLICY tenant_isolation ON model_versions
  USING (tenant_id = current_setting('app.tenant_id')::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id')::uuid);

-- TRD 2.7 — tenant_id is NULLABLE on this one ("tenant_id where applicable"),
-- so rows with no tenant match no policy and are invisible to every
-- tenant-scoped connection. That is correct: platform-level failures are not
-- tenant-visible. It also means dead-letter alerting must run under a
-- BYPASSRLS role. See the docstring on app/models/dead_letter.py.
ALTER TABLE dead_letter_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE dead_letter_tasks FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON dead_letter_tasks;
CREATE POLICY tenant_isolation ON dead_letter_tasks
  USING (tenant_id = current_setting('app.tenant_id')::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id')::uuid);


-- ---------------------------------------------------------------------------
-- 4. Fail the migration if any table was missed
-- ---------------------------------------------------------------------------
-- RLS exists here to catch the mistake application code made. This catches
-- the mistake THIS FILE can make: a table added by some future migration
-- whose author did not think to come back and add a policy. Without this,
-- that table ships unprotected and nothing anywhere says so.
--
-- EXCLUDED_TABLES below is that "written reason" this comment invites: every
-- name in it is a table with NO tenant_id and NO policy, on purpose, not a
-- gap this guard failed to catch.
--
--   reference_pincodes  — migration 0004. A global, un-tenanted lookup table
--     (the All-India Pincode dataset): every tenant reads the same rows, so
--     there is no tenant to scope a policy to. Table-level GRANTs from
--     section 1 above are what make it readable; RLS does not apply to it at
--     all. See app/models/reference_pincode.py.

DO $guard_forced$
DECLARE
  unprotected text;
  excluded_tables CONSTANT text[] := ARRAY['reference_pincodes'];
BEGIN
  SELECT string_agg(c.relname, ', ' ORDER BY c.relname)
    INTO unprotected
  FROM pg_class c
  JOIN pg_namespace n ON n.oid = c.relnamespace
  WHERE n.nspname = 'public'
    AND c.relkind = 'r'
    -- Alembic's own bookkeeping table has no tenant and needs no policy.
    AND c.relname <> 'alembic_version'
    AND NOT (c.relname = ANY(excluded_tables))
    AND NOT (c.relrowsecurity AND c.relforcerowsecurity);

  IF unprotected IS NOT NULL THEN
    RAISE EXCEPTION
      'Row-level security is not both enabled and forced on: %. Every table '
      'in this schema must carry a tenant policy (TRD NFR-02). Add it to '
      'migrations/sql/0002_rls_policies.sql, or exclude it here with a '
      'written reason.', unprotected;
  END IF;
END
$guard_forced$;

-- And the same check for the policy itself: ENABLE with no policy denies
-- everything. Safe, but broken in a way that presents as an application bug.

DO $guard_policy$
DECLARE
  policyless text;
  excluded_tables CONSTANT text[] := ARRAY['reference_pincodes'];
BEGIN
  SELECT string_agg(c.relname, ', ' ORDER BY c.relname)
    INTO policyless
  FROM pg_class c
  JOIN pg_namespace n ON n.oid = c.relnamespace
  WHERE n.nspname = 'public'
    AND c.relkind = 'r'
    AND c.relname <> 'alembic_version'
    AND NOT (c.relname = ANY(excluded_tables))
    AND NOT EXISTS (SELECT 1 FROM pg_policy p WHERE p.polrelid = c.oid);

  IF policyless IS NOT NULL THEN
    RAISE EXCEPTION
      'Tables have RLS enabled but no policy, so all access to them is '
      'denied: %', policyless;
  END IF;
END
$guard_policy$;
