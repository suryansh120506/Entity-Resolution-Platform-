#!/usr/bin/env bash
# ============================================================================
# ER-PLATFORM-001 · initdb 002 — database roles
#
# WHY THIS EXISTS — this is the most important detail in the whole RLS setup,
# and the one most often missed:
#
#   PostgreSQL silently EXEMPTS a table's owner from that table's row-level
#   security policies. If FastAPI and the Celery workers connect using the
#   same role that owns the tables, every policy in FIG 2.8 is installed,
#   reported as enabled by \d+, and enforces absolutely nothing.
#
# So the platform uses three roles, one per plane:
#
#   ${POSTGRES_USER}   OWNER. Owns the schema, runs Alembic. Not used at
#                      runtime by anything.
#
#   ${APP_DB_ROLE}     DATA PLANE. What the API and the Celery workers connect
#                      as. Owns nothing and is NOBYPASSRLS, so every FIG 2.8
#                      policy binds to it. This is the role that must never be
#                      able to see another tenant's rows.
#
#   ${PLATFORM_DB_ROLE} CONTROL PLANE. BYPASSRLS. For the three operations that
#                      are legitimately cross-tenant and cannot be anything
#                      else:
#                        - provisioning a tenant (there is no tenant to scope
#                          to yet, and `tenants` is itself RLS-protected)
#                        - reading dead_letter_tasks rows whose tenant_id is
#                          NULL (TRD 2.7 — platform-level failures)
#                        - enumerating tenants for the nightly per-tenant
#                          retrain (SAD 3.6)
#
# Splitting the third role out is the difference between "RLS has a documented
# exception, visible in pg_roles, held by one DSN" and "RLS has an undocumented
# exception, which is whoever happens to be connected". Migration 0002 also
# applies FORCE ROW LEVEL SECURITY, which subjects even the owner to policy —
# that is what makes this split load-bearing rather than decorative.
# ============================================================================
set -euo pipefail

APP_DB_ROLE="${APP_DB_ROLE:-er_app}"
APP_DB_PASSWORD="${APP_DB_PASSWORD:?APP_DB_PASSWORD must be set}"
PLATFORM_DB_ROLE="${PLATFORM_DB_ROLE:-er_platform}"
PLATFORM_DB_PASSWORD="${PLATFORM_DB_PASSWORD:?PLATFORM_DB_PASSWORD must be set}"

psql -v ON_ERROR_STOP=1 \
     --username "$POSTGRES_USER" \
     --dbname "$POSTGRES_DB" \
     -v app_role="$APP_DB_ROLE" \
     -v app_password="$APP_DB_PASSWORD" \
     -v platform_role="$PLATFORM_DB_ROLE" \
     -v platform_password="$PLATFORM_DB_PASSWORD" <<'EOSQL'
-- ---------------------------------------------------------------------------
-- Data-plane role: the API and the workers
-- ---------------------------------------------------------------------------
SELECT format(
  'CREATE ROLE %I LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS',
  :'app_role', :'app_password'
)
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'app_role')
\gexec

-- Idempotent: keeps the password in sync if the volume outlives an env change.
-- NOBYPASSRLS is reasserted every time on purpose — it is the one attribute on
-- this role whose loss would be silent and total.
SELECT format('ALTER ROLE %I LOGIN PASSWORD %L NOBYPASSRLS', :'app_role', :'app_password')
\gexec

SELECT format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), :'app_role')
\gexec

SELECT format('GRANT USAGE ON SCHEMA public TO %I', :'app_role')
\gexec

-- The runtime role must never create an object, because it would then own one,
-- and an owner escapes RLS.
SELECT format('REVOKE CREATE ON SCHEMA public FROM %I', :'app_role')
\gexec

-- Tables do not exist yet at initdb time. These default privileges apply to
-- every table the owner creates from here on, so a table added by a future
-- migration is reachable without a manual GRANT — while still being governed
-- by that table's policy.
SELECT format(
  'ALTER DEFAULT PRIVILEGES FOR ROLE %I IN SCHEMA public '
  'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO %I',
  current_user, :'app_role'
)
\gexec


-- ---------------------------------------------------------------------------
-- Control-plane role: tenant provisioning, dead-letter triage, retrain fan-out
-- ---------------------------------------------------------------------------
-- BYPASSRLS. This role can read every tenant's data. It exists so that the
-- three legitimately cross-tenant operations have somewhere to run that is
-- named, auditable, and separate from the API's credentials.
SELECT format(
  'CREATE ROLE %I LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE BYPASSRLS',
  :'platform_role', :'platform_password'
)
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'platform_role')
\gexec

SELECT format(
  'ALTER ROLE %I LOGIN PASSWORD %L BYPASSRLS', :'platform_role', :'platform_password'
)
\gexec

SELECT format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), :'platform_role')
\gexec

SELECT format('GRANT USAGE ON SCHEMA public TO %I', :'platform_role')
\gexec

SELECT format('REVOKE CREATE ON SCHEMA public FROM %I', :'platform_role')
\gexec

SELECT format(
  'ALTER DEFAULT PRIVILEGES FOR ROLE %I IN SCHEMA public '
  'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO %I',
  current_user, :'platform_role'
)
\gexec
EOSQL

echo "initdb 002: data-plane role '${APP_DB_ROLE}' (NOBYPASSRLS, non-owner)"
echo "initdb 002: control-plane role '${PLATFORM_DB_ROLE}' (BYPASSRLS, non-owner)"
