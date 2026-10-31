"""Does the tenancy boundary actually hold? — TRD FIG 2.8 / NFR-02.

Every test here is written against the database, not against application code,
because the claim being tested is about the database: "a single missed
`WHERE tenant_id = ?` in application code is not a data breach" (NFR-02).
The queries below deliberately omit that clause. If a test fails, the missed
clause IS a data breach and the second layer is not there.

Run:  docker compose up -d && alembic upgrade head && pytest tests/ -v
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import DatabaseError, IntegrityError
from sqlalchemy.orm import Session

from app.core.constants import TENANT_SCOPED_TABLES

pytestmark = pytest.mark.integration


def _bind(session: Session, tenant_id: uuid.UUID) -> None:
    """The same transaction-local binding app/db/session.py performs."""
    session.execute(
        text("SELECT set_config('app.tenant_id', :tid, true)"),
        {"tid": str(tenant_id)},
    )


# ---------------------------------------------------------------------------
# The core claim
# ---------------------------------------------------------------------------


def test_tenant_cannot_see_another_tenants_records(
    app_engine: Engine, two_tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    """A query with NO tenant filter returns only the bound tenant's rows."""
    tenant_a, tenant_b = two_tenants

    with Session(app_engine) as session:
        _bind(session, tenant_a)
        # Note the absence of a WHERE clause. This is the careless query
        # NFR-02 says must not be a breach.
        rows = session.execute(text("SELECT tenant_id FROM school_records")).scalars().all()

    assert rows, "Tenant A should see its own seeded record"
    assert set(rows) == {tenant_a}
    assert tenant_b not in rows


def test_tenant_cannot_see_another_tenants_jobs(
    app_engine: Engine, two_tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    tenant_a, tenant_b = two_tenants

    with Session(app_engine) as session:
        _bind(session, tenant_b)
        rows = session.execute(text("SELECT tenant_id FROM jobs")).scalars().all()

    assert set(rows) == {tenant_b}


def test_tenant_cannot_see_other_tenants_in_tenants_table(
    app_engine: Engine, two_tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    """The customer list is not enumerable from a tenant's own connection.

    This covers the policy on `tenants` itself, which TRD 2.4 does not
    specify — see section 2 of migrations/sql/0002_rls_policies.sql.
    """
    tenant_a, tenant_b = two_tenants

    with Session(app_engine) as session:
        _bind(session, tenant_a)
        ids = session.execute(text("SELECT id FROM tenants")).scalars().all()

    assert set(ids) == {tenant_a}


# ---------------------------------------------------------------------------
# The write half — WITH CHECK, which FIG 2.8 does not show
# ---------------------------------------------------------------------------


def test_cannot_insert_a_row_stamped_with_another_tenant(
    app_engine: Engine, two_tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    """USING alone would allow this. WITH CHECK is what stops it.

    Without WITH CHECK the insert succeeds, the row becomes invisible to the
    tenant that wrote it, and the bug surfaces much later as "my data
    vanished".
    """
    tenant_a, tenant_b = two_tenants

    with Session(app_engine) as session:
        _bind(session, tenant_a)
        with pytest.raises(DatabaseError):
            session.execute(
                text("INSERT INTO jobs (tenant_id, status) VALUES (:tenant_id, 'PENDING')"),
                {"tenant_id": tenant_b},
            )
        session.rollback()


def test_cannot_move_a_row_into_another_tenant(
    app_engine: Engine, two_tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    """An UPDATE cannot produce a row the writer could not read."""
    tenant_a, tenant_b = two_tenants

    with Session(app_engine) as session:
        _bind(session, tenant_a)
        with pytest.raises(DatabaseError):
            session.execute(text("UPDATE jobs SET tenant_id = :other"), {"other": tenant_b})
        session.rollback()


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_unset_tenant_guc_raises_rather_than_returning_rows(
    app_engine: Engine, two_tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    """A query with no tenant bound must abort, not return an empty set.

    This is the behaviour the `current_setting('app.tenant_id')::uuid` form
    buys, and the reason the migration does not use the NULLIF/missing_ok
    variant. An empty result set reads as "this tenant has no data"; an error
    reads as "you forgot to bind a tenant", which is what actually happened.
    """
    with Session(app_engine) as session:
        with pytest.raises(DatabaseError):
            session.execute(text("SELECT count(*) FROM school_records")).scalar()
        session.rollback()


def test_binding_does_not_leak_through_the_connection_pool(
    app_engine: Engine, two_tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    """A tenant's binding must not follow its connection back into the pool.

    This is the property `set_config(..., is_local => true)` buys, and it is
    the one that matters most under a pool: a connection handed back still
    carrying a tenant scope would give that scope to whichever request picked
    it up next — a cross-tenant leak with no bad query anywhere in it.

    Note what is NOT asserted here: that the value disappears after a commit
    on the *same* session. It does not, and deliberately so — the `after_begin`
    listener in app/db/session.py re-applies it to each new transaction, which
    is what lets a Celery task commit per batch without silently losing its
    scope. The guarantee is per connection, not per statement.
    """
    tenant_a, _ = two_tenants

    with Session(app_engine) as session:
        _bind(session, tenant_a)
        assert session.execute(text("SELECT current_setting('app.tenant_id')")).scalar() == str(
            tenant_a
        )
        session.commit()
    # Session closed: the connection is back in the pool.

    # A fresh session with no binding. With a pool size of 1 this would be the
    # very same connection; with a larger pool it is very likely to be.
    with Session(app_engine) as session:
        leaked = session.execute(text("SELECT current_setting('app.tenant_id', true)")).scalar()

    assert leaked in (None, ""), (
        f"tenant binding survived into a new session: {leaked!r}. A pooled "
        "connection is carrying a tenant scope between callers."
    )


# ---------------------------------------------------------------------------
# Schema-level guards: catch the drift, not just the leak
# ---------------------------------------------------------------------------


def test_every_tenant_scoped_table_has_rls_enabled_and_forced(
    owner_engine: Engine,
) -> None:
    """Both flags, on every table in TENANT_SCOPED_TABLES plus `tenants`.

    ENABLE without FORCE is the silent-failure case: policies present,
    correct, and inert for the owner. Asserting both is the only way to tell
    the difference from outside.
    """
    expected = set(TENANT_SCOPED_TABLES) | {"tenants"}

    with Session(owner_engine) as session:
        rows = session.execute(
            text(
                "SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity "
                "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'public' AND c.relkind = 'r'"
            )
        ).all()

    state = {name: (enabled, forced) for name, enabled, forced in rows}

    missing = expected - state.keys()
    assert not missing, f"Tables absent from the schema entirely: {sorted(missing)}"

    for table in sorted(expected):
        enabled, forced = state[table]
        assert enabled, f"{table}: ROW LEVEL SECURITY is not enabled"
        assert forced, f"{table}: ROW LEVEL SECURITY is enabled but not FORCED"


#: Tables with no tenant_id and no RLS policy, on purpose — kept in sync with
#: the `excluded_tables` array in both guard blocks of
#: migrations/sql/0002_rls_policies.sql. See reference_pincode.py for why.
GLOBALLY_READABLE_TABLES: tuple[str, ...] = ("reference_pincodes",)


def test_no_table_in_the_schema_is_left_unprotected(owner_engine: Engine) -> None:
    """Catches a table added by a future migration with no policy.

    The same check runs inside migration 0002, which is the right place to
    fail. This duplicates it as a test so that the failure is also visible to
    anyone running the suite against an already-migrated database.
    """
    with Session(owner_engine) as session:
        unprotected = (
            session.execute(
                text(
                    "SELECT c.relname FROM pg_class c "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE n.nspname = 'public' AND c.relkind = 'r' "
                    "AND c.relname <> 'alembic_version' "
                    "AND NOT (c.relrowsecurity AND c.relforcerowsecurity)"
                )
            )
            .scalars()
            .all()
        )

    assert set(unprotected) <= set(GLOBALLY_READABLE_TABLES), (
        f"Tables without enforced RLS: {sorted(unprotected)}. Add a policy in "
        "migrations/sql/0002_rls_policies.sql, or add the table to "
        "GLOBALLY_READABLE_TABLES here (and its excluded_tables array there) "
        "with a written reason."
    )


def test_constants_match_the_database(owner_engine: Engine) -> None:
    """TENANT_SCOPED_TABLES is not allowed to drift from the actual schema.

    The constant is what app-layer code reasons about; the schema is what
    enforces. A table present in one and not the other is how a gap opens.
    """
    with Session(owner_engine) as session:
        with_tenant_col = set(
            session.execute(
                # Base tables only. A view (splink_level_estimates) carries a
                # tenant_id column but no policy of its own: it is declared
                # security_invoker, so the policies on its base tables apply.
                text(
                    "SELECT c.table_name FROM information_schema.columns c "
                    "JOIN information_schema.tables t "
                    "  ON t.table_schema = c.table_schema AND t.table_name = c.table_name "
                    "WHERE c.table_schema = 'public' AND c.column_name = 'tenant_id' "
                    "AND t.table_type = 'BASE TABLE'"
                )
            )
            .scalars()
            .all()
        )

    assert with_tenant_col == set(TENANT_SCOPED_TABLES), (
        "app.core.constants.TENANT_SCOPED_TABLES disagrees with the schema. "
        f"Only in schema: {sorted(with_tenant_col - set(TENANT_SCOPED_TABLES))}; "
        f"only in constant: {sorted(set(TENANT_SCOPED_TABLES) - with_tenant_col)}"
    )


def test_data_plane_role_cannot_bypass_rls(app_engine: Engine) -> None:
    """The one role attribute whose loss would be silent and total."""
    with Session(app_engine) as session:
        bypasses = session.execute(
            text("SELECT rolbypassrls FROM pg_roles WHERE rolname = current_user")
        ).scalar()

    assert bypasses is False, (
        "The application's database role has BYPASSRLS. Every policy in "
        "FIG 2.8 is inert. Check which DSN the app is using."
    )


def test_data_plane_role_owns_nothing(app_engine: Engine) -> None:
    """An owner escapes its own policies, so the app role must own no table."""
    with Session(app_engine) as session:
        owned = (
            session.execute(
                text(
                    "SELECT c.relname FROM pg_class c "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE n.nspname = 'public' AND c.relkind = 'r' "
                    "AND pg_get_userbyid(c.relowner) = current_user"
                )
            )
            .scalars()
            .all()
        )

    assert not owned, f"The application role owns tables: {sorted(owned)}"


# ---------------------------------------------------------------------------
# reference_pincodes: the documented RLS exception
# ---------------------------------------------------------------------------


def test_reference_pincodes_has_no_rls_policy(owner_engine: Engine) -> None:
    """The exception is real: no policy, not just a permissive one.

    A mistakenly-added `tenant_isolation` policy here would make every
    tenant's pincode lookups silently empty, since the table has no
    `tenant_id` for `current_setting('app.tenant_id')` to match against.
    """
    with Session(owner_engine) as session:
        row = session.execute(
            text(
                "SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
                "WHERE relname = 'reference_pincodes'"
            )
        ).one()
        policy_count = session.execute(
            text(
                "SELECT count(*) FROM pg_policy p "
                "JOIN pg_class c ON c.oid = p.polrelid "
                "WHERE c.relname = 'reference_pincodes'"
            )
        ).scalar()

    assert row == (False, False), f"reference_pincodes has RLS flags set: {row}"
    assert policy_count == 0, "reference_pincodes has an RLS policy attached"


def test_reference_pincodes_is_readable_without_a_tenant_bound(
    app_engine: Engine,
) -> None:
    """A tenant-scoped connection can read it with no `app.tenant_id` set.

    Every other table in this schema raises on an unbound GUC
    (`test_unset_tenant_guc_raises_rather_than_returning_rows`). This table
    must NOT — the whole point of a shared reference table is that no tenant
    context is required to use it.
    """
    with Session(app_engine) as session:
        # Must not raise, unlike the equivalent query against any
        # tenant-scoped table with no GUC bound.
        session.execute(text("SELECT count(*) FROM reference_pincodes")).scalar()


# ---------------------------------------------------------------------------
# pgvector: the dimension is part of the contract
# ---------------------------------------------------------------------------


def test_embedding_columns_are_vector_768(owner_engine: Engine) -> None:
    """FIG 2.4. A vector column that lost its dimension accepts anything.

    Worth asserting because the way it breaks is quiet: autogenerate without
    the `render_item` hook in migrations/env.py renders these as NullType,
    which applies cleanly and produces columns that cannot hold an embedding.
    """
    with Session(owner_engine) as session:
        rows = session.execute(
            text(
                "SELECT a.attname, format_type(a.atttypid, a.atttypmod) "
                "FROM pg_attribute a "
                "JOIN pg_class c ON c.oid = a.attrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'public' AND c.relname = 'record_embeddings' "
                "AND a.attname IN ('name_embedding', 'address_embedding')"
            )
        ).all()

    types = dict(rows)
    assert types == {
        "name_embedding": "vector(768)",
        "address_embedding": "vector(768)",
    }, f"Unexpected embedding column types: {types}"


def test_no_unread_vector_index_is_maintained(owner_engine: Engine) -> None:
    """Migration 0006 dropped the HNSW pair: blocking is exact cosine inside
    location blocks, so an ANN index had no reader and taxed every insert.

    If ANN search comes back, remember pgvector filters RLS *after* the index
    scan; use iterative scans or per-tenant partitioning, then replace this.
    """
    with Session(owner_engine) as session:
        defs = session.execute(
            text(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname = 'public' AND indexdef ILIKE '%USING hnsw%'"
            )
        ).all()

    assert defs == [], f"HNSW indexes with no reader: {defs}"


def test_only_one_active_model_version_per_tenant_and_type(
    platform_engine: Engine, two_tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    """FIG 2.7's "only one active version per (tenant, model_type)".

    Enforced by a partial unique index, so the second active row is rejected
    by the database rather than by whichever code path happened to remember.
    """
    tenant_a, _ = two_tenants
    insert = text(
        "INSERT INTO model_versions "
        "(tenant_id, model_type, version_tag, is_active) "
        "VALUES (:tid, 'classifier', :tag, true)"
    )

    with Session(platform_engine) as session:
        session.execute(insert, {"tid": tenant_a, "tag": "v1"})
        session.commit()

        with pytest.raises(IntegrityError):
            session.execute(insert, {"tid": tenant_a, "tag": "v2"})
            session.commit()
        session.rollback()

    # An inactive second version is fine — that is what makes a rollback a
    # one-row UPDATE rather than a delete-and-reinsert.
    with Session(platform_engine) as session:
        session.execute(
            text(
                "INSERT INTO model_versions "
                "(tenant_id, model_type, version_tag, is_active) "
                "VALUES (:tid, 'classifier', 'v3', false)"
            ),
            {"tid": tenant_a},
        )
        session.commit()
