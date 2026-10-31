"""Engines, session factories, and tenant-scoped transaction plumbing.

This module is the load-bearing half of FIG 2.8. The policy in the migration
is inert on its own — it reads `current_setting('app.tenant_id')`, and
something has to put a value there. That something is here.

Three engines, one per plane:

  async     FastAPI request handlers, on the data-plane role. Light reads and
            enqueues only; NFR-05 forbids CPU-bound work on this path.
  sync      Queue workers, also data-plane. They own every CPU-bound step
            (SAD 3.2) and have no reason to be async.
  platform  The BYPASSRLS control-plane role, built lazily. Tenant
            provisioning, dead-letter triage, retrain fan-out — nothing else.

The first two use the same psycopg 3 driver and the same DSN.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.core.constants import TENANT_GUC

_settings = get_settings()

# --- Data-plane engines --------------------------------------------------------

async_engine: AsyncEngine = create_async_engine(
    _settings.async_database_url,
    echo=_settings.sql_echo,
    pool_size=_settings.db_pool_size,
    max_overflow=_settings.db_max_overflow,
    pool_timeout=_settings.db_pool_timeout,
    pool_recycle=_settings.db_pool_recycle,
    pool_pre_ping=True,
)

sync_engine: Engine = create_engine(
    _settings.sync_database_url,
    echo=_settings.sql_echo,
    pool_size=_settings.db_pool_size,
    max_overflow=_settings.db_max_overflow,
    pool_timeout=_settings.db_pool_timeout,
    pool_recycle=_settings.db_pool_recycle,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    class_=Session,
    expire_on_commit=False,
    autoflush=False,
)

# --- Tenant GUC binding --------------------------------------------------------

_SET_TENANT_SQL = text(f"SELECT set_config('{TENANT_GUC}', :tenant_id, true)")

# NOTE on `set_config(..., is_local => true)` versus the literal `SET LOCAL`
# shown in FIG 2.8: they are the same thing — both scope the value to the
# current transaction. The function form is used because `SET LOCAL` is
# utility syntax that cannot take a bind parameter, which would mean
# interpolating a tenant id into a SQL string. A parameterised call removes
# that whole class of mistake from the one place where making it is a
# cross-tenant leak.

_SESSION_TENANT_KEY = "er_tenant_id"


@event.listens_for(Session, "after_begin")
def _apply_tenant_guc(session: Session, transaction: Any, connection: Connection) -> None:
    """Re-apply the tenant GUC at the start of every transaction on a session.

    Setting it once when the session is created is not enough: the value is
    transaction-local, so the first `commit()` discards it and any query after
    that would run with the GUC unset. Under the FIG 2.8 policy an unset GUC
    raises rather than returning rows, so the failure is loud rather than
    silent — but it is still a failure, and a long-running queue task that
    commits per batch would hit it constantly.

    Binding to `after_begin` means the value is restored for every transaction
    the session opens, for as long as the session lives. It also means no
    caller has to remember to re-set it, and no caller can forget.

    Registered on the sync `Session` class, which is what `AsyncSession` drives
    underneath — so this covers both data-plane engines.
    """
    tenant_id = session.info.get(_SESSION_TENANT_KEY)
    if tenant_id is not None:
        connection.execute(_SET_TENANT_SQL, {"tenant_id": str(tenant_id)})


def bind_tenant(session: Session | AsyncSession, tenant_id: uuid.UUID | str) -> None:
    """Pin a session to one tenant for its entire lifetime.

    The value must come from a verified JWT claim. TRD 2.5: tenant id is never
    accepted from a request body or a query parameter.
    """
    # Validates the shape as a side effect — a non-UUID never reaches SQL.
    tenant_uuid = tenant_id if isinstance(tenant_id, uuid.UUID) else uuid.UUID(str(tenant_id))
    session.info[_SESSION_TENANT_KEY] = tenant_uuid


# --- Data-plane sessions -------------------------------------------------------


@asynccontextmanager
async def tenant_async_session(tenant_id: uuid.UUID | str) -> AsyncIterator[AsyncSession]:
    """Async session scoped to one tenant.

    `app.api.deps.get_db` delegates to this, so the request path and any async
    script share one implementation of the binding.

    There is no eager "SELECT 1" to force the transaction open: the
    `after_begin` listener fires before the first statement of every
    transaction, so the GUC is always in place by the time a query runs.
    Opening a transaction eagerly would add a round trip to every request to
    re-prove something the listener already guarantees.
    """
    session = AsyncSessionLocal()
    bind_tenant(session, tenant_id)
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


@contextmanager
def tenant_sync_session(tenant_id: uuid.UUID | str) -> Iterator[Session]:
    """Sync session scoped to one tenant. Used by queue tasks.

    Every task that touches tenant data takes `tenant_id` as an explicit
    argument and opens its session through here — a worker has no request to
    infer the tenant from, so it must be carried in the task payload.
    """
    session = SyncSessionLocal()
    bind_tenant(session, tenant_id)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# --- Control plane -------------------------------------------------------------
# A separate engine on the BYPASSRLS role, built lazily because most processes
# never need it and should not hold a pool against it.

_platform_engine: Engine | None = None


def _get_platform_engine() -> Engine:
    global _platform_engine
    if _platform_engine is None:
        if _settings.platform_database_url is None:
            raise RuntimeError(
                "PLATFORM_DATABASE_URL is not configured. Cross-tenant "
                "operations require the control-plane role (see "
                "docker/postgres/initdb/002_app_role.sh). Deliberately not "
                "falling back to the data-plane connection: under the FIG 2.8 "
                "policies that fallback would return zero rows or raise, and "
                "either way it would look like a bug in the caller rather "
                "than a missing credential."
            )
        _platform_engine = create_engine(
            str(_settings.platform_database_url),
            echo=_settings.sql_echo,
            # Small pool on purpose: this is for provisioning and triage, not
            # for serving traffic. A large pool here is a sign of misuse.
            pool_size=2,
            max_overflow=3,
            pool_pre_ping=True,
        )
    return _platform_engine


@contextmanager
def platform_session() -> Iterator[Session]:
    """Session with NO tenant scoping, on the BYPASSRLS control-plane role.

    There are exactly four legitimate callers, all cross-tenant by nature and
    none of them on a request path:

      1. Provisioning a tenant. There is no tenant to scope to yet, and
         `tenants` is itself RLS-protected.
      2. Reading `dead_letter_tasks` rows whose tenant_id is NULL — TRD 2.7's
         platform-level failures, which no tenant-scoped connection can see.
      3. Enumerating tenants to fan out the nightly retrain (SAD 3.6).
      4. The queue worker's claim / complete / retry / reap transitions on
         `task_queue` (app/queue/core.py). It touches only queue rows and
         dead letters; the task itself always runs on a tenant-bound session.

    Anything else belongs in `tenant_sync_session`. This function is named for
    what it is rather than what it does — `unscoped_session` would read like a
    convenience, and it is not one: every row it returns is a row that crossed
    a tenant boundary.
    """
    session = Session(bind=_get_platform_engine(), expire_on_commit=False, autoflush=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
