"""Test fixtures.

These are integration tests. They need the real Postgres from
docker-compose.yml with migrations applied, because the thing under test is
Postgres behaviour — a policy evaluated by the server. There is no way to test
row-level security against a mock, and a test that mocked it would pass on a
schema with no policies at all, which is precisely the failure it exists to
catch.

    docker compose up -d
    alembic upgrade head
    pytest
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session

# Load the repo-root .env before the DSNs are read.
#
# Without this the three fixtures below find nothing in the environment and
# every integration test *skips* — a fully green run that exercised none of
# the behaviour it claims to cover, which is a worse outcome than a red one.
# `override=False` so a DSN already exported (CI, or a developer pointing at
# a scratch database) still wins over the file.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

PLATFORM_URL = os.environ.get("PLATFORM_DATABASE_URL")
APP_URL = os.environ.get("DATABASE_URL")
OWNER_URL = os.environ.get("MIGRATION_DATABASE_URL")


@pytest.fixture(scope="session")
def platform_engine() -> Iterator[Engine]:
    """Control-plane engine (BYPASSRLS) — used only to seed and to clean up."""
    if not PLATFORM_URL:
        pytest.skip("PLATFORM_DATABASE_URL is not set")
    engine = create_engine(PLATFORM_URL)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def app_engine() -> Iterator[Engine]:
    """Data-plane engine — the role under test. Governed by every policy."""
    if not APP_URL:
        pytest.skip("DATABASE_URL is not set")
    engine = create_engine(APP_URL)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def owner_engine() -> Iterator[Engine]:
    """Owner engine — used to assert that FORCE applies to the owner too."""
    if not OWNER_URL:
        pytest.skip("MIGRATION_DATABASE_URL is not set")
    engine = create_engine(OWNER_URL)
    yield engine
    engine.dispose()


@pytest.fixture
def two_tenants(platform_engine: Engine) -> Iterator[tuple[uuid.UUID, uuid.UUID]]:
    """Two tenants, each with one job and one record. Cleaned up afterwards.

    Seeded through the control-plane role because `tenants` is RLS-protected
    and there is no tenant to scope to while creating one. That is not a
    workaround for the test — it is how tenant provisioning works in this
    system, which is why the role exists.
    """
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()

    with Session(platform_engine) as session:
        for tid, slug in ((tenant_a, "tenant-a"), (tenant_b, "tenant-b")):
            session.execute(
                text("INSERT INTO tenants (id, name, slug) VALUES (:id, :name, :slug)"),
                {"id": tid, "name": f"Test {slug}", "slug": f"{slug}-{tid.hex[:8]}"},
            )
            job_id = uuid.uuid4()
            session.execute(
                text(
                    "INSERT INTO jobs (id, tenant_id, status, total_rows) "
                    "VALUES (:id, :tenant_id, 'PENDING', 1)"
                ),
                {"id": job_id, "tenant_id": tid},
            )
            session.execute(
                text(
                    "INSERT INTO school_records (tenant_id, job_id, raw_row) "
                    "VALUES (:tenant_id, :job_id, CAST(:raw_row AS jsonb))"
                ),
                {
                    "tenant_id": tid,
                    "job_id": job_id,
                    "raw_row": '{"name": "KENDRIYA VIDYALAYA", "pincode": "000000"}',
                },
            )
        session.commit()

    yield tenant_a, tenant_b

    with Session(platform_engine) as session:
        # ON DELETE CASCADE from tenants takes jobs, records and clusters with it.
        session.execute(
            text("DELETE FROM tenants WHERE id = ANY(:ids)"),
            {"ids": [tenant_a, tenant_b]},
        )
        session.commit()
