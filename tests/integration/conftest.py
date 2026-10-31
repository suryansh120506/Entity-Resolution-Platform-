"""Fixtures for the live-service suite.

Everything here talks to the real containers from docker-compose.yml. The
session fixtures in `tests/conftest.py` provide the engines; these provide
the tenants and the cleanup.

Cleanup is by tenant deletion rather than by TRUNCATE of every table. The
`ON DELETE CASCADE` chain from `tenants` is the same mechanism production
uses to remove a tenant, so exercising it between tests means the teardown
path is itself under test — and a cascade that stopped working would show up
here as leaked rows rather than silently orphaning data in production.
`reference_pincodes` is untenanted and deliberately survives.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

#: Marks the whole package. `-m "not integration"` skips it in one move.
pytestmark = pytest.mark.integration


@pytest.fixture(scope="session", autouse=True)
def no_live_queue_worker(owner_engine: Engine) -> None:
    """Refuse to run beside a live `app.queue.worker` on this database.

    A live worker serves every tenant, so it claims the tasks these tests
    enqueue: it runs them with the real 1 GB encoder instead of the stub,
    races the test's in-process drain, and writes rows after teardown has
    started. The resulting failures look like pipeline bugs and are not.
    Failing here, with the reason, beats that diagnosis.
    """
    with Session(owner_engine) as session:
        live = session.execute(
            text(
                "SELECT count(*) FROM pg_stat_activity "
                "WHERE datname = current_database() AND application_name LIKE 'er-worker:%'"
            )
        ).scalar_one()
    if live:
        pytest.fail(
            f"{live} live queue worker(s) are attached to this database. Stop "
            "`python -m app.queue.worker` before running the integration suite: it "
            "would claim the tests' tasks.",
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never reach a real LLM or DuckDuckGo from the suite.

    Settings come from the developer's `.env`; once LLM_BASE_URL is set
    there, every pipeline test would route its middle band to Tier 2 and
    search the live web. Tests that exercise triage opt back in with fakes
    (see test_automation_e2e.py).
    """
    from app.core.config import get_settings
    from app.enrichment import sources

    monkeypatch.setattr(get_settings(), "llm_base_url", None)
    # Some tests rebuild Settings (get_settings.cache_clear()); the env var,
    # which outranks .env, keeps a rebuilt instance off the real LLM too.
    monkeypatch.setenv("LLM_BASE_URL", "")

    # A tenant's Premium key reaches a provider without any server setting,
    # so blocking LLM_BASE_URL is not enough. Every network path is closed;
    # a test that needs one fakes the method above it (same_entity, extract,
    # check, _web_search) and never reaches these.
    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("an integration test tried to reach a real LLM or search engine")

    monkeypatch.setattr(sources.LlmClient, "chat", no_network)
    monkeypatch.setattr(sources, "ddgs_search", no_network)
    import app.workers.tasks.enrichment as enrichment_tasks

    monkeypatch.setattr(enrichment_tasks, "ddgs_search", no_network)  # imported by name there


def _create_tenant(engine: Engine, slug: str) -> uuid.UUID:
    tenant_id = uuid.uuid4()
    with Session(engine) as session:
        session.execute(
            text(
                "INSERT INTO tenants (id, name, slug, confidence_auto_merge, "
                "confidence_api_floor) VALUES (:id, :name, :slug, :auto, :floor)"
            ),
            {
                "id": tenant_id,
                "name": f"E2E {slug}",
                "slug": f"{slug}-{tenant_id.hex[:8]}",
                # Explicit rather than relying on the column defaults: the
                # confidence gate reads these, and a test that silently
                # depended on a default would stop meaning anything the day
                # the default changed.
                "auto": "0.950",
                # 0: Tier 1 separates nothing, so with no LLM (the suite
                # default) every cluster under auto-merge waits for a
                # steward, as the steward-flow tests need. Tests of the
                # automated tiers set their own thresholds.
                "floor": "0.000",
            },
        )
        session.commit()
    return tenant_id


def _drop_tenant(engine: Engine, tenant_id: uuid.UUID) -> None:
    with Session(engine) as session:
        session.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant_id})
        session.commit()


@pytest.fixture(scope="session", autouse=True)
def seeded_pincodes(owner_engine: Engine) -> Iterator[None]:
    """Populate `reference_pincodes` for the PINs the fixtures use.

    Not optional set-dressing. With this table empty every PIN fails
    validation, `lf_complete_record` abstains on every row, and the
    classifier labels an entire upload `Missing_Context` — no embeddings, no
    clusters, and a job that succeeds having resolved nothing. A suite that
    ran against an unseeded table would be testing that degraded path while
    appearing to test the real one.

    Seeded once per session and left in place: the table is global and
    un-tenanted, so it is not part of any tenant's teardown.
    """
    with Session(owner_engine) as session:
        session.execute(
            text(
                "INSERT INTO reference_pincodes "
                "(pincode, office_name, district_name, state_name) "
                "SELECT lpad(g::text, 6, '0'), 'E2E Office ' || g, 'Bangalore', 'Karnataka' "
                "FROM generate_series(560000, 560050) AS g "
                "ON CONFLICT DO NOTHING"
            )
        )
        session.commit()

    # The validity cache is process-global and may already hold "not found"
    # for these PINs from an earlier test in the same session.
    from app.ml_pipeline.utils import clear_pincode_cache

    clear_pincode_cache()

    yield

    with Session(owner_engine) as session:
        session.execute(text("DELETE FROM reference_pincodes WHERE office_name LIKE 'E2E %'"))
        session.commit()


@pytest.fixture
def e2e_tenant(platform_engine: Engine) -> Iterator[uuid.UUID]:
    """One disposable tenant, removed with all its data afterwards."""
    tenant_id = _create_tenant(platform_engine, "primary")
    yield tenant_id
    _drop_tenant(platform_engine, tenant_id)


@pytest.fixture
def two_e2e_tenants(platform_engine: Engine) -> Iterator[tuple[uuid.UUID, uuid.UUID]]:
    """Two disposable tenants, for the isolation tests."""
    first = _create_tenant(platform_engine, "alpha")
    second = _create_tenant(platform_engine, "beta")
    yield first, second
    _drop_tenant(platform_engine, first)
    _drop_tenant(platform_engine, second)


_ROW_COUNT_SQL = text(
    "SELECT (SELECT count(*) FROM jobs) + (SELECT count(*) FROM school_records) "
    "+ (SELECT count(*) FROM clusters) + (SELECT count(*) FROM record_embeddings) "
    "+ (SELECT count(*) FROM tenant_ai_settings) + (SELECT count(*) FROM tenant_ontologies)"
)


@pytest.fixture
def assert_no_leaked_rows(platform_engine: Engine) -> Iterator[None]:
    """Fail a test that left tenant-scoped rows behind.

    State leaking between tests in a suite that shares one database is the
    classic way an integration suite becomes order-dependent — and the way
    it then passes locally and fails in CI, or vice versa.

    Measured as a delta against a baseline taken before the test, not against
    zero. The database is shared with whatever else the developer has been
    doing — a dev tenant created for the frontend, a half-finished manual
    experiment — and an absolute assertion would fail on their data rather
    than on the leak it is looking for. What matters is that this test put
    the count back where it found it.
    """
    with Session(platform_engine) as session:
        before = session.execute(_ROW_COUNT_SQL).scalar_one()

    yield

    with Session(platform_engine) as session:
        after = session.execute(_ROW_COUNT_SQL).scalar_one()

    assert after == before, (
        f"{after - before} tenant-scoped rows survived teardown (before={before}, after={after})"
    )
