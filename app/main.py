"""FastAPI application factory.

SAD 3.2 / NFR-05 constrain what may ever live behind this module: auth
verification, request validation, enqueue-and-return. No pandas, no torch, no
xgboost, and no Celery task logic reachable from an `async def` route. The
packaging split in pyproject.toml (ML libraries confined to the [ml] extra,
which the API image does not install) is what keeps that enforceable rather
than aspirational.

The v1 surface from TRD 2.5 is mounted here: uploads, job status and stats,
the keyset-paginated review queue, steward feedback, and the streaming
export. Health routes stay unauthenticated; everything under /v1 carries the
tenant-binding dependency at the router level.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.routes import clusters, health, jobs, ontologies
from app.api.v1.routes import settings as settings_routes
from app.core.config import get_settings
from app.db.session import async_engine

logger = logging.getLogger(__name__)


# ON WINDOWS, THE API MUST BE STARTED WITH:
#
#     uvicorn app.main:app --loop asyncio:SelectorEventLoop
#
# and not with a bare `uvicorn app.main:app`. Uvicorn hardcodes
# `asyncio.ProactorEventLoop` on Windows (uvicorn/loops/asyncio.py), and
# psycopg's async mode refuses to run on it — every request that touches the
# database fails with "Psycopg cannot use the 'ProactorEventLoop' to run in
# async mode". That is not a degradation; it is the whole async API being
# non-functional on the platform.
#
# It cannot be fixed from inside this module. Uvicorn passes an explicit
# `loop_factory` to `asyncio.run`, which bypasses the event loop *policy*
# entirely, so `asyncio.set_event_loop_policy(...)` here would look like a
# fix and do nothing. The `--loop` flag takes a `module:attr` import string,
# and `asyncio:SelectorEventLoop` is already the default everywhere except
# Windows, so the same flag is correct on every platform. The Makefile's
# `api` target passes it.
#
# Worth knowing why no test caught this: the API tests exercise the auth
# boundary (which never reaches the database) and the integration tests use
# the *sync* engine via Celery. Nothing opened an async connection until the
# frontend sent a real request.


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    logger.info("ER-PLATFORM API starting (environment=%s)", settings.environment)
    yield
    # Close pooled connections on the way out so a rolling deploy does not
    # leave sockets open against Postgres.
    await async_engine.dispose()
    logger.info("ER-PLATFORM API stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Entity Resolution Platform",
        description=("Multi-tenant entity resolution & record linkage API (ER-PLATFORM-001)."),
        version="0.1.0",
        lifespan=lifespan,
        # Interactive docs are useful locally and a needless surface in prod.
        docs_url="/docs" if settings.environment in {"local", "ci"} else None,
        redoc_url=None,
    )
    app.include_router(health.router)
    app.include_router(jobs.router)
    app.include_router(clusters.router)
    app.include_router(settings_routes.router)
    app.include_router(ontologies.router)
    return app


app = create_app()
