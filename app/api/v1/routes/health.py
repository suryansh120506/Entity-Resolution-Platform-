"""Liveness and readiness. The only routes that exist in Step 1.

Deliberately unauthenticated and deliberately not dependent on get_db — a
readiness probe that needs a tenant-bound session could not run, since there
is no tenant behind an orchestrator's health check.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import async_engine
from app.queue.topology import WORKER_APPLICATION_NAME_PREFIX

router = APIRouter(tags=["health"])


@router.get("/healthz", summary="Liveness — process is up")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness — dependencies are reachable")
async def readyz() -> JSONResponse:
    """Postgres is the only backing service: it is the database *and* the
    task queue, so an API that can reach it can both store and enqueue."""
    checks: dict[str, str] = {}

    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
        checks["postgres"] = f"error: {type(exc).__name__}"

    healthy = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "ready" if healthy else "degraded", "checks": checks},
    )


@router.get("/system/status", summary="Is every part of the platform running?")
async def system_status() -> dict[str, object]:
    """Database, broker, and whether any pipeline worker is listening.

    Separate from `/readyz` on purpose: readiness gates traffic to *this*
    process, and the API is perfectly ready without a worker. But an upload
    accepted with no worker running sits at PENDING forever with nothing
    anywhere saying why — the single most common "it's broken" report this
    platform has had — so the UI polls this and says so up front.

    Always 200; the body carries the verdict. No tenant data is involved,
    which is why it needs no token — and why it lives beside `/healthz`
    rather than under `/v1`, where every route is required to authenticate.
    """
    ready = await readyz()
    checks: dict[str, str] = json.loads(bytes(ready.body)).get("checks", {})

    workers = 0
    if checks.get("postgres") == "ok":
        try:
            # Each live worker holds a LISTEN connection on this same role,
            # named er-worker:<id>. A dead worker's connection, and so its
            # row here, goes with it: no heartbeat table to go stale.
            async with async_engine.connect() as conn:
                workers = (
                    await conn.execute(
                        text(
                            "SELECT count(*) FROM pg_stat_activity "
                            "WHERE datname = current_database() "
                            "AND application_name LIKE :prefix"
                        ),
                        {"prefix": f"{WORKER_APPLICATION_NAME_PREFIX}%"},
                    )
                ).scalar_one()
        except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
            checks["worker"] = f"error: {type(exc).__name__}"
    checks.setdefault("worker", "ok" if workers else "down")

    return {
        "status": "ok" if all(v == "ok" for v in checks.values()) else "degraded",
        "checks": checks,
        "workers": workers,
        "enrichment_llm": bool(get_settings().llm_base_url),
    }
