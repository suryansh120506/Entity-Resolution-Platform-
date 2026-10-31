"""Enqueue Silver backfills for jobs whose records lack stable identities.

Jobs ingested before the Silver layer (or before stable ids) are invisible to
gold linking: the pool is built from Silver rows with a stable id. This
finds every such job and enqueues `silver.backfill_job` for it — re-runnable
and idempotent (the task upserts; the enqueue is deduplicated per job).
A running queue worker picks the tasks up; job statuses are not touched.

    python scripts/backfill_silver.py            # every tenant
    python scripts/backfill_silver.py --dry-run  # list, enqueue nothing
"""

from __future__ import annotations

import click
from sqlalchemy import text

from app.db.session import platform_session, tenant_sync_session
from app.queue.core import enqueue, stage_dedupe_key

_NEEDS_BACKFILL = text(
    """
    SELECT j.tenant_id, j.id, count(r.id) AS records, count(s.stable_id) AS identified
      FROM jobs j
      JOIN school_records r ON r.job_id = j.id
      LEFT JOIN silver_records s ON s.record_id = r.id
     WHERE j.status IN ('AWAITING_REVIEW', 'COMPLETED')
     GROUP BY j.tenant_id, j.id
    HAVING count(s.stable_id) < count(r.id)
     ORDER BY j.tenant_id, j.id
    """
)


@click.command()
@click.option("--dry-run", is_flag=True, help="List the jobs, enqueue nothing.")
def main(dry_run: bool) -> None:
    # Cross-tenant discovery is a control-plane read; each enqueue then runs
    # tenant-bound, so RLS's WITH CHECK still vouches for the task's tenant.
    with platform_session() as session:
        jobs = session.execute(_NEEDS_BACKFILL).all()
    if not jobs:
        click.echo("Every finished job already has stable identities.")
        return
    for tenant_id, job_id, records, identified in jobs:
        click.echo(f"{job_id}  tenant={tenant_id}  identified {identified}/{records}")
        if dry_run:
            continue
        with tenant_sync_session(tenant_id) as session:
            enqueue(
                session,
                tenant_id=tenant_id,
                task_name="silver.backfill_job",
                payload={"job_id": str(job_id)},
                job_id=job_id,
                dedupe_key=stage_dedupe_key("silver.backfill_job", job_id),
            )
    if not dry_run:
        click.echo(f"Enqueued {len(jobs)} backfill task(s); a running worker will process them.")


if __name__ == "__main__":
    main()
