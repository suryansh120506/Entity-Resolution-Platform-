"""Re-run the confidence gate on a job waiting for review.

For jobs that reached AWAITING_REVIEW under the old gate, where everything
below auto-merge went to a steward. Re-gating applies the automated tiers
to the clusters no one has decided yet:

    below the tenant's floor (0.80)  -> kept separate      (Tier 1)
    the middle band, suspected chains -> automated triage  (Tier 2, needs an LLM)

Clusters a steward already decided are never touched. Without an LLM
configured the middle band stays in review; run this before configuring
one only if you want Tier 1 applied now and the rest left for people.

    python scripts/regate_job.py <job_id> --dry-run   # what would happen
    python scripts/regate_job.py <job_id>
"""

from __future__ import annotations

import uuid

import click
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import platform_session, tenant_sync_session
from app.queue.core import enqueue, stage_dedupe_key

_TASK = "resolution.resolve_by_confidence"


@click.command()
@click.argument("job_id", type=click.UUID)
@click.option("--dry-run", is_flag=True)
def main(job_id: uuid.UUID, dry_run: bool) -> None:
    with platform_session() as session:
        job = session.execute(
            text(
                "SELECT j.tenant_id, j.status, t.confidence_api_floor, t.confidence_auto_merge "
                "FROM jobs j JOIN tenants t ON t.id = j.tenant_id WHERE j.id = :j"
            ),
            {"j": job_id},
        ).first()
        if job is None:
            raise click.ClickException(f"No job {job_id}")
        tenant_id, status, floor, auto_merge = job
        if status != "AWAITING_REVIEW":
            raise click.ClickException(f"Job is {status}; only AWAITING_REVIEW can be re-gated.")
        below, middle = session.execute(
            text(
                "SELECT count(*) FILTER (WHERE confidence_score IS NULL "
                "  OR review_reason IS NOT NULL OR confidence_score < :floor), "
                "count(*) FILTER (WHERE confidence_score >= :floor AND review_reason IS NULL) "
                "FROM clusters WHERE job_id = :j AND status IN ('PENDING_REVIEW', 'PENDING_API')"
            ),
            {"j": job_id, "floor": floor},
        ).one()

    triage = get_settings().auto_triage_active
    click.echo(f"Floor {floor}, auto-merge {auto_merge}")
    click.echo(f"Kept separate by Tier 1       : {below}")
    click.echo(
        f"Sent to automated triage       : {middle}"
        if triage
        else f"Left for review (no LLM set)   : {middle}"
    )
    if dry_run:
        return

    with tenant_sync_session(tenant_id) as session:
        enqueue(
            session,
            tenant_id=tenant_id,
            task_name=_TASK,
            payload={"job_id": str(job_id)},
            job_id=job_id,
            dedupe_key=stage_dedupe_key(_TASK, job_id),
        )
    click.echo("Re-gate enqueued. A running worker will pick it up.")


if __name__ == "__main__":
    main()
