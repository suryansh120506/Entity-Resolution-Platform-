"""Retry the stage a FAILED job stopped at.

A task that exhausts its attempts fails the job, and FAILED is terminal: the
state machine (rightly) refuses to move a finished job. That is the wrong
answer when the cause was the environment (the laptop slept, the network
dropped) and the stage is idempotent. This reopens the job at the status the
failed stage expects to find, clears the recorded error, and enqueues a fresh
task. The failed task row and its dead letter stay as history.

    python scripts/retry_job.py <job_id>
    python scripts/retry_job.py <job_id> --dry-run
"""

from __future__ import annotations

import uuid

import click
from sqlalchemy import text

from app.db.session import platform_session, tenant_sync_session
from app.queue.core import enqueue, stage_dedupe_key

#: The job status each stage's `advance_status` call starts from.
_REOPEN_AT = {
    "ingest.ingest_and_classify": "PENDING",
    "silver.standardize": "PREPROCESSING",
    "embeddings.embeddings_stage": "PREPROCESSING",
    "blocking.blocking_stage": "EMBEDDING",
    "matching.probabilistic_match": "BLOCKING",
    "gold.link_job": "MATCHING",
    "resolution.resolve_by_confidence": "MATCHING",
    # Tier 2 runs on a RESOLVING job and resumes from the undecided clusters.
    "enrichment.triage_job": "RESOLVING",
}


@click.command()
@click.argument("job_id", type=click.UUID)
@click.option("--dry-run", is_flag=True)
def main(job_id: uuid.UUID, dry_run: bool) -> None:
    with platform_session() as session:
        job = session.execute(
            text("SELECT tenant_id, status FROM jobs WHERE id = :j"), {"j": job_id}
        ).first()
        if job is None:
            raise click.ClickException(f"No job {job_id}")
        tenant_id, status = job
        if status != "FAILED":
            raise click.ClickException(f"Job is {status}; only a FAILED job can be retried.")
        failed = session.execute(
            text(
                "SELECT task_name, payload, last_error FROM task_queue "
                "WHERE job_id = :j AND status = 'failed' ORDER BY finished_at DESC LIMIT 1"
            ),
            {"j": job_id},
        ).first()
    if failed is None:
        raise click.ClickException("No failed task found for this job.")
    task_name, payload, last_error = failed
    reopen_at = _REOPEN_AT.get(task_name)
    if reopen_at is None:
        raise click.ClickException(f"Don't know how to reopen the job for {task_name}.")

    click.echo(f"Failed task : {task_name}\nLast error  : {last_error}\nReopen at   : {reopen_at}")
    if dry_run:
        return

    with tenant_sync_session(tenant_id) as session:
        session.execute(
            text(
                "UPDATE jobs SET status = :s, completed_at = NULL, "
                "stage_progress = stage_progress - 'error' WHERE id = :j"
            ),
            {"s": reopen_at, "j": job_id},
        )
        enqueue(
            session,
            tenant_id=tenant_id,
            task_name=task_name,
            payload=dict(payload),
            job_id=job_id,
            dedupe_key=stage_dedupe_key(task_name, job_id),
        )
    click.echo("Re-enqueued. A running worker will pick it up.")


if __name__ == "__main__":
    main()
