"""`jobs` queue — lifecycle transitions that happen after the pipeline.

`finalize_reviewed` moves a job out of AWAITING_REVIEW once its last cluster
reaches a decision. Enqueued two ways, both deduplicated on the job id:

* event-driven, by the feedback route in the same transaction as the
  steward's verdict, so the job completes the moment the queue empties;
* by every worker's periodic sweep (`core.enqueue_reviewed_job_finalizers`),
  which catches whatever the event path missed — a review made before this
  existed, or a verdict written by a script.

Before this, nothing ever advanced AWAITING_REVIEW: a fully reviewed job sat
there forever.
"""

from __future__ import annotations

import logging
from typing import Any

from app.db.session import tenant_sync_session
from app.models.enums import JobStatus
from app.queue.registry import task
from app.workers.job_state import JobMissingError, advance_status, load_job
from app.workers.tasks.resolution import GOLDEN_TASK, enqueue_stage, outstanding_reviews

logger = logging.getLogger(__name__)


@task("jobs.finalize_reviewed")
def finalize_reviewed(job_id: str, tenant_id: str) -> dict[str, Any]:
    with tenant_sync_session(tenant_id) as session:
        try:
            job = load_job(session, job_id)
        except JobMissingError:
            return {"job_id": job_id, "skipped": "job deleted"}

        if JobStatus(job.status) is not JobStatus.AWAITING_REVIEW:
            return {"job_id": job_id, "skipped": f"job is {job.status}"}

        pending = outstanding_reviews(session, job_id)
        if pending:
            return {"job_id": job_id, "pending": pending}

        advance_status(session, job_id, JobStatus.RESOLVING)
        advance_status(session, job_id, JobStatus.COMPLETED)
        # Tier 3 runs on every completed job, however it got there.
        enqueue_stage(session, tenant_id, job_id, GOLDEN_TASK)
        logger.info("Job %s: every cluster decided; COMPLETED", job_id)
        return {"job_id": job_id, "completed": True}
