"""Unit tests for the queue topology and the worker stages — FIG 3.2.

No database. What is tested here is everything the worker layer decides
*before* it touches one: which queue a task lands on, what follows what, how
the job state machine refuses illegal transitions, and the pure
grouping/routing helpers each stage is built from. Queue semantics that only
Postgres can get wrong (SKIP LOCKED, fencing, dedupe) are in
tests/integration/test_medallion_e2e.py.

The stage bodies themselves are integration surface — they open a
tenant-bound session and run real ML — and are covered by the end-to-end
suite against Docker rather than mocked into meaninglessness here.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from app.models.enums import JobStatus
from app.models.job import Job
from app.queue.core import retry_delay
from app.queue.registry import REGISTRY, load_task_modules
from app.queue.topology import PIPELINE, PIPELINE_ENTRYPOINT, QUEUES, UnknownTaskError, queue_for
from app.workers.job_state import (
    PIPELINE_ORDER,
    JobStateError,
    StaleStageError,
    advance_status,
    mark_failed,
    record_progress,
)

JOB_ID = "11111111-1111-1111-1111-111111111111"
TENANT_ID = "22222222-2222-2222-2222-222222222222"


class FakeResult:
    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar_one_or_none(self) -> Any:
        return self._value


class FakeSession:
    """Returns one job and records flushes. No database involved."""

    def __init__(self, job: Job | None) -> None:
        self.job = job
        self.flushes = 0

    def execute(self, *args: Any, **kwargs: Any) -> FakeResult:
        return FakeResult(self.job)

    def flush(self) -> None:
        self.flushes += 1


def _job(status: JobStatus = JobStatus.PENDING) -> Job:
    job = Job()
    job.id = JOB_ID
    job.status = status.value
    job.stage_progress = {}
    return job


# --- Queue topology ------------------------------------------------------------


def test_every_pipeline_task_is_registered_on_its_queue() -> None:
    """A successor the worker cannot find dead-letters every job at that stage."""
    registry = load_task_modules()
    named = {PIPELINE_ENTRYPOINT, *PIPELINE, *(s for nxt in PIPELINE.values() for s in nxt)}

    for task_name in named | {"jobs.finalize_reviewed"}:
        assert task_name in registry, f"{task_name} is not registered"
        assert registry[task_name].queue == queue_for(task_name) in QUEUES


def test_the_pipeline_runs_in_figure_3_2_order() -> None:
    order, current = [PIPELINE_ENTRYPOINT], PIPELINE_ENTRYPOINT
    while current in PIPELINE:
        current = PIPELINE[current][0]
        order.append(current)

    assert order == [
        "ingest.ingest_and_classify",
        "silver.standardize",
        "embeddings.embeddings_stage",
        "blocking.blocking_stage",
        "matching.probabilistic_match",
        "gold.link_job",
        "resolution.resolve_by_confidence",
    ]


def test_enrichment_fans_out_beside_the_chain_not_in_it() -> None:
    """Network-bound, rate-limited enrichment must never gate matching."""
    assert PIPELINE["silver.standardize"][0] == "embeddings.embeddings_stage"
    assert "enrichment.enrich_job" in PIPELINE["silver.standardize"]
    assert "enrichment.enrich_job" not in PIPELINE


def test_the_pipeline_has_no_cycle() -> None:
    seen: set[str] = set()
    frontier = [PIPELINE_ENTRYPOINT]
    while frontier:
        task_name = frontier.pop()
        assert task_name not in seen, f"{task_name} is reached twice"
        seen.add(task_name)
        frontier.extend(PIPELINE.get(task_name, ()))


def test_enrichment_is_capped_to_one_running_task_platform_wide() -> None:
    """Registry politeness and free-tier LLM limits are global constraints."""
    assert QUEUES["enrichment"].max_running == 1


@pytest.mark.parametrize("bad", ["nodot", "unknownqueue.task", ".leading"])
def test_unroutable_task_names_are_refused(bad: str) -> None:
    with pytest.raises(UnknownTaskError):
        queue_for(bad)


def test_retry_delay_grows_and_is_capped() -> None:
    spec = QUEUES["ingest"]
    first = [retry_delay(spec, 1) for _ in range(50)]
    third = [retry_delay(spec, 3) for _ in range(50)]

    assert max(first) <= spec.backoff_base_seconds * 1.25
    assert min(third) >= spec.backoff_base_seconds * 16 * 0.75
    assert max(retry_delay(spec, 20) for _ in range(50)) <= spec.backoff_cap_seconds * 1.25
    assert len(set(first)) > 1, "no jitter: synchronised retry storms"


def test_a_duplicate_registration_is_refused() -> None:
    from app.queue.registry import task

    load_task_modules()
    with pytest.raises(ValueError, match="registered twice"):
        task("jobs.finalize_reviewed")(lambda job_id, tenant_id: None)
    assert REGISTRY["jobs.finalize_reviewed"].fn.__name__ == "finalize_reviewed"


# --- Job state machine ---------------------------------------------------------


def test_status_advances_forward() -> None:
    session = FakeSession(_job(JobStatus.PENDING))

    job = advance_status(session, JOB_ID, JobStatus.PREPROCESSING)  # type: ignore[arg-type]

    assert job.status == JobStatus.PREPROCESSING.value


def test_re_entering_the_same_status_is_a_no_op() -> None:
    """`task_acks_late` makes redelivery routine, not exceptional.

    A worker that did the work and died before acking gets the task again.
    The stage must not fail on the status it already set.
    """
    session = FakeSession(_job(JobStatus.EMBEDDING))

    job = advance_status(session, JOB_ID, JobStatus.EMBEDDING)  # type: ignore[arg-type]

    assert job.status == JobStatus.EMBEDDING.value
    assert session.flushes == 0, "a no-op must not write"


def test_backwards_transitions_raise_the_stale_variant() -> None:
    """A backwards move is a *late duplicate*, and the type says so.

    The distinction is load-bearing rather than cosmetic: stages catch
    `StaleStageError` and return quietly, while a general `JobStateError`
    fails the job. Collapsing the two would let a stray re-dispatch of a
    finished chain mark a healthy AWAITING_REVIEW job as FAILED.
    """
    session = FakeSession(_job(JobStatus.MATCHING))

    with pytest.raises(StaleStageError, match="already at"):
        advance_status(session, JOB_ID, JobStatus.EMBEDDING)  # type: ignore[arg-type]

    # Still a JobStateError, so any caller catching the base class is safe.
    assert issubclass(StaleStageError, JobStateError)


@pytest.mark.parametrize("terminal", sorted(JobStatus.terminal()))
def test_terminal_jobs_cannot_be_resurrected(terminal: JobStatus) -> None:
    """A finished job reappearing as running is worse than a refused task."""
    session = FakeSession(_job(terminal))

    with pytest.raises(JobStateError, match="terminal"):
        advance_status(session, JOB_ID, JobStatus.BLOCKING)  # type: ignore[arg-type]


def test_failure_is_reachable_from_any_stage() -> None:
    for status in PIPELINE_ORDER:
        if status in JobStatus.terminal():
            continue
        session = FakeSession(_job(status))
        job = advance_status(session, JOB_ID, JobStatus.FAILED)  # type: ignore[arg-type]
        assert job.status == JobStatus.FAILED.value


def test_missing_job_is_an_explicit_error() -> None:
    """Under RLS another tenant's job is invisible, so this is also the
    "dispatched with the wrong tenant_id" case — worth saying out loud."""
    session = FakeSession(None)

    with pytest.raises(JobStateError, match="not found"):
        advance_status(session, JOB_ID, JobStatus.BLOCKING)  # type: ignore[arg-type]


def test_a_late_failure_cannot_turn_a_completed_job_into_a_failed_one() -> None:
    """Regression: a stray re-dispatch of a finished job used to mark it FAILED."""
    job = _job(JobStatus.COMPLETED)
    session = FakeSession(job)

    mark_failed(session, JOB_ID, "stray task")  # type: ignore[arg-type]

    assert job.status == JobStatus.COMPLETED.value
    assert "error" not in job.stage_progress


def test_a_stage_arriving_after_completion_is_stale() -> None:
    session = FakeSession(_job(JobStatus.COMPLETED))

    with pytest.raises(StaleStageError):
        advance_status(session, JOB_ID, JobStatus.PREPROCESSING)  # type: ignore[arg-type]


def test_a_deleted_jobs_tasks_skip_rather_than_fail() -> None:
    """Missing is a stale-stage case: stages catch StaleStageError and return."""
    session = FakeSession(None)

    with pytest.raises(StaleStageError):
        advance_status(session, JOB_ID, JobStatus.EMBEDDING)  # type: ignore[arg-type]


def test_mark_failed_records_the_reason_and_completes_the_job() -> None:
    """TRD 2.7: a failed job must be diagnosable from Postgres alone."""
    session = FakeSession(_job(JobStatus.EMBEDDING))

    job = mark_failed(session, JOB_ID, "CUDA out of memory")  # type: ignore[arg-type]

    assert job.status == JobStatus.FAILED.value
    assert "CUDA out of memory" in job.stage_progress["error"]["message"]
    assert isinstance(job.completed_at, datetime)
    assert job.completed_at.tzinfo is not None


def test_mark_failed_is_idempotent() -> None:
    already = _job(JobStatus.FAILED)
    already.stage_progress = {"error": {"message": "first", "at": "t0"}}
    session = FakeSession(already)

    job = mark_failed(session, JOB_ID, "second")  # type: ignore[arg-type]

    assert job.stage_progress["error"]["message"] == "first", "first cause overwritten"


def test_progress_merges_rather_than_replacing() -> None:
    """stage_progress accumulates across the run; a stage that assigned the
    whole column would erase the stages before it."""
    job = _job(JobStatus.EMBEDDING)
    job.stage_progress = {"ingest": {"done": 100, "total": 100}}
    session = FakeSession(job)

    updated = record_progress(session, JOB_ID, "embedding", done=50, total=100)  # type: ignore[arg-type]

    assert updated.stage_progress["ingest"] == {"done": 100, "total": 100}
    embedding = updated.stage_progress["embedding"]
    assert (embedding["done"], embedding["total"]) == (50, 100)


def test_progress_keeps_the_stage_start_time_across_reports() -> None:
    """ETA needs a rate, and a rate needs when the stage *started* — server-side."""
    job = _job(JobStatus.EMBEDDING)
    session = FakeSession(job)

    first = record_progress(session, JOB_ID, "embedding", done=10, total=100)  # type: ignore[arg-type]
    started = first.stage_progress["embedding"]["started_at"]
    second = record_progress(session, JOB_ID, "embedding", done=40, total=100)  # type: ignore[arg-type]

    assert second.stage_progress["embedding"]["started_at"] == started
    assert "updated_at" in second.stage_progress["embedding"]


def test_progress_restarts_the_clock_when_a_stage_re_runs() -> None:
    """A count going backwards is a redelivered stage starting over."""
    job = _job(JobStatus.EMBEDDING)
    job.stage_progress = {"embedding": {"done": 90, "started_at": "2000-01-01T00:00:00+00:00"}}
    session = FakeSession(job)

    updated = record_progress(session, JOB_ID, "embedding", done=5, total=100)  # type: ignore[arg-type]

    assert updated.stage_progress["embedding"]["started_at"] != "2000-01-01T00:00:00+00:00"


def test_progress_details_survive_a_later_counter_update() -> None:
    job = _job(JobStatus.PREPROCESSING)
    session = FakeSession(job)

    record_progress(session, JOB_ID, "classify", done=1, details={"k": 1})  # type: ignore[arg-type]
    updated = record_progress(session, JOB_ID, "classify", done=2)  # type: ignore[arg-type]

    assert updated.stage_progress["classify"]["details"] == {"k": 1}


def test_progress_assigns_a_new_dict() -> None:
    """SQLAlchemy does not track in-place mutation of JSONB.

    A mutated dict is a silently discarded write — the counter appears to
    advance in Python and never lands in the database.
    """
    job = _job()
    original = job.stage_progress
    session = FakeSession(job)

    record_progress(session, JOB_ID, "ingest", done=1)  # type: ignore[arg-type]

    assert job.stage_progress is not original


# --- Ingest helpers ------------------------------------------------------------


def test_csv_is_streamed_in_chunks(tmp_path: Path) -> None:
    csv_file = tmp_path / "schools.csv"
    rows = "\n".join(f"School {i},Road {i},560001" for i in range(250))
    csv_file.write_text(f"name,address,pincode\n{rows}\n", encoding="utf-8")

    from app.workers.tasks.ingest import iter_csv_chunks

    chunks = list(iter_csv_chunks(csv_file, chunk_size=100))

    assert [len(c) for c in chunks] == [100, 100, 50]
    assert chunks[0][0] == {"name": "School 0", "address": "Road 0", "pincode": "560001"}


def test_csv_without_a_header_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")

    from app.workers.tasks.ingest import iter_csv_chunks

    with pytest.raises(ValueError, match="no header"):
        list(iter_csv_chunks(empty, chunk_size=10))


def test_ragged_rows_do_not_produce_null_keys(tmp_path: Path) -> None:
    """Extra fields land under a None key, which JSONB cannot represent."""
    csv_file = tmp_path / "ragged.csv"
    csv_file.write_text("name,pincode\nA,560001,EXTRA\n", encoding="utf-8")

    from app.workers.tasks.ingest import iter_csv_chunks

    chunk = next(iter(iter_csv_chunks(csv_file, chunk_size=10)))

    assert None not in chunk[0]


# --- Matching helpers ----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_row", "expected"),
    [
        ({"pincode": "560001"}, "560001"),
        ({"Pincode": " 560001 "}, "560001"),
        ({"PIN": "110085"}, "110085"),
        ({"postal_code": "635109"}, "635109"),
        ({"pinCode": "125120"}, "125120"),
        ({"name": "no pin here"}, ""),
    ],
)
def test_pincode_is_found_whatever_the_header(raw_row: dict, expected: str) -> None:
    """The matcher reads pincodes through the one alias table."""
    from app.ml_pipeline.fields import canonical_fields

    assert canonical_fields(raw_row)["pincode"] == expected


def test_pair_scores_are_banded_for_the_results_page() -> None:
    import pandas as pd

    from app.workers.tasks.matching import _band_counts

    bands = _band_counts(pd.Series([0.1, 0.49, 0.5, 0.8, 0.95, 1.0, None]))

    assert bands == {"distinct": 3, "possible": 1, "likely": 1, "certain": 2}
