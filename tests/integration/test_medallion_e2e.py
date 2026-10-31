"""Live-database tests for migration 0006: queue semantics, the Silver layer,
the Fellegi-Sunter feedback loop, stable entity ids, and job finalisation.

Everything that can only be wrong *in Postgres* — SKIP LOCKED partitioning,
lease fencing, the dedupe index, the stats trigger, RLS on the new tables —
is exercised against the real server, scoped to disposable tenants.
"""

from __future__ import annotations

import asyncio
import selectors
import uuid
from typing import Any

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

# Same directory, no package: pytest's rootdir-relative import puts this
# folder on sys.path. `stub_encoder` is a fixture; importing it registers it.
from test_pipeline_e2e import _make_job, _run_pipeline, _write_csv, stub_encoder  # noqa: F401

from app.core.security import Principal, Role
from app.db.session import platform_session, tenant_async_session, tenant_sync_session
from app.models.enums import ClusterStatus, FeedbackDecision, JobStatus
from app.queue import core
from app.queue.worker import Worker
from app.schemas.feedback import FeedbackRequest

pytestmark = pytest.mark.integration


def _run_async(coro: Any) -> Any:
    """psycopg's async mode cannot use Windows' default Proactor loop."""
    return asyncio.run(
        coro, loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())
    )


def _scalar(engine: Engine, sql: str, **params: Any) -> Any:
    with Session(engine) as session:
        return session.execute(text(sql), params).scalar_one()


# --- Queue semantics -------------------------------------------------------------


def _enqueue_jobs_task(tenant: uuid.UUID, job_id: uuid.UUID, *, dedupe: str | None) -> Any:
    with tenant_sync_session(tenant) as session:
        return core.enqueue(
            session,
            tenant_id=tenant,
            task_name="jobs.finalize_reviewed",
            payload={"job_id": str(job_id)},
            job_id=job_id,
            dedupe_key=dedupe,
        )


@pytest.mark.usefixtures("assert_no_leaked_rows")
def test_live_dedupe_key_admits_one_task(e2e_tenant: uuid.UUID) -> None:
    with tenant_sync_session(e2e_tenant) as session:
        job_id = _make_job(session, e2e_tenant)

    first = _enqueue_jobs_task(e2e_tenant, job_id, dedupe="k")
    second = _enqueue_jobs_task(e2e_tenant, job_id, dedupe="k")

    assert first is not None
    assert second is None, "a live dedupe key admitted a duplicate"


@pytest.mark.usefixtures("assert_no_leaked_rows")
def test_concurrent_claims_never_share_a_row(e2e_tenant: uuid.UUID) -> None:
    """SKIP LOCKED: two open claim transactions partition the ready set."""
    with tenant_sync_session(e2e_tenant) as session:
        job_id = _make_job(session, e2e_tenant)
    for i in range(6):
        _enqueue_jobs_task(e2e_tenant, job_id, dedupe=f"c{i}")

    with platform_session() as first, platform_session() as second:
        a = core.claim(first, worker_id="w1", queue="jobs", limit=4, tenant_id=e2e_tenant)
        # `first` still holds its row locks: `second` must skip them.
        b = core.claim(second, worker_id="w2", queue="jobs", limit=4, tenant_id=e2e_tenant)

    assert len(a) == 4 and len(b) == 2
    assert not {t.id for t in a} & {t.id for t in b}


@pytest.mark.usefixtures("assert_no_leaked_rows")
def test_a_reaped_lease_fences_out_the_old_holder(e2e_tenant: uuid.UUID) -> None:
    with tenant_sync_session(e2e_tenant) as session:
        job_id = _make_job(session, e2e_tenant)
    _enqueue_jobs_task(e2e_tenant, job_id, dedupe=None)

    with platform_session() as session:
        (task,) = core.claim(session, worker_id="dead", queue="jobs", limit=1, tenant_id=e2e_tenant)
        session.execute(
            text(
                "UPDATE task_queue SET lease_expires_at = now() - interval '1 second' "
                "WHERE id = :id"
            ),
            {"id": task.id},
        )
    with platform_session() as session:
        reaped = core.reap_expired(session)
    assert (task.id, "queued") in {(t.id, outcome) for t, outcome in reaped}

    with platform_session() as session:
        (again,) = core.claim(
            session, worker_id="alive", queue="jobs", limit=1, tenant_id=e2e_tenant
        )
    assert again.id == task.id and again.attempts == 2

    with platform_session() as session:
        assert core.complete(session, task=task, worker_id="dead") is False
        assert core.heartbeat(session, task_id=task.id, worker_id="dead", lease_seconds=60) is False
        assert core.complete(session, task=again, worker_id="alive") is True


@pytest.mark.usefixtures("assert_no_leaked_rows")
def test_a_live_worker_never_reaps_its_own_stalled_task(e2e_tenant: uuid.UUID) -> None:
    """After a stall that freezes the whole process (a laptop sleeping), every
    lease lapses and the reaper can run before the heartbeat does. Reaping its
    own task burned attempts and failed a 51k-record job; a worker that is
    running the reaper is alive, so only *other* workers' tasks are reaped."""
    with tenant_sync_session(e2e_tenant) as session:
        job_id = _make_job(session, e2e_tenant)
    _enqueue_jobs_task(e2e_tenant, job_id, dedupe=None)

    with platform_session() as session:
        (task,) = core.claim(session, worker_id="me", queue="jobs", limit=1, tenant_id=e2e_tenant)
        session.execute(
            text(
                "UPDATE task_queue SET lease_expires_at = now() - interval '1 hour' WHERE id = :id"
            ),
            {"id": task.id},
        )

    with platform_session() as session:
        assert core.reap_expired(session, exclude_worker="me") == []
    with platform_session() as session:
        # The heartbeat revives the lapsed lease: it does not check expiry.
        assert core.heartbeat(session, task_id=task.id, worker_id="me", lease_seconds=60)

    with platform_session() as session:
        session.execute(
            text(
                "UPDATE task_queue SET lease_expires_at = now() - interval '1 hour' WHERE id = :id"
            ),
            {"id": task.id},
        )
    with platform_session() as session:
        reaped = core.reap_expired(session, exclude_worker="someone-else")
    assert [(t.id, outcome) for t, outcome in reaped] == [(task.id, "queued")]


@pytest.mark.usefixtures("assert_no_leaked_rows")
def test_the_data_plane_cannot_claim_or_delete_queue_rows(
    e2e_tenant: uuid.UUID, app_engine: Engine
) -> None:
    """Migration 0006 revokes UPDATE/DELETE on task_queue from er_app."""
    with tenant_sync_session(e2e_tenant) as session:
        job_id = _make_job(session, e2e_tenant)
    _enqueue_jobs_task(e2e_tenant, job_id, dedupe=None)

    with Session(app_engine) as session:
        session.execute(
            text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(e2e_tenant)}
        )
        with pytest.raises(Exception, match="permission denied"):
            session.execute(text("UPDATE task_queue SET status = 'running'"))
        session.rollback()


def test_the_estimates_view_does_not_leak_across_tenants(
    two_e2e_tenants: tuple[uuid.UUID, uuid.UUID], app_engine: Engine
) -> None:
    """A view runs as its owner unless security_invoker is set, and the
    owner role would see every tenant's statistics."""
    tenant_a, tenant_b = two_e2e_tenants
    with platform_session() as session:
        session.execute(
            text(
                "INSERT INTO splink_settings (tenant_id, settings_hash, comparison_names, "
                "level_counts, settings) VALUES (:t, :h, ARRAY['name'], ARRAY[3]::smallint[], '{}')"
            ),
            {"t": tenant_a, "h": "a" * 64},
        )
        session.execute(
            text(
                "INSERT INTO splink_level_stats (tenant_id, settings_hash, comparison_idx, level, "
                "match_w) VALUES (:t, :h, 0, 2, 5)"
            ),
            {"t": tenant_a, "h": "a" * 64},
        )

    def visible(tenant: uuid.UUID) -> int:
        with Session(app_engine) as session:
            session.execute(
                text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(tenant)}
            )
            return int(
                session.execute(text("SELECT count(*) FROM splink_level_estimates")).scalar_one()
            )

    assert visible(tenant_a) == 1
    assert visible(tenant_b) == 0


# --- Pipeline: Silver, pair scores, entity ids -------------------------------------


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_the_pipeline_writes_every_medallion_layer(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    csv_path = tmp_path / "schools.csv"
    rows = _write_csv(csv_path)
    with tenant_sync_session(e2e_tenant) as session:
        job_id = _make_job(session, e2e_tenant)

    outcomes = _run_pipeline(job_id, e2e_tenant, csv_path)

    with Session(owner_engine) as session:
        tasks = dict(
            session.execute(
                text("SELECT task_name, status FROM task_queue WHERE job_id = :j"), {"j": job_id}
            ).all()
        )
        silver = session.execute(
            text(
                "SELECT count(*), count(DISTINCT ontology_version), min(ontology) "
                "FROM silver_records WHERE job_id = :j"
            ),
            {"j": job_id},
        ).one()
        gammas = session.execute(
            text(
                "SELECT ps.gamma, cfg.comparison_names FROM pair_scores ps "
                "JOIN splink_settings cfg USING (tenant_id, settings_hash) WHERE ps.job_id = :j"
            ),
            {"j": job_id},
        ).all()
        densities = (
            session.execute(text("SELECT density FROM clusters WHERE job_id = :j"), {"j": job_id})
            .scalars()
            .all()
        )

    assert set(tasks.values()) == {"succeeded"}, tasks
    assert {
        "ingest.ingest_and_classify",
        "silver.standardize",
        "enrichment.enrich_job",
        "embeddings.embeddings_stage",
        "blocking.blocking_stage",
        "matching.probabilistic_match",
        "resolution.resolve_by_confidence",
    } <= set(tasks), tasks
    assert outcomes == {"succeeded": len(tasks)}

    assert silver == (rows, 1, "education")
    assert gammas, "matching wrote no pair_scores"
    for gamma, names in gammas:
        assert len(gamma) == len(names) == 3
    assert densities and all(d is not None and 0 <= d <= 1 for d in densities)


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_rematching_keeps_entity_ids_stable(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    from app.workers.tasks.matching import probabilistic_match

    csv_path = tmp_path / "schools.csv"
    _write_csv(csv_path)
    with tenant_sync_session(e2e_tenant) as session:
        job_id = _make_job(session, e2e_tenant)
    _run_pipeline(job_id, e2e_tenant, csv_path)

    def memberships() -> dict[frozenset[str], str]:
        with Session(owner_engine) as session:
            rows = session.execute(
                text(
                    "SELECT c.entity_id::text, array_agg(m.record_id::text) FROM clusters c "
                    "JOIN cluster_members m ON m.cluster_id = c.id "
                    "WHERE c.job_id = :j AND c.status NOT IN ('RESOLVED', 'REJECTED') "
                    "GROUP BY c.entity_id"
                ),
                {"j": job_id},
            ).all()
        return {frozenset(members): entity for entity, members in rows}

    before = memberships()
    with tenant_sync_session(e2e_tenant) as session:
        session.execute(text("UPDATE jobs SET status = 'BLOCKING' WHERE id = :j"), {"j": job_id})
    probabilistic_match(str(job_id), str(e2e_tenant))
    after = memberships()

    assert before and after.keys() == before.keys()
    assert after == before, "a rebuild with identical members re-issued entity ids"


# --- Feedback loop -------------------------------------------------------------------


def _pending_cluster(
    owner_engine: Engine, job_id: uuid.UUID, min_size: int
) -> tuple[uuid.UUID, list[uuid.UUID]]:
    with Session(owner_engine) as session:
        row = session.execute(
            text(
                "SELECT c.id, array_agg(m.record_id ORDER BY m.record_id) FROM clusters c "
                "JOIN cluster_members m ON m.cluster_id = c.id "
                "WHERE c.job_id = :j AND c.status = 'PENDING_REVIEW' "
                "GROUP BY c.id HAVING count(*) >= :n ORDER BY c.id LIMIT 1"
            ),
            {"j": job_id, "n": min_size},
        ).first()
    if row is None:
        pytest.skip("pipeline produced no reviewable cluster of that size")
    return row[0], list(row[1])


def _submit(tenant: uuid.UUID, cluster_id: uuid.UUID, payload: FeedbackRequest) -> Any:
    from app.api.v1.routes.clusters import submit_feedback

    principal = Principal(subject="steward@test", tenant_id=tenant, role=Role.DATA_STEWARD)

    async def go() -> Any:
        async with tenant_async_session(tenant) as db:
            return await submit_feedback(
                db=db, principal=principal, cluster_id=cluster_id, payload=payload
            )

    return _run_async(go())


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_split_without_a_master_resolves_instead_of_500ing(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    """Regression: SPLIT with no master_record_id violated
    `resolved_requires_master_record` and surfaced as a 500."""
    csv_path = tmp_path / "schools.csv"
    _write_csv(csv_path)
    with tenant_sync_session(e2e_tenant) as session:
        job_id = _make_job(session, e2e_tenant)
        session.execute(
            text("UPDATE tenants SET confidence_auto_merge = 0.999 WHERE id = :t"),
            {"t": e2e_tenant},
        )
    _run_pipeline(job_id, e2e_tenant, csv_path)
    cluster_id, members = _pending_cluster(owner_engine, job_id, min_size=2)

    response = _submit(
        e2e_tenant,
        cluster_id,
        FeedbackRequest(decision=FeedbackDecision.SPLIT, excluded_record_ids=[members[-1]]),
    )

    assert response.cluster_status == ClusterStatus.RESOLVED.value
    with Session(owner_engine) as session:
        master, remaining = session.execute(
            text(
                "SELECT c.master_record_id, array_agg(m.record_id) FROM clusters c "
                "JOIN cluster_members m ON m.cluster_id = c.id WHERE c.id = :c "
                "GROUP BY c.master_record_id"
            ),
            {"c": cluster_id},
        ).one()
        event = session.execute(
            text("SELECT decision, partition FROM feedback_events WHERE cluster_id = :c"),
            {"c": cluster_id},
        ).one()
        cannot = session.execute(
            text(
                "SELECT count(*) FROM entity_constraints WHERE kind = 'cannot_link' "
                "AND event_id IN (SELECT id FROM feedback_events WHERE cluster_id = :c)"
            ),
            {"c": cluster_id},
        ).scalar_one()
        identity = dict(
            session.execute(
                text("SELECT record_id, stable_id FROM silver_records WHERE record_id = ANY(:r)"),
                {"r": members},
            ).all()
        )

    assert master is not None and master in remaining and members[-1] not in remaining
    assert event.decision == "SPLIT" and len(event.partition) == 2
    # Never-merges bind identities, so only kept/excluded pairs with *distinct*
    # identities can be recorded; a pair sharing one is indistinguishable in
    # every future upload and is (correctly) left unbound.
    distinct = {
        tuple(sorted((identity[k], identity[members[-1]])))
        for k in members[:-1]
        if identity[k] != identity[members[-1]]
    }
    assert cannot == len(distinct)


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_labels_update_the_sufficient_statistics_incrementally(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    """A MERGE on a scored pair adds exactly its weight to one level of each
    comparison; nothing is retrained, nothing is rescanned."""
    csv_path = tmp_path / "schools.csv"
    _write_csv(csv_path)
    with tenant_sync_session(e2e_tenant) as session:
        job_id = _make_job(session, e2e_tenant)
        session.execute(
            text("UPDATE tenants SET confidence_auto_merge = 0.999 WHERE id = :t"),
            {"t": e2e_tenant},
        )
    _run_pipeline(job_id, e2e_tenant, csv_path)
    cluster_id, members = _pending_cluster(owner_engine, job_id, min_size=2)

    def total_match_weight() -> float:
        return float(
            _scalar(
                owner_engine,
                "SELECT coalesce(sum(match_w), 0) FROM splink_level_stats WHERE tenant_id = :t",
                t=e2e_tenant,
            )
        )

    before = total_match_weight()
    _submit(
        e2e_tenant,
        cluster_id,
        FeedbackRequest(decision=FeedbackDecision.MERGE, master_record_id=members[0]),
    )

    with Session(owner_engine) as session:
        labels = session.execute(
            text(
                "SELECT label, gamma FROM pair_labels WHERE tenant_id = :t AND event_id IN "
                "(SELECT id FROM feedback_events WHERE cluster_id = :c)"
            ),
            {"t": e2e_tenant, "c": cluster_id},
        ).all()

    assert labels and all(label for label, _ in labels)
    expected = sum(sum(1 for g in gamma if g >= 0) for _, gamma in labels)
    assert total_match_weight() - before == pytest.approx(expected)


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_reviewing_the_last_cluster_completes_the_job(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    """Regression: nothing ever moved a job out of AWAITING_REVIEW."""
    csv_path = tmp_path / "schools.csv"
    _write_csv(csv_path)
    with tenant_sync_session(e2e_tenant) as session:
        job_id = _make_job(session, e2e_tenant)
        session.execute(
            text("UPDATE tenants SET confidence_auto_merge = 0.999 WHERE id = :t"),
            {"t": e2e_tenant},
        )
    _run_pipeline(job_id, e2e_tenant, csv_path)
    assert _scalar(owner_engine, "SELECT status FROM jobs WHERE id = :j", j=job_id) == (
        JobStatus.AWAITING_REVIEW.value
    )

    with Session(owner_engine) as session:
        pending = (
            session.execute(
                text("SELECT id FROM clusters WHERE job_id = :j AND status = 'PENDING_REVIEW'"),
                {"j": job_id},
            )
            .scalars()
            .all()
        )
    for cluster_id in pending:
        _submit(e2e_tenant, cluster_id, FeedbackRequest(decision=FeedbackDecision.REJECT))
        Worker(tenant_id=e2e_tenant).run_until_idle()

    assert _scalar(owner_engine, "SELECT status FROM jobs WHERE id = :j", j=job_id) == (
        JobStatus.COMPLETED.value
    )
