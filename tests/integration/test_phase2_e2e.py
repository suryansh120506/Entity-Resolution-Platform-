"""Live tests for Phase 2: the closed learning loop, constraint enforcement in
the matching stage, and presigned direct-to-bucket uploads.

The S3 side runs against moto's server mode: a real HTTP endpoint on
localhost, so the presigned PUT is an actual HTTP request with the URL the
API signed — the same path a browser takes — with no cloud account.
"""

from __future__ import annotations

import socket
import uuid
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

# Same directory, no package (see test_medallion_e2e.py).
from test_medallion_e2e import _run_async, _submit  # noqa: F401
from test_pipeline_e2e import _make_job, _run_pipeline, _write_csv, stub_encoder  # noqa: F401

from app.core.config import get_settings
from app.core.security import Principal, Role
from app.db.session import tenant_async_session, tenant_sync_session
from app.models.enums import FeedbackDecision, JobStatus
from app.queue.worker import Worker
from app.schemas.feedback import FeedbackRequest
from app.schemas.jobs import PresignRequest

pytestmark = pytest.mark.integration


def _model(owner_engine: Engine, job_id: uuid.UUID) -> dict[str, Any]:
    with Session(owner_engine) as session:
        return session.execute(
            text("SELECT stage_progress->'matching'->'details'->'model' FROM jobs WHERE id = :j"),
            {"j": job_id},
        ).scalar_one()


def _new_job(tenant: uuid.UUID) -> uuid.UUID:
    with tenant_sync_session(tenant) as session:
        return _make_job(session, tenant)


def _hold_for_review(tenant: uuid.UUID) -> None:
    """Nothing auto-merges, so every cluster is a reviewable one."""
    with tenant_sync_session(tenant) as session:
        session.execute(
            text("UPDATE tenants SET confidence_auto_merge = 0.999 WHERE id = :t"), {"t": tenant}
        )


# --- Learning loop -------------------------------------------------------------------


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_em_runs_once_per_tenant_then_the_stored_model_is_reused(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "schools.csv"
    _write_csv(csv_path)

    first = _new_job(e2e_tenant)
    _run_pipeline(first, e2e_tenant, csv_path)
    cold = _model(owner_engine, first)

    assert cold["source"] == "cold_start_em" and cold["em_run"] is True
    with Session(owner_engine) as session:
        active = session.execute(
            text(
                "SELECT version_tag, layout_hash, parameters ? 'comparisons' FROM model_versions "
                "WHERE tenant_id = :t AND model_type = 'splink' AND is_active"
            ),
            {"t": e2e_tenant},
        ).one()
    assert active[0] == cold["version_tag"] and len(active[1]) == 64 and active[2]

    # From here EM is forbidden: any call fails the job, and the test.
    from app.ml_pipeline.matching.matcher import RecordMatcher

    def no_em(self: Any) -> dict[str, Any]:
        raise AssertionError("EM ran for a tenant that already has a base model")

    monkeypatch.setattr(RecordMatcher, "train", no_em)

    second = _new_job(e2e_tenant)
    _run_pipeline(second, e2e_tenant, csv_path)
    warm = _model(owner_engine, second)

    assert warm["source"] == "stored" and warm["em_run"] is False
    assert warm["version_tag"] == cold["version_tag"]


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_steward_labels_flow_into_the_next_jobs_model(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    csv_path = tmp_path / "schools.csv"
    _write_csv(csv_path)
    _hold_for_review(e2e_tenant)

    first = _new_job(e2e_tenant)
    _run_pipeline(first, e2e_tenant, csv_path)
    with Session(owner_engine) as session:
        cluster_id, members = session.execute(
            text(
                "SELECT c.id, array_agg(m.record_id ORDER BY m.record_id) FROM clusters c "
                "JOIN cluster_members m ON m.cluster_id = c.id "
                "WHERE c.job_id = :j AND c.status = 'PENDING_REVIEW' GROUP BY c.id LIMIT 1"
            ),
            {"j": first},
        ).one()
    _submit(
        e2e_tenant,
        cluster_id,
        FeedbackRequest(decision=FeedbackDecision.MERGE, master_record_id=members[0]),
    )

    second = _new_job(e2e_tenant)
    _run_pipeline(second, e2e_tenant, csv_path)
    warm = _model(owner_engine, second)

    assert warm["source"] == "stored+feedback"
    assert warm["feedback_weight"] >= 1
    assert warm["em_run"] is False


# --- Constraints in the matching stage -----------------------------------------------


def write_variant_csv(path: Any, groups: int = 10) -> int:
    """Duplicates that match but carry *distinct identities*.

    Each group is one school four times: "School" twice (an exact duplicate,
    which gives EM exact-name pairs to learn the name comparison from) and
    the near-misses "Schol" and "Scool" — close enough for the matcher,
    different enough to fingerprint apart, so identity-level rules can
    separate them. Plus two unrelated-looking rows sharing one UDISE code (a
    natural-key identity), and junk so the cold-start classifier sees more
    than one class.
    """
    rows = ["name,address,district,state,pincode,udise_code"]
    for i in range(groups):
        for spelling in ("School", "School", "Schol", "Scool"):
            rows.append(
                f"Govt Higher Primary {spelling} Unit {i},Main Road Block {i},"
                f"Bangalore,Karnataka,56{i:04d},"
            )
    rows.append("Sunrise Academy,Lake View,Mysuru,Karnataka,570001,29281200612")
    rows.append("Old Mission Building,Hill Road,Mysuru,Karnataka,570002,29281200612")
    for i in range(6):
        rows.append(f"N/A,,,,{'0' * 6},")
        rows.append(f"TEST {i},,,,,")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows) - 1


def identities_of(owner_engine: Engine, records: list[Any]) -> dict[str, str]:
    with Session(owner_engine) as session:
        return {
            str(record): stable_id
            for record, stable_id in session.execute(
                text("SELECT record_id, stable_id FROM silver_records WHERE record_id = ANY(:r)"),
                {"r": [uuid.UUID(str(r)) for r in records]},
            ).all()
        }


def _constrain(
    tenant: uuid.UUID,
    job_id: uuid.UUID,
    pairs: list[tuple[Any, Any, str]],
    owner_engine: Engine,
) -> None:
    """Steward rules between the *identities* of the given records."""
    identity = identities_of(owner_engine, [r for a, b, _ in pairs for r in (a, b)])
    with tenant_sync_session(tenant) as session:
        event = session.execute(
            text(
                "INSERT INTO feedback_events (tenant_id, job_id, actor_type, actor_id, decision, "
                "partition) VALUES (:t, :j, 'steward', 'test', 'SPLIT', '[[]]') RETURNING id"
            ),
            {"t": tenant, "j": job_id},
        ).scalar_one()
        for a, b, kind in pairs:
            left, right = sorted((identity[str(a)], identity[str(b)]))
            assert left != right, "test records must carry distinct identities"
            session.execute(
                text(
                    "INSERT INTO entity_constraints "
                    "(tenant_id, identity_l, identity_r, kind, event_id) "
                    "VALUES (:t, :l, :r, :k, :e)"
                ),
                {"t": tenant, "l": left, "r": right, "k": kind, "e": event},
            )


def _rematch(tenant: uuid.UUID, job_id: uuid.UUID) -> None:
    from app.workers.tasks.matching import probabilistic_match

    with tenant_sync_session(tenant) as session:
        session.execute(text("UPDATE jobs SET status = 'BLOCKING' WHERE id = :j"), {"j": job_id})
    probabilistic_match(str(job_id), str(tenant))


def _cluster_of(owner_engine: Engine, job_id: uuid.UUID) -> dict[str, str]:
    with Session(owner_engine) as session:
        rows = session.execute(
            text(
                "SELECT m.record_id::text, c.id::text FROM clusters c "
                "JOIN cluster_members m ON m.cluster_id = c.id WHERE c.job_id = :j"
            ),
            {"j": job_id},
        ).all()
    return dict(rows)


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_matching_cuts_cannot_links_and_forces_must_links(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    csv_path = tmp_path / "schools.csv"
    write_variant_csv(csv_path)
    _hold_for_review(e2e_tenant)
    job_id = _new_job(e2e_tenant)
    _run_pipeline(job_id, e2e_tenant, csv_path)

    groups = distinct_identity_clusters(owner_engine, job_id, 2)
    assert len(groups) >= 2, "the variant CSV should form several multi-identity clusters"
    (a, b), (c, _) = groups[0], groups[1]

    # Split a/b apart; join a to c, which the model never linked.
    _constrain(e2e_tenant, job_id, [(a, b, "cannot_link"), (a, c, "must_link")], owner_engine)
    _rematch(e2e_tenant, job_id)
    clusters = _cluster_of(owner_engine, job_id)

    assert clusters.get(a) != clusters.get(b), "a cannot-link pair was clustered together"
    assert clusters[a] == clusters[c], "a must-link pair was left apart"
    with Session(owner_engine) as session:
        details = session.execute(
            text("SELECT stage_progress->'matching'->'details' FROM jobs WHERE id = :j"),
            {"j": job_id},
        ).scalar_one()
    assert details["cut_links"] >= 1 and details["forced_links"] >= 1


def distinct_identity_clusters(owner_engine: Engine, job_id: uuid.UUID, n: int) -> list[list[str]]:
    """For each cluster with ≥ n distinct identities: n members, one per
    identity (lowest record id each), clusters in a stable order."""
    with Session(owner_engine) as session:
        rows = session.execute(
            text(
                "SELECT c.id::text, m.record_id::text, s.stable_id FROM clusters c "
                "JOIN cluster_members m ON m.cluster_id = c.id "
                "JOIN silver_records s ON s.record_id = m.record_id "
                "WHERE c.job_id = :j ORDER BY c.id, s.stable_id, m.record_id"
            ),
            {"j": job_id},
        ).all()
    picked: dict[str, dict[str, str]] = {}
    for cluster_id, record_id, stable_id in rows:
        picked.setdefault(cluster_id, {}).setdefault(stable_id, record_id)
    return sorted(
        (
            list(by_identity.values())[:n]
            for by_identity in picked.values()
            if len(by_identity) >= n
        ),
        key=lambda members: members[0],
    )


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_contradictory_constraints_keep_the_records_apart(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    csv_path = tmp_path / "schools.csv"
    write_variant_csv(csv_path)
    job_id = _new_job(e2e_tenant)
    _run_pipeline(job_id, e2e_tenant, csv_path)

    with Session(owner_engine) as session:
        a, b, c = (
            session.execute(
                text(
                    "SELECT DISTINCT ON (s.stable_id) s.record_id::text FROM silver_records s "
                    "JOIN school_records r ON r.id = s.record_id WHERE s.job_id = :j "
                    "AND r.classification = 'Valid' ORDER BY s.stable_id LIMIT 3"
                ),
                {"j": job_id},
            )
            .scalars()
            .all()
        )
    _constrain(
        e2e_tenant,
        job_id,
        [(a, b, "must_link"), (b, c, "must_link"), (a, c, "cannot_link")],
        owner_engine,
    )
    _rematch(e2e_tenant, job_id)
    from app.workers.tasks.resolution import resolve_by_confidence

    resolve_by_confidence(str(job_id), str(e2e_tenant))

    with Session(owner_engine) as session:
        status, reason = session.execute(
            text(
                "SELECT c.status, c.review_reason FROM clusters c "
                "JOIN cluster_members m ON m.cluster_id = c.id WHERE m.record_id = :a"
            ),
            {"a": a},
        ).one()
    # Contradictory steward rules cannot be settled automatically: the
    # records are kept apart (migration 0010), not queued for a human.
    assert (status, reason) == ("REJECTED", "constraint_conflict")


# --- Presigned direct uploads ------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def s3_bucket(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    from moto.server import ThreadedMotoServer

    from app import storage

    port = _free_port()
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=port, verbose=False)
    server.start()
    bucket = "er-test-uploads"
    for key, value in {
        "OBJECT_STORAGE_ENDPOINT_URL": f"http://127.0.0.1:{port}",
        "OBJECT_STORAGE_BUCKET": bucket,
        "OBJECT_STORAGE_ACCESS_KEY_ID": "test",
        "OBJECT_STORAGE_SECRET_ACCESS_KEY": "test",
        # moto refuses R2's "auto"; any real region exercises the same path.
        "OBJECT_STORAGE_REGION": "us-east-1",
    }.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    storage.reset_client()
    storage.s3_client().create_bucket(Bucket=bucket)
    try:
        yield bucket
    finally:
        server.stop()
        monkeypatch.undo()
        get_settings.cache_clear()
        storage.reset_client()


def _principal(tenant: uuid.UUID) -> Principal:
    return Principal(subject="engineer@test", tenant_id=tenant, role=Role.DATA_ENGINEER)


def _presign(tenant: uuid.UUID, size: int) -> Any:
    from app.api.v1.routes.jobs import create_direct_upload

    async def go() -> Any:
        async with tenant_async_session(tenant) as db:
            return await create_direct_upload(
                db=db,
                principal=_principal(tenant),
                payload=PresignRequest(
                    filename="schools.csv", size_bytes=size, ontology="education"
                ),
            )

    return _run_async(go())


def _confirm(tenant: uuid.UUID, job_id: Any) -> Any:
    from app.api.v1.routes.jobs import confirm_direct_upload

    async def go() -> Any:
        async with tenant_async_session(tenant) as db:
            return await confirm_direct_upload(db=db, principal=_principal(tenant), job_id=job_id)

    return _run_async(go())


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_a_presigned_upload_goes_browser_to_bucket_to_worker(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any, s3_bucket: str
) -> None:
    csv_path = tmp_path / "schools.csv"
    rows = _write_csv(csv_path)
    body = csv_path.read_bytes()

    presigned = _presign(e2e_tenant, len(body))
    assert f"/{s3_bucket}/uploads/{e2e_tenant}/{presigned.job_id}.csv" in presigned.upload_url

    # Exactly what the browser does: PUT the bytes to the signed URL with the
    # signed headers, and no credentials of its own.
    response = httpx.put(presigned.upload_url, content=body, headers=presigned.headers)
    assert response.status_code == 200, response.text

    accepted = _confirm(e2e_tenant, presigned.job_id)
    assert accepted.total_bytes == len(body)

    outcomes = Worker(tenant_id=e2e_tenant).run_until_idle()
    assert outcomes.get("failed", 0) == 0, outcomes

    with Session(owner_engine) as session:
        status, total, source = session.execute(
            text("SELECT status, total_rows, source_uri FROM jobs WHERE id = :j"),
            {"j": presigned.job_id},
        ).one()
    assert source == f"s3://{s3_bucket}/uploads/{e2e_tenant}/{presigned.job_id}.csv"
    assert total == rows
    assert status in {JobStatus.AWAITING_REVIEW.value, JobStatus.COMPLETED.value}


@pytest.mark.usefixtures("assert_no_leaked_rows")
def test_a_size_mismatch_is_refused_and_the_object_discarded(
    e2e_tenant: uuid.UUID, s3_bucket: str
) -> None:
    from fastapi import HTTPException

    from app import storage

    presigned = _presign(e2e_tenant, 10)
    # moto does not verify signatures, so it accepts a body the real bucket
    # would reject; the confirm step must catch it independently.
    httpx.put(presigned.upload_url, content=b"x" * 999, headers=presigned.headers)

    with pytest.raises(HTTPException) as refused:
        _confirm(e2e_tenant, presigned.job_id)
    assert refused.value.status_code == 422

    with pytest.raises(storage.UploadMissingError):
        storage.object_size(f"s3://{s3_bucket}/uploads/{e2e_tenant}/{presigned.job_id}.csv")


@pytest.mark.usefixtures("assert_no_leaked_rows")
def test_confirming_before_the_put_is_a_conflict_not_a_job(
    e2e_tenant: uuid.UUID, s3_bucket: str, owner_engine: Engine
) -> None:
    from fastapi import HTTPException

    presigned = _presign(e2e_tenant, 10)
    with pytest.raises(HTTPException) as early:
        _confirm(e2e_tenant, presigned.job_id)

    assert early.value.status_code == 409
    with Session(owner_engine) as session:
        queued = session.execute(
            text("SELECT count(*) FROM task_queue WHERE job_id = :j"), {"j": presigned.job_id}
        ).scalar_one()
    assert queued == 0


def test_without_a_bucket_direct_upload_is_refused(
    e2e_tenant: uuid.UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    monkeypatch.setenv("OBJECT_STORAGE_BUCKET", "")
    get_settings.cache_clear()
    try:
        assert not get_settings().object_storage_enabled
        with pytest.raises(HTTPException) as refused:
            _presign(e2e_tenant, 10)
        assert refused.value.status_code == 409
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
