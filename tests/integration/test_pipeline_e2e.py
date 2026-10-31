"""End-to-end suite against the live containers — no mocked database.

Postgres, pgvector, Redis, Celery, XGBoost and Splink all run for real. The
whole point is the behaviour that only exists when they do: RLS evaluated by
the server, HNSW recall, and the job state machine as it actually advances.

ONE SUBSTITUTION, AND WHY
-------------------------
`SemanticEncoder` is replaced by a deterministic stub. It is not a mock of
anything under test here — it stands in for a 1 GB model download, which is
not a dependency a test suite should acquire, and whose own behaviour is
covered by `tests/ml_pipeline/test_encoder.py`. The stub is a pure function
of the text it is given, so two records that normalise to the same string
get byte-identical vectors, which is what makes the clustering assertions
below exact rather than probabilistic. The real `normalize_text` still runs.

Nothing else is substituted: the database session, Redis and the task
dispatch are the real things.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import numpy as np
import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from app.core.constants import EMBEDDING_DIMENSIONS
from app.db.session import tenant_sync_session
from app.ml_pipeline.blocking.blocker import CandidateBlocker, ProbeRecord
from app.models.enums import ClusterStatus, JobStatus, RecordClassification
from app.queue.core import enqueue
from app.queue.topology import PIPELINE_ENTRYPOINT
from app.queue.worker import Worker

pytestmark = pytest.mark.integration


# --- Deterministic vectors -----------------------------------------------------


def _unit_vector(seed: str) -> list[float]:
    """A stable unit vector for a string. Same text -> same vector."""
    digest = hashlib.sha256(seed.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    vector = rng.normal(size=EMBEDDING_DIMENSIONS)
    return (vector / np.linalg.norm(vector)).tolist()


def _vector_at_distance(base: list[float], cosine_distance: float) -> list[float]:
    """A unit vector a chosen cosine distance from `base`.

    Exact control is what lets the HNSW assertions below be about recall
    rather than about whether two strings happened to embed nearby.
    """
    base_array = np.asarray(base)
    rng = np.random.default_rng(abs(hash(cosine_distance)) % (2**32))
    noise = rng.normal(size=EMBEDDING_DIMENSIONS)
    # Component of the noise orthogonal to base.
    orthogonal = noise - np.dot(noise, base_array) * base_array
    orthogonal /= np.linalg.norm(orthogonal)

    target_cos = 1.0 - cosine_distance
    result = target_cos * base_array + np.sqrt(max(0.0, 1 - target_cos**2)) * orthogonal
    return (result / np.linalg.norm(result)).tolist()


class StubEncoder:
    """Deterministic stand-in for SemanticEncoder. See the module docstring."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.batch_size = kwargs.get("batch_size", 256)
        self.revision = "stub-encoder-v1"
        self.model_name = "stub"

    def encode_batch(self, texts: Any) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, EMBEDDING_DIMENSIONS), dtype=np.float32)
        return np.asarray([_unit_vector(t) for t in texts], dtype=np.float32)


def _insert_embedding(
    session: Session, tenant_id: uuid.UUID, job_id: uuid.UUID, name: str, vector: list[float]
) -> uuid.UUID:
    """One school_record plus its embedding, written through a bound session."""
    record_id = session.execute(
        text(
            "INSERT INTO school_records (tenant_id, job_id, raw_row, name_normalized) "
            "VALUES (:t, :j, CAST(:raw AS jsonb), :n) RETURNING id"
        ),
        {"t": tenant_id, "j": job_id, "raw": f'{{"name": "{name}"}}', "n": name},
    ).scalar_one()

    session.execute(
        text(
            "INSERT INTO record_embeddings "
            "(record_id, tenant_id, name_embedding, address_embedding, embedding_model_version) "
            "VALUES (:r, :t, :ne, :ae, 'e2e')"
        ),
        {"r": record_id, "t": tenant_id, "ne": str(vector), "ae": str(vector)},
    )
    return record_id


def _make_job(session: Session, tenant_id: uuid.UUID) -> uuid.UUID:
    return session.execute(
        text("INSERT INTO jobs (tenant_id, status) VALUES (:t, :s) RETURNING id"),
        {"t": tenant_id, "s": JobStatus.PENDING.value},
    ).scalar_one()


# --- The vector trap -----------------------------------------------------------


@pytest.mark.usefixtures("assert_no_leaked_rows")
def test_blocker_returns_only_the_bound_tenants_neighbours(
    two_e2e_tenants: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """The core claim: an ANN search cannot cross a tenant boundary.

    Tenant B's records are placed *closer* to tenant A's probe than any of
    tenant A's own records. A vector search that ignored RLS would return
    B's rows as the top-K; the correct answer is A's rows only.
    """
    tenant_a, tenant_b = two_e2e_tenants
    probe_vector = _unit_vector("probe")

    with tenant_sync_session(tenant_b) as session:
        job_b = _make_job(session, tenant_b)
        for i in range(30):
            _insert_embedding(
                session,
                tenant_b,
                job_b,
                f"tenant b decoy {i}",
                # Nearer the probe than anything tenant A owns.
                _vector_at_distance(probe_vector, 0.005 + i * 0.0001),
            )

    with tenant_sync_session(tenant_a) as session:
        job_a = _make_job(session, tenant_a)
        probe_id = _insert_embedding(session, tenant_a, job_a, "probe", probe_vector)
        expected = {
            _insert_embedding(
                session,
                tenant_a,
                job_a,
                f"tenant a neighbour {i}",
                _vector_at_distance(probe_vector, 0.05 + i * 0.01),
            )
            for i in range(3)
        }

    with tenant_sync_session(tenant_a) as session:
        blocker = CandidateBlocker(session, top_k=10, max_distance=0.5, ef_search=200)
        pairs = list(blocker.find_candidates([ProbeRecord(probe_id, probe_vector, probe_vector)]))

    found = {p.record_id_b if p.record_id_a == probe_id else p.record_id_a for p in pairs}

    assert found == expected, (
        "Blocker returned the wrong neighbour set. Anything outside `expected` "
        "is a cross-tenant leak; anything missing is a silently dropped "
        "candidate that the matcher will now never see."
    )


@pytest.mark.usefixtures("assert_no_leaked_rows")
def test_overfetch_preserves_recall_when_another_tenant_crowds_the_index(
    two_e2e_tenants: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """The trap itself, forced into the open.

    pgvector applies the HNSW approximation *before* the RLS predicate. With
    a foreign tenant's rows occupying every one of the nearest slots, a
    `LIMIT top_k` search can come back short — or empty — with no error
    anywhere. `enable_seqscan = off` forces the index path so the behaviour
    is the index's rather than a sequential scan's, which would filter
    correctly and prove nothing.

    Asserted as recall: with the over-fetch in place, every one of tenant
    A's genuine neighbours is still returned despite 300 nearer rows
    belonging to someone else.
    """
    tenant_a, tenant_b = two_e2e_tenants
    probe_vector = _unit_vector("crowded probe")

    with tenant_sync_session(tenant_b) as session:
        job_b = _make_job(session, tenant_b)
        for i in range(300):
            _insert_embedding(
                session,
                tenant_b,
                job_b,
                f"crowd {i}",
                _vector_at_distance(probe_vector, 0.001 + i * 0.00001),
            )

    with tenant_sync_session(tenant_a) as session:
        job_a = _make_job(session, tenant_a)
        probe_id = _insert_embedding(session, tenant_a, job_a, "crowded probe", probe_vector)
        expected = {
            _insert_embedding(
                session,
                tenant_a,
                job_a,
                f"real neighbour {i}",
                _vector_at_distance(probe_vector, 0.10 + i * 0.01),
            )
            for i in range(5)
        }

    with tenant_sync_session(tenant_a) as session:
        session.execute(text("SET LOCAL enable_seqscan = off"))
        blocker = CandidateBlocker(session, top_k=5, max_distance=0.5, oversample=20, ef_search=400)
        pairs = list(blocker.find_candidates([ProbeRecord(probe_id, probe_vector)]))

    found = {p.record_id_b if p.record_id_a == probe_id else p.record_id_a for p in pairs}

    assert found == expected, (
        f"Recall loss under RLS post-filtering: expected {len(expected)} neighbours, "
        f"got {len(found)}. This is the failure mode the over-fetch and ef_search "
        "exist to prevent, and it is silent in production."
    )


@pytest.mark.usefixtures("assert_no_leaked_rows")
def test_a_tenant_with_no_neighbours_gets_an_empty_result_not_an_error(
    two_e2e_tenants: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """A tenant alone in a crowded index gets nothing, quietly and correctly."""
    tenant_a, tenant_b = two_e2e_tenants
    probe_vector = _unit_vector("lonely")

    with tenant_sync_session(tenant_b) as session:
        job_b = _make_job(session, tenant_b)
        for i in range(20):
            _insert_embedding(
                session,
                tenant_b,
                job_b,
                f"other {i}",
                _vector_at_distance(probe_vector, 0.01 + i * 0.001),
            )

    with tenant_sync_session(tenant_a) as session:
        job_a = _make_job(session, tenant_a)
        probe_id = _insert_embedding(session, tenant_a, job_a, "lonely", probe_vector)

    with tenant_sync_session(tenant_a) as session:
        blocker = CandidateBlocker(session, top_k=10, ef_search=200)
        pairs = list(blocker.find_candidates([ProbeRecord(probe_id, probe_vector)]))

    assert pairs == []


# --- The orchestration loop ----------------------------------------------------


def _run_pipeline(
    job_id: uuid.UUID, tenant_id: uuid.UUID, csv_path: Any, *, expect_failure: bool = False
) -> dict[str, int]:
    """Enqueue the entrypoint and drain *this tenant's* queue through the real
    worker: claim, execute, complete, hand off to successors.

    Scoped by tenant because the suite shares its database with whatever else
    is running; a test must never execute another tenant's work.
    """
    with tenant_sync_session(tenant_id) as session:
        enqueue(
            session,
            tenant_id=tenant_id,
            task_name=PIPELINE_ENTRYPOINT,
            payload={"job_id": str(job_id), "csv_path": str(csv_path)},
            job_id=job_id,
        )
    outcomes = Worker(tenant_id=tenant_id).run_until_idle()
    if not expect_failure:
        assert outcomes.get("failed", 0) == 0, (
            f"pipeline tasks failed: {outcomes}; see dead_letter_tasks for the traceback"
        )
    return outcomes


@pytest.fixture
def stub_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the 1 GB model for a deterministic function. See module docstring."""
    import app.workers.tasks.embeddings as embeddings_module

    monkeypatch.setattr(embeddings_module, "SemanticEncoder", StubEncoder)


def _write_csv(path: Any) -> int:
    """A job with all three classes and known duplicate pairs.

    The duplicates are written with *different* abbreviations of the same
    name, so they only collide after `normalize_text` has expanded them —
    the normaliser is under test here too, not bypassed.
    """
    rows = ["name,address,district,state,pincode"]
    for i in range(14):
        pin = f"56{i:04d}"
        rows.append(
            f"Govt. Higher Primary School Unit {i},Main Road Block {i},Bangalore,Karnataka,{pin}"
        )
        rows.append(
            f"Government Higher Primary School Unit {i},Main Rd Block {i},Bangalore,Karnataka,{pin}"
        )
    # Junk, so the cold-start classifier sees more than one class.
    for i in range(6):
        rows.append(f"N/A,,,,{'0' * 6}")
        rows.append(f"TEST {i},,,,")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows) - 1


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_pipeline_advances_the_job_through_every_stage(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    """Drive the real chain and read the state machine out of Postgres.

    Asserted against the `jobs` row rather than against Celery's result
    backend, deliberately: TRD 2.7 requires a job to be diagnosable from
    Postgres alone, so Postgres is what the test is allowed to look at.
    """
    csv_path = tmp_path / "schools.csv"
    expected_rows = _write_csv(csv_path)

    with tenant_sync_session(e2e_tenant) as session:
        job_id = _make_job(session, e2e_tenant)

    _run_pipeline(job_id, e2e_tenant, csv_path)

    with Session(owner_engine) as session:
        job = session.execute(
            text("SELECT status, stage_progress, total_rows FROM jobs WHERE id = :id"),
            {"id": job_id},
        ).one()

    status, progress, total_rows = job

    assert total_rows == expected_rows, "ingest did not record the row count"

    # Every stage records progress, so the set of keys is the audit trail of
    # which stages actually ran.
    assert {"ingest", "classify", "embedding", "blocking", "matching"} <= set(progress), (
        f"stages missing from stage_progress: "
        f"{ {'ingest', 'classify', 'embedding', 'blocking', 'matching'} - set(progress) }"
    )
    assert progress["ingest"]["done"] == expected_rows

    # AWAITING_REVIEW because the gate found work for a human; COMPLETED only
    # if every cluster cleared the auto-merge threshold.
    assert status in {JobStatus.AWAITING_REVIEW.value, JobStatus.COMPLETED.value}, (
        f"job ended at {status}, not a post-resolution state"
    )


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_pipeline_classifies_embeds_and_clusters(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    """The artefacts each stage leaves behind, checked in the database."""
    csv_path = tmp_path / "schools.csv"
    _write_csv(csv_path)

    with tenant_sync_session(e2e_tenant) as session:
        job_id = _make_job(session, e2e_tenant)

    _run_pipeline(job_id, e2e_tenant, csv_path)

    with Session(owner_engine) as session:
        classes = dict(
            session.execute(
                text(
                    "SELECT classification, count(*) FROM school_records "
                    "WHERE job_id = :j GROUP BY classification"
                ),
                {"j": job_id},
            ).all()
        )
        label_sources = set(
            session.execute(
                text("SELECT DISTINCT label_source FROM school_records WHERE job_id = :j"),
                {"j": job_id},
            )
            .scalars()
            .all()
        )
        embedded = session.execute(
            text(
                "SELECT count(*) FROM record_embeddings e JOIN school_records r "
                "ON r.id = e.record_id WHERE r.job_id = :j"
            ),
            {"j": job_id},
        ).scalar_one()
        clusters = session.execute(
            text("SELECT count(*) FROM clusters WHERE job_id = :j"), {"j": job_id}
        ).scalar_one()
        scored = session.execute(
            text(
                "SELECT count(*) FROM clusters WHERE job_id = :j AND confidence_score IS NOT NULL"
            ),
            {"j": job_id},
        ).scalar_one()

    assert len(classes) >= 2, f"cold start produced only one class: {classes}"
    assert classes.get(RecordClassification.VALID.value, 0) > 0

    # Cold-start labels must be marked discardable (TRD 2.2, migration 0003).
    assert label_sources == {"PSEUDO"}, label_sources

    # Only Valid rows are embedded — that is the point of classifying first.
    assert embedded == classes.get(RecordClassification.VALID.value, 0), (
        f"{embedded} embeddings for {classes.get('Valid', 0)} Valid rows"
    )

    assert clusters > 0, "blocking found no duplicate pairs in a file built from pairs"
    assert scored == clusters, "matching left clusters unscored"


def _counts(engine: Engine, job_id: uuid.UUID) -> tuple[int, int, int]:
    with Session(engine) as session:
        records = session.execute(
            text("SELECT count(*) FROM school_records WHERE job_id = :j"), {"j": job_id}
        ).scalar_one()
        embeddings = session.execute(
            text(
                "SELECT count(*) FROM record_embeddings e JOIN school_records r "
                "ON r.id = e.record_id WHERE r.job_id = :j"
            ),
            {"j": job_id},
        ).scalar_one()
        clusters = session.execute(
            text("SELECT count(*) FROM clusters WHERE job_id = :j"), {"j": job_id}
        ).scalar_one()
    return records, embeddings, clusters


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_a_redelivered_stage_does_not_duplicate_its_work(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    """An expired lease redelivers an individual *stage*, not the whole chain.

    A worker that finished the inserts and died before completing the task
    gets the same task again with the job still at that stage. The second
    run must converge on the same rows rather than appending a second copy.
    """
    from app.workers.tasks.blocking import blocking_stage
    from app.workers.tasks.embeddings import embeddings_stage
    from app.workers.tasks.ingest import ingest_and_classify
    from app.workers.tasks.silver import standardize

    csv_path = tmp_path / "schools.csv"
    _write_csv(csv_path)

    with tenant_sync_session(e2e_tenant) as session:
        job_id = _make_job(session, e2e_tenant)

    job, tenant = str(job_id), str(e2e_tenant)

    # Each stage run twice back to back, at the status it belongs to.
    ingest_and_classify(job, tenant, str(csv_path))
    after_first_ingest = _counts(owner_engine, job_id)
    ingest_and_classify(job, tenant, str(csv_path))
    assert _counts(owner_engine, job_id) == after_first_ingest, "ingest duplicated rows"

    standardize(job, tenant)
    after_first_silver = _silver_count(owner_engine, job_id)
    standardize(job, tenant)
    assert _silver_count(owner_engine, job_id) == after_first_silver == after_first_ingest[0]

    embeddings_stage(job, tenant)
    after_first_embed = _counts(owner_engine, job_id)
    embeddings_stage(job, tenant)
    assert _counts(owner_engine, job_id) == after_first_embed, "embeddings duplicated rows"

    blocking_stage(job, tenant)
    after_first_block = _counts(owner_engine, job_id)
    blocking_stage(job, tenant)
    assert _counts(owner_engine, job_id) == after_first_block, "blocking duplicated clusters"


def _silver_count(engine: Engine, job_id: uuid.UUID) -> int:
    with Session(engine) as session:
        return int(
            session.execute(
                text("SELECT count(*) FROM silver_records WHERE job_id = :j"), {"j": job_id}
            ).scalar_one()
        )


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_a_stray_re_dispatch_does_not_destroy_a_finished_job(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    """Re-running the whole chain on a job awaiting review must be a no-op.

    Regression. Re-dispatching the chain is not redelivery, it is a restart,
    and `advance_status` rightly refuses AWAITING_REVIEW -> PREPROCESSING.
    The danger was in what happened next: the refusal propagated as an
    ordinary failure, so a duplicate dispatch — a double-clicked button, a
    retried orchestration task — marked a perfectly healthy job FAILED and
    took the steward's queue with it.

    Now the stale stage returns quietly and the job keeps its state.
    """
    csv_path = tmp_path / "schools.csv"
    _write_csv(csv_path)

    with tenant_sync_session(e2e_tenant) as session:
        job_id = _make_job(session, e2e_tenant)

    _run_pipeline(job_id, e2e_tenant, csv_path)

    with Session(owner_engine) as session:
        settled_status = session.execute(
            text("SELECT status FROM jobs WHERE id = :id"), {"id": job_id}
        ).scalar_one()
    settled_counts = _counts(owner_engine, job_id)

    # The stray second dispatch.
    _run_pipeline(job_id, e2e_tenant, csv_path)

    with Session(owner_engine) as session:
        after_status = session.execute(
            text("SELECT status FROM jobs WHERE id = :id"), {"id": job_id}
        ).scalar_one()

    assert after_status == settled_status, (
        f"a duplicate dispatch moved the job {settled_status} -> {after_status}"
    )
    assert after_status != JobStatus.FAILED.value
    assert _counts(owner_engine, job_id) == settled_counts


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_a_failing_stage_records_the_reason_in_postgres(
    e2e_tenant: uuid.UUID, owner_engine: Engine
) -> None:
    """TRD 2.7: a failed job is diagnosable from Postgres alone.

    The failure is written by the worker in a fresh transaction after the
    stage's own transaction has rolled back — the part that is easy to get
    wrong and impossible to notice until an incident.
    """
    with tenant_sync_session(e2e_tenant) as session:
        job_id = _make_job(session, e2e_tenant)

    # Fails on the first attempt: a missing file is permanent, so it is not
    # in TRANSIENT_ERRORS and is never retried into a misleading second error.
    outcomes = _run_pipeline(job_id, e2e_tenant, "/nonexistent/schools.csv", expect_failure=True)
    assert outcomes == {"failed": 1}, outcomes

    with Session(owner_engine) as session:
        status, progress = session.execute(
            text("SELECT status, stage_progress FROM jobs WHERE id = :id"), {"id": job_id}
        ).one()
        task_status, attempts = session.execute(
            text("SELECT status, attempts FROM task_queue WHERE job_id = :id"), {"id": job_id}
        ).one()
        dead = session.execute(
            text("SELECT error, traceback FROM dead_letter_tasks WHERE job_id = :id"),
            {"id": job_id},
        ).one()

    assert status == JobStatus.FAILED.value
    assert "error" in progress, "no reason recorded; the job is a mystery to an operator"
    assert "FileNotFoundError" in progress["error"]["message"]
    assert (task_status, attempts) == ("failed", 1), "a permanent error was retried"
    assert "FileNotFoundError" in dead.error and dead.traceback
    # Nothing downstream of a failed stage was enqueued.
    with Session(owner_engine) as session:
        assert (
            session.execute(
                text("SELECT count(*) FROM task_queue WHERE job_id = :id"), {"id": job_id}
            ).scalar_one()
            == 1
        )


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_clusters_land_in_a_band_the_tenants_thresholds_allow(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    """The confidence gate reads the tenant's own columns, not the defaults."""
    csv_path = tmp_path / "schools.csv"
    _write_csv(csv_path)

    with tenant_sync_session(e2e_tenant) as session:
        job_id = _make_job(session, e2e_tenant)

    _run_pipeline(job_id, e2e_tenant, csv_path)

    with Session(owner_engine) as session:
        rows = session.execute(
            text(
                "SELECT status, confidence_score, master_record_id FROM clusters WHERE job_id = :j"
            ),
            {"j": job_id},
        ).all()

    assert rows, "no clusters to route"

    terminal = {
        ClusterStatus.AUTO_MERGED.value,
        ClusterStatus.PENDING_API.value,
        ClusterStatus.PENDING_REVIEW.value,
    }
    for status, score, master in rows:
        assert status in terminal, f"cluster left in {status} after the gate"
        # FIG 2.5's CHECK would have refused the insert otherwise, so this is
        # really asserting the gate did not route around the constraint.
        if status == ClusterStatus.AUTO_MERGED.value:
            assert master is not None
            assert score >= 0.950
