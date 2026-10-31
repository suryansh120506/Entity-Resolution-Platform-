"""Phase 3 against the live database: stable identity, decisions that carry
across uploads, the rejected pool, and link_only against the gold layer.

Every test uploads the same file twice (jobs A and B) and asks the question
the phase exists to answer: does what was decided on A bind B?
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

# Same directory, no package (see test_medallion_e2e.py).
from test_medallion_e2e import _submit
from test_phase2_e2e import distinct_identity_clusters, identities_of, write_variant_csv
from test_pipeline_e2e import _make_job, _run_pipeline, stub_encoder  # noqa: F401

from app.db.session import tenant_sync_session
from app.models.enums import FeedbackDecision
from app.schemas.feedback import FeedbackRequest

pytestmark = pytest.mark.integration


def _upload(tenant: uuid.UUID, csv_path: Any) -> uuid.UUID:
    with tenant_sync_session(tenant) as session:
        job_id = _make_job(session, tenant)
    _run_pipeline(job_id, tenant, csv_path)
    return job_id


def _set_auto_merge(tenant: uuid.UUID, value: str) -> None:
    with tenant_sync_session(tenant) as session:
        session.execute(
            text("UPDATE tenants SET confidence_auto_merge = :v WHERE id = :t"),
            {"v": value, "t": tenant},
        )


def _matching(owner_engine: Engine, job_id: uuid.UUID) -> dict[str, Any]:
    with Session(owner_engine) as session:
        return session.execute(
            text("SELECT stage_progress->'matching'->'details' FROM jobs WHERE id = :j"),
            {"j": job_id},
        ).scalar_one()


def _cluster_identity_sets(owner_engine: Engine, job_id: uuid.UUID) -> list[frozenset[str]]:
    with Session(owner_engine) as session:
        rows = session.execute(
            text(
                "SELECT c.id, s.stable_id FROM clusters c "
                "JOIN cluster_members m ON m.cluster_id = c.id "
                "JOIN silver_records s ON s.record_id = m.record_id WHERE c.job_id = :j"
            ),
            {"j": job_id},
        ).all()
    grouped: dict[Any, set[str]] = {}
    for cluster_id, stable_id in rows:
        grouped.setdefault(cluster_id, set()).add(stable_id)
    return [frozenset(ids) for ids in grouped.values()]


def _members(owner_engine: Engine, cluster_record: str) -> list[str]:
    with Session(owner_engine) as session:
        return [
            str(r)
            for r in session.execute(
                text(
                    "SELECT m.record_id FROM cluster_members m WHERE m.cluster_id = "
                    "(SELECT cluster_id FROM cluster_members WHERE record_id = :r) "
                    "ORDER BY m.record_id"
                ),
                {"r": uuid.UUID(cluster_record)},
            ).scalars()
        ]


def _cluster_id(owner_engine: Engine, record: str) -> uuid.UUID:
    with Session(owner_engine) as session:
        return session.execute(
            text("SELECT cluster_id FROM cluster_members WHERE record_id = :r"),
            {"r": uuid.UUID(record)},
        ).scalar_one()


# --- Stable identity -------------------------------------------------------------------


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_identity_is_deterministic_and_tiered(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    csv_path = tmp_path / "schools.csv"
    write_variant_csv(csv_path)
    first, second = _upload(e2e_tenant, csv_path), _upload(e2e_tenant, csv_path)

    def identities(job_id: uuid.UUID) -> list[tuple[str, str, str]]:
        with Session(owner_engine) as session:
            return session.execute(
                text(
                    "SELECT r.raw_row->>'name', s.stable_id, s.identity_kind "
                    "FROM silver_records s JOIN school_records r ON r.id = s.record_id "
                    "WHERE s.job_id = :j ORDER BY 1, 2"
                ),
                {"j": job_id},
            ).all()

    a, b = identities(first), identities(second)
    assert [row[1:] for row in a if not row[1].startswith("rec:")] == [
        row[1:] for row in b if not row[1].startswith("rec:")
    ], "the same file must yield the same identities in every upload"

    kinds = {name: kind for name, _, kind in a}
    assert kinds["Sunrise Academy"] == kinds["Old Mission Building"] == "natural_key"
    assert {sid for name, sid, _ in a if name in ("Sunrise Academy", "Old Mission Building")} == {
        "nk:udise_code:29281200612"
    }
    assert kinds["Govt Higher Primary Schol Unit 0"] == "fingerprint"
    # Junk with no name anchor has no stable basis: record-scoped.
    assert all(kind == "record" for name, _, kind in a if name.startswith("TEST"))


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_records_sharing_a_natural_key_merge_whatever_their_names(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    csv_path = tmp_path / "schools.csv"
    write_variant_csv(csv_path)
    job_id = _upload(e2e_tenant, csv_path)

    with Session(owner_engine) as session:
        clusters = (
            session.execute(
                text(
                    "SELECT m.cluster_id FROM cluster_members m "
                    "JOIN school_records r ON r.id = m.record_id "
                    "WHERE r.job_id = :j AND r.raw_row->>'udise_code' = '29281200612'"
                ),
                {"j": job_id},
            )
            .scalars()
            .all()
        )
    assert len(clusters) == 2 and clusters[0] == clusters[1]


# --- Decisions carry across uploads ------------------------------------------------------


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_a_split_on_one_upload_binds_the_next(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    csv_path = tmp_path / "schools.csv"
    write_variant_csv(csv_path)
    _set_auto_merge(e2e_tenant, "0.999")
    job_a = _upload(e2e_tenant, csv_path)

    keep, split_off = distinct_identity_clusters(owner_engine, job_a, 2)[0]
    identity = identities_of(owner_engine, [keep, split_off])
    excluded = [
        r
        for r in _members(owner_engine, keep)
        if identities_of(owner_engine, [r])[r] == identity[split_off]
    ]
    _submit(
        e2e_tenant,
        _cluster_id(owner_engine, keep),
        FeedbackRequest(
            decision=FeedbackDecision.SPLIT, excluded_record_ids=[uuid.UUID(r) for r in excluded]
        ),
    )

    # Without the rule, upload A put these identities in one cluster, so
    # upload B (the same file) would too: the rule must be what separates them.
    job_b = _upload(e2e_tenant, csv_path)
    together = [
        ids
        for ids in _cluster_identity_sets(owner_engine, job_b)
        if {identity[keep], identity[split_off]} <= ids
    ]
    assert together == [], "a never-merge decided on upload A was ignored on upload B"
    details = _matching(owner_engine, job_b)
    assert details["cannot_link_constraints"] >= 1 and details["cut_links"] >= 1


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_a_rejected_trio_is_never_re_proposed_but_its_pairs_may_be(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    csv_path = tmp_path / "schools.csv"
    write_variant_csv(csv_path)
    _set_auto_merge(e2e_tenant, "0.999")
    job_a = _upload(e2e_tenant, csv_path)

    trio = distinct_identity_clusters(owner_engine, job_a, 3)[0]
    rejected = frozenset(identities_of(owner_engine, trio).values())
    _submit(
        e2e_tenant,
        _cluster_id(owner_engine, trio[0]),
        FeedbackRequest(decision=FeedbackDecision.REJECT),
    )

    with Session(owner_engine) as session:
        stored = (
            session.execute(
                text("SELECT identities FROM entity_group_constraints WHERE tenant_id = :t"),
                {"t": e2e_tenant},
            )
            .scalars()
            .all()
        )
    assert [frozenset(g) for g in stored] == [rejected]

    job_b = _upload(e2e_tenant, csv_path)
    sets = _cluster_identity_sets(owner_engine, job_b)
    assert not any(rejected <= ids for ids in sets), "the rejected grouping came back"
    assert any(len(rejected & ids) == 2 for ids in sets), (
        "not-all-together must leave proper subsets free to merge"
    )
    details = _matching(owner_engine, job_b)
    assert details["not_all_together_constraints"] == 1 and details["cut_links"] >= 1


# --- Gold layer linking ------------------------------------------------------------------


def _links(owner_engine: Engine, job_id: uuid.UUID) -> list[Any]:
    with Session(owner_engine) as session:
        return session.execute(
            text(
                "SELECT g.record_id::text, g.target_record_id::text, g.target_kind, g.status, "
                "g.reason, g.target_stable_id, s.stable_id AS own_stable_id "
                "FROM gold_links g JOIN silver_records s ON s.record_id = g.record_id "
                "WHERE g.job_id = :j"
            ),
            {"j": job_id},
        ).all()


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_a_reupload_links_to_existing_entities_and_inherits_their_ids(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    csv_path = tmp_path / "schools.csv"
    write_variant_csv(csv_path)
    _set_auto_merge(e2e_tenant, "0.700")
    job_a = _upload(e2e_tenant, csv_path)
    job_b = _upload(e2e_tenant, csv_path)

    links = _links(owner_engine, job_b)
    auto = [link for link in links if link.status == "auto_linked"]
    assert auto, "a re-upload of the same file linked to nothing in the gold layer"
    assert any(link.target_kind == "gold_master" for link in auto)
    assert any(link.reason == "natural_key" for link in auto)

    with Session(owner_engine) as session:
        entities_a = set(
            session.execute(
                text("SELECT entity_id FROM clusters WHERE job_id = :j"), {"j": job_a}
            ).scalars()
        )
        entities_b = set(
            session.execute(
                text("SELECT entity_id FROM clusters WHERE job_id = :j"), {"j": job_b}
            ).scalars()
        )
        details = session.execute(
            text("SELECT stage_progress->'gold_linking'->'details' FROM jobs WHERE id = :j"),
            {"j": job_b},
        ).scalar_one()
    assert details["entities_inherited"] >= 1
    assert entities_a & entities_b, "the same schools got new entity ids on re-upload"
    assert details["model"], "gold linking ran without the tenant's model"


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_released_records_are_linkable_but_never_merges_still_hold(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    """A SPLIT releases the excluded record into the gold pool. The next
    upload's copy of it links to that released record — and never to the
    entity the steward split it away from."""
    csv_path = tmp_path / "schools.csv"
    write_variant_csv(csv_path)
    _set_auto_merge(e2e_tenant, "0.999")
    job_a = _upload(e2e_tenant, csv_path)

    keep, split_off = distinct_identity_clusters(owner_engine, job_a, 2)[0]
    identity = identities_of(owner_engine, [keep, split_off])
    excluded = [
        r
        for r in _members(owner_engine, keep)
        if identities_of(owner_engine, [r])[r] == identity[split_off]
    ]
    _submit(
        e2e_tenant,
        _cluster_id(owner_engine, keep),
        FeedbackRequest(
            decision=FeedbackDecision.SPLIT, excluded_record_ids=[uuid.UUID(r) for r in excluded]
        ),
    )

    job_b = _upload(e2e_tenant, csv_path)
    from_split_identity = [
        link for link in _links(owner_engine, job_b) if link.own_stable_id == identity[split_off]
    ]
    assert any(
        link.target_kind == "released" and link.target_stable_id == identity[split_off]
        for link in from_split_identity
    ), "the released record was not offered back to the next upload"
    assert not any(link.target_stable_id == identity[keep] for link in from_split_identity), (
        "a gold link re-proposed a merge the steward split apart"
    )


# --- Backfill ------------------------------------------------------------------------------


@pytest.mark.usefixtures("stub_encoder", "assert_no_leaked_rows")
def test_backfill_restores_silver_without_touching_the_job(
    e2e_tenant: uuid.UUID, owner_engine: Engine, tmp_path: Any
) -> None:
    from app.workers.tasks.silver import backfill_job

    csv_path = tmp_path / "schools.csv"
    rows = write_variant_csv(csv_path)
    job_id = _upload(e2e_tenant, csv_path)
    with Session(owner_engine) as session:
        status = session.execute(
            text("SELECT status FROM jobs WHERE id = :j"), {"j": job_id}
        ).scalar_one()
        session.execute(text("DELETE FROM silver_records WHERE job_id = :j"), {"j": job_id})
        session.commit()

    result = backfill_job(str(job_id), str(e2e_tenant))

    with Session(owner_engine) as session:
        restored, with_identity, after = session.execute(
            text(
                "SELECT count(*), count(stable_id), (SELECT status FROM jobs WHERE id = :j) "
                "FROM silver_records WHERE job_id = :j"
            ),
            {"j": job_id},
        ).one()
    assert restored == with_identity == rows == result["records"]
    assert after == status
