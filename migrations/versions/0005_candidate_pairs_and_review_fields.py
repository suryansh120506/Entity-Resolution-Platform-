"""candidate_pairs, jobs.original_filename, clusters.golden_fields,
human_feedback.field_corrections

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-26

Four changes, all additive:

* `candidate_pairs` — the blocking stage's output, so matching can score
  pairs *before* anything groups records into clusters (see
  app/models/candidate_pair.py). Tenant-scoped, so it gets the same
  ENABLE + FORCE + USING/WITH CHECK policy every table in migration 0002
  carries; the drift tests in tests/test_rls_isolation.py fail otherwise.
  Table GRANTs need no statement here: 0002's ALTER DEFAULT PRIVILEGES covers
  every table the owner creates afterwards.
* `jobs.original_filename` — so a job list can show what was uploaded.
* `clusters.golden_fields` / `human_feedback.field_corrections` — a steward's
  corrections to the surviving record, stored beside it rather than written
  into `raw_row`, which TRD 2.3 keeps immutable.

New nullable columns carry no default, so each ALTER is a catalog-only change
with no table rewrite. RLS on the three existing tables is table-wide and
covers the new columns without any change.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "candidate_pairs",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("record_id_a", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("record_id_b", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("distance", sa.Float(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.CheckConstraint("record_id_a < record_id_b", name="ck_candidate_pairs_canonical_order"),
        sa.CheckConstraint(
            "source IN ('name', 'address', 'both')", name="ck_candidate_pairs_source_valid"
        ),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"], name="fk_candidate_pairs_job_id_jobs", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["record_id_a"],
            ["school_records.id"],
            name="fk_candidate_pairs_record_id_a_school_records",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["record_id_b"],
            ["school_records.id"],
            name="fk_candidate_pairs_record_id_b_school_records",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name="fk_candidate_pairs_tenant_id_tenants",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("job_id", "record_id_a", "record_id_b", name="pk_candidate_pairs"),
    )
    op.create_index(
        "ix_candidate_pairs_tenant_id_job_id", "candidate_pairs", ["tenant_id", "job_id"]
    )

    op.execute("ALTER TABLE candidate_pairs ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE candidate_pairs FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY tenant_isolation ON candidate_pairs "
        "USING (tenant_id = current_setting('app.tenant_id')::uuid) "
        "WITH CHECK (tenant_id = current_setting('app.tenant_id')::uuid)"
    )

    op.add_column("jobs", sa.Column("original_filename", sa.Text(), nullable=True))
    op.add_column(
        "clusters",
        sa.Column("golden_fields", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "human_feedback",
        sa.Column("field_corrections", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("human_feedback", "field_corrections")
    op.drop_column("clusters", "golden_fields")
    op.drop_column("jobs", "original_filename")
    op.drop_table("candidate_pairs")
