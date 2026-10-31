"""Add school_records.label_source — provenance of `classification`

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-16

Adds a NOT NULL TEXT column recording whether a row's `classification` came
from the UNLABELED default, a PSEUDO label the classifier assigned, or a
STEWARD's review of a cluster.

Two things worth stating about this being safe on a live table:

1. Adding a NOT NULL column with a server default is a metadata-only change
   on Postgres 11+ — the default is stored once in the catalog, not written
   into every existing row, so this does not take a table-rewrite lock.

2. RLS is untouched. The `tenant_isolation` policy on `school_records`
   (migrations/sql/0002_rls_policies.sql) is defined on the table as a whole,
   keyed on `tenant_id` — it has no per-column scope, so a new column is
   covered by it automatically and needs no migration of its own.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "school_records",
        sa.Column(
            "label_source",
            sa.Text(),
            server_default=sa.text("'UNLABELED'"),
            nullable=False,
        ),
    )
    op.create_check_constraint(
        "ck_school_records_label_source_valid",
        "school_records",
        "label_source IN ('UNLABELED', 'PSEUDO', 'STEWARD')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_school_records_label_source_valid", "school_records", type_="check")
    op.drop_column("school_records", "label_source")
