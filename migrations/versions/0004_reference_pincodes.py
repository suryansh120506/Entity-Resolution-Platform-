"""Reference pincode lookup table — global, un-tenanted (All-India Pincode data)

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-16

`reference_pincodes` is a shared, read-mostly lookup table seeded from the
government All-India Pincode CSV (scripts/seed_pincodes.py). It carries no
`tenant_id` — every tenant validates addresses against the same PIN code
list — so unlike every table in migration 0001 it is NOT covered by RLS and
is NOT added to `app.core.constants.TENANT_SCOPED_TABLES`.

That is a documented exception, not a gap: migrations/sql/0002_rls_policies.sql
section 4 fails the migration (and `make rls-verify`) if any table in the
schema lacks an enforced RLS policy, specifically so a table added by a later
migration cannot ship unprotected by accident. This migration's counterpart
edit to that file names `reference_pincodes` explicitly as the exception, with
the reason above, so the guard keeps catching real gaps instead of also
tripping on this one every time it runs.

Table privileges need no separate GRANT here — migration 0002's
`ALTER DEFAULT PRIVILEGES ... GRANT ... ON TABLES` already covers every table
the owner role creates from here on, which is exactly how `er_app` gets
read access to this table despite it existing entirely outside RLS.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "reference_pincodes",
        sa.Column(
            "id",
            sa.BigInteger(),
            sa.Identity(always=False),
            nullable=False,
        ),
        sa.Column("pincode", sa.String(length=6), nullable=False),
        sa.Column("office_name", sa.Text(), nullable=False),
        sa.Column("district_name", sa.Text(), nullable=False),
        sa.Column("state_name", sa.Text(), nullable=False),
        sa.Column("taluk", sa.Text(), nullable=True),
        sa.Column("delivery_status", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_reference_pincodes"),
    )
    # B-Tree (Postgres' default index type) on both lookup columns. Not
    # unique on either: one pincode covers several offices, and office names
    # repeat across districts.
    op.create_index("ix_reference_pincodes_pincode", "reference_pincodes", ["pincode"])
    op.create_index("ix_reference_pincodes_district_name", "reference_pincodes", ["district_name"])


def downgrade() -> None:
    op.drop_table("reference_pincodes")
