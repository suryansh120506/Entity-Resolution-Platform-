"""Row-Level Security — TRD FIG 2.8

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-16

The SQL lives in migrations/sql/0002_rls_policies.sql, not inline in this
file. That is so the security control can be read, reviewed and run as SQL —
`psql -1 -f migrations/sql/0002_rls_policies.sql` — without anyone having to
mentally strip Python string quoting out of the policy bodies first. This
module is the Alembic entry point to it, and there is exactly one copy of the
statements.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def _run(filename: str) -> None:
    path = SQL_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Migration SQL not found: {path}. This file is the RLS policy "
            "itself — a migration that silently skipped it would report "
            "success while leaving every table unprotected."
        )
    op.execute(path.read_text(encoding="utf-8"))


def upgrade() -> None:
    _run("0002_rls_policies.sql")


def downgrade() -> None:
    _run("0002_rls_policies_downgrade.sql")
