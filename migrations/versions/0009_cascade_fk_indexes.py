"""Index the referencing column of every cascading foreign key.

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-24
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def _run(filename: str) -> None:
    path = SQL_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Migration SQL not found: {path}")
    op.execute(path.read_text(encoding="utf-8"))


def upgrade() -> None:
    _run("0009_cascade_fk_indexes.sql")


def downgrade() -> None:
    _run("0009_cascade_fk_indexes_downgrade.sql")
