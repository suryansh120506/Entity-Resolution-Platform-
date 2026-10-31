"""Zero-cost medallion: Postgres task queue, Silver layer, enrichment cache,
Fellegi-Sunter feedback tables, stable cluster entity ids, HNSW drop.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-23

The statements live in migrations/sql/0006_zero_cost_medallion.sql, for the
same reason 0002's do: RLS policies, triggers and privilege changes should be
reviewable and runnable as plain SQL.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def _run(filename: str) -> None:
    path = SQL_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Migration SQL not found: {path}")
    op.execute(path.read_text(encoding="utf-8"))


def upgrade() -> None:
    _run("0006_zero_cost_medallion.sql")


def downgrade() -> None:
    _run("0006_zero_cost_medallion_downgrade.sql")
