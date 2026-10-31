"""Automated triage and golden enrichment: audit columns on clusters, the
'golden' enrichment cache strategy, and a 0.80 Tier 1 floor.

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-24
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def _run(filename: str) -> None:
    path = SQL_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Migration SQL not found: {path}")
    op.execute(path.read_text(encoding="utf-8"))


def upgrade() -> None:
    _run("0010_automated_triage.sql")


def downgrade() -> None:
    _run("0010_automated_triage_downgrade.sql")
