"""Cerebras as a bring-your-own-key provider (widens the provider CHECK).

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-25
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def _run(filename: str) -> None:
    path = SQL_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Migration SQL not found: {path}")
    op.execute(path.read_text(encoding="utf-8"))


def upgrade() -> None:
    _run("0012_free_providers.sql")


def downgrade() -> None:
    _run("0012_free_providers_downgrade.sql")
