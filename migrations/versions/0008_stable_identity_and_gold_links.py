"""Stable identities on Silver, identity-keyed constraints, gold links.

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-23
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def _run(filename: str) -> None:
    path = SQL_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Migration SQL not found: {path}")
    op.execute(path.read_text(encoding="utf-8"))


def upgrade() -> None:
    _run("0008_stable_identity_and_gold_links.sql")


def downgrade() -> None:
    _run("0008_stable_identity_and_gold_links_downgrade.sql")
