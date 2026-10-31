"""Alembic environment for ER-PLATFORM-001.

Two things here are not boilerplate and are worth reading before editing:

1. The connection URL comes from MIGRATION_DATABASE_URL — the *owner* role.
   Never from DATABASE_URL. The runtime role has no CREATE privilege by
   design (see migrations/sql/0002_rls_policies.sql), so pointing Alembic at
   it fails; more importantly, an owner-role DSN sitting in the application's
   own setting is how the RLS bypass in the FORCE comment happens.

2. `render_item` teaches autogenerate to emit `pgvector.sqlalchemy.Vector`.
   Without it, autogenerate renders the 768-dim embedding columns as
   `sa.NullType()` and silently drops the dimension — the resulting migration
   applies cleanly and produces a table that cannot hold an embedding.
"""

from __future__ import annotations

import os
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import engine_from_config, pool

# Importing the models package registers every table on Base.metadata.
from app.models import Base  # noqa: F401  (imported for its side effect)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """Resolve the migration DSN, owner role only."""
    url = os.environ.get("MIGRATION_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "MIGRATION_DATABASE_URL is not set. Alembic must connect as the "
            "schema owner; see .env.example. (DATABASE_URL is accepted as a "
            "fallback for local convenience, but the runtime role it normally "
            "points at has no CREATE privilege.)"
        )
    return url


def render_item(type_: str, obj: Any, autogen_context: Any) -> str | bool:
    """Render pgvector's Vector type with its dimension intact."""
    if type_ == "type" and obj.__class__.__module__.startswith("pgvector"):
        autogen_context.imports.add("import pgvector.sqlalchemy")
        return f"pgvector.sqlalchemy.Vector(dim={obj.dim})"
    # False = fall through to Alembic's default rendering.
    return False


def include_object(
    obj: Any, name: str | None, type_: str, reflected: bool, compare_to: Any
) -> bool:
    """Keep autogenerate's hands off objects it did not create.

    The HNSW indexes in FIG 2.4 are declared with `postgresql_using="hnsw"`
    plus operator classes and build parameters. Alembic's reflection of an
    index that specific is not reliable enough to diff against, and a spurious
    drop/recreate of an HNSW index on a 50k-row table is a multi-minute
    migration nobody asked for. They are managed explicitly in the version
    files instead.
    """
    return not (type_ == "index" and name and name.endswith("_hnsw"))


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        render_item=render_item,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            render_item=render_item,
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
