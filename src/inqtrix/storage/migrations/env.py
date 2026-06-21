"""Alembic environment for the inqtrix platform schema (async).

URL resolution order:

1. A live connection injected via ``config.attributes["connection"]``
   — the test harness drives migrations over its own engine.
2. ``sqlalchemy.url`` set programmatically on the config
   (:func:`inqtrix.storage.migrate.run_migrations`).
3. The ``INQTRIX_DATABASE_URL`` environment variable — the one
   documented exception to the constructor-first rule, because the
   ``alembic`` CLI has no settings bridge.
"""

from __future__ import annotations

import asyncio
import os

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from inqtrix.storage.content_orm import content_metadata
from inqtrix.storage.runs_orm import runs_metadata
from inqtrix.storage.auth_orm import auth_metadata
from inqtrix.storage.identity_orm import identity_metadata
from inqtrix.storage.indexing_orm import indexing_metadata
from inqtrix.storage.knowledge_orm import knowledge_metadata
from inqtrix.storage.chat_orm import chat_metadata
from inqtrix.storage.editor_orm import editor_metadata
from inqtrix.storage.asset_records_orm import asset_metadata
from inqtrix.storage.vector_index_orm import vector_index_metadata
from inqtrix.storage.account_orm import account_metadata

config = context.config

target_metadata = [
    identity_metadata,
    content_metadata,
    runs_metadata,
    auth_metadata,
    knowledge_metadata,
    indexing_metadata,
    chat_metadata,
    editor_metadata,
    asset_metadata,
    vector_index_metadata,
    account_metadata,
]


def _resolve_url() -> str:
    configured = config.get_main_option("sqlalchemy.url")
    if configured:
        return configured
    env_url = os.environ.get("INQTRIX_DATABASE_URL", "")
    if env_url:
        return env_url
    raise RuntimeError(
        "No database URL: set sqlalchemy.url or export "
        "INQTRIX_DATABASE_URL."
    )


def run_migrations_offline() -> None:
    """Emit migration SQL without a live connection (--sql mode)."""
    context.configure(
        url=_resolve_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    # NullPool on purpose: migrations must not share or strand pooled
    # connections with a running application loop.
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _resolve_url()
    connectable = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations over an injected connection or a fresh engine."""
    connection = config.attributes.get("connection", None)
    if connection is not None:
        do_run_migrations(connection)
        return
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
