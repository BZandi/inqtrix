"""Async engine construction and the tenant-scoped transaction helper.

The tenant helper is the single sanctioned way to touch tenant tables:
every call opens ONE transaction, switches to the restricted
application role, and sets the transaction-local ``inqtrix.tenant_id``
GUC the row-level-security policies read. Both steps are
deliberately transaction-scoped:

* ``SET LOCAL ROLE`` reverts at COMMIT/ROLLBACK, so pooled
  connections never leak the restricted role into unrelated work
  (e.g. migrations sharing a pool in tests).
* ``set_config(..., is_local => true)`` reverts the tenant id the
  same way — the only safe form under connection pooling, where a
  session-scoped value would leak the previous tenant to the next
  request (PgBouncer transaction mode skips reset queries entirely).

The policies fail closed on top of this: the
``inqtrix_current_tenant_id()`` helper raises when the GUC is unset
*or* empty (after a transaction-local value reverts, the parameter
survives as an empty string for the connection lifetime — the
documented ``current_setting`` gotcha), so a forgotten
``tenant_session`` produces a loud error, never another tenant's rows.

Transaction-pooler contract (PgBouncer ``pool_mode=transaction``): this
module is deliberately pooler-safe — no session-scoped SET, no
LISTEN/NOTIFY, no session advisory locks — and MUST stay that way; a
future session-scoped feature would silently break behind a pooler.
Two operational requirements remain on the DEPLOYMENT side: (1) asyncpg
prepared statements need either ``?prepared_statement_cache_size=0`` in
the URL or PgBouncer >= 1.21 with ``max_prepared_statements`` tracking
(the bundled compose/helm pgbouncer sets both); (2) Alembic migrations
(``inqtrix-migrate``) connect DIRECTLY to Postgres, never through the
pooler — DDL and long transactions do not multiplex.
"""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

TENANT_GUC = "inqtrix.tenant_id"
"""Name of the per-transaction GUC carrying the tenant id for RLS."""

_SAFE_ROLE_PATTERN = re.compile(r"^[a-z_][a-z0-9_]*$")
"""Allowed shape for the application role identifier. ``SET ROLE`` is a
utility statement and cannot take bind parameters, so the identifier is
validated against this conservative pattern before interpolation."""


def build_engine(
    database_url: str,
    *,
    null_pool: bool = False,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout: float = 30.0,
) -> AsyncEngine:
    """Create the async engine for the platform persistence layer.

    Args:
        database_url: SQLAlchemy async URL
            (``postgresql+asyncpg://...``). Constructor argument by
            design — only the settings bridge reads the environment.
        null_pool: When ``True``, use a :class:`NullPool` (a fresh
            connection per operation, none cached). asyncpg connections
            are event-loop-affine; a pooled connection reused across
            loops fails. NullPool makes the engine loop-AGNOSTIC, which
            a store that is called both from the async request loop AND
            from a sync worker thread via ``asyncio.run`` requires (the
            quota store; the durable run store solves the same problem
            with its own dedicated loop instead). The pool sizing
            arguments are ignored on this branch — NullPool holds no
            connections.
        pool_size: Persistent connections this engine keeps open.
            Default 5 (the SQLAlchemy default) keeps existing
            deployments byte-identical. Operators size it through
            ``StorageSettings.pool_kwargs()`` so the per-process budget
            — ``pooled_engines x (pool_size + max_overflow)`` — stays
            reviewable against Postgres ``max_connections``.
        max_overflow: Burst connections beyond ``pool_size``, closed
            again when idle (default 10, the SQLAlchemy default).
        pool_timeout: Seconds a caller waits for a free pooled
            connection before failing loudly (default 30, the
            SQLAlchemy default). Bounds the queueing latency a
            too-small pool would otherwise convert errors into.

    Returns:
        An :class:`AsyncEngine` with pre-ping (survives idle-timeout
        connection kills) and a 30-minute recycle below common
        infrastructure idle limits.
    """
    if null_pool:
        return create_async_engine(database_url, poolclass=NullPool)
    return create_async_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
    )


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create the session factory bound to *engine*.

    ``expire_on_commit=False`` is mandatory for async SQLAlchemy:
    attribute access after commit must not trigger implicit IO
    (``MissingGreenlet``).
    """
    return async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def tenant_session(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    tenant_id: str,
    app_role: str,
) -> AsyncIterator[AsyncSession]:
    """One tenant-scoped transaction against the identity schema.

    Opens a session, begins a transaction, switches to *app_role*
    (when non-empty), sets the transaction-local tenant GUC, and
    yields the session. Commit happens on clean exit, rollback on
    exception. Callers must finish their work inside the context —
    a second transaction on the same session would run without the
    GUC and fail loudly in the RLS helper.

    Args:
        session_factory: Factory from :func:`build_session_factory`.
        tenant_id: Tenant whose rows this transaction may touch.
            Must be non-empty — an empty tenant id would otherwise
            silently match zero rows instead of failing.
        app_role: Restricted Postgres role to run as (see
            ``StorageSettings.app_role``). Empty skips the switch.

    Raises:
        ValueError: On an empty *tenant_id* or an *app_role* that does
            not match the conservative identifier pattern (defense
            against SQL injection through configuration).
    """
    if not tenant_id.strip():
        raise ValueError("tenant_id must be non-empty for a tenant session")
    if app_role and not _SAFE_ROLE_PATTERN.fullmatch(app_role):
        raise ValueError(f"app_role has an unsafe identifier shape: {app_role!r}")
    async with session_factory() as session:
        async with session.begin():
            if app_role:
                # Identifier validated above; SET ROLE cannot be
                # parameterized (Postgres utility statement).
                await session.execute(text(f'SET LOCAL ROLE "{app_role}"'))
            await session.execute(
                text("SELECT set_config(:guc, :tenant, true)"),
                {"guc": TENANT_GUC, "tenant": tenant_id},
            )
            yield session
