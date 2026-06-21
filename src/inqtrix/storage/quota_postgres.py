"""Postgres implementation of the quota store.

Same conventions as the other durable repositories: every operation
runs in one tenant-scoped transaction under the restricted app role,
RLS is the second defense behind the explicit ``tenant_id`` predicates.

Lazy rollover falls out of the schema for free: the window
(``period_start``) is part of the counter primary key, so a new month
is simply a new row at 0 — old months linger as harmless history and
``read_usage`` only ever looks at the active window.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from inqtrix.quota.models import QuotaDimension, current_period_start
from inqtrix.quota.models import DEFAULT_SUBJECT, STOCK_PERIOD
from inqtrix.storage.db import (
    build_engine,
    build_session_factory,
    tenant_session,
)
from inqtrix.storage.quota_orm import quota_limits, quota_usage_counters

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession


def _active_period(dimension: QuotaDimension, now: float) -> float:
    return STOCK_PERIOD if dimension.is_stock else current_period_start(now)


class PostgresQuotaStore:
    """Quota usage + limits over the ``quota_*`` tables.

    Owns a NullPool engine on purpose (built from *database_url*): the
    quota service is called both from the async request loop (admission)
    AND from the sync run-execution thread via ``asyncio.run`` (token
    recording at completion). asyncpg connections are loop-affine, so a
    cached pool would fail across loops; NullPool opens a fresh
    connection each call, making the store loop-agnostic. Quota
    operations are single-row and infrequent, so the per-call connect
    cost is negligible.

    Args:
        database_url: Async SQLAlchemy URL (``postgresql+asyncpg://...``).
        app_role: Restricted Postgres role for the tenant sessions.
    """

    def __init__(self, *, database_url: str, app_role: str) -> None:
        self._engine = build_engine(database_url, null_pool=True)
        self._session_factory = build_session_factory(self._engine)
        self._app_role = app_role

    async def aclose(self) -> None:
        """Dispose the owned engine at shutdown (lifecycle parity).

        Must run on a live event loop, not a throwaway ``asyncio.run``
        one — the API lifespan teardown calls this. NullPool holds no
        idle connections, so this only releases the engine/dialect
        machinery, but keeps the store's lifecycle consistent with the
        run store's ``close``.
        """
        await self._engine.dispose()

    def _session(
        self, tenant_id: str
    ) -> "AbstractAsyncContextManager[AsyncSession]":
        return tenant_session(
            self._session_factory,
            tenant_id=tenant_id,
            app_role=self._app_role,
        )

    async def add_usage(
        self,
        *,
        tenant_id: str,
        subject_sub: str,
        dimension: QuotaDimension,
        period_start: float,
        amount: int,
    ) -> int:
        now = time.time()
        seed = max(0, amount)
        async with self._session(tenant_id) as session:
            stmt = pg_insert(quota_usage_counters).values(
                tenant_id=tenant_id,
                subject_sub=subject_sub,
                dimension=dimension.value,
                period_start=period_start,
                used=seed,
                updated_at=now,
            )
            # Clamp at 0 so a stock release never drives the level
            # negative; a new window simply lands as a fresh row.
            stmt = stmt.on_conflict_do_update(
                index_elements=[
                    quota_usage_counters.c.tenant_id,
                    quota_usage_counters.c.subject_sub,
                    quota_usage_counters.c.dimension,
                    quota_usage_counters.c.period_start,
                ],
                set_={
                    "used": func.greatest(
                        0, quota_usage_counters.c.used + amount
                    ),
                    "updated_at": now,
                },
            ).returning(quota_usage_counters.c.used)
            new_used = (await session.execute(stmt)).scalar_one()
        return int(new_used)

    async def read_usage(
        self,
        *,
        tenant_id: str,
        subject_subs: Sequence[str],
        dimensions: Sequence[QuotaDimension],
        now: float,
    ) -> dict[str, dict[QuotaDimension, int]]:
        subs = list(subject_subs)
        dim_values = [d.value for d in dimensions]
        # Pre-fill 0 for every requested dimension so the shape matches
        # the memory store (a missing counter reads as 0, not absent).
        result: dict[str, dict[QuotaDimension, int]] = {
            sub: {d: 0 for d in dimensions} for sub in subs
        }
        if not subs or not dim_values:
            return result
        active = {d: _active_period(d, now) for d in dimensions}
        async with self._session(tenant_id) as session:
            rows = (
                await session.execute(
                    select(
                        quota_usage_counters.c.subject_sub,
                        quota_usage_counters.c.dimension,
                        quota_usage_counters.c.period_start,
                        quota_usage_counters.c.used,
                    ).where(
                        quota_usage_counters.c.tenant_id == tenant_id,
                        quota_usage_counters.c.subject_sub.in_(subs),
                        quota_usage_counters.c.dimension.in_(dim_values),
                    )
                )
            ).all()
        for sub, dim_value, period_start, used in rows:
            dimension = QuotaDimension(dim_value)
            # Only the active window counts; a stale-period row reads as
            # absent (lazy rollover) without being rewritten here.
            if period_start == active[dimension]:
                result[sub][dimension] = int(used)
        return result

    async def reset_usage(
        self,
        *,
        tenant_id: str,
        subject_sub: str,
        dimension: QuotaDimension,
        now: float,
    ) -> None:
        if dimension.is_stock:
            # Stock is freed by deletion, never reset; zeroing it would
            # decouple the counter from real occupancy. Loud bug signal.
            raise ValueError(
                f"cannot reset stock dimension {dimension.value}"
            )
        active = _active_period(dimension, now)
        stamped = time.time()
        async with self._session(tenant_id) as session:
            stmt = pg_insert(quota_usage_counters).values(
                tenant_id=tenant_id,
                subject_sub=subject_sub,
                dimension=dimension.value,
                period_start=active,
                used=0,
                updated_at=stamped,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[
                    quota_usage_counters.c.tenant_id,
                    quota_usage_counters.c.subject_sub,
                    quota_usage_counters.c.dimension,
                    quota_usage_counters.c.period_start,
                ],
                set_={"used": 0, "updated_at": stamped},
            )
            await session.execute(stmt)

    async def get_limits(
        self,
        *,
        tenant_id: str,
        subject_subs: Sequence[str],
        dimensions: Sequence[QuotaDimension],
    ) -> dict[str, dict[QuotaDimension, int]]:
        subs = list(subject_subs)
        dim_values = [d.value for d in dimensions]
        result: dict[str, dict[QuotaDimension, int]] = {}
        if not subs or not dim_values:
            return result
        async with self._session(tenant_id) as session:
            rows = (
                await session.execute(
                    select(
                        quota_limits.c.subject_sub,
                        quota_limits.c.dimension,
                        quota_limits.c.limit_value,
                    ).where(
                        quota_limits.c.tenant_id == tenant_id,
                        quota_limits.c.subject_sub.in_(subs),
                        quota_limits.c.dimension.in_(dim_values),
                    )
                )
            ).all()
        for sub, dim_value, value in rows:
            result.setdefault(sub, {})[QuotaDimension(dim_value)] = int(value)
        return result

    async def set_limit(
        self,
        *,
        tenant_id: str,
        subject_sub: str,
        dimension: QuotaDimension,
        value: int,
        set_by_sub: str,
    ) -> None:
        now = time.time()
        async with self._session(tenant_id) as session:
            stmt = pg_insert(quota_limits).values(
                tenant_id=tenant_id,
                subject_sub=subject_sub,
                dimension=dimension.value,
                limit_value=value,
                set_by_sub=set_by_sub,
                set_at=now,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[
                    quota_limits.c.tenant_id,
                    quota_limits.c.subject_sub,
                    quota_limits.c.dimension,
                ],
                set_={
                    "limit_value": value,
                    "set_by_sub": set_by_sub,
                    "set_at": now,
                },
            )
            await session.execute(stmt)

    async def clear_limit(
        self,
        *,
        tenant_id: str,
        subject_sub: str,
        dimension: QuotaDimension,
    ) -> None:
        async with self._session(tenant_id) as session:
            await session.execute(
                delete(quota_limits).where(
                    quota_limits.c.tenant_id == tenant_id,
                    quota_limits.c.subject_sub == subject_sub,
                    quota_limits.c.dimension == dimension.value,
                )
            )

    async def list_subjects(self, *, tenant_id: str) -> list[str]:
        async with self._session(tenant_id) as session:
            counter_subs = (
                await session.execute(
                    select(quota_usage_counters.c.subject_sub)
                    .where(quota_usage_counters.c.tenant_id == tenant_id)
                    .distinct()
                )
            ).scalars().all()
            limit_subs = (
                await session.execute(
                    select(quota_limits.c.subject_sub)
                    .where(quota_limits.c.tenant_id == tenant_id)
                    .distinct()
                )
            ).scalars().all()
        subs = set(counter_subs) | set(limit_subs)
        subs.discard(DEFAULT_SUBJECT)
        return sorted(subs)
