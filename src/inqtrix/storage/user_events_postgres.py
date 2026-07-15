"""PostgreSQL persistence for the user invalidation stream."""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Sequence

from sqlalchemy import delete, func, insert, select

from inqtrix.storage.db import tenant_session
from inqtrix.storage.identity_orm import users
from inqtrix.storage.user_event_orm import user_events
from inqtrix.user_events import (
    USER_EVENT_REPLAY_LIMIT,
    USER_EVENT_RETENTION_SECONDS,
    UserEventPage,
    UserInvalidation,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


def _event_from_row(row: Any) -> UserInvalidation:
    return UserInvalidation(
        id=int(row.id),
        tenant_id=str(row.tenant_id),
        target_user_id=row.target_user_id,
        scope=str(row.scope),
        resource_type=(
            str(row.resource_type) if row.resource_type is not None else None
        ),
        resource_id=(
            str(row.resource_id) if row.resource_id is not None else None
        ),
        created_at=(
            row.created_at.timestamp() if row.created_at is not None else 0.0
        ),
    )


async def append_user_invalidation(
    session: "AsyncSession",
    *,
    tenant_id: str,
    target_user_id: uuid.UUID,
    scope: str,
    resource_type: str | None = None,
    resource_id: str | None = None,
) -> UserInvalidation:
    """Append an invalidation inside the caller's existing transaction.

    Mutation repositories use this helper after their resource/audit writes.
    A rollback therefore removes the invalidation too; there is no outbox race
    and no second event authority.
    """
    normalized_scope = scope.strip()
    if not normalized_scope:
        raise ValueError("scope must be non-empty")
    row = (
        await session.execute(
            insert(user_events)
            .values(
                tenant_id=tenant_id,
                target_user_id=target_user_id,
                scope=normalized_scope,
                resource_type=resource_type,
                resource_id=resource_id,
            )
            .returning(user_events)
        )
    ).one()
    return _event_from_row(row)


async def append_instance_admin_invalidations(
    session: "AsyncSession",
    *,
    tenant_id: str,
    target_user_ids: Sequence[uuid.UUID] = (),
    scope: str,
    resource_type: str | None = None,
    resource_id: str | None = None,
) -> tuple[UserInvalidation, ...]:
    """Invalidate affected users and every active instance administrator.

    Administrative list/detail views are instance-wide, so workspace and user
    mutations must wake administrators who are neither the actor nor a
    workspace member. The event remains content-free: it carries only the
    existing refetch coordinates, never mutation data or authorization state.
    """
    admin_user_ids = (
        await session.execute(
            select(users.c.id)
            .where(
                users.c.tenant_id == tenant_id,
                users.c.instance_role == "admin",
                users.c.disabled_at.is_(None),
            )
            .order_by(users.c.id)
        )
    ).scalars()
    targets = set(target_user_ids)
    targets.update(admin_user_ids)
    appended = []
    for target_user_id in sorted(targets, key=str):
        appended.append(
            await append_user_invalidation(
                session,
                tenant_id=tenant_id,
                target_user_id=target_user_id,
                scope=scope,
                resource_type=resource_type,
                resource_id=resource_id,
            )
        )
    return tuple(appended)


class PostgresUserEventStore:
    """Tenant-scoped event replay with bounded polling and lazy retention."""

    def __init__(
        self,
        *,
        session_factory: "async_sessionmaker[AsyncSession]",
        app_role: str,
        retention_seconds: float = USER_EVENT_RETENTION_SECONDS,
        poll_seconds: float = 0.5,
    ) -> None:
        if retention_seconds <= 0:
            raise ValueError("retention_seconds must be positive")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self._session_factory = session_factory
        self._app_role = app_role
        self._retention_seconds = float(retention_seconds)
        self._poll_seconds = float(poll_seconds)

    def _session(
        self, tenant_id: str
    ) -> "AbstractAsyncContextManager[AsyncSession]":
        return tenant_session(
            self._session_factory,
            tenant_id=tenant_id,
            app_role=self._app_role,
        )

    async def append(
        self,
        *,
        tenant_id: str,
        target_user_id: uuid.UUID,
        scope: str,
        resource_type: str | None = None,
        resource_id: str | None = None,
    ) -> UserInvalidation:
        async with self._session(tenant_id) as session:
            await self._cleanup(session)
            return await append_user_invalidation(
                session,
                tenant_id=tenant_id,
                target_user_id=target_user_id,
                scope=scope,
                resource_type=resource_type,
                resource_id=resource_id,
            )

    async def page_after(
        self,
        *,
        tenant_id: str,
        target_user_id: uuid.UUID,
        cursor: int,
        limit: int = USER_EVENT_REPLAY_LIMIT,
    ) -> UserEventPage:
        if cursor < 0:
            raise ValueError("cursor must be non-negative")
        bounded_limit = max(1, min(int(limit), USER_EVENT_REPLAY_LIMIT))
        async with self._session(tenant_id) as session:
            await self._cleanup(session)
            bounds = (
                await session.execute(
                    select(
                        func.min(user_events.c.id),
                        func.max(user_events.c.id),
                    ).where(user_events.c.tenant_id == tenant_id)
                )
            ).one()
            oldest = int(bounds[0]) if bounds[0] is not None else 0
            current = int(bounds[1]) if bounds[1] is not None else 0
            if cursor > current and cursor != 0:
                return UserEventPage((), current, reset_required=True)
            if cursor and oldest and cursor < oldest - 1:
                return UserEventPage((), current, reset_required=True)
            rows = (
                await session.execute(
                    select(user_events)
                    .where(
                        user_events.c.tenant_id == tenant_id,
                        user_events.c.target_user_id == target_user_id,
                        user_events.c.id > cursor,
                    )
                    .order_by(user_events.c.id)
                    .limit(bounded_limit + 1)
                )
            ).all()
            if len(rows) > bounded_limit:
                return UserEventPage((), current, reset_required=True)
            return UserEventPage(
                tuple(_event_from_row(row) for row in rows), current
            )

    async def current_cursor(self, *, tenant_id: str) -> int:
        async with self._session(tenant_id) as session:
            await self._cleanup(session)
            value = await session.scalar(
                select(func.max(user_events.c.id)).where(
                    user_events.c.tenant_id == tenant_id
                )
            )
            return int(value or 0)

    async def wait_for_change(
        self,
        *,
        tenant_id: str,
        target_user_id: uuid.UUID,
        cursor: int,
        timeout: float,
    ) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            async with self._session(tenant_id) as session:
                exists = await session.scalar(
                    select(user_events.c.id)
                    .where(
                        user_events.c.tenant_id == tenant_id,
                        user_events.c.target_user_id == target_user_id,
                        user_events.c.id > cursor,
                    )
                    .limit(1)
                )
            if exists is not None:
                return
            await asyncio.sleep(
                min(self._poll_seconds, max(0.0, deadline - time.monotonic()))
            )

    async def _cleanup(self, session: "AsyncSession") -> None:
        cutoff = datetime.now(UTC) - timedelta(seconds=self._retention_seconds)
        await session.execute(
            delete(user_events).where(user_events.c.created_at < cutoff)
        )
