"""Postgres-backed agent-session store (durable Agent-Desk tier).

Sessions persist relationally, scoped per ``(tenant_id, created_by_user_id,
workspace_id)`` with the inherited tenant-session lifecycle
(:class:`BaseSessionStore`). ``list_sessions`` SELECTs metadata columns only
(NOT the heavy ``items_json``); ``get_session`` SELECTs the full row.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from inqtrix.project.base_session_store import (
    BaseSessionStore,
    DEFAULT_TENANT as _DEFAULT_TENANT,
)
from inqtrix.project.agent_sessions_ports import (
    AgentSession,
    AgentSessionGroup,
    AgentSessionGroupNotFound,
    AgentSessionNotFound,
)
from inqtrix.project.scoped_upsert import (
    ResourceScope,
    delete_scoped_postgres,
    require_scoped_parent,
    scoped_postgres_upsert,
)
from inqtrix.storage.agent_sessions_orm import (
    agent_session_groups,
    agent_sessions,
)

# Metadata columns (everything EXCEPT the heavy items_json) for the list path.
_META_COLUMNS = (
    agent_sessions.c.id,
    agent_sessions.c.tenant_id,
    agent_sessions.c.created_by_user_id,
    agent_sessions.c.workspace_id,
    agent_sessions.c.title,
    agent_sessions.c.group_id,
    agent_sessions.c.created_at,
    agent_sessions.c.updated_at,
)


class PostgresAgentSessionStore(BaseSessionStore):
    """Durable
    :class:`~inqtrix.project.agent_sessions_ports.AgentSessionStore`."""

    async def claim_session(
        self, *, id: str, title: str, created_at: float,
        created_by_user_id: uuid.UUID | None, workspace_id: str | None,
    ) -> AgentSession:
        values = dict(
            id=id,
            tenant_id=_DEFAULT_TENANT,
            created_by_user_id=created_by_user_id,
            workspace_id=workspace_id,
            title=title,
            group_id=None,
            items_json="[]",
            created_at=created_at,
            updated_at=created_at,
        )
        async with self._session() as session:
            row = (
                await session.execute(
                    pg_insert(agent_sessions)
                    .values(**values)
                    .on_conflict_do_nothing(index_elements=[agent_sessions.c.id])
                    .returning(agent_sessions)
                )
            ).first()
            if row is None:
                row = (
                    await session.execute(
                        select(agent_sessions).where(
                            agent_sessions.c.tenant_id == _DEFAULT_TENANT,
                            agent_sessions.c.id == id,
                        )
                    )
                ).one()
        return _from_row(row)

    async def upsert_session(
        self, *, id: str, title: str, items_json: str, group_id: str | None,
        created_at: float,
        updated_at: float,
        created_by_user_id: uuid.UUID | None,
        workspace_id: str | None,
    ) -> AgentSession:
        values = dict(
            id=id, tenant_id=_DEFAULT_TENANT, created_by_user_id=created_by_user_id,
            workspace_id=workspace_id, title=title, group_id=group_id,
            items_json=items_json,
            created_at=created_at, updated_at=updated_at,
        )
        mutable = ["title", "group_id", "items_json", "updated_at"]
        stmt = scoped_postgres_upsert(
            pg_insert(agent_sessions), agent_sessions, values, mutable
        ).returning(agent_sessions)
        async with self._session() as session:
            if group_id is not None:
                await require_scoped_parent(
                    session,
                    table=agent_session_groups,
                    parent_id=group_id,
                    tenant_id=_DEFAULT_TENANT,
                    created_by_user_id=created_by_user_id,
                    workspace_id=workspace_id,
                    not_found=AgentSessionGroupNotFound,
                )
            row = (await session.execute(stmt)).first()
        if row is None:
            raise AgentSessionNotFound(id)
        return _from_row(row)

    async def list_sessions(
        self, *, created_by_user_id: uuid.UUID | None, workspace_id: str | None
    ) -> list[AgentSession]:
        query = _scoped_query(
            select(*_META_COLUMNS), agent_sessions, created_by_user_id, workspace_id
        )
        query = query.order_by(
            agent_sessions.c.updated_at.desc(), agent_sessions.c.id.desc()
        )
        async with self._session() as session:
            rows = (await session.execute(query)).all()
        return [_from_row(row) for row in rows]

    async def get_session(self, session_id: str) -> AgentSession:
        query = select(agent_sessions).where(
            agent_sessions.c.tenant_id == _DEFAULT_TENANT,
            agent_sessions.c.id == session_id,
        )
        async with self._session() as session:
            row = (await session.execute(query)).first()
        if row is None:
            raise AgentSessionNotFound(session_id)
        return _from_row(row)

    async def delete_session(
        self, session_id: str, *, scope: ResourceScope
    ) -> None:
        async with self._session() as session:
            await delete_scoped_postgres(
                session, table=agent_sessions, resource_id=session_id,
                tenant_id=_DEFAULT_TENANT, scope=scope,
                not_found=AgentSessionNotFound,
            )

    async def upsert_group(
        self, *, id: str, title: str, created_at: float, updated_at: float,
        created_by_user_id: uuid.UUID | None, workspace_id: str | None,
    ) -> AgentSessionGroup:
        values = dict(
            id=id, tenant_id=_DEFAULT_TENANT, created_by_user_id=created_by_user_id,
            workspace_id=workspace_id, title=title, created_at=created_at,
            updated_at=updated_at,
        )
        stmt = scoped_postgres_upsert(
            pg_insert(agent_session_groups),
            agent_session_groups,
            values,
            ["title", "updated_at"],
        ).returning(agent_session_groups)
        async with self._session() as session:
            row = (await session.execute(stmt)).first()
        if row is None:
            raise AgentSessionGroupNotFound(id)
        return _group_from_row(row)

    async def claim_group(
        self, *, id: str, title: str, created_at: float,
        created_by_user_id: uuid.UUID | None, workspace_id: str | None,
    ) -> AgentSessionGroup:
        values = dict(
            id=id, tenant_id=_DEFAULT_TENANT,
            created_by_user_id=created_by_user_id, workspace_id=workspace_id,
            title=title, created_at=created_at, updated_at=created_at,
        )
        async with self._session() as session:
            row = (
                await session.execute(
                    pg_insert(agent_session_groups)
                    .values(**values)
                    .on_conflict_do_nothing(
                        index_elements=[agent_session_groups.c.id]
                    )
                    .returning(agent_session_groups)
                )
            ).first()
            if row is None:
                row = (
                    await session.execute(
                        select(agent_session_groups).where(
                            agent_session_groups.c.tenant_id
                            == _DEFAULT_TENANT,
                            agent_session_groups.c.id == id,
                        )
                    )
                ).one()
        return _group_from_row(row)

    async def list_groups(
        self, *, created_by_user_id: uuid.UUID | None, workspace_id: str | None
    ) -> list[AgentSessionGroup]:
        query = _scoped_query(
            select(agent_session_groups), agent_session_groups,
            created_by_user_id, workspace_id,
        )
        query = query.order_by(
            agent_session_groups.c.created_at.desc(),
            agent_session_groups.c.id.desc(),
        )
        async with self._session() as session:
            rows = (await session.execute(query)).all()
        return [_group_from_row(row) for row in rows]

    async def delete_group(
        self, group_id: str, *, scope: ResourceScope
    ) -> None:
        async with self._session() as session:
            await delete_scoped_postgres(
                session, table=agent_session_groups, resource_id=group_id,
                tenant_id=_DEFAULT_TENANT, scope=scope,
                not_found=AgentSessionGroupNotFound,
            )


def _from_row(row) -> AgentSession:
    return AgentSession(
        id=row.id,
        title=row.title,
        group_id=row.group_id,
        items_json=getattr(row, "items_json", "[]") or "[]",
        created_at=row.created_at,
        updated_at=row.updated_at,
        tenant_id=row.tenant_id,
        created_by_user_id=row.created_by_user_id,
        workspace_id=row.workspace_id,
    )


def _group_from_row(row) -> AgentSessionGroup:
    return AgentSessionGroup(
        id=row.id,
        title=row.title,
        created_at=row.created_at,
        updated_at=row.updated_at,
        tenant_id=row.tenant_id,
        created_by_user_id=row.created_by_user_id,
        workspace_id=row.workspace_id,
    )


def _scoped_query(
    query, table, created_by_user_id: uuid.UUID | None, workspace_id
):
    query = query.where(table.c.tenant_id == _DEFAULT_TENANT)
    if created_by_user_id is not None:
        query = query.where(table.c.created_by_user_id == created_by_user_id)
    if workspace_id is not None:
        query = query.where(table.c.workspace_id == workspace_id)
    return query
