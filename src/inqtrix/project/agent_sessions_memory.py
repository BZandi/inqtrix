"""In-memory agent-session store (tier without Postgres + offline test).

Mirrors :class:`~inqtrix.project.agent_sessions_postgres.PostgresAgentSessionStore`:
``list_sessions`` returns metadata only (empty ``items_json``); ``get_session``
returns the full row. Process-local, not durable.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import replace

from inqtrix.project.agent_sessions_ports import (
    AgentSession,
    AgentSessionGroup,
    AgentSessionGroupNotFound,
    AgentSessionNotFound,
)
from inqtrix.project.scoped_upsert import ResourceScope, require_memory_scope


class MemoryAgentSessionStore:
    """Process-local
    :class:`~inqtrix.project.agent_sessions_ports.AgentSessionStore`."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, AgentSession] = {}
        self._groups: dict[str, AgentSessionGroup] = {}

    async def claim_session(
        self, *, id: str, title: str, created_at: float,
        created_by_user_id: uuid.UUID | None, workspace_id: str | None,
    ) -> AgentSession:
        with self._lock:
            existing = self._sessions.get(id)
            if existing is not None:
                return existing
            session = AgentSession(
                id=id,
                title=title,
                items_json="[]",
                created_at=created_at,
                updated_at=created_at,
                created_by_user_id=created_by_user_id,
                workspace_id=workspace_id,
            )
            self._sessions[id] = session
            return session

    async def upsert_session(
        self, *, id: str, title: str, items_json: str, group_id: str | None,
        created_at: float,
        updated_at: float,
        created_by_user_id: uuid.UUID | None,
        workspace_id: str | None,
    ) -> AgentSession:
        with self._lock:
            if group_id is not None:
                require_memory_scope(
                    self._groups.get(group_id),
                    created_by_user_id=created_by_user_id,
                    workspace_id=workspace_id,
                    resource_id=group_id,
                    not_found=AgentSessionGroupNotFound,
                )
            existing = self._sessions.get(id)
            if existing is not None:
                require_memory_scope(
                    existing,
                    created_by_user_id=created_by_user_id,
                    workspace_id=workspace_id,
                    resource_id=id,
                    not_found=AgentSessionNotFound,
                )
                session = replace(
                    existing, title=title, items_json=items_json, group_id=group_id,
                    updated_at=updated_at,
                )
            else:
                session = AgentSession(
                    id=id, title=title, items_json=items_json, created_at=created_at,
                    group_id=group_id, updated_at=updated_at, created_by_user_id=created_by_user_id,
                    workspace_id=workspace_id,
                )
            self._sessions[id] = session
            return session

    async def list_sessions(
        self, *, created_by_user_id: uuid.UUID | None, workspace_id: str | None
    ) -> list[AgentSession]:
        with self._lock:
            items = _scoped(self._sessions.values(), created_by_user_id, workspace_id)
        items.sort(key=lambda s: (s.updated_at, s.id), reverse=True)
        # Metadata only — the items body loads via get_session.
        return [replace(session, items_json="[]") for session in items]

    async def get_session(self, session_id: str) -> AgentSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise AgentSessionNotFound(session_id)
        return session

    async def delete_session(
        self, session_id: str, *, scope: ResourceScope
    ) -> None:
        with self._lock:
            require_memory_scope(
                self._sessions.get(session_id),
                created_by_user_id=scope.created_by_user_id,
                workspace_id=scope.workspace_id,
                resource_id=session_id,
                not_found=AgentSessionNotFound,
            )
            self._sessions.pop(session_id, None)

    async def upsert_group(
        self, *, id: str, title: str, created_at: float, updated_at: float,
        created_by_user_id: uuid.UUID | None, workspace_id: str | None,
    ) -> AgentSessionGroup:
        with self._lock:
            existing = self._groups.get(id)
            if existing is not None:
                require_memory_scope(
                    existing,
                    created_by_user_id=created_by_user_id,
                    workspace_id=workspace_id,
                    resource_id=id,
                    not_found=AgentSessionGroupNotFound,
                )
                group = replace(existing, title=title, updated_at=updated_at)
            else:
                group = AgentSessionGroup(
                    id=id, title=title, created_at=created_at,
                    updated_at=updated_at, created_by_user_id=created_by_user_id,
                    workspace_id=workspace_id,
                )
            self._groups[id] = group
            return group

    async def claim_group(
        self, *, id: str, title: str, created_at: float,
        created_by_user_id: uuid.UUID | None, workspace_id: str | None,
    ) -> AgentSessionGroup:
        with self._lock:
            existing = self._groups.get(id)
            if existing is not None:
                return existing
            group = AgentSessionGroup(
                id=id, title=title, created_at=created_at,
                updated_at=created_at, created_by_user_id=created_by_user_id,
                workspace_id=workspace_id,
            )
            self._groups[id] = group
            return group

    async def list_groups(
        self, *, created_by_user_id: uuid.UUID | None, workspace_id: str | None
    ) -> list[AgentSessionGroup]:
        with self._lock:
            items = _scoped(
                self._groups.values(), created_by_user_id, workspace_id
            )
        items.sort(key=lambda group: (group.created_at, group.id), reverse=True)
        return items

    async def delete_group(
        self, group_id: str, *, scope: ResourceScope
    ) -> None:
        with self._lock:
            require_memory_scope(
                self._groups.get(group_id),
                created_by_user_id=scope.created_by_user_id,
                workspace_id=scope.workspace_id,
                resource_id=group_id,
                not_found=AgentSessionGroupNotFound,
            )
            self._groups.pop(group_id, None)
            for session_id, session in list(self._sessions.items()):
                if session.group_id == group_id:
                    self._sessions[session_id] = replace(
                        session, group_id=None
                    )

    async def aclose(self) -> None:
        return None


def _scoped(
    values,
    created_by_user_id: uuid.UUID | None,
    workspace_id: str | None,
):
    items = list(values)
    if created_by_user_id is not None:
        items = [i for i in items if i.created_by_user_id == created_by_user_id]
    if workspace_id is not None:
        items = [i for i in items if i.workspace_id == workspace_id]
    return items
