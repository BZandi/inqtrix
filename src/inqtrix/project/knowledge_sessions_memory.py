"""In-memory knowledge-session store (tier without Postgres + offline test).

Mirrors :class:`~inqtrix.project.knowledge_sessions_postgres.PostgresKnowledgeSessionStore`:
``list_sessions`` returns metadata only (empty ``items_json``); ``get_session``
returns the full row. Process-local, not durable.
"""

from __future__ import annotations

from dataclasses import replace

from inqtrix.project.knowledge_sessions_ports import (
    KnowledgeSession,
    KnowledgeSessionGroup,
    KnowledgeSessionNotFound,
)


class MemoryKnowledgeSessionStore:
    """Process-local
    :class:`~inqtrix.project.knowledge_sessions_ports.KnowledgeSessionStore`."""

    def __init__(self) -> None:
        self._sessions: dict[str, KnowledgeSession] = {}
        self._groups: dict[str, KnowledgeSessionGroup] = {}

    async def upsert_session(
        self, *, id: str, title: str, items_json: str, group_id: str | None,
        created_at: float, updated_at: float, created_by_sub: str | None,
        workspace_id: str | None,
    ) -> KnowledgeSession:
        existing = self._sessions.get(id)
        if existing is not None:
            session = replace(
                existing, title=title, items_json=items_json, group_id=group_id,
                updated_at=updated_at,
            )
        else:
            session = KnowledgeSession(
                id=id, title=title, items_json=items_json, created_at=created_at,
                group_id=group_id, updated_at=updated_at, created_by_sub=created_by_sub,
                workspace_id=workspace_id,
            )
        self._sessions[id] = session
        return session

    async def list_sessions(
        self, *, created_by_sub: str | None, workspace_id: str | None
    ) -> list[KnowledgeSession]:
        items = _scoped(self._sessions.values(), created_by_sub, workspace_id)
        items.sort(key=lambda s: (s.updated_at, s.id), reverse=True)
        # Metadata only — the items body loads via get_session.
        return [replace(session, items_json="[]") for session in items]

    async def get_session(self, session_id: str) -> KnowledgeSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KnowledgeSessionNotFound(session_id)
        return session

    async def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def upsert_group(
        self, *, id: str, title: str, created_at: float, updated_at: float,
        created_by_sub: str | None, workspace_id: str | None,
    ) -> KnowledgeSessionGroup:
        existing = self._groups.get(id)
        if existing is not None:
            group = replace(existing, title=title, updated_at=updated_at)
        else:
            group = KnowledgeSessionGroup(
                id=id, title=title, created_at=created_at, updated_at=updated_at,
                created_by_sub=created_by_sub, workspace_id=workspace_id,
            )
        self._groups[id] = group
        return group

    async def list_groups(
        self, *, created_by_sub: str | None, workspace_id: str | None
    ) -> list[KnowledgeSessionGroup]:
        items = _scoped(self._groups.values(), created_by_sub, workspace_id)
        items.sort(key=lambda group: (group.created_at, group.id), reverse=True)
        return items

    async def delete_group(self, group_id: str) -> None:
        self._groups.pop(group_id, None)
        for session_id, session in list(self._sessions.items()):
            if session.group_id == group_id:
                self._sessions[session_id] = replace(session, group_id=None)

    async def aclose(self) -> None:
        return None


def _scoped(
    values,
    created_by_sub: str | None,
    workspace_id: str | None,
):
    items = list(values)
    if created_by_sub is not None:
        items = [i for i in items if i.created_by_sub == created_by_sub]
    if workspace_id is not None:
        items = [i for i in items if i.workspace_id == workspace_id]
    return items
