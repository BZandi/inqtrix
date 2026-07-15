"""Knowledge-session persistence service (Wissensmodus saved sessions).

The Ask-view counterpart of the chat-history service: a thin owner-scoped layer
over a :class:`~inqtrix.project.knowledge_sessions_ports.KnowledgeSessionStore`.
Sessions are PRIVATE per user (no sharing surface, like the asset tier in M6c);
every denial is the indistinct not-found.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from inqtrix.auth.permissions import require_owned_access
from inqtrix.project.knowledge_sessions_ports import (
    KnowledgeSession,
    KnowledgeSessionGroup,
    KnowledgeSessionGroupNotFound,
    KnowledgeSessionNotFound,
    KnowledgeSessionStore,
)
from inqtrix.project.scoped_upsert import ResourceScope
from inqtrix.services.workspace_guard import deny_cross_workspace

if TYPE_CHECKING:
    from inqtrix.auth.principal import UserContext


class KnowledgeSessionsService:
    """Application service over a :class:`KnowledgeSessionStore`."""

    def __init__(self, *, store: KnowledgeSessionStore, durable: bool = False) -> None:
        self._store = store
        self._durable = durable

    @property
    def store(self) -> KnowledgeSessionStore:
        return self._store

    @property
    def durable(self) -> bool:
        return self._durable

    async def list_sessions(
        self, *, caller_user_id: uuid.UUID | None, workspace_id: str | None
    ) -> list[KnowledgeSession]:
        return await self._store.list_sessions(
            created_by_user_id=caller_user_id, workspace_id=workspace_id
        )

    async def get_session(
        self, session_id: str, *, visible_to: "UserContext | None"
    ) -> KnowledgeSession:
        session = await self._store.get_session(session_id)
        require_owned_access(
            owner_user_id=session.created_by_user_id, resource_tenant_id=session.tenant_id,
            resource_id=session.id, visible_to=visible_to,
            not_found=KnowledgeSessionNotFound,
        )
        return session

    async def save_session(
        self, *, id: str, title: str, items_json: str, group_id: str | None,
        created_at: float,
        updated_at: float,
        caller_user_id: uuid.UUID | None,
        workspace_id: str | None, visible_to: "UserContext | None",
    ) -> KnowledgeSession:
        try:
            existing = await self._store.get_session(id)
        except KnowledgeSessionNotFound:
            existing = None
        if existing is not None:
            require_owned_access(
                owner_user_id=existing.created_by_user_id, resource_tenant_id=existing.tenant_id,
                resource_id=existing.id, visible_to=visible_to,
                not_found=KnowledgeSessionNotFound,
            )
            owner_user_id, owner_ws = existing.created_by_user_id, existing.workspace_id
        else:
            owner_user_id, owner_ws = caller_user_id, workspace_id
        return await self._store.upsert_session(
            id=id, title=title, items_json=items_json, group_id=group_id,
            created_at=created_at, updated_at=updated_at, created_by_user_id=owner_user_id,
            workspace_id=owner_ws,
        )

    async def delete_session(
        self, session_id, *, visible_to: "UserContext | None",
        request_workspace_id=None,
    ) -> None:
        session = await self._store.get_session(session_id)
        require_owned_access(
            owner_user_id=session.created_by_user_id, resource_tenant_id=session.tenant_id,
            resource_id=session.id, visible_to=visible_to,
            not_found=KnowledgeSessionNotFound,
        )
        deny_cross_workspace(
            resource_workspace_id=session.workspace_id,
            request_workspace_id=request_workspace_id,
            not_found=lambda: KnowledgeSessionNotFound(session_id),
        )
        await self._store.delete_session(
            session_id, scope=ResourceScope.from_record(session)
        )

    async def save_group(
        self, *, id: str, title: str, created_at: float, updated_at: float,
        caller_user_id: uuid.UUID | None, workspace_id: str | None,
        visible_to: "UserContext | None",
    ) -> KnowledgeSessionGroup:
        existing = None
        for group in await self._store.list_groups(created_by_user_id=None, workspace_id=None):
            if group.id == id:
                existing = group
                break
        if existing is not None:
            require_owned_access(
                owner_user_id=existing.created_by_user_id,
                resource_tenant_id=existing.tenant_id,
                resource_id=existing.id,
                visible_to=visible_to,
                not_found=KnowledgeSessionGroupNotFound,
            )
            owner_user_id, owner_ws = existing.created_by_user_id, existing.workspace_id
        else:
            owner_user_id, owner_ws = caller_user_id, workspace_id
        return await self._store.upsert_group(
            id=id, title=title, created_at=created_at, updated_at=updated_at,
            created_by_user_id=owner_user_id, workspace_id=owner_ws,
        )

    async def list_groups(
        self, *, caller_user_id: uuid.UUID | None, workspace_id: str | None
    ) -> list[KnowledgeSessionGroup]:
        return await self._store.list_groups(
            created_by_user_id=caller_user_id, workspace_id=workspace_id
        )

    async def delete_group(
        self, group_id: str, *, visible_to: "UserContext | None",
        request_workspace_id=None,
    ) -> None:
        existing = None
        for group in await self._store.list_groups(created_by_user_id=None, workspace_id=None):
            if group.id == group_id:
                existing = group
                break
        if existing is None:
            raise KnowledgeSessionGroupNotFound(group_id)
        require_owned_access(
            owner_user_id=existing.created_by_user_id,
            resource_tenant_id=existing.tenant_id,
            resource_id=existing.id,
            visible_to=visible_to,
            not_found=KnowledgeSessionGroupNotFound,
        )
        deny_cross_workspace(
            resource_workspace_id=existing.workspace_id,
            request_workspace_id=request_workspace_id,
            not_found=lambda: KnowledgeSessionGroupNotFound(group_id),
        )
        await self._store.delete_group(
            group_id, scope=ResourceScope.from_record(existing)
        )
