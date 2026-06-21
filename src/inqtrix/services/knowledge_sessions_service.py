"""Knowledge-session persistence service (Wissensmodus saved sessions).

The Ask-view counterpart of the chat-history service: a thin owner-scoped layer
over a :class:`~inqtrix.project.knowledge_sessions_ports.KnowledgeSessionStore`.
Sessions are PRIVATE per user (no sharing surface, like the asset tier in M6c);
every denial is the indistinct not-found.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from inqtrix.auth.permissions import resolve_owned_access
from inqtrix.project.knowledge_sessions_ports import (
    KnowledgeSession,
    KnowledgeSessionGroup,
    KnowledgeSessionGroupNotFound,
    KnowledgeSessionNotFound,
    KnowledgeSessionStore,
)
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
        self, *, caller_sub: str | None, workspace_id: str | None
    ) -> list[KnowledgeSession]:
        return await self._store.list_sessions(
            created_by_sub=caller_sub, workspace_id=workspace_id
        )

    async def get_session(
        self, session_id: str, *, visible_to: "UserContext | None", also_visible=None
    ) -> KnowledgeSession:
        session = await self._store.get_session(session_id)
        shared = resolve_owned_access(
            owner_sub=session.created_by_sub, resource_tenant_id=session.tenant_id,
            resource_id=session.id, visible_to=visible_to, also_visible=also_visible,
            not_found=KnowledgeSessionNotFound,
        )
        if shared is not None:
            raise KnowledgeSessionNotFound(session_id)
        return session

    async def save_session(
        self, *, id: str, title: str, items_json: str, group_id: str | None,
        created_at: float, updated_at: float, caller_sub: str | None,
        workspace_id: str | None, visible_to: "UserContext | None",
    ) -> KnowledgeSession:
        try:
            existing = await self._store.get_session(id)
        except KnowledgeSessionNotFound:
            existing = None
        if existing is not None:
            shared = resolve_owned_access(
                owner_sub=existing.created_by_sub, resource_tenant_id=existing.tenant_id,
                resource_id=existing.id, visible_to=visible_to, also_visible=None,
                not_found=KnowledgeSessionNotFound,
            )
            if shared is not None:
                raise KnowledgeSessionNotFound(id)
            owner_sub, owner_ws = existing.created_by_sub, existing.workspace_id
        else:
            owner_sub, owner_ws = caller_sub, workspace_id
        if group_id is not None:
            await self._require_group_for_owner(
                group_id, owner_sub=owner_sub, owner_workspace=owner_ws,
                visible_to=visible_to,
            )
        return await self._store.upsert_session(
            id=id, title=title, items_json=items_json, group_id=group_id,
            created_at=created_at, updated_at=updated_at, created_by_sub=owner_sub,
            workspace_id=owner_ws,
        )

    async def delete_session(
        self, session_id, *, visible_to: "UserContext | None", also_visible=None,
        request_workspace_id=None,
    ) -> None:
        session = await self._store.get_session(session_id)
        shared = resolve_owned_access(
            owner_sub=session.created_by_sub, resource_tenant_id=session.tenant_id,
            resource_id=session.id, visible_to=visible_to, also_visible=also_visible,
            not_found=KnowledgeSessionNotFound,
        )
        if shared is not None:
            raise KnowledgeSessionNotFound(session_id)
        deny_cross_workspace(
            resource_workspace_id=session.workspace_id,
            request_workspace_id=request_workspace_id,
            not_found=lambda: KnowledgeSessionNotFound(session_id),
        )
        await self._store.delete_session(session_id)

    async def save_group(
        self, *, id: str, title: str, created_at: float, updated_at: float,
        caller_sub: str | None, workspace_id: str | None,
        visible_to: "UserContext | None",
    ) -> KnowledgeSessionGroup:
        existing = None
        for group in await self._store.list_groups(created_by_sub=None, workspace_id=None):
            if group.id == id:
                existing = group
                break
        if existing is not None:
            shared = resolve_owned_access(
                owner_sub=existing.created_by_sub,
                resource_tenant_id=existing.tenant_id,
                resource_id=existing.id,
                visible_to=visible_to,
                also_visible=None,
                not_found=KnowledgeSessionGroupNotFound,
            )
            if shared is not None:
                raise KnowledgeSessionGroupNotFound(id)
            owner_sub, owner_ws = existing.created_by_sub, existing.workspace_id
        else:
            owner_sub, owner_ws = caller_sub, workspace_id
        return await self._store.upsert_group(
            id=id, title=title, created_at=created_at, updated_at=updated_at,
            created_by_sub=owner_sub, workspace_id=owner_ws,
        )

    async def list_groups(
        self, *, caller_sub: str | None, workspace_id: str | None
    ) -> list[KnowledgeSessionGroup]:
        return await self._store.list_groups(
            created_by_sub=caller_sub, workspace_id=workspace_id
        )

    async def delete_group(
        self, group_id: str, *, visible_to: "UserContext | None",
        request_workspace_id=None,
    ) -> None:
        existing = None
        for group in await self._store.list_groups(created_by_sub=None, workspace_id=None):
            if group.id == group_id:
                existing = group
                break
        if existing is None:
            raise KnowledgeSessionGroupNotFound(group_id)
        shared = resolve_owned_access(
            owner_sub=existing.created_by_sub,
            resource_tenant_id=existing.tenant_id,
            resource_id=existing.id,
            visible_to=visible_to,
            also_visible=None,
            not_found=KnowledgeSessionGroupNotFound,
        )
        if shared is not None:
            raise KnowledgeSessionGroupNotFound(group_id)
        deny_cross_workspace(
            resource_workspace_id=existing.workspace_id,
            request_workspace_id=request_workspace_id,
            not_found=lambda: KnowledgeSessionGroupNotFound(group_id),
        )
        await self._store.delete_group(group_id)

    async def _require_group_for_owner(
        self, group_id: str, *, owner_sub: str | None, owner_workspace: str | None,
        visible_to: "UserContext | None",
    ) -> None:
        group = None
        for candidate in await self._store.list_groups(created_by_sub=None, workspace_id=None):
            if candidate.id == group_id:
                group = candidate
                break
        if group is None:
            raise KnowledgeSessionGroupNotFound(group_id)
        shared = resolve_owned_access(
            owner_sub=group.created_by_sub,
            resource_tenant_id=group.tenant_id,
            resource_id=group.id,
            visible_to=visible_to,
            also_visible=None,
            not_found=KnowledgeSessionGroupNotFound,
        )
        if (
            shared is not None
            or group.created_by_sub != owner_sub
            or group.workspace_id != owner_workspace
        ):
            raise KnowledgeSessionGroupNotFound(group_id)
