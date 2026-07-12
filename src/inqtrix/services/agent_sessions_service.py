"""Agent-session persistence service (Agent-Desk saved sessions).

The Agent-Desk counterpart of the chat-history service: a thin owner-scoped layer
over a :class:`~inqtrix.project.agent_sessions_ports.AgentSessionStore`.
Sessions are PRIVATE per user (no sharing surface, like the asset tier in M6c);
every denial is the indistinct not-found.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from inqtrix.auth.permissions import resolve_owned_access
from inqtrix.project.agent_sessions_ports import (
    AgentSession,
    AgentSessionGroup,
    AgentSessionGroupNotFound,
    AgentSessionNotFound,
    AgentSessionStore,
)
from inqtrix.services.workspace_guard import deny_cross_workspace

if TYPE_CHECKING:
    from inqtrix.auth.principal import UserContext
    from inqtrix.runs.ports import RunStorePort


class AgentSessionsService:
    """Application service over a :class:`AgentSessionStore`."""

    def __init__(
        self,
        *,
        store: AgentSessionStore,
        run_store: "RunStorePort | None" = None,
        durable: bool = False,
    ) -> None:
        self._store = store
        self._run_store = run_store
        self._durable = durable

    @property
    def store(self) -> AgentSessionStore:
        return self._store

    @property
    def durable(self) -> bool:
        return self._durable

    async def claim_session(
        self,
        session_id: str,
        *,
        title: str,
        caller_sub: str | None,
        workspace_id: str | None,
        visible_to: "UserContext | None",
        created_at: float | None = None,
    ) -> AgentSession:
        """Atomically claim or validate an agent execution session.

        A deleted registry row does not erase its historical ownership: when
        runs with the same id remain, only their one unambiguous owner may
        recreate it.  The final store claim is insert-if-absent, and the
        returned row is checked again so concurrent foreign claimants cannot
        win through a check/insert race.
        """
        owners = (
            self._run_store.session_owners(session_id)
            if self._run_store is not None
            else set()
        )
        try:
            existing = await self._store.get_session(session_id)
        except AgentSessionNotFound:
            existing = None
        if existing is not None:
            self._require_owner(existing, visible_to=visible_to)
            if owners and visible_to is not None:
                if len(owners) != 1:
                    raise AgentSessionNotFound(session_id)
                historical_tenant, historical_sub = next(iter(owners))
                if (
                    existing.created_by_sub != historical_sub
                    or historical_tenant
                    not in (None, existing.tenant_id)
                ):
                    raise AgentSessionNotFound(session_id)
            return existing

        if owners:
            if len(owners) != 1:
                raise AgentSessionNotFound(session_id)
            historical_tenant, historical_sub = next(iter(owners))
            if visible_to is not None:
                principal = visible_to.principal
                if (
                    historical_sub != principal.sub
                    or historical_tenant not in (None, principal.tenant_id)
                ):
                    raise AgentSessionNotFound(session_id)

        claimed = await self._store.claim_session(
            id=session_id,
            title=title,
            created_at=created_at if created_at is not None else time.time(),
            created_by_sub=caller_sub,
            workspace_id=workspace_id,
        )
        self._require_owner(claimed, visible_to=visible_to)
        return claimed

    async def list_sessions(
        self, *, caller_sub: str | None, workspace_id: str | None
    ) -> list[AgentSession]:
        return await self._store.list_sessions(
            created_by_sub=caller_sub, workspace_id=workspace_id
        )

    async def get_session(
        self, session_id: str, *, visible_to: "UserContext | None", also_visible=None
    ) -> AgentSession:
        session = await self._store.get_session(session_id)
        shared = resolve_owned_access(
            owner_sub=session.created_by_sub, resource_tenant_id=session.tenant_id,
            resource_id=session.id, visible_to=visible_to, also_visible=also_visible,
            not_found=AgentSessionNotFound,
        )
        if shared is not None:
            raise AgentSessionNotFound(session_id)
        return session

    async def save_session(
        self, *, id: str, title: str, items_json: str, group_id: str | None,
        created_at: float, updated_at: float, caller_sub: str | None,
        workspace_id: str | None, visible_to: "UserContext | None",
    ) -> AgentSession:
        claimed = await self.claim_session(
            id,
            title=title,
            caller_sub=caller_sub,
            workspace_id=workspace_id,
            visible_to=visible_to,
            created_at=created_at,
        )
        owner_sub, owner_ws = claimed.created_by_sub, claimed.workspace_id
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
            not_found=AgentSessionNotFound,
        )
        if shared is not None:
            raise AgentSessionNotFound(session_id)
        deny_cross_workspace(
            resource_workspace_id=session.workspace_id,
            request_workspace_id=request_workspace_id,
            not_found=lambda: AgentSessionNotFound(session_id),
        )
        await self._store.delete_session(session_id)

    async def save_group(
        self, *, id: str, title: str, created_at: float, updated_at: float,
        caller_sub: str | None, workspace_id: str | None,
        visible_to: "UserContext | None",
    ) -> AgentSessionGroup:
        claimed = await self._store.claim_group(
            id=id,
            title=title,
            created_at=created_at,
            created_by_sub=caller_sub,
            workspace_id=workspace_id,
        )
        self._require_group_owner(claimed, visible_to=visible_to)
        owner_sub, owner_ws = claimed.created_by_sub, claimed.workspace_id
        return await self._store.upsert_group(
            id=id, title=title, created_at=created_at, updated_at=updated_at,
            created_by_sub=owner_sub, workspace_id=owner_ws,
        )

    async def list_groups(
        self, *, caller_sub: str | None, workspace_id: str | None
    ) -> list[AgentSessionGroup]:
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
            raise AgentSessionGroupNotFound(group_id)
        shared = resolve_owned_access(
            owner_sub=existing.created_by_sub,
            resource_tenant_id=existing.tenant_id,
            resource_id=existing.id,
            visible_to=visible_to,
            also_visible=None,
            not_found=AgentSessionGroupNotFound,
        )
        if shared is not None:
            raise AgentSessionGroupNotFound(group_id)
        deny_cross_workspace(
            resource_workspace_id=existing.workspace_id,
            request_workspace_id=request_workspace_id,
            not_found=lambda: AgentSessionGroupNotFound(group_id),
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
            raise AgentSessionGroupNotFound(group_id)
        shared = resolve_owned_access(
            owner_sub=group.created_by_sub,
            resource_tenant_id=group.tenant_id,
            resource_id=group.id,
            visible_to=visible_to,
            also_visible=None,
            not_found=AgentSessionGroupNotFound,
        )
        if (
            shared is not None
            or group.created_by_sub != owner_sub
            or group.workspace_id != owner_workspace
        ):
            raise AgentSessionGroupNotFound(group_id)

    @staticmethod
    def _require_owner(
        session: AgentSession, *, visible_to: "UserContext | None"
    ) -> None:
        shared = resolve_owned_access(
            owner_sub=session.created_by_sub,
            resource_tenant_id=session.tenant_id,
            resource_id=session.id,
            visible_to=visible_to,
            also_visible=None,
            not_found=AgentSessionNotFound,
        )
        if shared is not None:
            raise AgentSessionNotFound(session.id)

    @staticmethod
    def _require_group_owner(
        group: AgentSessionGroup, *, visible_to: "UserContext | None"
    ) -> None:
        shared = resolve_owned_access(
            owner_sub=group.created_by_sub,
            resource_tenant_id=group.tenant_id,
            resource_id=group.id,
            visible_to=visible_to,
            also_visible=None,
            not_found=AgentSessionGroupNotFound,
        )
        if shared is not None:
            raise AgentSessionGroupNotFound(group.id)
