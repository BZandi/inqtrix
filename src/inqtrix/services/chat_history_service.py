"""Chat-history persistence service (M6a project tier).

Owns what the chat-history router delegates: payload validation, the
owner/share access rule, and the resolution of "which threads belong to
this caller" before the store's keyset query runs. Distinct from the AI
:class:`~inqtrix.services.chat_service.ChatService` (completions) — this
service never calls a model; it persists and reads the conversation
record.

Ownership model (the established sharing rule, reused verbatim):
threads/groups carry ``created_by_sub``. ``None`` (anonymous/static
principals) stays visible to every caller — the compatibility rule.
Owned threads are visible to their creator and to share recipients (the
router resolves grants into ``also_visible``); messages inherit
visibility from their parent thread. Saving a thread/message through a
share needs at least an edit grant; deleting stays owner-only. Every
denial is the indistinct :class:`ThreadNotFound`/
:class:`ThreadGroupNotFound` (existence is not disclosed) — the same
hide-on-deny rule as collections.
"""

from __future__ import annotations

from typing import Any, Mapping, TYPE_CHECKING

from inqtrix.auth.permissions import SharePermission, resolve_owned_access
from inqtrix.services.workspace_guard import deny_cross_workspace
from inqtrix.project.chat_ports import (
    ChatMessage,
    ChatStore,
    ChatThread,
    ChatThreadGroup,
    ThreadGroupNotFound,
    ThreadNotFound,
)

if TYPE_CHECKING:
    from inqtrix.auth.principal import UserContext

_VALID_ROLES = frozenset({"user", "assistant"})
_VALID_SOURCES = frozenset({"api", "imported", "mock"})


class ChatValidationError(ValueError):
    """Raised for client-payload problems (maps to HTTP 400).

    The role/source domain is rejected here, before the database CHECK
    constraint, so a bad value is a clean 400 instead of an opaque 500
    (No Silent Fallbacks: the failure is visible and attributable)."""


class ChatHistoryService:
    """Application service over a :class:`ChatStore`.

    Args:
        store: The wired chat store (memory or Postgres).
        durable: Whether the store survives a restart (Postgres). The
            capability manifest advertises project persistence only when
            this is ``True`` — a volatile store must not invite the
            frontend to abandon its durable local markdown project for an
            ephemeral server tier (the prompt-template ``durable`` rule).
    """

    def __init__(self, *, store: ChatStore, durable: bool = False) -> None:
        self._store = store
        self._durable = durable

    @property
    def store(self) -> ChatStore:
        """The wired chat store (shutdown disposes its engine)."""
        return self._store

    @property
    def durable(self) -> bool:
        """Whether the backing store survives a restart (capability gate)."""
        return self._durable

    # -- threads ---------------------------------------------------------- #

    async def save_thread(
        self,
        *,
        id: str,
        title: str,
        preview: str,
        source: str,
        group_id: str | None,
        created_at: float,
        updated_at: float,
        caller_sub: str | None,
        workspace_id: str | None,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> ChatThread:
        """Create or update a thread (idempotent autosave).

        A new id is owned by the caller; an existing id requires at least
        an edit grant and keeps its original owner/workspace.
        """
        if source not in _VALID_SOURCES:
            raise ChatValidationError(f"unknown thread source: {source!r}")
        try:
            existing = await self._store.get_thread(id)
        except ThreadNotFound:
            existing = None
        if existing is not None:
            shared = resolve_owned_access(
                owner_sub=existing.created_by_sub,
                resource_tenant_id=existing.tenant_id,
                resource_id=existing.id,
                visible_to=visible_to,
                also_visible=also_visible,
                not_found=ThreadNotFound,
            )
            if shared is not None and not shared.at_least(SharePermission.EDIT):
                raise ThreadNotFound(id)
            owner_sub = existing.created_by_sub
            owner_workspace = existing.workspace_id
        else:
            owner_sub = caller_sub
            owner_workspace = workspace_id
        return await self._store.upsert_thread(
            id=id,
            title=title,
            preview=preview,
            source=source,
            group_id=group_id,
            created_at=created_at,
            updated_at=updated_at,
            created_by_sub=owner_sub,
            workspace_id=owner_workspace,
        )

    async def list_threads(
        self,
        *,
        caller_sub: str | None,
        workspace_id: str | None,
        limit: int,
        after: tuple[float, str] | None,
    ) -> tuple[list[ChatThread], str | None]:
        """One keyset page of the caller's own threads (newest first)."""
        return await self._store.list_threads_page(
            created_by_sub=caller_sub,
            workspace_id=workspace_id,
            limit=limit,
            after=after,
        )

    async def get_thread(
        self,
        thread_id: str,
        *,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> ChatThread:
        """One thread the caller may see, or :class:`ThreadNotFound`."""
        thread = await self._store.get_thread(thread_id)
        resolve_owned_access(
            owner_sub=thread.created_by_sub,
            resource_tenant_id=thread.tenant_id,
            resource_id=thread.id,
            visible_to=visible_to,
            also_visible=also_visible,
            not_found=ThreadNotFound,
        )
        return thread

    async def delete_thread(
        self,
        thread_id: str,
        *,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
        request_workspace_id: str | None = None,
    ) -> None:
        """Delete a thread (owner-only) with its messages (cascade)."""
        thread = await self._store.get_thread(thread_id)
        shared = resolve_owned_access(
            owner_sub=thread.created_by_sub,
            resource_tenant_id=thread.tenant_id,
            resource_id=thread.id,
            visible_to=visible_to,
            also_visible=also_visible,
            not_found=ThreadNotFound,
        )
        if shared is not None:
            raise ThreadNotFound(thread_id)
        deny_cross_workspace(
            resource_workspace_id=thread.workspace_id,
            request_workspace_id=request_workspace_id,
            not_found=lambda: ThreadNotFound(thread_id),
        )
        await self._store.delete_thread(thread_id)

    # -- messages --------------------------------------------------------- #

    async def append_messages(
        self,
        thread_id: str,
        *,
        messages: list[dict[str, Any]],
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> list[ChatMessage]:
        """Append/upsert messages into a thread the caller may edit."""
        thread = await self._store.get_thread(thread_id)
        shared = resolve_owned_access(
            owner_sub=thread.created_by_sub,
            resource_tenant_id=thread.tenant_id,
            resource_id=thread.id,
            visible_to=visible_to,
            also_visible=also_visible,
            not_found=ThreadNotFound,
        )
        if shared is not None and not shared.at_least(SharePermission.EDIT):
            raise ThreadNotFound(thread_id)
        parsed = [self._parse_message(thread_id, raw) for raw in messages]
        return await self._store.append_messages(parsed)

    async def delete_message(
        self,
        thread_id: str,
        message_id: str,
        *,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
        request_workspace_id: str | None = None,
    ) -> None:
        """Delete one message from a thread the caller may edit.

        Editing access, not owner-only: deleting a message is the inverse
        of appending one, so it mirrors :meth:`append_messages` (owner or
        at-least-edit share) rather than the owner-only thread/group
        delete. Only an inaccessible/unknown thread raises (the indistinct
        :class:`ThreadNotFound`, hide-on-deny preserved); a missing
        message is the store's no-op (the idempotency rule the port
        documents).
        """
        thread = await self._store.get_thread(thread_id)
        shared = resolve_owned_access(
            owner_sub=thread.created_by_sub,
            resource_tenant_id=thread.tenant_id,
            resource_id=thread.id,
            visible_to=visible_to,
            also_visible=also_visible,
            not_found=ThreadNotFound,
        )
        if shared is not None and not shared.at_least(SharePermission.EDIT):
            raise ThreadNotFound(thread_id)
        deny_cross_workspace(
            resource_workspace_id=thread.workspace_id,
            request_workspace_id=request_workspace_id,
            not_found=lambda: ThreadNotFound(thread_id),
        )
        await self._store.delete_message(thread_id, message_id)

    async def list_messages(
        self,
        thread_id: str,
        *,
        limit: int,
        after: tuple[float, str] | None,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> tuple[list[ChatMessage], str | None]:
        """One keyset page of a readable thread's messages (newest first)."""
        await self.get_thread(
            thread_id, visible_to=visible_to, also_visible=also_visible
        )
        return await self._store.list_messages_page(
            thread_id, limit=limit, after=after
        )

    # -- groups ----------------------------------------------------------- #

    async def save_group(
        self,
        *,
        id: str,
        title: str,
        created_at: float,
        updated_at: float,
        caller_sub: str | None,
        workspace_id: str | None,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> ChatThreadGroup:
        """Create or update a thread group (idempotent)."""
        existing = None
        for group in await self._store.list_groups(
            created_by_sub=None, workspace_id=None
        ):
            if group.id == id:
                existing = group
                break
        if existing is not None:
            resolve_owned_access(
                owner_sub=existing.created_by_sub,
                resource_tenant_id=existing.tenant_id,
                resource_id=existing.id,
                visible_to=visible_to,
                also_visible=also_visible,
                not_found=ThreadGroupNotFound,
            )
            owner_sub = existing.created_by_sub
            owner_workspace = existing.workspace_id
        else:
            owner_sub = caller_sub
            owner_workspace = workspace_id
        return await self._store.upsert_group(
            id=id,
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            created_by_sub=owner_sub,
            workspace_id=owner_workspace,
        )

    async def list_groups(
        self,
        *,
        caller_sub: str | None,
        workspace_id: str | None,
    ) -> list[ChatThreadGroup]:
        """All of the caller's thread groups (newest first)."""
        return await self._store.list_groups(
            created_by_sub=caller_sub, workspace_id=workspace_id
        )

    async def delete_group(
        self,
        group_id: str,
        *,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
        request_workspace_id: str | None = None,
    ) -> None:
        """Delete a group (its threads orphan to ungrouped)."""
        existing = None
        for group in await self._store.list_groups(
            created_by_sub=None, workspace_id=None
        ):
            if group.id == group_id:
                existing = group
                break
        if existing is None:
            raise ThreadGroupNotFound(group_id)
        shared = resolve_owned_access(
            owner_sub=existing.created_by_sub,
            resource_tenant_id=existing.tenant_id,
            resource_id=existing.id,
            visible_to=visible_to,
            also_visible=also_visible,
            not_found=ThreadGroupNotFound,
        )
        if shared is not None:
            raise ThreadGroupNotFound(group_id)
        deny_cross_workspace(
            resource_workspace_id=existing.workspace_id,
            request_workspace_id=request_workspace_id,
            not_found=lambda: ThreadGroupNotFound(group_id),
        )
        await self._store.delete_group(group_id)

    # -- helpers ---------------------------------------------------------- #

    @staticmethod
    def _parse_message(thread_id: str, raw: dict[str, Any]) -> ChatMessage:
        message_id = raw.get("id")
        role = raw.get("role")
        if not isinstance(message_id, str) or not message_id:
            raise ChatValidationError("message id is required")
        if role not in _VALID_ROLES:
            raise ChatValidationError(f"unknown message role: {role!r}")
        content = raw.get("content_markdown", "")
        if not isinstance(content, str):
            raise ChatValidationError("content_markdown must be a string")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ChatValidationError("message metadata must be an object")
        created_at = raw.get("created_at")
        if not isinstance(created_at, (int, float)):
            raise ChatValidationError("message created_at must be a number")
        return ChatMessage(
            id=message_id,
            thread_id=thread_id,
            role=role,
            content_markdown=content,
            metadata=dict(metadata),
            created_at=float(created_at),
        )
