"""In-memory chat-history store (the tier without Postgres).

The fallback when ``INQTRIX_STORAGE_BACKEND`` is not ``postgres`` and the
offline test backend for the port contract. Mirrors the visibility and
keyset semantics of :class:`~inqtrix.project.chat_postgres.PostgresChatStore`
byte-for-byte so the two tiers return identical pages and cursors —
filter BEFORE slice (never under-fill a page) and reuse the shared
:func:`~inqtrix.pagination.keyset_page`. Process-local and not durable:
data is lost on restart, which is exactly why the durable tier exists.
"""

from __future__ import annotations

from dataclasses import replace

from inqtrix.pagination import keyset_page
from inqtrix.project.chat_ports import (
    ChatMessage,
    ChatThread,
    ChatThreadGroup,
    ThreadNotFound,
)


class MemoryChatStore:
    """Process-local :class:`~inqtrix.project.chat_ports.ChatStore`."""

    def __init__(self) -> None:
        self._threads: dict[str, ChatThread] = {}
        self._groups: dict[str, ChatThreadGroup] = {}
        # Keyed by the COMPOSITE (thread_id, id), mirroring the Postgres
        # composite PK: a re-used id from another thread is a distinct row,
        # never an overwrite of the foreign owner's message.
        self._messages: dict[tuple[str, str], ChatMessage] = {}

    async def upsert_thread(
        self,
        *,
        id: str,
        title: str,
        preview: str,
        source: str,
        group_id: str | None,
        created_at: float,
        updated_at: float,
        created_by_sub: str | None,
        workspace_id: str | None,
    ) -> ChatThread:
        existing = self._threads.get(id)
        if existing is not None:
            thread = replace(
                existing,
                title=title,
                preview=preview,
                source=source,
                group_id=group_id,
                updated_at=updated_at,
            )
        else:
            thread = ChatThread(
                id=id,
                title=title,
                preview=preview,
                source=source,
                group_id=group_id,
                created_at=created_at,
                updated_at=updated_at,
                created_by_sub=created_by_sub,
                workspace_id=workspace_id,
            )
        self._threads[id] = thread
        return thread

    async def list_threads_page(
        self,
        *,
        created_by_sub: str | None,
        workspace_id: str | None,
        limit: int,
        after: tuple[float, str] | None,
    ) -> tuple[list[ChatThread], str | None]:
        items = list(self._threads.values())
        if created_by_sub is not None:
            items = [t for t in items if t.created_by_sub == created_by_sub]
        if workspace_id is not None:
            items = [t for t in items if t.workspace_id == workspace_id]
        items.sort(key=lambda t: (t.created_at, t.id), reverse=True)
        return keyset_page(
            items,
            limit=limit,
            after=after,
            created_at_of=lambda t: t.created_at,
            id_of=lambda t: t.id,
        )

    async def get_thread(self, thread_id: str) -> ChatThread:
        try:
            return self._threads[thread_id]
        except KeyError as exc:
            raise ThreadNotFound(thread_id) from exc

    async def delete_thread(self, thread_id: str) -> None:
        self._threads.pop(thread_id, None)
        self._messages = {
            key: message
            for key, message in self._messages.items()
            if message.thread_id != thread_id
        }

    async def append_messages(
        self, messages: list[ChatMessage]
    ) -> list[ChatMessage]:
        stored: list[ChatMessage] = []
        for message in messages:
            key = (message.thread_id, message.id)
            existing = self._messages.get(key)
            if existing is not None:
                merged = replace(
                    existing,
                    role=message.role,
                    content_markdown=message.content_markdown,
                    metadata=dict(message.metadata),
                )
            else:
                merged = replace(message, metadata=dict(message.metadata))
            self._messages[key] = merged
            stored.append(merged)
        return stored

    async def delete_message(self, thread_id: str, message_id: str) -> None:
        self._messages.pop((thread_id, message_id), None)

    async def list_messages_page(
        self,
        thread_id: str,
        *,
        limit: int,
        after: tuple[float, str] | None,
    ) -> tuple[list[ChatMessage], str | None]:
        items = [m for m in self._messages.values() if m.thread_id == thread_id]
        items.sort(key=lambda m: (m.created_at, m.id), reverse=True)
        return keyset_page(
            items,
            limit=limit,
            after=after,
            created_at_of=lambda m: m.created_at,
            id_of=lambda m: m.id,
        )

    async def upsert_group(
        self,
        *,
        id: str,
        title: str,
        created_at: float,
        updated_at: float,
        created_by_sub: str | None,
        workspace_id: str | None,
    ) -> ChatThreadGroup:
        existing = self._groups.get(id)
        if existing is not None:
            group = replace(existing, title=title, updated_at=updated_at)
        else:
            group = ChatThreadGroup(
                id=id,
                title=title,
                created_at=created_at,
                updated_at=updated_at,
                created_by_sub=created_by_sub,
                workspace_id=workspace_id,
            )
        self._groups[id] = group
        return group

    async def list_groups(
        self,
        *,
        created_by_sub: str | None,
        workspace_id: str | None,
    ) -> list[ChatThreadGroup]:
        items = list(self._groups.values())
        if created_by_sub is not None:
            items = [g for g in items if g.created_by_sub == created_by_sub]
        if workspace_id is not None:
            items = [g for g in items if g.workspace_id == workspace_id]
        items.sort(key=lambda g: (g.created_at, g.id), reverse=True)
        return items

    async def delete_group(self, group_id: str) -> None:
        self._groups.pop(group_id, None)
        for tid, thread in list(self._threads.items()):
            if thread.group_id == group_id:
                self._threads[tid] = replace(thread, group_id=None)

    async def aclose(self) -> None:
        return None
