"""Contracts of the knowledge-session store (Wissensmodus saved sessions).

Mirrors :mod:`inqtrix.project.asset_records_ports`: the store owns persistence
only; scoping lives in
:class:`~inqtrix.services.knowledge_sessions_service.KnowledgeSessionsService`
and the wire shape in the router. Two implementations behind one port:
:class:`~inqtrix.project.knowledge_sessions_memory.MemoryKnowledgeSessionStore`
(offline/test) and
:class:`~inqtrix.project.knowledge_sessions_postgres.PostgresKnowledgeSessionStore`.

A session carries a heavy ``items_json`` body: ``list_sessions`` returns
metadata only (``items_json="[]"``); ``get_session`` returns the full row with
the items (load-on-open).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class KnowledgeSessionNotFound(KeyError):
    """Raised when a session id is unknown to the store (HTTP 404)."""


class KnowledgeSessionGroupNotFound(KeyError):
    """Raised when a session group id is unknown to the store (HTTP 404)."""


@dataclass(frozen=True)
class KnowledgeSessionGroup:
    """One grouping of a user's knowledge sessions.

    Attributes:
        id: Client-supplied id (``knowledge-session-group-...``), the primary
            key.
        title: User-facing folder label in the Knowledge Desk history panel.
        created_at/updated_at: Unix timestamps.
        tenant_id/created_by_sub/workspace_id: The owner scope.
    """

    id: str
    title: str
    created_at: float
    updated_at: float
    tenant_id: str = "default"
    created_by_sub: str | None = None
    workspace_id: str | None = None


@dataclass(frozen=True)
class KnowledgeSession:
    """One saved Ask session: a titled Q&A conversation.

    Attributes:
        id: Client-supplied id (``ks_...``), the primary key.
        title: Session label shown in the history panel.
        group_id: Owning :class:`KnowledgeSessionGroup` id, or ``None`` when
            ungrouped.
        items_json: The ordered Q&A items serialized as a JSON array
            (question + the rendered answer record). ``"[]"`` on list rows
            (load-on-open); the full array on get.
        created_at/updated_at: Unix timestamps.
        tenant_id/created_by_sub/workspace_id: The owner scope.
    """

    id: str
    title: str
    created_at: float
    updated_at: float
    group_id: str | None = None
    items_json: str = field(repr=False, default="[]")
    tenant_id: str = "default"
    created_by_sub: str | None = None
    workspace_id: str | None = None


@runtime_checkable
class KnowledgeSessionStore(Protocol):
    """Persistence port for knowledge sessions."""

    async def upsert_session(
        self,
        *,
        id: str,
        title: str,
        items_json: str,
        group_id: str | None,
        created_at: float,
        updated_at: float,
        created_by_sub: str | None,
        workspace_id: str | None,
    ) -> KnowledgeSession: ...

    async def list_sessions(
        self, *, created_by_sub: str | None, workspace_id: str | None
    ) -> list[KnowledgeSession]:
        """Sessions for the scope (newest-updated first), METADATA ONLY
        (``items_json="[]"`` — the items load via get_session)."""
        ...

    async def get_session(self, session_id: str) -> KnowledgeSession:
        """One session WITH its items (load-on-open), or
        :class:`KnowledgeSessionNotFound`."""
        ...

    async def delete_session(self, session_id: str) -> None: ...

    async def upsert_group(
        self,
        *,
        id: str,
        title: str,
        created_at: float,
        updated_at: float,
        created_by_sub: str | None,
        workspace_id: str | None,
    ) -> KnowledgeSessionGroup: ...

    async def list_groups(
        self, *, created_by_sub: str | None, workspace_id: str | None
    ) -> list[KnowledgeSessionGroup]:
        """Groups for the scope, newest-created first."""
        ...

    async def delete_group(self, group_id: str) -> None:
        """Delete a group; its sessions orphan to ungrouped."""
        ...

    async def aclose(self) -> None: ...
