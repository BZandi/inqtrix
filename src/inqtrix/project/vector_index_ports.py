"""Contracts of the vector-index-record store (M6c).

Mirrors asset_records_ports: the store owns persistence only; scoping lives
in :class:`~inqtrix.services.vector_index_service.VectorIndexService` and
the wire shape in the router. Two implementations behind one port:
:class:`~inqtrix.project.vector_index_memory.MemoryVectorIndexStore`
(offline/test) and
:class:`~inqtrix.project.vector_index_postgres.PostgresVectorIndexStore`.

Unlike editor documents / assets, a vector index has NO heavy body — the
record is small (its members + capped history travel with it). The list
endpoint returns full records (eager load per the storage matrix); members
and history are replaced wholesale on each upsert.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class VectorIndexNotFound(KeyError):
    """Raised when an index id is unknown to the store (HTTP 404)."""


@dataclass(frozen=True)
class VectorIndexMember:
    """One document referenced by an index.

    Attributes:
        file_id: The asset id referenced by the index.
        state: ``pending`` (queued), ``embedded`` (vectors present), or
            ``skipped`` (no extractable text — terminal, never embeds).
        server_document_id: The backend knowledge-document id this member was
            ingested as, once known; ``None`` when ingested before this was
            tracked (offline / older index). Lets "remove from index" delete the
            exact document without a full rebuild; persisted so it survives a
            reload (else removal degrades to local-only).
    """

    file_id: str
    state: str
    server_document_id: str | None = None


@dataclass(frozen=True)
class VectorIndexHistoryEntry:
    """One finished reindex run, shown in the inline history (newest first).

    Attributes:
        result: ``ok`` / ``cancelled`` / ``error``.
        documents: Documents processed in the run.
        duration_ms: Wall-clock run duration in milliseconds.
        error: Failure message when ``result == "error"``, else ``None``.
        started_at/finished_at: Unix timestamps.
    """

    result: str
    documents: int
    duration_ms: int
    error: str | None
    started_at: float
    finished_at: float


@dataclass(frozen=True)
class VectorIndexRecord:
    """One vector index (RAG file<->collection mapping).

    Attributes:
        id: The opaque client-supplied id (e.g. ``vector-index-...``), the PK.
        title/handle: Display label and stable handle.
        model: Embedding model id.
        dims: Embedding dimensionality.
        status: ``error``/``indexing``/``ready``/``stale``.
        server_collection_id: Backend knowledge-collection id, or ``None``
            for a simulated (demo/offline) index.
        server_collection_model: Embedding model the server collection was
            built with, or ``None``. Lets a reindex distinguish "documents
            added" from "model changed" — must persist so the incremental
            path survives a reload.
        last_error: Visible last-run failure message, or ``None``.
        members: The referenced documents (n:m).
        history: Past reindex runs, newest first (capped client-side).
        created_at/updated_at: Unix timestamps.
    """

    id: str
    title: str
    handle: str
    model: str
    dims: int
    status: str
    server_collection_id: str | None
    server_collection_model: str | None
    last_error: str | None
    members: tuple[VectorIndexMember, ...] = ()
    history: tuple[VectorIndexHistoryEntry, ...] = field(default=(), repr=False)
    created_at: float = 0.0
    updated_at: float = 0.0
    tenant_id: str = "default"
    created_by_sub: str | None = None
    workspace_id: str | None = None


@runtime_checkable
class VectorIndexStore(Protocol):
    """Persistence port for vector-index records + their members/history."""

    async def upsert_index(
        self,
        *,
        id: str,
        title: str,
        handle: str,
        model: str,
        dims: int,
        status: str,
        server_collection_id: str | None,
        server_collection_model: str | None,
        last_error: str | None,
        members: tuple[VectorIndexMember, ...],
        history: tuple[VectorIndexHistoryEntry, ...],
        created_at: float,
        updated_at: float,
        created_by_sub: str | None,
        workspace_id: str | None,
    ) -> VectorIndexRecord:
        """Insert or update the record AND replace its members + history
        wholesale, atomically. Never reassigns ``created_at`` / ownership."""
        ...

    async def list_indexes_page(
        self,
        *,
        created_by_sub: str | None,
        workspace_id: str | None,
        limit: int,
        after: tuple[float, str] | None,
    ) -> tuple[list[VectorIndexRecord], str | None]:
        """One keyset page of the caller's indexes (newest first), each a
        FULL record (members + history) — there is no heavy body to defer."""
        ...

    async def get_index(self, index_id: str) -> VectorIndexRecord:
        """One full record, or :class:`VectorIndexNotFound`. Used by the
        service for the owner check on upsert / delete."""
        ...

    async def delete_index(self, index_id: str) -> None: ...

    async def aclose(self) -> None: ...
