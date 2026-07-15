"""In-memory vector-index-record store (tier without Postgres + offline test).

Mirrors :class:`~inqtrix.project.vector_index_postgres.PostgresVectorIndexStore`
byte-for-byte (filter before slice; reuse keyset_page; members + history
replaced wholesale on upsert). Process-local, not durable.
"""

from __future__ import annotations

import uuid
from dataclasses import replace

from inqtrix.pagination import keyset_page
from inqtrix.project.vector_index_ports import (
    VectorIndexNotFound,
    VectorIndexRecord,
)
from inqtrix.project.scoped_upsert import ResourceScope, require_memory_scope


class MemoryVectorIndexStore:
    """Process-local :class:`~inqtrix.project.vector_index_ports.VectorIndexStore`."""

    def __init__(self) -> None:
        self._indexes: dict[str, VectorIndexRecord] = {}

    async def upsert_index(
        self, *, id, title, handle, model, dims, status, server_collection_id,
        server_collection_model, last_error, members, history, created_at,
        updated_at, created_by_user_id: uuid.UUID | None, workspace_id,
    ) -> VectorIndexRecord:
        existing = self._indexes.get(id)
        mutable = dict(
            title=title, handle=handle, model=model, dims=dims, status=status,
            server_collection_id=server_collection_id,
            server_collection_model=server_collection_model, last_error=last_error,
            members=tuple(members), history=tuple(history), updated_at=updated_at,
        )
        if existing is not None:
            require_memory_scope(
                existing,
                created_by_user_id=created_by_user_id,
                workspace_id=workspace_id,
                resource_id=id,
                not_found=VectorIndexNotFound,
            )
            record = replace(existing, **mutable)
        else:
            record = VectorIndexRecord(
                id=id, created_at=created_at, created_by_user_id=created_by_user_id,
                workspace_id=workspace_id, **mutable,
            )
        self._indexes[id] = record
        return record

    async def list_indexes_page(
        self, *, created_by_user_id: uuid.UUID | None, workspace_id, limit, after
    ) -> tuple[list[VectorIndexRecord], str | None]:
        items = _scoped(self._indexes.values(), created_by_user_id, workspace_id)
        items.sort(key=lambda r: (r.created_at, r.id), reverse=True)
        return keyset_page(
            items, limit=limit, after=after,
            created_at_of=lambda r: r.created_at, id_of=lambda r: r.id,
        )

    async def get_index(self, index_id: str) -> VectorIndexRecord:
        try:
            return self._indexes[index_id]
        except KeyError as exc:
            raise VectorIndexNotFound(index_id) from exc

    async def delete_index(
        self, index_id: str, *, scope: ResourceScope
    ) -> None:
        require_memory_scope(
            self._indexes.get(index_id),
            created_by_user_id=scope.created_by_user_id,
            workspace_id=scope.workspace_id,
            resource_id=index_id,
            not_found=VectorIndexNotFound,
        )
        self._indexes.pop(index_id, None)

    async def aclose(self) -> None:
        return None


def _scoped(values, created_by_user_id: uuid.UUID | None, workspace_id):
    items = list(values)
    if created_by_user_id is not None:
        items = [i for i in items if i.created_by_user_id == created_by_user_id]
    if workspace_id is not None:
        items = [i for i in items if i.workspace_id == workspace_id]
    return items
