"""Vector-index-record persistence service (M6c project tier).

The vector-index counterpart of the asset-records service: payload
validation, the shared owner/share access rule
(:func:`~inqtrix.auth.permissions.resolve_owned_access`), and owner
resolution before the store query. Indexes are private per-user in M6c (no
sharing surface — the indexed *collections* are shared via the knowledge
layer, not these client records); all denials are the indistinct not-found.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from inqtrix.auth.permissions import SharePermission, resolve_owned_access
from inqtrix.services.workspace_guard import deny_cross_workspace
from inqtrix.project.vector_index_ports import (
    VectorIndexHistoryEntry,
    VectorIndexMember,
    VectorIndexNotFound,
    VectorIndexRecord,
    VectorIndexStore,
)

if TYPE_CHECKING:
    from inqtrix.auth.principal import UserContext

_VALID_STATUS = frozenset({"error", "indexing", "ready", "stale"})
# Mirrors the frontend VectorIndexMemberState union and the DB CHECK
# (ck_vector_index_members_state). 'skipped' = a no-text document, terminal
# (can never embed); persisted so it survives a reload.
_VALID_MEMBER_STATE = frozenset({"pending", "embedded", "skipped"})
_VALID_RESULT = frozenset({"cancelled", "error", "ok"})


class VectorIndexValidationError(ValueError):
    """Raised for client-payload problems (maps to HTTP 400)."""


class VectorIndexService:
    """Application service over a :class:`VectorIndexStore`."""

    def __init__(self, *, store: VectorIndexStore, durable: bool = False) -> None:
        self._store = store
        self._durable = durable

    @property
    def store(self) -> VectorIndexStore:
        return self._store

    @property
    def durable(self) -> bool:
        return self._durable

    async def save_index(
        self, *, id, title, handle, model, dims, status, server_collection_id,
        server_collection_model, last_error, members, history, created_at,
        updated_at, caller_sub, workspace_id, visible_to, also_visible=None,
    ) -> VectorIndexRecord:
        if status not in _VALID_STATUS:
            raise VectorIndexValidationError(f"unknown index status: {status!r}")
        # Members are an n:m set keyed on file_id; the wire shape is a list,
        # so a duplicate file_id is expressible. Collapse first-occurrence-wins
        # (the frontend reducer's own rule) so both store tiers receive a clean
        # set — the composite (index_id, file_id) PK can never be violated and
        # memory/postgres stay observably equivalent.
        members = _dedupe_members(members)
        for member in members:
            if member.state not in _VALID_MEMBER_STATE:
                raise VectorIndexValidationError(
                    f"unknown member state: {member.state!r}"
                )
        for entry in history:
            if entry.result not in _VALID_RESULT:
                raise VectorIndexValidationError(
                    f"unknown run result: {entry.result!r}"
                )
        try:
            existing = await self._store.get_index(id)
        except VectorIndexNotFound:
            existing = None
        if existing is not None:
            shared = resolve_owned_access(
                owner_sub=existing.created_by_sub, resource_tenant_id=existing.tenant_id,
                resource_id=existing.id, visible_to=visible_to,
                also_visible=also_visible, not_found=VectorIndexNotFound,
            )
            if shared is not None and not shared.at_least(SharePermission.EDIT):
                raise VectorIndexNotFound(id)
            owner_sub, owner_ws = existing.created_by_sub, existing.workspace_id
        else:
            owner_sub, owner_ws = caller_sub, workspace_id
        return await self._store.upsert_index(
            id=id, title=title, handle=handle, model=model, dims=dims,
            status=status, server_collection_id=server_collection_id,
            server_collection_model=server_collection_model,
            last_error=last_error, members=tuple(members), history=tuple(history),
            created_at=created_at, updated_at=updated_at, created_by_sub=owner_sub,
            workspace_id=owner_ws,
        )

    async def list_indexes(self, *, caller_sub, workspace_id, limit, after):
        return await self._store.list_indexes_page(
            created_by_sub=caller_sub, workspace_id=workspace_id, limit=limit, after=after
        )

    async def delete_index(
        self, index_id, *, visible_to, also_visible=None, request_workspace_id=None
    ) -> None:
        index = await self._store.get_index(index_id)
        shared = resolve_owned_access(
            owner_sub=index.created_by_sub, resource_tenant_id=index.tenant_id,
            resource_id=index.id, visible_to=visible_to, also_visible=also_visible,
            not_found=VectorIndexNotFound,
        )
        if shared is not None:
            raise VectorIndexNotFound(index_id)
        deny_cross_workspace(
            resource_workspace_id=index.workspace_id,
            request_workspace_id=request_workspace_id,
            not_found=lambda: VectorIndexNotFound(index_id),
        )
        await self._store.delete_index(index_id)


def _dedupe_members(members) -> tuple[VectorIndexMember, ...]:
    """Collapse duplicate ``file_id`` members, keeping the first occurrence
    (and its order) — the same rule the frontend ``createVectorIndex`` reducer
    applies. Idempotent on an already-unique set."""
    seen: dict[str, VectorIndexMember] = {}
    for member in members:
        if member.file_id not in seen:
            seen[member.file_id] = member
    return tuple(seen.values())
