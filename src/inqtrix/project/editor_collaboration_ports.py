"""Contracts for durable editor live-collaboration persistence.

The collaboration server never receives database credentials. It calls the
internal FastAPI surface, whose service persists through this port. Keeping the
transaction boundary in Python makes FastAPI the sole authority for identity,
sharing, sequence allocation, fencing, and retention while Hocuspocus remains
a replaceable synchronization component.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

CollaborationPermission = Literal["view", "suggest", "edit"]
CollaborationActorKind = Literal["human", "assistant", "agent", "system"]
CollaborationChangeKind = Literal["direct", "suggestion", "decision", "system"]
CollaborationDecision = Literal["accept", "reject"]
CollaborationSuggestionKind = Literal["insertion", "deletion", "modification"]


class CollaborationDocumentNotFound(KeyError):
    """Raised when a collaboration document is absent or tenant-invisible."""


class CollaborationConflict(RuntimeError):
    """Raised when a generation, revision, mode, schema, or command conflicts.

    Attributes:
        reason: Stable machine-readable conflict reason.
        current_sequence: Current durable sequence when relevant.
    """

    def __init__(self, reason: str, *, current_sequence: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.current_sequence = current_sequence


class CollaborationLeaseInvalid(PermissionError):
    """Raised when a lease is unknown, expired, revoked, or mismatched."""

    def __init__(self, reason: str = "lease_invalid") -> None:
        super().__init__(reason)
        self.reason = reason


class CollaborationInstanceFenced(RuntimeError):
    """Raised when a stale Node instance attempts a durable write."""


class CollaborationRateLimited(RuntimeError):
    """Raised when lease issuance or concurrent-session limits are exceeded."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class CollaborationDocumentState:
    """Durable state required to load or authorize one collaboration room."""

    document_id: str
    tenant_id: str
    generation: int
    schema_version: int
    schema_hash: str
    persisted_sequence: int
    projection_sequence: int
    content_markdown: str
    projection_updated_at: float | None
    owner_user_id: uuid.UUID | None
    deleted_at: float | None = None


@dataclass(frozen=True)
class CollaborationUpdate:
    """One idempotently persisted binary Yjs update and its audit metadata."""

    document_id: str
    tenant_id: str
    generation: int
    sequence: int
    update_hash: str
    update_bytes: bytes | None = field(repr=False)
    actor_user_id: uuid.UUID | None
    actor_kind: CollaborationActorKind
    change_kind: CollaborationChangeKind
    suggestion_ids: tuple[str, ...] = ()
    command_id: uuid.UUID | None = None
    created_at: float = 0.0
    payload_pruned_at: float | None = None


@dataclass(frozen=True)
class CollaborationSnapshot:
    """Verified binary Yjs state covering all updates through a sequence."""

    document_id: str
    tenant_id: str
    generation: int
    covered_sequence: int
    state_update: bytes = field(repr=False)
    state_vector: bytes = field(repr=False)
    state_hash: str
    projection_hash: str
    schema_version: int
    schema_hash: str
    created_at: float


@dataclass(frozen=True)
class CollaborationSnapshotCandidate:
    """One verified snapshot and a complete hash-verifiable durable tail."""

    snapshot: CollaborationSnapshot
    updates: tuple[CollaborationUpdate, ...]


@dataclass(frozen=True)
class CollaborationLoadedState:
    """Latest verified snapshot plus fallback candidates and update tails."""

    document: CollaborationDocumentState
    snapshot: CollaborationSnapshot
    updates: tuple[CollaborationUpdate, ...]
    fallback_candidates: tuple[CollaborationSnapshotCandidate, ...] = ()


@dataclass(frozen=True)
class CollaborationLease:
    """Database-backed, document-scoped authorization lease."""

    lease_id: uuid.UUID
    token_hash: str = field(repr=False)
    tenant_id: str
    document_id: str
    generation: int
    user_id: uuid.UUID
    permission: CollaborationPermission
    session_id: str
    issued_at: float
    expires_at: float
    last_validated_at: float
    rotation_command_id: uuid.UUID | None = None
    rotated_from_lease_id: uuid.UUID | None = None
    revoked_at: float | None = None


@dataclass(frozen=True)
class CollaborationInstanceLease:
    """Single-writer fencing lease held by the active Node process."""

    instance_id: str
    epoch: int
    lease_expires_at: float
    updated_at: float


@dataclass(frozen=True)
class CollaborationSuggestion:
    """One changed suggestion descriptor validated by the Node sidecar."""

    suggestion_id: str
    patch_id: str
    author_id: uuid.UUID
    created_at: float
    kind: CollaborationSuggestionKind


@dataclass(frozen=True)
class CollaborationPatchState:
    """Current active suggestion membership for one affected patch."""

    patch_id: str
    author_id: uuid.UUID
    created_at: float
    active_suggestion_ids: tuple[str, ...]
    kinds: tuple[CollaborationSuggestionKind, ...]


@dataclass(frozen=True)
class PersistCollaborationUpdate:
    """Input contract for one durable update append."""

    tenant_id: str
    document_id: str
    generation: int
    instance_id: str
    instance_epoch: int
    lease_id: uuid.UUID | None
    actor_user_id: uuid.UUID
    update_hash: str
    update_bytes: bytes = field(repr=False)
    actor_kind: CollaborationActorKind
    change_kind: CollaborationChangeKind
    suggestion_ids: tuple[str, ...] = ()
    suggestions: tuple[CollaborationSuggestion, ...] = ()
    patches: tuple[CollaborationPatchState, ...] = ()
    decision: CollaborationDecision | None = None
    command_id: uuid.UUID | None = None
    command_payload_hash: str | None = None
    expected_sequence: int | None = None
    now: float = 0.0


@dataclass(frozen=True)
class PersistedCollaborationUpdate:
    """Idempotent append result returned for the durable acknowledgement."""

    sequence: int
    persisted_sequence: int
    duplicate: bool
    persisted_at: float


@dataclass(frozen=True)
class CollaborationUpdateLookup:
    """Sequence coordinate for a previously persisted update digest."""

    update_hash: str
    sequence: int


@dataclass(frozen=True)
class CollaborationPersistedCommand:
    """Durable idempotency record returned before a server-side mutation."""

    actor_kind: CollaborationActorKind
    actor_user_id: uuid.UUID
    change_kind: Literal["decision", "suggestion"]
    command_id: uuid.UUID
    command_payload_hash: str
    decision: CollaborationDecision | None
    generation: int
    patch_ids: tuple[str, ...]
    sequence: int
    suggestion_ids: tuple[str, ...]
    update_hash: str


@dataclass(frozen=True)
class CollaborationActivity:
    """Content-bounded activity row used by the Changes inspector."""

    sequence: int
    actor_user_id: uuid.UUID | None
    actor_kind: CollaborationActorKind
    change_kind: CollaborationChangeKind
    suggestion_ids: tuple[str, ...]
    command_id: uuid.UUID | None
    created_at: float


@dataclass(frozen=True)
class CollaborationOpenPatch:
    """One pending collaboration patch for the open-changes view."""

    patch_id: str
    author_user_id: uuid.UUID
    created_at: float
    suggestion_ids: tuple[str, ...]
    kinds: tuple[CollaborationSuggestionKind, ...]
    exact_edits: tuple[dict[str, Any], ...] | None = field(repr=False)


@dataclass(frozen=True)
class CollaborationOpenPatchPage:
    """One bounded raw keyset page of published pending patches."""

    patches: tuple[CollaborationOpenPatch, ...]
    next_cursor: tuple[float, str] | None


@dataclass(frozen=True)
class CollaborationPolicyEvent:
    """Content-free user-event coordinate relevant to live connections."""

    id: int
    target_user_id: uuid.UUID
    scope: str
    resource_type: str
    resource_id: str | None


@dataclass(frozen=True)
class CollaborationPolicyPage:
    """Ordered policy-event page with a global durable cursor."""

    events: tuple[CollaborationPolicyEvent, ...]
    current_cursor: int
    reset_required: bool = False


@runtime_checkable
class EditorCollaborationStore(Protocol):
    """Persistence boundary for collaboration state and authorization."""

    async def load_state(
        self,
        *,
        tenant_id: str,
        document_id: str,
        generation: int | None = None,
    ) -> CollaborationLoadedState:
        """Load the latest snapshot and complete update tail."""
        ...

    async def enable_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
        owner_user_id: uuid.UUID,
        expected_revision: int,
        expected_metadata_revision: int,
        schema_version: int,
        schema_hash: str,
        snapshot: CollaborationSnapshot,
        projection_markdown: str,
        now: float,
    ) -> CollaborationDocumentState:
        """Atomically convert one unshared Markdown document to generation one."""
        ...

    async def issue_lease(
        self,
        lease: CollaborationLease,
        *,
        max_active: int,
        max_issued_per_window: int,
        issued_since: float,
    ) -> CollaborationLease:
        """Persist a bounded lease while enforcing durable issuance limits."""
        ...

    async def rotate_lease(
        self,
        *,
        previous_lease_id: uuid.UUID,
        previous_token_hash: str,
        replacement: CollaborationLease,
        max_issued_per_window: int,
        issued_since: float,
    ) -> CollaborationLease:
        """Atomically revoke one live lease and insert its replacement."""
        ...

    async def introspect_lease(
        self,
        *,
        tenant_id: str,
        lease_id: uuid.UUID,
        token_hash: str,
        now: float,
    ) -> CollaborationLease:
        """Return a currently valid lease and update its validation time."""
        ...

    async def revoke_leases(
        self,
        *,
        tenant_id: str,
        document_id: str,
        user_id: uuid.UUID | None,
        now: float,
    ) -> int:
        """Revoke active leases for one document, optionally one user."""
        ...

    async def acquire_instance(
        self,
        *,
        tenant_id: str,
        instance_id: str,
        now: float,
        lease_seconds: float,
    ) -> CollaborationInstanceLease:
        """Acquire the single-writer slot and advance its fencing epoch."""
        ...

    async def renew_instance(
        self,
        *,
        tenant_id: str,
        instance_id: str,
        epoch: int,
        now: float,
        lease_seconds: float,
    ) -> CollaborationInstanceLease:
        """Renew an unexpired slot without changing its epoch."""
        ...

    async def get_current_instance(
        self,
        *,
        tenant_id: str,
        now: float,
    ) -> CollaborationInstanceLease | None:
        """Return the authoritative unexpired single-writer lease, if any."""
        ...

    async def validate_instance(
        self,
        *,
        tenant_id: str,
        instance_id: str,
        epoch: int,
        now: float,
    ) -> CollaborationInstanceLease:
        """Require the currently active instance without extending its lease."""
        ...

    async def append_update(
        self, update: PersistCollaborationUpdate
    ) -> PersistedCollaborationUpdate:
        """Fence, authorize, sequence, and idempotently append one Yjs update."""
        ...

    async def lookup_command(
        self,
        *,
        tenant_id: str,
        document_id: str,
        generation: int,
        command_id: uuid.UUID,
        command_payload_hash: str,
    ) -> CollaborationPersistedCommand | None:
        """Return an exact prior server command or fail on identity reuse."""
        ...

    async def lookup_updates_by_hashes(
        self,
        *,
        tenant_id: str,
        document_id: str,
        generation: int,
        update_hashes: tuple[str, ...],
        instance_id: str,
        instance_epoch: int,
        now: float,
    ) -> tuple[CollaborationUpdateLookup, ...]:
        """Resolve hashes within one atomically fenced tenant generation."""
        ...

    async def store_snapshot(
        self,
        snapshot: CollaborationSnapshot,
        *,
        projection_markdown: str,
        instance_id: str,
        instance_epoch: int,
        now: float,
    ) -> None:
        """Fence and persist a verified snapshot in one transaction."""
        ...

    async def update_projection(
        self,
        *,
        tenant_id: str,
        document_id: str,
        generation: int,
        covered_sequence: int,
        content_markdown: str,
        projection_hash: str,
        now: float,
    ) -> CollaborationDocumentState:
        """Publish a canonical Markdown projection through a durable sequence."""
        ...

    async def list_activity(
        self,
        *,
        tenant_id: str,
        document_id: str,
        generation: int,
        before_sequence: int | None,
        author_user_id: uuid.UUID | None,
        change_kind: CollaborationChangeKind | None,
        limit: int,
    ) -> tuple[CollaborationActivity, ...]:
        """List durable activity metadata without returning binary payloads."""
        ...

    async def list_open_patches(
        self,
        *,
        tenant_id: str,
        document_id: str,
        generation: int,
        before: tuple[float, str] | None,
        author_user_id: uuid.UUID | None,
        suggestion_kind: CollaborationSuggestionKind | None,
        limit: int,
    ) -> CollaborationOpenPatchPage:
        """List one bounded keyset page of published pending patches."""
        ...

    async def list_open_patch_ids_at_sequence(
        self,
        *,
        tenant_id: str,
        document_id: str,
        generation: int,
        expected_sequence: int,
        limit: int,
    ) -> tuple[str, ...]:
        """Select all published pending patch IDs at one authoritative sequence."""
        ...

    async def lookup_decision_command_by_id(
        self,
        *,
        tenant_id: str,
        document_id: str,
        generation: int,
        command_id: uuid.UUID,
    ) -> CollaborationPersistedCommand | None:
        """Reconstruct a prior decision without requiring its payload hash."""
        ...

    async def policy_events_after(
        self,
        *,
        tenant_id: str,
        cursor: int,
        limit: int,
    ) -> CollaborationPolicyPage:
        """Read editor/user invalidations from the existing user-events feed."""
        ...

    async def compact(
        self,
        *,
        tenant_id: str,
        document_id: str,
        generation: int,
        instance_id: str,
        instance_epoch: int,
        now: float,
        payload_retention_seconds: float,
        metadata_retention_seconds: float,
    ) -> tuple[int, int]:
        """Fence and prune covered payloads and metadata in one transaction."""
        ...

    async def purge_tombstones(
        self,
        *,
        tenant_id: str,
        instance_id: str,
        instance_epoch: int,
        now: float,
        retention_seconds: float,
    ) -> int:
        """Fence and remove collaboration tombstones in one transaction."""
        ...

    async def tombstone_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
        owner_user_id: uuid.UUID,
        now: float,
    ) -> int:
        """Delete logically, advance generation, revoke shares, and invalidate."""
        ...
