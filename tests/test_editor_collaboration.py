"""Observable HTTP contracts for editor live collaboration."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import uuid
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inqtrix.auth.permissions import AccessMode, ResourceAccess
from inqtrix.auth.principal import Principal, UserContext
from inqtrix.project.editor_collaboration_ports import (
    CollaborationActivity,
    CollaborationConflict,
    CollaborationDocumentNotFound,
    CollaborationDocumentState,
    CollaborationInstanceFenced,
    CollaborationInstanceLease,
    CollaborationLease,
    CollaborationLeaseInvalid,
    CollaborationLoadedState,
    CollaborationOpenPatch,
    CollaborationOpenPatchPage,
    CollaborationPersistedCommand,
    CollaborationRateLimited,
    CollaborationSnapshot,
    CollaborationSnapshotCandidate,
    CollaborationUpdate,
    CollaborationUpdateLookup,
    PersistCollaborationUpdate,
    PersistedCollaborationUpdate,
)
from inqtrix.project.editor_ports import DocumentNotFound, EditorDocument
from inqtrix.server.routers import capabilities as capabilities_router
from inqtrix.server.routers import editor_collaboration, internal_collaboration
from inqtrix.services.collaboration_client import (
    CollaborationConversion,
    CollaborationDecisionResult,
    CollaborationProjection,
    CollaborationSuggestionResult,
)
from inqtrix.services.editor_collaboration_service import (
    CollaborationAuthenticationRequired,
    CollaborationDocumentTooLarge,
    CollaborationProtocolConflict,
    EditorCollaborationService,
)
from inqtrix.settings import CollaborationSettings


DOCUMENT_ID = "ed_collaboration"
USER_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
ACTOR_ID = uuid.UUID("22222222-2222-4222-8222-222222222222")
SESSION_ID = "session-collaboration-test"
INTERNAL_SECRET = "collaboration-internal-test-secret"
INTERNAL_HEADERS = {"Authorization": f"Bearer {INTERNAL_SECRET}"}
INTERNAL_TENANT_ID = "tenant-collaboration-test"


def _sha256(value: bytes) -> str:
    """Return the canonical lowercase digest used by collaboration payloads."""
    return hashlib.sha256(value).hexdigest()


def test_collaboration_settings_canonicalize_the_deployment_tenant() -> None:
    """The API and Node share one bounded nonblank tenant setting."""
    settings = CollaborationSettings.model_validate(
        {"tenant_id": "  tenant-primary  "}
    )

    assert settings.tenant_id == "tenant-primary"
    with pytest.raises(ValueError, match="must not be blank"):
        CollaborationSettings.model_validate({"tenant_id": "   "})


def _cookie_principal() -> Principal:
    """Return a fully scoped browser-session principal."""
    return Principal(
        user_id=USER_ID,
        kind="oidc_session",
        tenant_id="default",
        role="member",
        display_name="Ada Editor",
        email="ada@example.test",
        session_id=SESSION_ID,
    )


def _document(*, collaboration: bool = False) -> EditorDocument:
    """Build one owner document in either Markdown or collaboration mode."""
    return EditorDocument(
        id=DOCUMENT_ID,
        title="Shared draft",
        content_markdown="# Shared draft\n\nInitial body.",
        revision=7,
        metadata_revision=3,
        created_by_user_id=USER_ID,
        content_mode="collaboration" if collaboration else "markdown",
        collaboration_generation=4 if collaboration else 0,
        collaboration_schema_version=1 if collaboration else None,
        collaboration_schema_hash=_sha256(b"schema") if collaboration else None,
        persisted_sequence=11 if collaboration else 0,
        projection_sequence=10 if collaboration else 0,
    )


class _Documents:
    """Narrow document port fake retaining the authoritative document record."""

    def __init__(self, document: EditorDocument) -> None:
        self.document = document
        self.get_calls: list[dict[str, Any]] = []

    async def get_document(
        self,
        document_id: str,
        *,
        visible_to: UserContext | None,
        minimum: Any,
    ) -> EditorDocument:
        """Return the document while recording the requested access floor."""
        self.get_calls.append(
            {
                "document_id": document_id,
                "visible_to": visible_to,
                "minimum": minimum,
            }
        )
        if document_id != self.document.id:
            raise DocumentNotFound(document_id)
        return self.document

    async def get_document_with_access(
        self,
        document_id: str,
        *,
        visible_to: UserContext | None,
        minimum: Any,
    ) -> tuple[EditorDocument, ResourceAccess]:
        """Return the document with owner access for session issuance."""
        document = await self.get_document(
            document_id,
            visible_to=visible_to,
            minimum=minimum,
        )
        return document, ResourceAccess(AccessMode.OWNER)


class _Node:
    """Side-effect-free collaboration sidecar fake used by the real service."""

    def __init__(self) -> None:
        self.ready = True
        self.convert_calls: list[dict[str, Any]] = []
        self.decide_calls: list[dict[str, Any]] = []
        self.project_calls: list[dict[str, Any]] = []
        self.projection_results: list[CollaborationProjection] = []
        self.publish_calls: list[dict[str, Any]] = []
        self.conversion = CollaborationConversion(
            schema_hash=_sha256(b"schema"),
            state_update=b"initial-yjs-state",
            state_vector=b"initial-yjs-vector",
            state_hash=_sha256(b"initial-yjs-state"),
            projection_markdown="# Shared draft\n\nInitial body.",
            projection_hash=_sha256(b"# Shared draft\n\nInitial body."),
        )

    async def available(self) -> bool:
        """Report the configured readiness state."""
        return self.ready

    async def convert(
        self,
        *,
        document_id: str,
        markdown: str,
        schema_version: int,
        max_document_bytes: int,
    ) -> CollaborationConversion:
        """Record conversion input and return a verified binary snapshot."""
        self.convert_calls.append(
            {
                "document_id": document_id,
                "markdown": markdown,
                "schema_version": schema_version,
                "max_document_bytes": max_document_bytes,
            }
        )
        return self.conversion

    async def decide(self, **kwargs: Any) -> CollaborationDecisionResult:
        """Record one public decision and return its durable command result."""
        self.decide_calls.append(dict(kwargs))
        return CollaborationDecisionResult(
            command_id=kwargs["command_id"],
            sequence=kwargs["expected_sequence"] + 1,
            suggestion_ids=("suggestion-decision",),
        )

    async def project(self, **kwargs: Any) -> CollaborationProjection:
        """Return a projection whose publication may race a later update."""
        self.project_calls.append(dict(kwargs))
        if self.projection_results:
            return self.projection_results.pop(0)
        return CollaborationProjection(
            generation=4,
            sequence=12,
            markdown="# Projected\n\nDurable body.",
            projection_hash=_sha256(b"# Projected\n\nDurable body."),
            schema_hash=_sha256(b"schema"),
        )

    async def publish_suggestion(
        self, **kwargs: Any
    ) -> CollaborationSuggestionResult:
        """Record a private-patch publication and return its durable identity."""
        self.publish_calls.append(dict(kwargs))
        return CollaborationSuggestionResult(
            command_id=kwargs["command_id"],
            patch_id=kwargs["patch_id"],
            sequence=12,
            suggestion_ids=(
                str(uuid.UUID("77777777-7777-4777-8777-777777777777")),
            ),
        )

    async def aclose(self) -> None:
        """Satisfy the service lifecycle contract."""


class _PublicStore:
    """Lease-aware collaboration port fake for public service behavior."""

    def __init__(self, documents: _Documents) -> None:
        self.documents = documents
        self.enable_calls: list[dict[str, Any]] = []
        self.issue_calls: list[CollaborationLease] = []
        self.rotate_calls: list[dict[str, Any]] = []
        self.leases: dict[uuid.UUID, CollaborationLease] = {}
        self.rotation_results: dict[uuid.UUID, CollaborationLease] = {}
        self.issue_error: Exception | None = None
        self.rotate_error: Exception | None = None
        self.activity_calls: list[dict[str, Any]] = []
        self.activity_rows: tuple[CollaborationActivity, ...] = ()
        self.open_patch_calls: list[dict[str, Any]] = []
        self.open_patch_rows: tuple[CollaborationOpenPatch, ...] = ()
        self.open_patch_next_cursor: tuple[float, str] | None = None
        self.open_patch_pages: dict[
            tuple[float, str] | None, CollaborationOpenPatchPage
        ] = {}
        self.all_open_patch_ids: tuple[str, ...] = ()
        self.all_open_error: Exception | None = None
        self.prior_decision: CollaborationPersistedCommand | None = None
        self.projection_authoritative_sequences: list[int] = []
        self.current_instance_results: list[
            CollaborationInstanceLease | None
        ] = []
        self.current_instance_calls: list[dict[str, Any]] = []
        self.instance_validation_calls: list[dict[str, Any]] = []
        self.compact_calls: list[dict[str, Any]] = []
        self.purge_calls: list[dict[str, Any]] = []

    async def enable_document(self, **kwargs: Any) -> CollaborationDocumentState:
        """Apply activation atomically to the fake document authority."""
        self.enable_calls.append(dict(kwargs))
        document = self.documents.document
        schema_hash = str(kwargs["schema_hash"])
        projection_markdown = str(kwargs["projection_markdown"])
        self.documents.document = replace(
            document,
            content_markdown=projection_markdown,
            content_mode="collaboration",
            collaboration_generation=1,
            collaboration_schema_version=int(kwargs["schema_version"]),
            collaboration_schema_hash=schema_hash,
            persisted_sequence=0,
            projection_sequence=0,
        )
        return CollaborationDocumentState(
            document_id=document.id,
            tenant_id=document.tenant_id,
            generation=1,
            schema_version=int(kwargs["schema_version"]),
            schema_hash=schema_hash,
            persisted_sequence=0,
            projection_sequence=0,
            content_markdown=projection_markdown,
            projection_updated_at=float(kwargs["now"]),
            owner_user_id=USER_ID,
        )

    async def issue_lease(
        self,
        lease: CollaborationLease,
        *,
        max_active: int,
        max_issued_per_window: int,
        issued_since: float,
    ) -> CollaborationLease:
        """Persist one lease or expose the configured durable rate error."""
        del max_active, max_issued_per_window, issued_since
        if self.issue_error is not None:
            raise self.issue_error
        self.issue_calls.append(lease)
        self.leases[lease.lease_id] = lease
        return lease

    async def rotate_lease(
        self,
        *,
        previous_lease_id: uuid.UUID,
        previous_token_hash: str,
        replacement: CollaborationLease,
        max_issued_per_window: int,
        issued_since: float,
    ) -> CollaborationLease:
        """Atomically replace exactly the live lease named by its token hash."""
        del max_issued_per_window, issued_since
        if self.rotate_error is not None:
            raise self.rotate_error
        if replacement.rotation_command_id in self.rotation_results:
            return self.rotation_results[replacement.rotation_command_id]
        previous = self.leases.get(previous_lease_id)
        if previous is None or previous.token_hash != previous_token_hash:
            raise CollaborationLeaseInvalid("lease_invalid")
        self.rotate_calls.append(
            {
                "previous_lease_id": previous_lease_id,
                "previous_token_hash": previous_token_hash,
                "replacement": replacement,
            }
        )
        del self.leases[previous_lease_id]
        self.leases[replacement.lease_id] = replacement
        if replacement.rotation_command_id is not None:
            self.rotation_results[replacement.rotation_command_id] = replacement
        return replacement

    async def list_activity(
        self, **kwargs: Any
    ) -> tuple[CollaborationActivity, ...]:
        """Return configured durable history after recording its filters."""
        self.activity_calls.append(dict(kwargs))
        return self.activity_rows

    async def list_open_patches(
        self, **kwargs: Any
    ) -> CollaborationOpenPatchPage:
        """Return configured pending patches after recording their filters."""
        self.open_patch_calls.append(dict(kwargs))
        if self.open_patch_pages:
            return self.open_patch_pages[kwargs["before"]]
        return CollaborationOpenPatchPage(
            patches=self.open_patch_rows,
            next_cursor=self.open_patch_next_cursor,
        )

    async def list_open_patch_ids_at_sequence(
        self, **kwargs: Any
    ) -> tuple[str, ...]:
        """Return one authoritative all-open decision selection."""
        self.open_patch_calls.append(dict(kwargs))
        if self.all_open_error is not None:
            raise self.all_open_error
        return self.all_open_patch_ids

    async def update_projection(
        self, **kwargs: Any
    ) -> CollaborationDocumentState:
        """Publish a projection while retaining the newest global sequence."""
        document = self.documents.document
        persisted_sequence = (
            self.projection_authoritative_sequences.pop(0)
            if self.projection_authoritative_sequences
            else max(document.persisted_sequence, kwargs["covered_sequence"])
        )
        self.documents.document = replace(
            document,
            content_markdown=kwargs["content_markdown"],
            persisted_sequence=persisted_sequence,
            projection_sequence=kwargs["covered_sequence"],
            projection_updated_at=kwargs["now"],
        )
        return CollaborationDocumentState(
            document_id=document.id,
            tenant_id=document.tenant_id,
            generation=document.collaboration_generation,
            schema_version=document.collaboration_schema_version or 1,
            schema_hash=document.collaboration_schema_hash or _sha256(b"schema"),
            persisted_sequence=persisted_sequence,
            projection_sequence=kwargs["covered_sequence"],
            content_markdown=kwargs["content_markdown"],
            projection_updated_at=kwargs["now"],
            owner_user_id=document.created_by_user_id,
        )

    async def lookup_decision_command_by_id(
        self, **kwargs: Any
    ) -> CollaborationPersistedCommand | None:
        """Return a configured prior all-open decision."""
        return self.prior_decision

    async def validate_instance(self, **kwargs: Any) -> None:
        """Record the concrete service's tenant-scoped instance fence."""
        self.instance_validation_calls.append(dict(kwargs))

    async def get_current_instance(
        self, **kwargs: Any
    ) -> CollaborationInstanceLease | None:
        """Return scripted authoritative instance reads for readiness probes."""
        self.current_instance_calls.append(dict(kwargs))
        if self.current_instance_results:
            return self.current_instance_results.pop(0)
        return None

    async def compact(self, **kwargs: Any) -> tuple[int, int]:
        """Record one tenant-scoped document compaction."""
        self.compact_calls.append(dict(kwargs))
        return 2, 3

    async def purge_tombstones(self, **kwargs: Any) -> int:
        """Record one tenant-scoped tombstone purge."""
        self.purge_calls.append(dict(kwargs))
        return 4


class _Users:
    """Minimal mirrored-user directory for activity labels."""

    async def profiles_for_user_ids(
        self, *, tenant_id: str, user_ids: tuple[uuid.UUID, ...]
    ) -> dict[uuid.UUID, Any]:
        del tenant_id
        return {
            user_id: SimpleNamespace(
                display_name=(
                    "Ada Editor" if user_id == USER_ID else "Other Editor"
                ),
                email=None,
            )
            for user_id in user_ids
        }


@dataclass(frozen=True)
class _PublicHarness:
    """Bound public API test client and its observable collaborators."""

    client: TestClient
    documents: _Documents
    node: _Node
    service: EditorCollaborationService
    store: _PublicStore


def _public_harness(
    *,
    principal: Principal | None = None,
    collaboration: bool = False,
    tenant_id: str = "default",
) -> _PublicHarness:
    """Wire the real public router and service around bounded port fakes."""
    active_principal = principal or _cookie_principal()
    documents = _Documents(_document(collaboration=collaboration))
    node = _Node()
    store = _PublicStore(documents)
    settings = SimpleNamespace(
        secret=INTERNAL_SECRET,
        tenant_id=tenant_id,
        protocol_version=1,
        schema_version=1,
        max_document_bytes=1024 * 1024,
        max_frame_bytes=64 * 1024,
        lease_ttl_seconds=60,
        token_refresh_seconds=17,
        provider_flush_ms=75,
        max_sessions_per_user_document=5,
        session_rate_per_minute=30,
        instance_lease_seconds=15.0,
        update_payload_retention_seconds=60.0,
        activity_retention_seconds=120.0,
        tombstone_retention_seconds=180.0,
    )
    service = EditorCollaborationService(
        store=store,
        documents=documents,
        node=node,
        settings=settings,
        users=_Users(),
    )

    def principal_dependency() -> Principal:
        return active_principal

    def user_context_dependency() -> UserContext | None:
        if active_principal.user_id is None:
            return None
        return UserContext(principal=active_principal)

    container = SimpleNamespace(
        editor_collaboration_service=service,
        principal_dependency=principal_dependency,
        settings=SimpleNamespace(collaboration=settings),
        user_context_dependency=user_context_dependency,
    )
    app = FastAPI()
    app.include_router(editor_collaboration.build_router(container))
    return _PublicHarness(
        client=TestClient(app, raise_server_exceptions=False),
        documents=documents,
        node=node,
        service=service,
        store=store,
    )


def _session_body(
    *,
    lease_token: str | None = None,
    rotation_command_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    """Return the public session request contract."""
    body: dict[str, Any] = {"protocol_version": 1, "schema_version": 1}
    if lease_token is not None:
        body["lease_token"] = lease_token
    if rotation_command_id is not None:
        body["rotation_command_id"] = str(rotation_command_id)
    return body


def test_public_activation_converts_owner_document_atomically() -> None:
    """Activation returns the durable collaboration state, not Node internals."""
    harness = _public_harness()

    with harness.client:
        response = harness.client.post(
            f"/v1/editor/documents/{DOCUMENT_ID}/collaboration:enable",
            json={
                "expected_revision": 7,
                "expected_metadata_revision": 3,
                "schema_version": 1,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "content_mode": "collaboration",
        "generation": 1,
        "schema_version": 1,
        "schema_hash": _sha256(b"schema"),
        "persisted_sequence": 0,
        "projection_sequence": 0,
    }
    assert harness.node.convert_calls == [
        {
            "document_id": DOCUMENT_ID,
            "markdown": "# Shared draft\n\nInitial body.",
            "schema_version": 1,
            "max_document_bytes": 1024 * 1024,
        }
    ]
    activation = harness.store.enable_calls[0]
    assert activation["expected_revision"] == 7
    assert activation["expected_metadata_revision"] == 3
    assert activation["snapshot"].state_update == b"initial-yjs-state"
    assert harness.documents.document.content_mode == "collaboration"


@pytest.mark.parametrize(
    ("path", "body"),
    [
        (
            f"/v1/editor/documents/{DOCUMENT_ID}/collaboration:enable",
            {
                "expected_revision": 7,
                "expected_metadata_revision": 3,
                "schema_version": 1,
            },
        ),
        (
            f"/v1/editor/documents/{DOCUMENT_ID}/collaboration/session",
            _session_body(),
        ),
        (
            f"/v1/editor/documents/{DOCUMENT_ID}/suggestions:publish",
            {
                "patch_id": str(
                    uuid.UUID("66666666-6666-4666-8666-666666666666")
                ),
                "command_id": str(
                    uuid.UUID("55555555-5555-4555-8555-555555555555")
                ),
                "actor_kind": "assistant",
                "expected_sequence": 11,
                "target_markdown": "# Suggested body",
            },
        ),
    ],
)
def test_public_collaboration_requires_cookie_session_not_pat(
    path: str,
    body: dict[str, Any],
) -> None:
    """A bearer PAT cannot activate or mint browser collaboration credentials."""
    pat = Principal(
        user_id=USER_ID,
        kind="pat",
        tenant_id="default",
        role="member",
        pat_id="pat_test",
    )
    harness = _public_harness(principal=pat, collaboration=True)

    with harness.client:
        response = harness.client.post(path, json=body)

    assert response.status_code == 403
    assert response.json()["error"] == {
        "message": "Live-Kollaboration erfordert eine aktive Browser-Sitzung.",
        "type": "forbidden",
        "reason": "cookie_session_required",
    }
    assert harness.store.issue_calls == []


@pytest.mark.asyncio
async def test_ready_instance_requires_a_stable_fence_around_node_readiness() -> None:
    """The public probe identity is read from the DB before and after Node."""
    harness = _public_harness(collaboration=True)
    lease = CollaborationInstanceLease(
        instance_id="node-ready",
        epoch=7,
        lease_expires_at=10_000.0,
        updated_at=9_000.0,
    )
    harness.store.current_instance_results = [lease, lease]

    result = await harness.service.ready_instance()

    assert result == lease
    assert [
        call["tenant_id"] for call in harness.store.current_instance_calls
    ] == ["default", "default"]


@pytest.mark.asyncio
async def test_ready_instance_rejects_a_takeover_during_the_probe() -> None:
    """A fencing change during the health request yields not-ready, not a mix."""
    harness = _public_harness(collaboration=True)
    harness.store.current_instance_results = [
        CollaborationInstanceLease("node-old", 7, 10_000.0, 9_000.0),
        CollaborationInstanceLease("node-new", 8, 10_100.0, 9_100.0),
    ]

    assert await harness.service.ready_instance() is None


@pytest.mark.asyncio
async def test_node_facing_maintenance_rejects_an_unconfigured_tenant() -> None:
    """A compromised sidecar cannot select another tenant for retention."""
    harness = _public_harness(collaboration=True)

    with pytest.raises(CollaborationProtocolConflict, match="tenant_conflict"):
        await harness.service.run_maintenance(
            tenant_id="foreign-tenant",
            document_id=None,
            generation=None,
            instance_id="node-a",
            epoch=1,
        )

    assert harness.store.purge_calls == []


def test_public_session_issues_then_rotates_one_opaque_lease() -> None:
    """Refresh replaces the old lease instead of consuming a session slot."""
    harness = _public_harness(collaboration=True)

    with harness.client:
        first = harness.client.post(
            f"/v1/editor/documents/{DOCUMENT_ID}/collaboration/session",
            json=_session_body(),
        )
        assert first.status_code == 200
        first_payload = first.json()
        second = harness.client.post(
            f"/v1/editor/documents/{DOCUMENT_ID}/collaboration/session",
            json=_session_body(lease_token=first_payload["lease_token"]),
        )

    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["lease_token"] != first_payload["lease_token"]
    assert second_payload == {
        "websocket_path": "/collaboration",
        "room": f"inqtrix-editor-v1:{DOCUMENT_ID}:g4",
        "lease_token": second_payload["lease_token"],
        "expires_at": second_payload["expires_at"],
        "refresh_after": second_payload["refresh_after"],
        "provider_flush_ms": 75,
        "access": "edit",
        "initial_write_mode": "edit",
        "user": {
            "id": str(USER_ID),
            "name": "Ada Editor",
            "color": EditorCollaborationService.user_color(USER_ID),
        },
        "protocol_version": 1,
        "schema_version": 1,
    }
    assert second_payload["refresh_after"] == pytest.approx(
        second_payload["expires_at"] - 43
    )
    assert len(harness.store.issue_calls) == 1
    assert len(harness.store.rotate_calls) == 1
    rotation = harness.store.rotate_calls[0]
    assert rotation["previous_lease_id"] == harness.store.issue_calls[0].lease_id
    assert len(harness.store.leases) == 1
    assert harness.store.issue_calls[0].lease_id not in harness.store.leases


def test_public_lease_rotation_rejects_malformed_token_as_401() -> None:
    """A malformed lease token yields the STRUCTURED lease error, never a 500.

    Pins the service's own token-validation raise sites: a missing
    ``CollaborationLeaseInvalid`` import once turned every invalid/expired
    lease into a NameError 500, so clients could never recover a lease and
    stayed read-only ("reconnecting read-only", live incident 2026-07-15).
    Fake stores raising the exception themselves cannot catch that class of
    regression — this drives the service's OWN ``_decode_token`` path.
    """
    harness = _public_harness(collaboration=True)

    with harness.client:
        response = harness.client.post(
            f"/v1/editor/documents/{DOCUMENT_ID}/collaboration/session",
            json=_session_body(lease_token="cl1.not-a-real-payload.bad-sig"),
        )

    assert response.status_code == 401
    payload = response.json()["error"]
    assert payload["type"] == "authentication_error"
    assert payload["reason"] == "lease_invalid"


def test_public_lease_rotation_reconstructs_a_lost_response_idempotently() -> None:
    """Retrying one rotation command returns the same durable replacement."""
    harness = _public_harness(collaboration=True)
    rotation_command_id = uuid.UUID("99999999-9999-4999-8999-999999999999")

    with harness.client:
        first = harness.client.post(
            f"/v1/editor/documents/{DOCUMENT_ID}/collaboration/session",
            json=_session_body(),
        )
        assert first.status_code == 200
        request = _session_body(
            lease_token=first.json()["lease_token"],
            rotation_command_id=rotation_command_id,
        )
        rotated = harness.client.post(
            f"/v1/editor/documents/{DOCUMENT_ID}/collaboration/session",
            json=request,
        )
        replayed = harness.client.post(
            f"/v1/editor/documents/{DOCUMENT_ID}/collaboration/session",
            json=request,
        )

    assert rotated.status_code == replayed.status_code == 200
    assert replayed.json()["lease_token"] == rotated.json()["lease_token"]
    assert replayed.json()["expires_at"] == rotated.json()["expires_at"]
    assert len(harness.store.rotate_calls) == 1
    assert len(harness.store.leases) == 1


@pytest.mark.parametrize("reason", ["lease_expired", "lease_revoked"])
def test_public_lease_rotation_preserves_terminal_lease_reason(reason: str) -> None:
    """Expired and revoked leases remain distinct public authentication errors."""
    harness = _public_harness(collaboration=True)

    with harness.client:
        first = harness.client.post(
            f"/v1/editor/documents/{DOCUMENT_ID}/collaboration/session",
            json=_session_body(),
        )
        assert first.status_code == 200
        harness.store.rotate_error = CollaborationLeaseInvalid(reason)
        response = harness.client.post(
            f"/v1/editor/documents/{DOCUMENT_ID}/collaboration/session",
            json=_session_body(
                lease_token=first.json()["lease_token"],
                rotation_command_id=uuid.uuid4(),
            ),
        )

    assert response.status_code == 401
    assert response.json()["error"]["reason"] == reason


@pytest.mark.parametrize("reason", ["session_limit", "session_rate_limited"])
def test_public_session_preserves_durable_rate_limit_reason(reason: str) -> None:
    """Both durable lease limits map to the public 429 envelope."""
    harness = _public_harness(collaboration=True)
    harness.store.issue_error = CollaborationRateLimited(reason)

    with harness.client:
        response = harness.client.post(
            f"/v1/editor/documents/{DOCUMENT_ID}/collaboration/session",
            json=_session_body(),
        )

    assert response.status_code == 429
    assert response.json()["error"] == {
        "message": "Zu viele Kollaborationssitzungen oder Sitzungsanfragen.",
        "type": "rate_limit_error",
        "reason": reason,
    }


def test_public_lease_rotation_preserves_rate_limit_error() -> None:
    """A refresh-window limit remains a public 429 instead of a generic conflict."""
    harness = _public_harness(collaboration=True)

    with harness.client:
        first = harness.client.post(
            f"/v1/editor/documents/{DOCUMENT_ID}/collaboration/session",
            json=_session_body(),
        )
        assert first.status_code == 200
        harness.store.rotate_error = CollaborationRateLimited(
            "session_rate_limited"
        )
        response = harness.client.post(
            f"/v1/editor/documents/{DOCUMENT_ID}/collaboration/session",
            json=_session_body(lease_token=first.json()["lease_token"]),
        )

    assert response.status_code == 429
    assert response.json()["error"] == {
        "message": "Zu viele Kollaborationssitzungen oder Sitzungsanfragen.",
        "type": "rate_limit_error",
        "reason": "session_rate_limited",
    }


def test_concrete_maintenance_service_propagates_explicit_tenant() -> None:
    """Fencing, compaction, and purge share the caller's canonical tenant."""
    harness = _public_harness(
        collaboration=True,
        tenant_id=INTERNAL_TENANT_ID,
    )

    result = asyncio.run(
        harness.service.run_maintenance(
            tenant_id=INTERNAL_TENANT_ID,
            document_id=DOCUMENT_ID,
            generation=4,
            instance_id="node-a",
            epoch=7,
        )
    )

    assert result == {
        "payloads_pruned": 2,
        "metadata_pruned": 3,
        "tombstones_purged": 4,
    }
    assert harness.store.compact_calls[0]["tenant_id"] == INTERNAL_TENANT_ID
    assert harness.store.purge_calls[0]["tenant_id"] == INTERNAL_TENANT_ID
    assert harness.store.compact_calls[0]["instance_id"] == "node-a"
    assert harness.store.compact_calls[0]["instance_epoch"] == 7
    assert harness.store.purge_calls[0]["instance_id"] == "node-a"
    assert harness.store.purge_calls[0]["instance_epoch"] == 7


def _publish_suggestion_body() -> dict[str, Any]:
    return {
        "patch_id": str(
            uuid.UUID("66666666-6666-4666-8666-666666666666")
        ),
        "command_id": str(
            uuid.UUID("55555555-5555-4555-8555-555555555555")
        ),
        "actor_kind": "assistant",
        "expected_sequence": 11,
        "target_markdown": "# Shared suggestion\n\nRevised body.",
    }


def test_public_publish_suggestion_accepts_private_assistant_patch() -> None:
    """A cookie user may publish one UUID-addressed private AI patch."""
    harness = _public_harness(collaboration=True)
    body = _publish_suggestion_body()

    with harness.client:
        response = harness.client.post(
            f"/v1/editor/documents/{DOCUMENT_ID}/suggestions:publish",
            json=body,
        )

    assert response.status_code == 200
    assert response.json() == {
        "command_id": body["command_id"],
        "patch_id": body["patch_id"],
        "sequence": 12,
        "suggestion_ids": [
            str(uuid.UUID("77777777-7777-4777-8777-777777777777"))
        ],
    }
    assert harness.node.publish_calls == [
        {
            "document_id": DOCUMENT_ID,
            "generation": 4,
            "expected_sequence": 11,
            "command_id": uuid.UUID(body["command_id"]),
            "patch_id": body["patch_id"],
            "actor_kind": "assistant",
            "actor_user_id": USER_ID,
            "target_markdown": body["target_markdown"],
        }
    ]


def test_public_projection_reprojects_until_markdown_is_globally_current() -> None:
    """A peer update racing publication forces a projection at the new watermark."""
    harness = _public_harness(collaboration=True)
    harness.store.projection_authoritative_sequences = [15]
    harness.node.projection_results = [
        CollaborationProjection(
            generation=4,
            sequence=12,
            markdown="# First projection",
            projection_hash=_sha256(b"# First projection"),
            schema_hash=_sha256(b"schema"),
        ),
        CollaborationProjection(
            generation=4,
            sequence=15,
            markdown="# Current projection",
            projection_hash=_sha256(b"# Current projection"),
            schema_hash=_sha256(b"schema"),
        ),
    ]

    with harness.client:
        response = harness.client.post(
            f"/v1/editor/documents/{DOCUMENT_ID}/collaboration/projection:flush"
        )

    assert response.status_code == 200
    assert response.json()["sequence"] == 15
    assert response.json()["authoritative_sequence"] == 15
    assert response.json()["content_markdown"] == "# Current projection"
    assert harness.node.project_calls == [
        {
            "document_id": DOCUMENT_ID,
            "generation": 4,
            "minimum_sequence": 11,
        },
        {
            "document_id": DOCUMENT_ID,
            "generation": 4,
            "minimum_sequence": 15,
        },
    ]


def test_public_projection_fails_visibly_when_updates_never_quiesce() -> None:
    """A continuously advancing document cannot yield falsely current Markdown."""
    harness = _public_harness(collaboration=True)
    harness.store.projection_authoritative_sequences = [13, 15, 17]
    harness.node.projection_results = [
        CollaborationProjection(
            generation=4,
            sequence=sequence,
            markdown=f"# Projection {sequence}",
            projection_hash=_sha256(f"# Projection {sequence}".encode()),
            schema_hash=_sha256(b"schema"),
        )
        for sequence in (12, 14, 16)
    ]

    with harness.client:
        response = harness.client.post(
            f"/v1/editor/documents/{DOCUMENT_ID}/collaboration/projection:flush"
        )

    assert response.status_code == 409
    assert response.json()["error"]["reason"] == "projection_not_current"
    assert response.json()["error"]["current_sequence"] == 17
    assert [call["minimum_sequence"] for call in harness.node.project_calls] == [
        11,
        13,
        15,
    ]


def test_public_patch_decision_accepts_an_explicit_uuid_batch() -> None:
    """The documented explicit batch reaches Node with one sequence precondition."""
    harness = _public_harness(collaboration=True)
    patch_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    decision_id = uuid.uuid4()

    with harness.client:
        response = harness.client.post(
            f"/v1/editor/documents/{DOCUMENT_ID}/patches:decide",
            json={
                "patch_ids": patch_ids,
                "decision": "accept",
                "expected_sequence": 11,
                "decision_id": str(decision_id),
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "decision_id": str(decision_id),
        "sequence": 12,
        "suggestion_ids": ["suggestion-decision"],
    }
    assert harness.node.decide_calls[0]["patch_ids"] == tuple(patch_ids)
    assert harness.node.decide_calls[0]["expected_sequence"] == 11


def test_public_patch_decision_selects_all_open_beyond_one_activity_page() -> None:
    """Confirmed all-open selection is not truncated at the 200-row UI page."""
    harness = _public_harness(collaboration=True)
    patch_ids = tuple(str(uuid.UUID(int=index + 1)) for index in range(205))
    decision_id = uuid.uuid4()
    harness.store.all_open_patch_ids = patch_ids

    with harness.client:
        response = harness.client.post(
            f"/v1/editor/documents/{DOCUMENT_ID}/patches:decide",
            json={
                "all_open": True,
                "confirm_all_open": True,
                "decision": "reject",
                "expected_sequence": 11,
                "decision_id": str(decision_id),
            },
        )

    assert response.status_code == 200
    assert harness.store.open_patch_calls == [
        {
            "tenant_id": "default",
            "document_id": DOCUMENT_ID,
            "generation": 4,
            "expected_sequence": 11,
            "limit": 5_000,
        }
    ]
    assert harness.node.decide_calls[0]["patch_ids"] == patch_ids


def test_public_all_open_decision_rejects_a_stale_global_sequence() -> None:
    """All-open selection fails before Node when the authoritative CAS moved."""
    harness = _public_harness(collaboration=True)
    harness.store.all_open_error = CollaborationConflict(
        "sequence_conflict", current_sequence=12
    )

    with harness.client:
        response = harness.client.post(
            f"/v1/editor/documents/{DOCUMENT_ID}/patches:decide",
            json={
                "all_open": True,
                "confirm_all_open": True,
                "decision": "accept",
                "expected_sequence": 11,
                "decision_id": str(uuid.uuid4()),
            },
        )

    assert response.status_code == 409
    assert response.json()["error"]["reason"] == "sequence_conflict"
    assert response.json()["error"]["current_sequence"] == 12
    assert harness.node.decide_calls == []


@pytest.mark.parametrize(
    ("field", "value", "expected_status"),
    [
        ("patch_id", "not-a-uuid", 400),
        ("command_id", "not-a-uuid", 400),
        ("actor_kind", "agent", 400),
        ("expected_sequence", -1, 400),
        ("expected_sequence", True, 400),
        ("target_markdown", "x" * (1_048_576 + 1), 413),
    ],
)
def test_public_publish_suggestion_rejects_invalid_browser_contract(
    field: str,
    value: Any,
    expected_status: int,
) -> None:
    """Only bounded assistant publications with canonical UUIDs reach Node."""
    harness = _public_harness(collaboration=True)
    body = _publish_suggestion_body()
    body[field] = value

    with harness.client:
        response = harness.client.post(
            f"/v1/editor/documents/{DOCUMENT_ID}/suggestions:publish",
            json=body,
        )

    assert response.status_code == expected_status
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert harness.node.publish_calls == []


def test_public_activity_history_applies_author_and_type_filters() -> None:
    """History filters are forwarded before adjacent-edit grouping."""
    harness = _public_harness(collaboration=True)
    harness.store.activity_rows = (
        CollaborationActivity(
            sequence=9,
            actor_user_id=ACTOR_ID,
            actor_kind="human",
            change_kind="direct",
            suggestion_ids=(),
            command_id=None,
            created_at=12.0,
        ),
    )

    with harness.client:
        response = harness.client.get(
            f"/v1/editor/documents/{DOCUMENT_ID}/activity",
            params={
                "view": "history",
                "author_id": str(ACTOR_ID),
                "type": "direct",
                "limit": 20,
            },
        )

    assert response.status_code == 200
    assert response.json()["data"] == [
        {
            "from_sequence": 9,
            "to_sequence": 9,
            "type": "direct",
            "actor_kind": "human",
            "actor": {"id": str(ACTOR_ID), "name": "Other Editor"},
            "suggestion_ids": [],
            "command_id": None,
            "created_at": 12.0,
        }
    ]
    assert harness.store.activity_calls == [
        {
            "tenant_id": "default",
            "document_id": DOCUMENT_ID,
            "generation": 4,
            "before_sequence": None,
            "author_user_id": ACTOR_ID,
            "change_kind": "direct",
            "limit": 21,
        }
    ]


def test_public_activity_history_cursor_uses_raw_boundary_before_grouping() -> None:
    """A full raw page keeps its continuation when direct rows collapse."""
    harness = _public_harness(collaboration=True)
    harness.store.activity_rows = tuple(
        CollaborationActivity(
            sequence=sequence,
            actor_user_id=ACTOR_ID,
            actor_kind="human",
            change_kind="direct",
            suggestion_ids=(),
            command_id=None,
            created_at=float(sequence),
        )
        for sequence in range(100, 49, -1)
    )

    with harness.client:
        response = harness.client.get(
            f"/v1/editor/documents/{DOCUMENT_ID}/activity",
            params={"view": "history", "limit": 50},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["next_cursor"] == "51"
    assert len(payload["data"]) == 1
    assert payload["data"][0]["from_sequence"] == 51
    assert payload["data"][0]["to_sequence"] == 100
    assert harness.store.activity_calls[0]["limit"] == 51


def test_public_open_activity_returns_exact_patch_preview_when_available() -> None:
    """Anchored AI edits remain inspectable while human descriptors stay bounded."""
    harness = _public_harness(collaboration=True)
    patch_id = "patch-open-exact"
    exact_edit = {
        "id": "edit-1",
        "find": "old text",
        "text": "new text",
        "position": "replace",
        "quote_before": "before",
        "quote_after": "after",
    }
    harness.store.open_patch_rows = (
        CollaborationOpenPatch(
            patch_id=patch_id,
            author_user_id=ACTOR_ID,
            created_at=13.0,
            suggestion_ids=("suggestion-1",),
            kinds=("modification",),
            exact_edits=(exact_edit,),
        ),
    )

    with harness.client:
        response = harness.client.get(
            f"/v1/editor/documents/{DOCUMENT_ID}/activity",
            params={
                "view": "open",
                "author_id": str(ACTOR_ID),
                "type": "modification",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {
                "patch_id": patch_id,
                "author": {"id": str(ACTOR_ID), "name": "Other Editor"},
                "created_at": 13.0,
                "suggestion_ids": ["suggestion-1"],
                "type": "modification",
                "types": ["modification"],
                "preview": {"edits": [exact_edit]},
            }
        ],
        "next_cursor": None,
    }
    assert harness.store.open_patch_calls == [
        {
            "tenant_id": "default",
            "document_id": DOCUMENT_ID,
            "generation": 4,
            "before": None,
            "author_user_id": ACTOR_ID,
            "suggestion_kind": "modification",
            "limit": 50,
        }
    ]


def test_public_open_activity_marks_human_exact_preview_unavailable() -> None:
    """Human Yjs patches expose metadata without promising stored exact text."""
    harness = _public_harness(collaboration=True)
    harness.store.open_patch_rows = (
        CollaborationOpenPatch(
            patch_id="patch-open-human",
            author_user_id=ACTOR_ID,
            created_at=14.0,
            suggestion_ids=("suggestion-human",),
            kinds=("insertion",),
            exact_edits=None,
        ),
    )

    with harness.client:
        response = harness.client.get(
            f"/v1/editor/documents/{DOCUMENT_ID}/activity",
            params={"view": "open"},
        )

    assert response.status_code == 200
    assert response.json()["data"][0]["preview"] is None


def test_public_open_activity_keyset_walks_beyond_two_hundred_rows() -> None:
    """The opaque cursor exposes every open patch without a hidden truncation."""
    harness = _public_harness(collaboration=True)
    patches = tuple(
        CollaborationOpenPatch(
            patch_id=str(uuid.UUID(int=index + 1)),
            author_user_id=ACTOR_ID,
            created_at=1_000.0 - index,
            suggestion_ids=(f"suggestion-{index}",),
            kinds=("insertion",),
            exact_edits=None,
        )
        for index in range(205)
    )
    boundary = (patches[199].created_at, patches[199].patch_id)
    harness.store.open_patch_pages = {
        None: CollaborationOpenPatchPage(
            patches=patches[:200], next_cursor=boundary
        ),
        boundary: CollaborationOpenPatchPage(
            patches=patches[200:], next_cursor=None
        ),
    }

    with harness.client:
        first = harness.client.get(
            f"/v1/editor/documents/{DOCUMENT_ID}/activity",
            params={"view": "open", "limit": 200},
        )
        assert first.status_code == 200
        second = harness.client.get(
            f"/v1/editor/documents/{DOCUMENT_ID}/activity",
            params={
                "view": "open",
                "limit": 200,
                "cursor": first.json()["next_cursor"],
            },
        )

    assert second.status_code == 200
    observed_ids = [
        row["patch_id"]
        for response in (first, second)
        for row in response.json()["data"]
    ]
    assert observed_ids == [patch.patch_id for patch in patches]
    assert first.json()["next_cursor"] is not None
    assert second.json()["next_cursor"] is None
    assert harness.store.open_patch_calls[1]["before"] == boundary


class _InternalService:
    """Internal API service fake exposing validated DTOs and injected errors."""

    def __init__(self) -> None:
        snapshot = CollaborationSnapshot(
            document_id=DOCUMENT_ID,
            tenant_id=INTERNAL_TENANT_ID,
            generation=4,
            covered_sequence=4,
            state_update=b"snapshot-state",
            state_vector=b"snapshot-vector",
            state_hash=_sha256(b"snapshot-state"),
            projection_hash=_sha256(b"projection"),
            schema_version=1,
            schema_hash=_sha256(b"schema"),
            created_at=10.0,
        )
        document = CollaborationDocumentState(
            document_id=DOCUMENT_ID,
            tenant_id=INTERNAL_TENANT_ID,
            generation=4,
            schema_version=1,
            schema_hash=_sha256(b"schema"),
            persisted_sequence=5,
            projection_sequence=4,
            content_markdown="projected",
            projection_updated_at=10.0,
            owner_user_id=USER_ID,
        )
        update = CollaborationUpdate(
            document_id=DOCUMENT_ID,
            tenant_id=INTERNAL_TENANT_ID,
            generation=4,
            sequence=5,
            update_hash=_sha256(b"tail-update"),
            update_bytes=b"tail-update",
            actor_user_id=ACTOR_ID,
            actor_kind="human",
            change_kind="direct",
            created_at=11.0,
        )
        fallback_snapshot = replace(
            snapshot,
            covered_sequence=3,
            state_update=b"fallback-snapshot-state",
            state_vector=b"fallback-snapshot-vector",
            state_hash=_sha256(b"fallback-snapshot-state"),
            created_at=9.0,
        )
        fallback_update = replace(
            update,
            sequence=4,
            update_hash=_sha256(b"fallback-tail-update"),
            update_bytes=b"fallback-tail-update",
            created_at=10.0,
        )
        self.loaded_state = CollaborationLoadedState(
            document=document,
            snapshot=snapshot,
            updates=(update,),
            fallback_candidates=(
                CollaborationSnapshotCandidate(
                    snapshot=fallback_snapshot,
                    updates=(fallback_update, update),
                ),
            ),
        )
        self.error: Exception | None = None
        self.load_calls: list[dict[str, Any]] = []
        self.persisted_updates: list[PersistCollaborationUpdate] = []
        self.command_lookups: list[dict[str, Any]] = []
        self.instance_acquires: list[dict[str, Any]] = []
        self.instance_renewals: list[dict[str, Any]] = []
        self.maintenance_calls: list[dict[str, Any]] = []
        self.policy_calls: list[dict[str, Any]] = []
        self.update_lookups: list[dict[str, Any]] = []
        self.command_result: CollaborationPersistedCommand | None = None
        self.persisted_result = PersistedCollaborationUpdate(
            sequence=12,
            persisted_sequence=12,
            duplicate=False,
            persisted_at=0.0,
        )
        self.stored_snapshots: list[
            tuple[CollaborationSnapshot, str, str, int, str]
        ] = []

    async def load_state(self, **kwargs: Any) -> CollaborationLoadedState:
        """Return the prepared state after recording the fenced coordinate."""
        self.load_calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        return self.loaded_state

    async def acquire_instance(self, **kwargs: Any) -> Any:
        """Record acquisition of the configured deployment tenant."""
        self.instance_acquires.append(dict(kwargs))
        return SimpleNamespace(
            instance_id=kwargs["instance_id"],
            epoch=3,
            lease_expires_at=20.0,
        )

    async def renew_instance(self, **kwargs: Any) -> Any:
        """Record renewal of the configured deployment tenant."""
        self.instance_renewals.append(dict(kwargs))
        return SimpleNamespace(
            instance_id=kwargs["instance_id"],
            epoch=kwargs["epoch"],
            lease_expires_at=25.0,
        )

    async def policy_events(self, **kwargs: Any) -> dict[str, Any]:
        """Record a tenant-scoped invalidation cursor read."""
        self.policy_calls.append(dict(kwargs))
        return {"events": [], "cursor": 0, "reset_required": False}

    async def run_maintenance(self, **kwargs: Any) -> dict[str, int]:
        """Record one tenant-scoped fenced retention pass."""
        self.maintenance_calls.append(dict(kwargs))
        return {
            "payloads_pruned": 0,
            "metadata_pruned": 0,
            "tombstones_purged": 0,
        }

    async def persist_update(
        self, *, update: PersistCollaborationUpdate
    ) -> PersistedCollaborationUpdate:
        """Record the parsed update contract and acknowledge it durably."""
        if self.error is not None:
            raise self.error
        self.persisted_updates.append(update)
        return replace(self.persisted_result, persisted_at=update.now)

    async def lookup_command(
        self, **kwargs: Any
    ) -> CollaborationPersistedCommand | None:
        """Record one idempotency lookup before a sidecar command replay."""
        if self.error is not None:
            raise self.error
        self.command_lookups.append(dict(kwargs))
        return self.command_result

    async def lookup_updates(
        self, **kwargs: Any
    ) -> tuple[CollaborationUpdateLookup, ...]:
        """Record one fenced hash reconciliation lookup."""
        if self.error is not None:
            raise self.error
        self.update_lookups.append(dict(kwargs))
        hashes = kwargs["update_hashes"]
        return (
            CollaborationUpdateLookup(update_hash=hashes[0], sequence=12),
        )

    async def store_snapshot(
        self,
        *,
        snapshot: CollaborationSnapshot,
        projection_markdown: str,
        instance_id: str,
        epoch: int,
        tenant_id: str,
    ) -> None:
        """Record the fully validated snapshot DTO."""
        if self.error is not None:
            raise self.error
        self.stored_snapshots.append(
            (snapshot, projection_markdown, instance_id, epoch, tenant_id)
        )

    async def introspect_lease(self, **kwargs: Any) -> dict[str, Any]:
        """Return a minimal successful lease or expose a mapped domain error."""
        if self.error is not None:
            raise self.error
        return {"valid": True, **kwargs}


def _internal_client(service: _InternalService) -> TestClient:
    """Bind the real internal router to an exact-secret test container."""
    container = SimpleNamespace(
        editor_collaboration_service=service,
        settings=SimpleNamespace(
            collaboration=SimpleNamespace(
                secret=INTERNAL_SECRET,
                max_document_bytes=10 * 1_048_576,
            )
        ),
    )
    app = FastAPI()
    app.include_router(internal_collaboration.build_router(container))
    return TestClient(app, raise_server_exceptions=False)


def _state_path() -> str:
    """Return a fully fenced internal state-load URL."""
    return (
        f"/internal/collaboration/documents/{DOCUMENT_ID}/state"
        "?generation=4&instance_id=node-a&epoch=7"
        f"&tenant_id={INTERNAL_TENANT_ID}"
    )


@pytest.mark.parametrize(
    "authorization",
    [None, "Bearer wrong", f"bearer {INTERNAL_SECRET}", f"Basic {INTERNAL_SECRET}"],
)
def test_internal_api_requires_the_exact_bearer_header(
    authorization: str | None,
) -> None:
    """Missing, wrong, or differently schemed credentials fail before service I/O."""
    service = _InternalService()
    client = _internal_client(service)
    headers = {"Authorization": authorization} if authorization is not None else {}

    with client:
        response = client.get(_state_path(), headers=headers)

    assert response.status_code == 401
    assert response.json()["error"] == {
        "message": "Internal authentication failed.",
        "type": "authentication_error",
        "reason": "internal_auth_failed",
    }
    assert service.load_calls == []


def test_internal_instance_policy_and_maintenance_carry_configured_tenant() -> None:
    """Every trusted sidecar control flow requires the same explicit tenant."""
    service = _InternalService()
    client = _internal_client(service)

    with client:
        acquired = client.post(
            "/internal/collaboration/instances/acquire",
            headers=INTERNAL_HEADERS,
            json={
                "tenant_id": INTERNAL_TENANT_ID,
                "instance_id": "node-a",
                "lease_seconds": 15,
                "protocol_version": 1,
                "schema_version": 1,
            },
        )
        renewed = client.post(
            "/internal/collaboration/instances/renew",
            headers=INTERNAL_HEADERS,
            json={
                "tenant_id": INTERNAL_TENANT_ID,
                "instance_id": "node-a",
                "epoch": 3,
                "lease_seconds": 15,
            },
        )
        policy = client.get(
            "/internal/collaboration/policy-events",
            headers=INTERNAL_HEADERS,
            params={"tenant_id": INTERNAL_TENANT_ID, "after_id": 0},
        )
        maintenance = client.post(
            "/internal/collaboration/maintenance:compact",
            headers=INTERNAL_HEADERS,
            json={
                "tenant_id": INTERNAL_TENANT_ID,
                "instance_id": "node-a",
                "epoch": 3,
            },
        )

    assert acquired.status_code == renewed.status_code == 200
    assert policy.status_code == maintenance.status_code == 200
    assert service.instance_acquires[0]["tenant_id"] == INTERNAL_TENANT_ID
    assert service.instance_renewals[0]["tenant_id"] == INTERNAL_TENANT_ID
    assert service.policy_calls == [
        {"tenant_id": INTERNAL_TENANT_ID, "cursor": 0, "limit": 500}
    ]
    assert service.maintenance_calls == [
        {
            "tenant_id": INTERNAL_TENANT_ID,
            "document_id": None,
            "generation": None,
            "instance_id": "node-a",
            "epoch": 3,
        }
    ]


def test_internal_state_load_serializes_snapshot_and_complete_update_tail() -> None:
    """The sidecar receives one canonical snapshot and every later update."""
    service = _InternalService()
    client = _internal_client(service)

    with client:
        response = client.get(_state_path(), headers=INTERNAL_HEADERS)

    assert response.status_code == 200
    assert response.json() == {
        "document_id": DOCUMENT_ID,
        "generation": 4,
        "persisted_sequence": 5,
        "schema_version": 1,
        "schema_hash": _sha256(b"schema"),
        "snapshot": {
            "covered_sequence": 4,
            "state_update_base64": base64.b64encode(b"snapshot-state").decode(),
            "state_vector_base64": base64.b64encode(b"snapshot-vector").decode(),
            "state_hash": _sha256(b"snapshot-state"),
        },
        "updates": [
            {
                "sequence": 5,
                "update_hash": _sha256(b"tail-update"),
                "update_base64": base64.b64encode(b"tail-update").decode(),
            }
        ],
        "snapshot_candidates": [
            {
                "covered_sequence": 4,
                "state_update_base64": base64.b64encode(
                    b"snapshot-state"
                ).decode(),
                "state_vector_base64": base64.b64encode(
                    b"snapshot-vector"
                ).decode(),
                "state_hash": _sha256(b"snapshot-state"),
                "updates": [
                    {
                        "sequence": 5,
                        "update_hash": _sha256(b"tail-update"),
                        "update_base64": base64.b64encode(
                            b"tail-update"
                        ).decode(),
                    }
                ],
            },
            {
                "covered_sequence": 3,
                "state_update_base64": base64.b64encode(
                    b"fallback-snapshot-state"
                ).decode(),
                "state_vector_base64": base64.b64encode(
                    b"fallback-snapshot-vector"
                ).decode(),
                "state_hash": _sha256(b"fallback-snapshot-state"),
                "updates": [
                    {
                        "sequence": 4,
                        "update_hash": _sha256(b"fallback-tail-update"),
                        "update_base64": base64.b64encode(
                            b"fallback-tail-update"
                        ).decode(),
                    },
                    {
                        "sequence": 5,
                        "update_hash": _sha256(b"tail-update"),
                        "update_base64": base64.b64encode(
                            b"tail-update"
                        ).decode(),
                    },
                ],
            },
        ],
    }
    assert service.load_calls == [
        {
            "tenant_id": INTERNAL_TENANT_ID,
            "document_id": DOCUMENT_ID,
            "generation": 4,
            "instance_id": "node-a",
            "epoch": 7,
        }
    ]


def _update_payload(update_bytes: bytes = b"binary-yjs-update") -> dict[str, Any]:
    """Return a valid internal update request."""
    suggestion_id = str(uuid.UUID("44444444-4444-4444-8444-444444444444"))
    patch_id = str(uuid.UUID("66666666-6666-4666-8666-666666666666"))
    return {
        "tenant_id": INTERNAL_TENANT_ID,
        "generation": 4,
        "instance_id": "node-a",
        "epoch": 7,
        "lease_id": str(uuid.UUID("33333333-3333-4333-8333-333333333333")),
        "actor_user_id": str(ACTOR_ID),
        "update_hash": _sha256(update_bytes),
        "update_base64": base64.b64encode(update_bytes).decode("ascii"),
        "actor_kind": "human",
        "change_kind": "suggestion",
        "suggestion_ids": [suggestion_id],
        "suggestions": [
            {
                "suggestion_id": suggestion_id,
                "patch_id": patch_id,
                "author_id": str(ACTOR_ID),
                "created_at": 1_784_112_000,
                "kind": "insertion",
            }
        ],
        "patches": [
            {
                "patch_id": patch_id,
                "author_id": str(ACTOR_ID),
                "created_at": 1_784_112_000,
                "active_suggestion_ids": [suggestion_id],
                "kinds": ["insertion"],
            }
        ],
        "decision": None,
        "command_id": str(uuid.UUID("55555555-5555-4555-8555-555555555555")),
        "command_payload_hash": _sha256(b"canonical-command"),
        "expected_sequence": 11,
    }


def test_internal_update_validates_and_preserves_binary_metadata() -> None:
    """A valid append reaches the service as the typed durable update contract."""
    service = _InternalService()
    client = _internal_client(service)

    with client:
        response = client.post(
            f"/internal/collaboration/documents/{DOCUMENT_ID}/updates",
            headers=INTERNAL_HEADERS,
            json=_update_payload(),
        )

    assert response.status_code == 200
    assert response.json() == {
        "sequence": 12,
        "persisted_sequence": 12,
        "duplicate": False,
    }
    update = service.persisted_updates[0]
    assert update.update_bytes == b"binary-yjs-update"
    assert update.tenant_id == INTERNAL_TENANT_ID
    assert update.update_hash == _sha256(b"binary-yjs-update")
    assert update.actor_user_id == ACTOR_ID
    assert update.suggestion_ids == (
        str(uuid.UUID("44444444-4444-4444-8444-444444444444")),
    )
    assert update.suggestions[0].kind == "insertion"
    assert update.patches[0].active_suggestion_ids == update.suggestion_ids
    assert update.decision is None
    assert update.command_payload_hash == _sha256(b"canonical-command")
    assert update.expected_sequence == 11


def test_internal_duplicate_update_preserves_current_document_watermark() -> None:
    """A replay returns its original coordinate and the locked room watermark."""
    service = _InternalService()
    service.persisted_result = PersistedCollaborationUpdate(
        sequence=4,
        persisted_sequence=9,
        duplicate=True,
        persisted_at=0.0,
    )
    client = _internal_client(service)

    with client:
        response = client.post(
            f"/internal/collaboration/documents/{DOCUMENT_ID}/updates",
            headers=INTERNAL_HEADERS,
            json=_update_payload(),
        )

    assert response.status_code == 200
    assert response.json() == {
        "sequence": 4,
        "persisted_sequence": 9,
        "duplicate": True,
    }


def test_internal_update_hash_lookup_is_fenced_and_tenant_scoped() -> None:
    """Reconnect reconciliation returns only hashes durably stored in scope."""
    service = _InternalService()
    client = _internal_client(service)
    persisted_hash = _sha256(b"persisted-update")
    missing_hash = _sha256(b"missing-update")

    with client:
        response = client.post(
            f"/internal/collaboration/documents/{DOCUMENT_ID}/updates:lookup",
            headers=INTERNAL_HEADERS,
            json={
                "tenant_id": INTERNAL_TENANT_ID,
                "generation": 4,
                "hashes": [persisted_hash, missing_hash],
                "instance_id": "node-a",
                "epoch": 7,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "updates": [{"hash": persisted_hash, "sequence": 12}]
    }
    assert service.update_lookups == [
        {
            "tenant_id": INTERNAL_TENANT_ID,
            "document_id": DOCUMENT_ID,
            "generation": 4,
            "update_hashes": (persisted_hash, missing_hash),
            "instance_id": "node-a",
            "epoch": 7,
        }
    ]


def test_internal_durable_update_flow_requires_explicit_tenant() -> None:
    """The trusted sidecar cannot fall through to an implicit deployment tenant."""
    service = _InternalService()
    client = _internal_client(service)
    payload = _update_payload()
    del payload["tenant_id"]

    with client:
        response = client.post(
            f"/internal/collaboration/documents/{DOCUMENT_ID}/updates",
            headers=INTERNAL_HEADERS,
            json=payload,
        )

    assert response.status_code == 400
    assert response.json()["error"]["reason"] == "invalid_tenant_id"
    assert service.persisted_updates == []


def test_internal_command_lookup_returns_exact_persisted_identity() -> None:
    """A lost sidecar response resolves only through its canonical payload hash."""
    service = _InternalService()
    command_id = uuid.UUID("55555555-5555-4555-8555-555555555555")
    command_hash = _sha256(b"canonical-command")
    service.command_result = CollaborationPersistedCommand(
        actor_kind="human",
        actor_user_id=ACTOR_ID,
        change_kind="decision",
        command_id=command_id,
        command_payload_hash=command_hash,
        decision="accept",
        generation=4,
        patch_ids=(str(uuid.UUID("66666666-6666-4666-8666-666666666666")),),
        sequence=12,
        suggestion_ids=(),
        update_hash=_sha256(b"binary-yjs-update"),
    )
    client = _internal_client(service)

    with client:
        response = client.post(
            f"/internal/collaboration/documents/{DOCUMENT_ID}/commands:lookup",
            headers=INTERNAL_HEADERS,
            json={
                "generation": 4,
                "tenant_id": INTERNAL_TENANT_ID,
                "command_id": str(command_id),
                "command_payload_hash": command_hash,
                "instance_id": "node-a",
                "epoch": 7,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "found": True,
        "actor_kind": "human",
        "actor_user_id": str(ACTOR_ID),
        "change_kind": "decision",
        "command_id": str(command_id),
        "command_payload_hash": command_hash,
        "decision": "accept",
        "generation": 4,
        "patch_ids": [
            str(uuid.UUID("66666666-6666-4666-8666-666666666666"))
        ],
        "sequence": 12,
        "suggestion_ids": [],
        "update_hash": _sha256(b"binary-yjs-update"),
    }
    assert service.command_lookups == [
        {
            "tenant_id": INTERNAL_TENANT_ID,
            "document_id": DOCUMENT_ID,
            "generation": 4,
            "command_id": command_id,
            "command_payload_hash": command_hash,
            "instance_id": "node-a",
            "epoch": 7,
        }
    ]


def test_internal_command_lookup_reports_missing_without_mutation() -> None:
    """An unseen command is a typed miss so the sidecar may execute it once."""
    service = _InternalService()
    client = _internal_client(service)

    with client:
        response = client.post(
            f"/internal/collaboration/documents/{DOCUMENT_ID}/commands:lookup",
            headers=INTERNAL_HEADERS,
            json={
                "generation": 4,
                "tenant_id": INTERNAL_TENANT_ID,
                "command_id": str(
                    uuid.UUID("55555555-5555-4555-8555-555555555555")
                ),
                "command_payload_hash": _sha256(b"canonical-command"),
                "instance_id": "node-a",
                "epoch": 7,
            },
        )

    assert response.status_code == 200
    assert response.json() == {"found": False}


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("update_base64", "not!base64", "invalid_update_base64"),
        ("update_base64", "YR==", "non_canonical_update_base64"),
        ("update_hash", "0" * 64, "update_hash_mismatch"),
        ("update_hash", "A" * 64, "invalid_update_hash"),
    ],
)
def test_internal_update_rejects_noncanonical_base64_and_bad_hashes(
    field: str,
    value: str,
    reason: str,
) -> None:
    """Malformed bytes and unverifiable digests never reach persistence."""
    service = _InternalService()
    client = _internal_client(service)
    payload = _update_payload()
    payload[field] = value

    with client:
        response = client.post(
            f"/internal/collaboration/documents/{DOCUMENT_ID}/updates",
            headers=INTERNAL_HEADERS,
            json=payload,
        )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "Invalid collaboration request.",
        "type": "invalid_request_error",
        "reason": reason,
    }
    assert service.persisted_updates == []


def _snapshot_payload(state_update: bytes = b"compacted-state") -> dict[str, Any]:
    """Return a valid internal snapshot request."""
    projection = "projection-at-12"
    return {
        "tenant_id": INTERNAL_TENANT_ID,
        "generation": 4,
        "covered_sequence": 12,
        "state_update_base64": base64.b64encode(state_update).decode("ascii"),
        "state_vector_base64": base64.b64encode(b"compacted-vector").decode(
            "ascii"
        ),
        "state_hash": _sha256(state_update),
        "projection_markdown": projection,
        "projection_hash": _sha256(projection.encode("utf-8")),
        "schema_version": 1,
        "schema_hash": _sha256(b"schema"),
        "instance_id": "node-a",
        "epoch": 7,
    }


def test_internal_snapshot_validates_binary_contract_before_acknowledgement() -> None:
    """A 204 means the service received a verified, fenced snapshot DTO."""
    service = _InternalService()
    client = _internal_client(service)

    with client:
        response = client.post(
            f"/internal/collaboration/documents/{DOCUMENT_ID}/snapshots",
            headers=INTERNAL_HEADERS,
            json=_snapshot_payload(),
        )

    assert response.status_code == 204
    snapshot, projection, instance_id, epoch, tenant_id = service.stored_snapshots[0]
    assert snapshot.state_update == b"compacted-state"
    assert snapshot.state_vector == b"compacted-vector"
    assert snapshot.state_hash == _sha256(b"compacted-state")
    assert snapshot.covered_sequence == 12
    assert projection == "projection-at-12"
    assert instance_id == "node-a"
    assert epoch == 7
    assert tenant_id == INTERNAL_TENANT_ID


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        (
            "state_update_base64",
            "YR==",
            "non_canonical_state_update_base64",
        ),
        (
            "state_vector_base64",
            "YR==",
            "non_canonical_state_vector_base64",
        ),
        ("state_hash", "0" * 64, "state_hash_mismatch"),
        ("projection_hash", "F" * 64, "invalid_projection_hash"),
    ],
)
def test_internal_snapshot_rejects_unverifiable_payloads(
    field: str,
    value: str,
    reason: str,
) -> None:
    """Snapshot validation is fail-closed for every binary authority field."""
    service = _InternalService()
    client = _internal_client(service)
    payload = _snapshot_payload()
    payload[field] = value

    with client:
        response = client.post(
            f"/internal/collaboration/documents/{DOCUMENT_ID}/snapshots",
            headers=INTERNAL_HEADERS,
            json=payload,
        )

    assert response.status_code == 400
    assert response.json()["error"]["reason"] == reason
    assert service.stored_snapshots == []


@pytest.mark.parametrize(
    ("error", "status", "reason", "current_sequence"),
    [
        (CollaborationDocumentNotFound(DOCUMENT_ID), 404, "not_found", None),
        (CollaborationLeaseInvalid("lease_expired"), 401, "lease_expired", None),
        (CollaborationLeaseInvalid("access_revoked"), 403, "access_revoked", None),
        (
            CollaborationAuthenticationRequired("inactive"),
            401,
            "lease_invalid",
            None,
        ),
        (CollaborationInstanceFenced("stale"), 409, "instance_fenced", None),
        (
            CollaborationConflict("sequence_conflict", current_sequence=19),
            409,
            "sequence_conflict",
            19,
        ),
        (CollaborationDocumentTooLarge("frame"), 413, "payload_too_large", None),
    ],
)
def test_internal_domain_errors_have_stable_http_semantics(
    error: Exception,
    status: int,
    reason: str,
    current_sequence: int | None,
) -> None:
    """The sidecar can distinguish auth, fencing, conflict, and size failures."""
    service = _InternalService()
    service.error = error
    client = _internal_client(service)

    with client:
        response = client.post(
            "/internal/collaboration/leases/introspect",
            headers=INTERNAL_HEADERS,
            json={
                "lease_token": "opaque-lease",
                "room": f"inqtrix-editor-v1:{DOCUMENT_ID}:g4",
                "instance_id": "node-a",
                "epoch": 7,
            },
        )

    assert response.status_code == status
    assert response.json()["error"]["reason"] == reason
    if current_sequence is not None:
        assert response.json()["error"]["current_sequence"] == current_sequence


class _Registry:
    """Empty algorithm registry required by the capability manifest."""

    def ids(self) -> tuple[str, ...]:
        """Return no optional algorithms."""
        return ()

    def manifest(self) -> list[dict[str, Any]]:
        """Return an empty public algorithm manifest."""
        return []


class _AvailabilityService:
    """Collaboration readiness probe fake."""

    def __init__(self, available: bool) -> None:
        self.available = available
        self.calls = 0

    async def service_available(self) -> bool:
        """Return the current sidecar readiness result."""
        self.calls += 1
        return self.available


def _capabilities_client(service: _AvailabilityService | None) -> TestClient:
    """Build the real manifest route with a minimal no-infrastructure container."""
    settings = SimpleNamespace(
        server=SimpleNamespace(enable_openapi=False),
        queue=SimpleNamespace(backend="memory"),
        storage=SimpleNamespace(backend="memory", max_file_bytes=1024),
        collaboration=SimpleNamespace(protocol_version=3, schema_version=5),
        agent=SimpleNamespace(
            agent_tier=None,
            depth="normal",
            max_total_seconds=300,
            reasoning_timeout=120,
            editor_assistant_timeout=120,
            search_timeout=90,
            claim_extract_timeout=60,
        ),
        agent_platform=SimpleNamespace(
            default_autonomy="balanced",
            default_agent_mode="workspace_agent",
            max_clarification_rounds=2,
            advanced_autonomy=False,
            max_parallel_children=2,
            discovery_max_tool_calls=4,
            max_plan_tasks=12,
            skills_max_attached=5,
            skills_disclosure_budget_chars=20_000,
        ),
    )
    registry = _Registry()
    container = SimpleNamespace(
        settings=settings,
        registry=registry,
        knowledge_service=None,
        knowledge_ceiling=None,
        file_service=None,
        object_store_backend="none",
        capability_registry=None,
        editor_collaboration_service=service,
        share_service=None,
        prompt_template_service=None,
        skill_service=None,
        quota_service=None,
        chat_history_service=None,
    )
    app = FastAPI()
    app.include_router(capabilities_router.build_router(container))
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    ("service", "configured", "available"),
    [
        (None, False, False),
        (_AvailabilityService(False), True, False),
        (_AvailabilityService(True), True, True),
    ],
)
def test_capabilities_publish_configured_and_live_collaboration_state(
    service: _AvailabilityService | None,
    configured: bool,
    available: bool,
) -> None:
    """Clients see disabled, degraded, and ready states without private URLs."""
    client = _capabilities_client(service)

    with client:
        response = client.get("/v1/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert payload["features"]["collaboration"] is available
    assert payload["collaboration"] == {
        "configured": configured,
        "service_available": available,
        "transport_path": "/collaboration",
        "protocol_version": 3,
        "schema_version": 5,
        "mode": "single_replica",
    }
    assert INTERNAL_SECRET not in response.text
