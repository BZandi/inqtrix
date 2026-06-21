"""Unit tests for the permission chokepoint and the memory backend."""

from __future__ import annotations

import logging

import pytest

from inqtrix.auth.identity_memory import MemoryIdentityStore
from inqtrix.auth.permissions import (
    PermissionService,
    ResourceNotFound,
    SharePermission,
    SubjectRef,
    WorkspaceNotFound,
    WorkspaceRole,
)
from inqtrix.auth.principal import ANONYMOUS_PRINCIPAL, STATIC_PRINCIPAL, Principal


SCOPED_KINDS = ("oidc_session", "pat")
"""Both scoped principal kinds — PAT must be filtered exactly like an
OIDC session (pins the exclusion set against an inclusion refactor)."""


def oidc_principal(
    sub: str, *, tenant_id: str = "default", kind: str = "oidc_session"
) -> Principal:
    return Principal(sub=sub, kind=kind, tenant_id=tenant_id, role="member")


def make_service(store: MemoryIdentityStore | None = None) -> tuple[
    PermissionService, MemoryIdentityStore
]:
    identity = store or MemoryIdentityStore()
    return (
        PermissionService(
            members=identity, groups=identity, shares=identity, audit=identity
        ),
        identity,
    )


# ------------------------------------------------------------------ #
# Ordered enums
# ------------------------------------------------------------------ #


def test_workspace_role_ordering_is_total_and_ascending():
    assert WorkspaceRole.OWNER.at_least(WorkspaceRole.VIEWER)
    assert WorkspaceRole.EDITOR.at_least(WorkspaceRole.COMMENTER)
    assert not WorkspaceRole.VIEWER.at_least(WorkspaceRole.COMMENTER)
    assert WorkspaceRole.VIEWER.at_least(WorkspaceRole.VIEWER)


def test_share_permission_ordering_is_total_and_ascending():
    assert SharePermission.MANAGE.at_least(SharePermission.VIEW)
    assert SharePermission.EDIT.at_least(SharePermission.COMMENT)
    assert not SharePermission.VIEW.at_least(SharePermission.EDIT)


# ------------------------------------------------------------------ #
# Legacy principals stay unscoped
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
@pytest.mark.parametrize("principal", [ANONYMOUS_PRINCIPAL, STATIC_PRINCIPAL])
async def test_legacy_principals_resolve_to_no_scoping(principal):
    service, _ = make_service()
    assert await service.resolve_user_context(principal) is None
    # Workspace checks and resource checks pass unconditionally.
    assert await service.resolve_workspace(principal, "ws_any") == "ws_any"
    assert await service.can(
        principal, SharePermission.MANAGE, resource_type="report", resource_id="r1"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", SCOPED_KINDS)
async def test_scoped_principal_resolves_memberships_server_side(kind):
    service, store = make_service()
    store.add_workspace("ws_a")
    store.add_member("ws_a", "alice", WorkspaceRole.EDITOR)
    store.add_group("g_legal", ["alice"])

    context = await service.resolve_user_context(
        oidc_principal("alice", kind=kind)
    )

    assert context is not None
    assert context.workspace_ids == ("ws_a",)
    assert context.groups == ("g_legal",)


# ------------------------------------------------------------------ #
# Workspace resolution: 404-not-403, indistinguishable absence
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", SCOPED_KINDS)
async def test_non_member_and_unknown_workspace_raise_identically(kind):
    service, store = make_service()
    store.add_workspace("ws_b")
    store.add_member("ws_b", "bob", WorkspaceRole.OWNER)
    alice = oidc_principal("alice", kind=kind)

    with pytest.raises(WorkspaceNotFound):
        await service.resolve_workspace(alice, "ws_b")
    with pytest.raises(WorkspaceNotFound):
        await service.resolve_workspace(alice, "ws_does_not_exist")


@pytest.mark.asyncio
async def test_min_role_below_threshold_hides_the_workspace():
    service, store = make_service()
    store.add_workspace("ws_a")
    store.add_member("ws_a", "alice", WorkspaceRole.VIEWER)
    alice = oidc_principal("alice")

    assert await service.resolve_workspace(alice, "ws_a") == "ws_a"
    with pytest.raises(WorkspaceNotFound):
        await service.resolve_workspace(
            alice, "ws_a", min_role=WorkspaceRole.EDITOR
        )


@pytest.mark.asyncio
async def test_tenant_mismatch_hides_the_workspace():
    service, store = make_service()
    store.add_workspace("ws_a", tenant_id="tenant-x")
    store.add_member("ws_a", "alice", WorkspaceRole.OWNER)

    with pytest.raises(WorkspaceNotFound):
        await service.resolve_workspace(
            oidc_principal("alice", tenant_id="default"), "ws_a"
        )


# ------------------------------------------------------------------ #
# Grant union: workspace role vs direct share vs group share
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_can_unions_role_share_and_group_grants_by_max_rank():
    service, store = make_service()
    store.add_workspace("ws_a")
    store.add_member("ws_a", "alice", WorkspaceRole.VIEWER)
    store.add_group("g_legal", ["alice"])
    store.add_share(
        subject_type="group",
        subject_id="g_legal",
        resource_type="report",
        resource_id="r1",
        permission=SharePermission.EDIT,
    )
    alice = oidc_principal("alice")

    # Workspace role alone: viewer -> view yes, edit no.
    assert await service.can(
        alice, SharePermission.VIEW,
        resource_type="report", resource_id="r2", workspace_id="ws_a",
    )
    assert not await service.can(
        alice, SharePermission.EDIT,
        resource_type="report", resource_id="r2", workspace_id="ws_a",
    )
    # Group share lifts r1 to edit even though the role only views.
    assert await service.can(
        alice, SharePermission.EDIT,
        resource_type="report", resource_id="r1", workspace_id="ws_a",
    )
    # Nothing grants manage.
    assert not await service.can(
        alice, SharePermission.MANAGE,
        resource_type="report", resource_id="r1", workspace_id="ws_a",
    )


@pytest.mark.asyncio
async def test_direct_share_wins_when_higher_than_group_share():
    service, store = make_service()
    store.add_group("g", ["alice"])
    store.add_share(
        subject_type="group", subject_id="g",
        resource_type="file", resource_id="f1",
        permission=SharePermission.VIEW,
    )
    store.add_share(
        subject_type="user", subject_id="alice",
        resource_type="file", resource_id="f1",
        permission=SharePermission.MANAGE,
    )

    assert await service.can(
        oidc_principal("alice"), SharePermission.MANAGE,
        resource_type="file", resource_id="f1",
    )


@pytest.mark.asyncio
async def test_revoked_share_stops_granting():
    service, store = make_service()
    store.add_share(
        subject_type="user", subject_id="alice",
        resource_type="file", resource_id="f1",
        permission=SharePermission.EDIT,
    )
    alice = oidc_principal("alice")
    assert await service.can(
        alice, SharePermission.EDIT, resource_type="file", resource_id="f1"
    )

    store.revoke_share(
        subject_type="user", subject_id="alice",
        resource_type="file", resource_id="f1",
    )
    assert not await service.can(
        alice, SharePermission.EDIT, resource_type="file", resource_id="f1"
    )


# ------------------------------------------------------------------ #
# require(): hidden from the client, loud for operators
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_require_denial_raises_not_found_and_is_audited(caplog):
    service, store = make_service()
    alice = oidc_principal("alice")

    with caplog.at_level(logging.WARNING, logger="inqtrix"):
        with pytest.raises(ResourceNotFound):
            await service.require(
                alice, SharePermission.VIEW,
                resource_type="report", resource_id="r1",
            )

    assert any("authz denied" in message for message in caplog.messages)
    assert len(store.audit_entries) == 1
    entry = store.audit_entries[0]
    assert entry.action == "authz.denied"
    assert entry.actor_sub == "alice"
    assert entry.resource_id == "r1"
    assert entry.detail == {"permission": "view"}


@pytest.mark.asyncio
async def test_require_passes_silently_when_granted():
    service, store = make_service()
    store.add_share(
        subject_type="user", subject_id="alice",
        resource_type="report", resource_id="r1",
        permission=SharePermission.VIEW,
    )

    await service.require(
        oidc_principal("alice"), SharePermission.VIEW,
        resource_type="report", resource_id="r1",
    )
    assert store.audit_entries == []


@pytest.mark.asyncio
async def test_shares_are_tenant_scoped():
    service, store = make_service()
    store.add_share(
        subject_type="user", subject_id="alice",
        resource_type="file", resource_id="f1",
        permission=SharePermission.MANAGE,
        tenant_id="tenant-x",
    )

    assert not await service.can(
        oidc_principal("alice", tenant_id="default"),
        SharePermission.VIEW,
        resource_type="file", resource_id="f1",
    )


# ------------------------------------------------------------------ #
# Settings bridge
# ------------------------------------------------------------------ #


def test_postgres_backend_without_url_fails_loudly():
    from inqtrix.server.container import build_permission_service
    from inqtrix.settings import Settings, StorageSettings

    with pytest.raises(RuntimeError, match="INQTRIX_DATABASE_URL"):
        build_permission_service(
            Settings(storage=StorageSettings(backend="postgres"))
        )


def test_storage_settings_survive_serialization_round_trip():
    from inqtrix.settings import Settings, StorageSettings

    settings = Settings(
        storage=StorageSettings(
            backend="postgres",
            database_url="postgresql+asyncpg://example/db",
        )
    )
    restored = Settings.model_validate(settings.model_dump())
    assert restored.storage.backend == "postgres"
    assert restored.storage.database_url == "postgresql+asyncpg://example/db"
    assert restored.storage.app_role == "inqtrix_app"


# ------------------------------------------------------------------ #
# Memory store arrangement guards
# ------------------------------------------------------------------ #


def test_add_member_to_unknown_workspace_fails_loudly():
    store = MemoryIdentityStore()
    with pytest.raises(KeyError, match="unknown workspace"):
        store.add_member("ws_missing", "alice", WorkspaceRole.OWNER)


@pytest.mark.asyncio
async def test_permission_for_with_no_subjects_is_none():
    store = MemoryIdentityStore()
    assert (
        await store.permission_for(
            tenant_id="default",
            resource_type="file",
            resource_id="f1",
            subjects=[],
        )
        is None
    )


@pytest.mark.asyncio
async def test_subject_ref_shape_matches_share_lookup():
    store = MemoryIdentityStore()
    store.add_share(
        subject_type="user", subject_id="alice",
        resource_type="file", resource_id="f1",
        permission=SharePermission.COMMENT,
    )
    grant = await store.permission_for(
        tenant_id="default",
        resource_type="file",
        resource_id="f1",
        subjects=[SubjectRef(subject_type="user", subject_id="alice")],
    )
    assert grant is SharePermission.COMMENT
