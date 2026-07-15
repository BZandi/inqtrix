"""Share-service contracts over the memory identity store.

The service exposes direct user shares only.  Canonical user UUIDs are the
authority boundary; external IdP subjects never enter share records.  Active
duplicates conflict, permission changes use optimistic revisions, acceptance
is idempotent, and either the owner or the recipient may remove a share.
"""

from __future__ import annotations

import uuid
from typing import cast

import pytest

from inqtrix.auth.directory import MemoryUserDirectory
from inqtrix.auth.identity_memory import MemoryIdentityStore
from inqtrix.auth.memory_authority import (
    MemoryAuthorityCoordinator,
    MemoryResourceSnapshot,
)
from inqtrix.auth.permissions import (
    AuthorizationService,
    SharePermission,
    WorkspaceRole,
)
from inqtrix.auth.principal import Principal
from inqtrix.auth.shares import (
    ShareBackendUnsupported,
    ShareConflict,
    ShareNotAllowed,
    ShareRecord,
    ShareService,
    ShareValidationError,
)

OWNER_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
RECIPIENT_ID = uuid.UUID("22222222-2222-4222-8222-222222222222")
SECOND_RECIPIENT_ID = uuid.UUID("33333333-3333-4333-8333-333333333333")
STRANGER_ID = uuid.UUID("44444444-4444-4444-8444-444444444444")
FOREIGN_OWNER_ID = uuid.UUID("55555555-5555-4555-8555-555555555555")
UNKNOWN_USER_ID = uuid.UUID("66666666-6666-4666-8666-666666666666")

OWNER = Principal(user_id=OWNER_ID, kind="oidc_session")
RECIPIENT = Principal(user_id=RECIPIENT_ID, kind="oidc_session")
SECOND_RECIPIENT = Principal(
    user_id=SECOND_RECIPIENT_ID, kind="oidc_session"
)
STRANGER = Principal(user_id=STRANGER_ID, kind="oidc_session")

RESOURCES = {
    "run_owned": OWNER_ID,
    "run_foreign": FOREIGN_OWNER_ID,
    "editor_owned": OWNER_ID,
}
KNOWN_USERS = {
    OWNER_ID,
    RECIPIENT_ID,
    SECOND_RECIPIENT_ID,
    STRANGER_ID,
}


def make_service(
    store: MemoryIdentityStore | None = None,
    *,
    unsupported_resource_types: tuple[str, ...] = (),
) -> tuple[ShareService, AuthorizationService, MemoryIdentityStore]:
    """Build the direct-share service with deterministic UUID identities."""
    identity = store or MemoryIdentityStore()
    permissions = AuthorizationService(
        members=identity,
        shares=identity,
        audit=identity,
    )

    async def resolve_resource_owner(
        tenant_id: str, resource_id: str
    ) -> uuid.UUID | None:
        return RESOURCES.get(resource_id)

    async def resolve_resource_title(
        tenant_id: str, resource_id: str
    ) -> str | None:
        return {
            "run_owned": "Meine Recherche",
            "editor_owned": "Draft report",
        }.get(resource_id)

    async def user_exists(tenant_id: str, user_id: uuid.UUID) -> bool:
        return user_id in KNOWN_USERS

    service = ShareService(
        shares=identity,
        permissions=permissions,
        owner_resolvers={
            "run": resolve_resource_owner,
            "editor_document": resolve_resource_owner,
        },
        title_resolvers={
            "run": resolve_resource_title,
            "editor_document": resolve_resource_title,
        },
        user_lookup=user_exists,
        audit=identity,
        unsupported_resource_types=unsupported_resource_types,
    )
    return service, permissions, identity


async def grant(
    service: ShareService,
    principal: Principal = OWNER,
    resource_type: str = "run",
    resource_id: str = "run_owned",
    invitees: tuple[tuple[uuid.UUID, SharePermission], ...] = (
        (RECIPIENT_ID, SharePermission.VIEW),
    ),
) -> tuple[ShareRecord, ...]:
    """Grant the selected resource to the default recipient."""
    return await service.grant(
        principal,
        resource_type=resource_type,
        resource_id=resource_id,
        invitees=list(invitees),
    )


class TestGrant:
    @pytest.mark.asyncio
    async def test_grant_is_pending_until_recipient_accepts(self):
        service, permissions, _identity = make_service()

        created = (await grant(service))[0]

        assert created.recipient_user_id == RECIPIENT_ID
        assert created.permission is SharePermission.VIEW
        assert created.accepted_at is None
        assert not await permissions.can(
            RECIPIENT,
            SharePermission.VIEW,
            owner_user_id=OWNER_ID,
            resource_tenant_id="default",
            resource_type="run",
            resource_id="run_owned",
        )

        accepted = await service.accept(RECIPIENT, share_id=created.id)

        assert accepted is not None
        assert accepted.accepted_at is not None
        assert await permissions.can(
            RECIPIENT,
            SharePermission.VIEW,
            owner_user_id=OWNER_ID,
            resource_tenant_id="default",
            resource_type="run",
            resource_id="run_owned",
        )

    @pytest.mark.asyncio
    async def test_editor_document_accepts_suggest_grant(self) -> None:
        service, permissions, _identity = make_service()

        created = (
            await grant(
                service,
                resource_type="editor_document",
                resource_id="editor_owned",
                invitees=((RECIPIENT_ID, SharePermission.SUGGEST),),
            )
        )[0]
        accepted = await service.accept(RECIPIENT, share_id=created.id)

        assert accepted is not None
        assert accepted.permission is SharePermission.SUGGEST
        assert await permissions.can(
            RECIPIENT,
            SharePermission.SUGGEST,
            owner_user_id=OWNER_ID,
            resource_tenant_id="default",
            resource_type="editor_document",
            resource_id="editor_owned",
        )
        assert not await permissions.can(
            RECIPIENT,
            SharePermission.EDIT,
            owner_user_id=OWNER_ID,
            resource_tenant_id="default",
            resource_type="editor_document",
            resource_id="editor_owned",
        )

    @pytest.mark.asyncio
    async def test_existing_resource_rejects_suggest_grant(self) -> None:
        service, _permissions, _identity = make_service()

        with pytest.raises(ShareValidationError, match="Ressourcentyp"):
            await grant(
                service,
                invitees=((RECIPIENT_ID, SharePermission.SUGGEST),),
            )

        assert await service.list_for_resource(
            OWNER,
            resource_type="run",
            resource_id="run_owned",
        ) == ()

    @pytest.mark.asyncio
    async def test_active_duplicate_conflicts_without_mutating_share(self):
        service, _permissions, _identity = make_service()
        original = (await grant(service))[0]

        with pytest.raises(ShareConflict) as error:
            await grant(
                service,
                invitees=((RECIPIENT_ID, SharePermission.EDIT),),
            )

        assert error.value.current_revision is None
        listed = await service.list_for_resource(
            OWNER,
            resource_type="run",
            resource_id="run_owned",
        )
        assert listed == (original,)

    @pytest.mark.asyncio
    async def test_only_owner_can_grant(self):
        service, _permissions, _identity = make_service()

        with pytest.raises(ShareNotAllowed):
            await grant(service, principal=STRANGER)
        with pytest.raises(ShareNotAllowed):
            await grant(service, resource_id="run_foreign")
        with pytest.raises(ShareNotAllowed):
            await grant(service, resource_id="run_missing")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("invitee", "permission"),
        [
            (OWNER_ID, SharePermission.VIEW),
            (RECIPIENT_ID, cast(SharePermission, "manage")),
            (UNKNOWN_USER_ID, SharePermission.VIEW),
        ],
    )
    async def test_invalid_grants_fail_loudly(
        self, invitee: uuid.UUID, permission: SharePermission
    ):
        service, _permissions, _identity = make_service()

        with pytest.raises(ShareValidationError):
            await grant(service, invitees=((invitee, permission),))

    @pytest.mark.asyncio
    async def test_unknown_resource_type_is_a_validation_error(self):
        service, _permissions, _identity = make_service()

        with pytest.raises(ShareValidationError, match="Ressourcentyp"):
            await service.grant(
                OWNER,
                resource_type="comet",
                resource_id="x",
                invitees=[(RECIPIENT_ID, SharePermission.VIEW)],
            )

    @pytest.mark.asyncio
    async def test_known_resource_without_transactional_backend_is_unsupported(self):
        service, _permissions, _identity = make_service(
            unsupported_resource_types=("knowledge_collection",)
        )

        with pytest.raises(ShareBackendUnsupported, match="knowledge_collection"):
            await service.grant(
                OWNER,
                resource_type="knowledge_collection",
                resource_id="kc_unsafe",
                invitees=[(RECIPIENT_ID, SharePermission.VIEW)],
            )
        assert await service.accepted_for_recipient(
            RECIPIENT, resource_type="knowledge_collection"
        ) == {}

    @pytest.mark.asyncio
    async def test_workspace_reconcile_uses_owner_not_historical_grantor(self):
        identity = MemoryIdentityStore(restrict_to_workspace_members=True)
        users = MemoryUserDirectory()
        for user_id, subject in (
            (OWNER_ID, "owner"),
            (RECIPIENT_ID, "recipient"),
            (STRANGER_ID, "stranger"),
        ):
            await users.record_login(
                tenant_id="default",
                issuer="https://issuer.example",
                subject=subject,
                email=f"{subject}@example.com",
                email_verified=True,
                display_name=subject.title(),
                canonical_user_id=user_id,
            )
        owners = {"run_owned": OWNER_ID}
        authority = MemoryAuthorityCoordinator()
        authority.bind_users(users)
        identity.bind_authority_coordinator(authority)
        authority.register_resource(
            "run",
            lambda tenant_id, resource_id: MemoryResourceSnapshot(
                exists=tenant_id == "default" and resource_id in owners,
                owner_user_id=owners.get(resource_id),
            ),
        )
        identity.add_workspace("workspace-a")
        identity.add_member("workspace-a", OWNER_ID, WorkspaceRole.OWNER)
        identity.add_member("workspace-a", RECIPIENT_ID, WorkspaceRole.VIEWER)
        identity.add_share(
            recipient_user_id=RECIPIENT_ID,
            resource_type="run",
            resource_id="run_owned",
            permission=SharePermission.VIEW,
            granted_by_user_id=STRANGER_ID,
        )

        revoked = await identity.reconcile_workspace_shares(
            tenant_id="default"
        )

        assert revoked == 0
        assert await identity.permission_for(
            tenant_id="default",
            resource_type="run",
            resource_id="run_owned",
            recipient_user_id=RECIPIENT_ID,
        ) is SharePermission.VIEW

        owners["run_owned"] = STRANGER_ID
        revoked = await identity.reconcile_workspace_shares(
            tenant_id="default"
        )

        assert revoked == 1
        assert await identity.permission_for(
            tenant_id="default",
            resource_type="run",
            resource_id="run_owned",
            recipient_user_id=RECIPIENT_ID,
        ) is None


class TestPermissionUpdate:
    @pytest.mark.asyncio
    async def test_owner_updates_permission_at_expected_revision(self):
        service, permissions, _identity = make_service()
        created = (await grant(service))[0]
        accepted = await service.accept(RECIPIENT, share_id=created.id)
        assert accepted is not None

        updated = await service.update_permission(
            OWNER,
            share_id=created.id,
            permission=SharePermission.EDIT,
            expected_revision=accepted.revision,
        )

        assert updated.id == created.id
        assert updated.permission is SharePermission.EDIT
        assert updated.revision == accepted.revision + 1
        assert updated.accepted_at == accepted.accepted_at
        assert await permissions.can(
            RECIPIENT,
            SharePermission.EDIT,
            owner_user_id=OWNER_ID,
            resource_tenant_id="default",
            resource_type="run",
            resource_id="run_owned",
        )

    @pytest.mark.asyncio
    async def test_stale_revision_conflicts_without_lost_update(self):
        service, _permissions, _identity = make_service()
        created = (await grant(service))[0]
        current = await service.update_permission(
            OWNER,
            share_id=created.id,
            permission=SharePermission.EDIT,
            expected_revision=created.revision,
        )

        with pytest.raises(ShareConflict) as error:
            await service.update_permission(
                OWNER,
                share_id=created.id,
                permission=SharePermission.VIEW,
                expected_revision=created.revision,
            )

        assert error.value.current_revision == current.revision
        listed = await service.list_for_resource(
            OWNER,
            resource_type="run",
            resource_id="run_owned",
        )
        assert listed[0].permission is SharePermission.EDIT
        assert listed[0].revision == current.revision

    @pytest.mark.asyncio
    async def test_non_owner_cannot_update_permission(self):
        service, _permissions, _identity = make_service()
        created = (await grant(service))[0]

        with pytest.raises(ShareNotAllowed):
            await service.update_permission(
                RECIPIENT,
                share_id=created.id,
                permission=SharePermission.EDIT,
                expected_revision=created.revision,
            )

    @pytest.mark.asyncio
    async def test_existing_resource_rejects_suggest_update(self) -> None:
        service, _permissions, _identity = make_service()
        created = (await grant(service))[0]

        with pytest.raises(ShareValidationError, match="Ressourcentyp"):
            await service.update_permission(
                OWNER,
                share_id=created.id,
                permission=SharePermission.SUGGEST,
                expected_revision=created.revision,
            )

        (unchanged,) = await service.list_for_resource(
            OWNER,
            resource_type="run",
            resource_id="run_owned",
        )
        assert unchanged.permission is SharePermission.VIEW
        assert unchanged.revision == created.revision

    @pytest.mark.asyncio
    async def test_editor_document_updates_to_suggest(self) -> None:
        service, _permissions, _identity = make_service()
        created = (
            await grant(
                service,
                resource_type="editor_document",
                resource_id="editor_owned",
            )
        )[0]

        updated = await service.update_permission(
            OWNER,
            share_id=created.id,
            permission=SharePermission.SUGGEST,
            expected_revision=created.revision,
        )

        assert updated.permission is SharePermission.SUGGEST
        assert updated.revision == created.revision + 1


class TestAcceptance:
    @pytest.mark.asyncio
    async def test_accept_is_idempotent_for_active_recipient_share(self):
        service, _permissions, _identity = make_service()
        created = (await grant(service))[0]

        first = await service.accept(RECIPIENT, share_id=created.id)
        second = await service.accept(RECIPIENT, share_id=created.id)

        assert first is not None
        assert second == first

    @pytest.mark.asyncio
    async def test_foreign_unknown_and_removed_shares_cannot_be_accepted(self):
        service, _permissions, _identity = make_service()
        created = (await grant(service))[0]

        assert await service.accept(OWNER, share_id=created.id) is None
        assert await service.accept(STRANGER, share_id=created.id) is None
        assert await service.accept(RECIPIENT, share_id="missing") is None
        assert await service.remove(OWNER, share_id=created.id) is not None
        assert await service.accept(RECIPIENT, share_id=created.id) is None


class TestRemoval:
    @pytest.mark.asyncio
    async def test_owner_withdraws_share_and_access_disappears(self):
        service, permissions, identity = make_service()
        created = (await grant(service))[0]
        assert await service.accept(RECIPIENT, share_id=created.id) is not None

        assert await service.remove(OWNER, share_id=created.id) is not None

        assert not await permissions.can(
            RECIPIENT,
            SharePermission.VIEW,
            owner_user_id=OWNER_ID,
            resource_tenant_id="default",
            resource_type="run",
            resource_id="run_owned",
        )
        assert "share.revoked" in [
            entry.action for entry in identity.audit_entries
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("accept_first", [False, True])
    async def test_recipient_can_decline_or_leave_own_share(
        self, accept_first: bool
    ):
        service, permissions, identity = make_service()
        created = (await grant(service))[0]
        if accept_first:
            assert (
                await service.accept(RECIPIENT, share_id=created.id)
                is not None
            )

        assert await service.remove(RECIPIENT, share_id=created.id) is not None

        assert await service.inbox(RECIPIENT) == ()
        assert not await permissions.can(
            RECIPIENT,
            SharePermission.VIEW,
            owner_user_id=OWNER_ID,
            resource_tenant_id="default",
            resource_type="run",
            resource_id="run_owned",
        )
        expected_action = "share.left" if accept_first else "share.declined"
        assert expected_action in [
            entry.action for entry in identity.audit_entries
        ]

    @pytest.mark.asyncio
    async def test_stranger_and_unknown_share_removal_are_hidden(self):
        service, _permissions, _identity = make_service()
        created = (await grant(service))[0]

        assert await service.remove(STRANGER, share_id=created.id) is None
        assert await service.remove(OWNER, share_id="missing") is None
        assert len(await service.inbox(RECIPIENT)) == 1


class TestListings:
    @pytest.mark.asyncio
    async def test_only_owner_can_list_resource_shares(self):
        service, _permissions, _identity = make_service()
        created = (await grant(service))[0]

        assert await service.list_for_resource(
            OWNER,
            resource_type="run",
            resource_id="run_owned",
        ) == (created,)
        with pytest.raises(ShareNotAllowed):
            await service.list_for_resource(
                RECIPIENT,
                resource_type="run",
                resource_id="run_owned",
            )

    @pytest.mark.asyncio
    async def test_inbox_keeps_pending_and_accepted_direct_shares(self):
        service, _permissions, _identity = make_service()
        created = (await grant(service))[0]

        pending = await service.inbox(RECIPIENT)
        assert len(pending) == 1
        assert pending[0].share_id == created.id
        assert pending[0].accepted_at is None

        assert await service.accept(RECIPIENT, share_id=created.id) is not None
        accepted = await service.inbox(RECIPIENT)
        assert len(accepted) == 1
        assert accepted[0].accepted_at is not None

    @pytest.mark.asyncio
    async def test_inbox_skips_titleless_resources(self):
        service, _permissions, identity = make_service()
        identity.add_share(
            recipient_user_id=RECIPIENT_ID,
            resource_type="run",
            resource_id="run_pruned",
            permission=SharePermission.VIEW,
            granted_by_user_id=OWNER_ID,
            accepted=False,
        )

        assert await service.inbox(RECIPIENT) == ()
