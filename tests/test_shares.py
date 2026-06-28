"""Share-service tests over the memory identity store.

Pins the v1 contracts: only owners (or manage grants) may grant and
revoke, re-grants replace the permission, denials are
indistinguishable from absence, shared-with-me reduces to the highest
grant, and consent gates access — a freshly minted share is pending and
grants nothing until the recipient accepts (one store backs both
surfaces — no split brain).
"""

from __future__ import annotations

import pytest

from inqtrix.auth.identity_memory import MemoryIdentityStore
from inqtrix.auth.permissions import (
    PermissionService,
    SharePermission,
)
from inqtrix.auth.principal import Principal
from inqtrix.auth.shares import (
    ShareNotAllowed,
    ShareService,
    ShareValidationError,
)

OWNER = Principal(sub="owner-1", kind="oidc_session")
RECIPIENT = Principal(sub="user-2", kind="oidc_session")
STRANGER = Principal(sub="user-3", kind="oidc_session")

RESOURCES = {"run_owned": "owner-1", "run_foreign": "someone-else"}
KNOWN_USERS = {"owner-1", "user-2", "user-3"}


def make_service(store: MemoryIdentityStore | None = None):
    identity = store or MemoryIdentityStore()
    permissions = PermissionService(
        members=identity, groups=identity, shares=identity, audit=identity
    )

    async def resolve_run_owner(tenant_id: str, resource_id: str):
        return RESOURCES.get(resource_id)

    async def resolve_run_title(tenant_id: str, resource_id: str):
        # Only the genuinely-owned resource carries a title; a share on a
        # resource without one (e.g. pruned) must be skipped by the listings.
        return {"run_owned": "Meine Recherche"}.get(resource_id)

    async def user_exists(tenant_id: str, sub: str) -> bool:
        return sub in KNOWN_USERS

    service = ShareService(
        shares=identity,
        permissions=permissions,
        owner_resolvers={"run": resolve_run_owner},
        title_resolvers={"run": resolve_run_title},
        user_lookup=user_exists,
        audit=identity,
    )
    return service, permissions, identity


async def grant(service, principal=OWNER, resource_id="run_owned",
                invitees=(("user-2", SharePermission.VIEW),)):
    return await service.grant(
        principal,
        resource_type="run",
        resource_id=resource_id,
        invitees=list(invitees),
    )


class TestGrant:
    @pytest.mark.asyncio
    async def test_grant_is_pending_until_accepted(self):
        """A minted share grants nothing until the recipient consents; the
        permission layer only honours it after :meth:`accept` (the consent
        gate, enforced once in ``permission_for``)."""
        service, permissions, _identity = make_service()
        created = await grant(service)
        assert len(created) == 1
        assert created[0].permission is SharePermission.VIEW
        assert created[0].accepted_at is None
        # Pending: no access yet.
        assert not await permissions.can(
            RECIPIENT,
            SharePermission.VIEW,
            resource_type="run",
            resource_id="run_owned",
        )
        # Consent flips access on; the level is still VIEW, not EDIT.
        assert await service.accept(RECIPIENT, share_id=created[0].id)
        assert await permissions.can(
            RECIPIENT,
            SharePermission.VIEW,
            resource_type="run",
            resource_id="run_owned",
        )
        assert not await permissions.can(
            RECIPIENT,
            SharePermission.EDIT,
            resource_type="run",
            resource_id="run_owned",
        )

    @pytest.mark.asyncio
    async def test_only_the_recipient_can_accept(self):
        """Consent is the recipient's alone: owner/stranger accept is a
        no-op ``False`` and never grants access."""
        service, permissions, _identity = make_service()
        created = await grant(service)
        assert not await service.accept(OWNER, share_id=created[0].id)
        assert not await service.accept(STRANGER, share_id=created[0].id)
        assert not await permissions.can(
            RECIPIENT,
            SharePermission.VIEW,
            resource_type="run",
            resource_id="run_owned",
        )
        # The genuine recipient still can; double-accept is a benign False.
        assert await service.accept(RECIPIENT, share_id=created[0].id)
        assert not await service.accept(RECIPIENT, share_id=created[0].id)

    @pytest.mark.asyncio
    async def test_regrant_preserves_acceptance(self):
        """A permission change on an already-accepted share must NOT drop the
        recipient back to pending — access stays live across the re-grant."""
        service, permissions, _identity = make_service()
        created = await grant(service)
        assert await service.accept(RECIPIENT, share_id=created[0].id)
        # Owner raises the level; the recipient does not re-consent.
        await grant(service, invitees=(("user-2", SharePermission.EDIT),))
        assert await permissions.can(
            RECIPIENT,
            SharePermission.EDIT,
            resource_type="run",
            resource_id="run_owned",
        )

    @pytest.mark.asyncio
    async def test_regrant_replaces_the_permission(self):
        service, permissions, _identity = make_service()
        await grant(service)
        await grant(service, invitees=(("user-2", SharePermission.EDIT),))
        listed = await service.list_for_resource(
            OWNER, resource_type="run", resource_id="run_owned"
        )
        assert len(listed) == 1
        assert listed[0].permission is SharePermission.EDIT

    @pytest.mark.asyncio
    async def test_non_owner_cannot_grant(self):
        service, _permissions, _identity = make_service()
        with pytest.raises(ShareNotAllowed):
            await grant(service, principal=STRANGER)

    @pytest.mark.asyncio
    async def test_foreign_resource_is_not_allowed(self):
        service, _permissions, _identity = make_service()
        with pytest.raises(ShareNotAllowed):
            await grant(service, resource_id="run_foreign")

    @pytest.mark.asyncio
    async def test_vanished_resource_is_indistinguishable(self):
        service, _permissions, _identity = make_service()
        with pytest.raises(ShareNotAllowed):
            await grant(service, resource_id="run_missing")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "invitee, permission",
        [
            ("owner-1", SharePermission.VIEW),   # owner needs no share
            ("user-2", SharePermission.MANAGE),  # not grantable in v1
            ("ghost", SharePermission.VIEW),     # unknown user
        ],
    )
    async def test_invalid_grants_fail_loudly(self, invitee, permission):
        service, _permissions, _identity = make_service()
        with pytest.raises(ShareValidationError):
            await grant(service, invitees=((invitee, permission),))

    @pytest.mark.asyncio
    async def test_manage_holder_cannot_grant_to_themselves(self):
        """A manage-grant holder must not escalate their own level by
        re-granting themselves — the self check fires before any
        write."""
        service, _permissions, identity = make_service()
        identity.add_share(
            subject_type="user",
            subject_id="user-2",
            resource_type="run",
            resource_id="run_owned",
            permission=SharePermission.MANAGE,
        )
        with pytest.raises(ShareValidationError):
            await service.grant(
                RECIPIENT,
                resource_type="run",
                resource_id="run_owned",
                invitees=[("user-2", SharePermission.EDIT)],
            )

    @pytest.mark.asyncio
    async def test_unknown_resource_type_is_a_validation_error(self):
        service, _permissions, _identity = make_service()
        with pytest.raises(ShareValidationError, match="Ressourcentyp"):
            await service.grant(
                OWNER,
                resource_type="comet",
                resource_id="x",
                invitees=[("user-2", SharePermission.VIEW)],
            )


class TestRevoke:
    @pytest.mark.asyncio
    async def test_owner_revokes_and_access_disappears(self):
        service, permissions, _identity = make_service()
        created = await grant(service)
        assert await service.accept(RECIPIENT, share_id=created[0].id)
        assert await permissions.can(
            RECIPIENT,
            SharePermission.VIEW,
            resource_type="run",
            resource_id="run_owned",
        )
        assert await service.revoke(OWNER, share_id=created[0].id)
        assert not await permissions.can(
            RECIPIENT,
            SharePermission.VIEW,
            resource_type="run",
            resource_id="run_owned",
        )

    @pytest.mark.asyncio
    async def test_recipient_cannot_revoke(self):
        """A view grant is not a manage grant — the recipient cannot
        cut off other recipients (or themselves silently)."""
        service, _permissions, _identity = make_service()
        created = await grant(service)
        assert not await service.revoke(RECIPIENT, share_id=created[0].id)

    @pytest.mark.asyncio
    async def test_unknown_share_id_is_false(self):
        service, _permissions, _identity = make_service()
        assert not await service.revoke(OWNER, share_id="missing")


class TestListings:
    @pytest.mark.asyncio
    async def test_shared_with_me_reduces_to_highest_grant(self):
        service, _permissions, identity = make_service()
        created = await grant(service)
        await service.accept(RECIPIENT, share_id=created[0].id)
        # A group grant on the same resource with a higher level (seed seam
        # arranges it as already accepted).
        identity.add_group("g1", ["user-2"])
        identity.add_share(
            subject_type="group",
            subject_id="g1",
            resource_type="run",
            resource_id="run_owned",
            permission=SharePermission.EDIT,
        )
        shared = await service.shared_with_me(
            RECIPIENT, resource_type="run"
        )
        assert set(shared) == {"run_owned"}
        assert shared["run_owned"].permission is SharePermission.EDIT

    @pytest.mark.asyncio
    async def test_shared_with_me_excludes_pending(self):
        """A pending share is not yet shared-in: the consent gate keeps it out
        of the visibility union until accepted."""
        service, _permissions, _identity = make_service()
        await grant(service)
        assert (
            await service.shared_with_me(RECIPIENT, resource_type="run") == {}
        )

    @pytest.mark.asyncio
    async def test_outgoing_counts_count_active_shares(self):
        service, _permissions, _identity = make_service()
        await grant(service)
        await grant(service, invitees=(("user-3", SharePermission.VIEW),))
        counts = await service.outgoing_counts(
            OWNER, resource_type="run", resource_ids=["run_owned", "other"]
        )
        assert counts == {"run_owned": 2}

    @pytest.mark.asyncio
    async def test_outgoing_counts_hide_foreign_resources(self):
        """Counts are an existence oracle — strangers learn nothing."""
        service, _permissions, _identity = make_service()
        await grant(service)
        probed = await service.outgoing_counts(
            STRANGER, resource_type="run", resource_ids=["run_owned"]
        )
        assert probed == {}
        with pytest.raises(ShareValidationError):
            await service.outgoing_counts(
                STRANGER, resource_type="comet", resource_ids=["x"]
            )

    @pytest.mark.asyncio
    async def test_outgoing_groups_with_pending_count(self):
        service, _permissions, _identity = make_service()
        first = await grant(service)  # user-2, pending
        await grant(service, invitees=(("user-3", SharePermission.VIEW),))
        await service.accept(RECIPIENT, share_id=first[0].id)
        items = await service.outgoing(OWNER)
        assert len(items) == 1
        item = items[0]
        assert item.resource_id == "run_owned"
        assert item.resource_title == "Meine Recherche"
        assert item.share_count == 2
        # user-3 has not consented yet.
        assert item.pending_count == 1

    @pytest.mark.asyncio
    async def test_list_for_resource_requires_view(self):
        service, _permissions, _identity = make_service()
        created = await grant(service)
        # A pending recipient has no view access yet, so cannot list either.
        with pytest.raises(ShareNotAllowed):
            await service.list_for_resource(
                RECIPIENT, resource_type="run", resource_id="run_owned"
            )
        await service.accept(RECIPIENT, share_id=created[0].id)
        listed = await service.list_for_resource(
            RECIPIENT, resource_type="run", resource_id="run_owned"
        )
        assert len(listed) == 1
        with pytest.raises(ShareNotAllowed):
            await service.list_for_resource(
                STRANGER, resource_type="run", resource_id="run_owned"
            )


class TestRecipientDrop:
    @pytest.mark.asyncio
    async def test_decline_pending_removes_invitation(self):
        service, _permissions, identity = make_service()
        created = await grant(service)
        assert await service.recipient_drop(
            RECIPIENT, share_id=created[0].id
        )
        assert await service.inbox(RECIPIENT) == ()
        assert "share.declined" in [
            entry.action for entry in identity.audit_entries
        ]

    @pytest.mark.asyncio
    async def test_leave_accepted_drops_access(self):
        service, permissions, identity = make_service()
        created = await grant(service)
        await service.accept(RECIPIENT, share_id=created[0].id)
        assert await service.recipient_drop(
            RECIPIENT, share_id=created[0].id
        )
        assert not await permissions.can(
            RECIPIENT,
            SharePermission.VIEW,
            resource_type="run",
            resource_id="run_owned",
        )
        assert "share.left" in [
            entry.action for entry in identity.audit_entries
        ]

    @pytest.mark.asyncio
    async def test_stranger_cannot_drop_someone_elses_share(self):
        service, _permissions, _identity = make_service()
        created = await grant(service)
        assert not await service.recipient_drop(
            STRANGER, share_id=created[0].id
        )
        # The invitation survives for the real recipient.
        assert len(await service.inbox(RECIPIENT)) == 1


class TestInbox:
    @pytest.mark.asyncio
    async def test_inbox_partitions_pending_then_accepted(self):
        service, _permissions, _identity = make_service()
        created = await grant(service)
        pending = await service.inbox(RECIPIENT)
        assert len(pending) == 1
        assert pending[0].resource_id == "run_owned"
        assert pending[0].resource_title == "Meine Recherche"
        assert pending[0].accepted_at is None

        await service.accept(RECIPIENT, share_id=created[0].id)
        accepted = await service.inbox(RECIPIENT)
        assert len(accepted) == 1
        assert accepted[0].accepted_at is not None

    @pytest.mark.asyncio
    async def test_inbox_skips_titleless_resources(self):
        service, _permissions, identity = make_service()
        # A share whose title resolver yields None (resource gone/untitled)
        # must not surface — there is nothing for the recipient to act on.
        identity.add_share(
            subject_type="user",
            subject_id="user-2",
            resource_type="run",
            resource_id="run_pruned",
            permission=SharePermission.VIEW,
            accepted=False,
        )
        assert await service.inbox(RECIPIENT) == ()

    @pytest.mark.asyncio
    async def test_inbox_excludes_group_shares(self):
        service, _permissions, identity = make_service()
        identity.add_group("g1", ["user-2"])
        identity.add_share(
            subject_type="group",
            subject_id="g1",
            resource_type="run",
            resource_id="run_owned",
            permission=SharePermission.VIEW,
            accepted=False,
        )
        # The inbox is the per-user consent surface: a group share the
        # recipient cannot individually accept or drop must not appear
        # (symmetry with recipient_drop, which also refuses group shares).
        assert await service.inbox(RECIPIENT) == ()
