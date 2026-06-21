"""Share-service tests over the memory identity store.

Pins the v1 contracts: only owners (or manage grants) may grant and
revoke, re-grants replace the permission, denials are
indistinguishable from absence, shared-with-me reduces to the highest
grant, and the permission layer immediately honours minted shares
(one store backs both surfaces — no split brain).
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

    async def user_exists(tenant_id: str, sub: str) -> bool:
        return sub in KNOWN_USERS

    service = ShareService(
        shares=identity,
        permissions=permissions,
        owner_resolvers={"run": resolve_run_owner},
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
    async def test_owner_grants_and_permission_layer_honours_it(self):
        service, permissions, _identity = make_service()
        created = await grant(service)
        assert len(created) == 1
        assert created[0].permission is SharePermission.VIEW
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
        await grant(service)
        # A group grant on the same resource with a higher level.
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
    async def test_list_for_resource_requires_view(self):
        service, _permissions, _identity = make_service()
        await grant(service)
        listed = await service.list_for_resource(
            RECIPIENT, resource_type="run", resource_id="run_owned"
        )
        assert len(listed) == 1
        with pytest.raises(ShareNotAllowed):
            await service.list_for_resource(
                STRANGER, resource_type="run", resource_id="run_owned"
            )
