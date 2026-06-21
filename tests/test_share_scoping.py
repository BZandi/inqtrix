"""Workspace-scoped sharing (opt-in): the grant + typeahead co-member boundary.

Pins ``settings.sharing.restrict_to_workspace_members``. With it OFF, sharing
is tenant-wide (byte-identical to before the knob existed). With it ON, a grant
and the typeahead are confined to the grantor's workspace co-members; the
grant-time check is the authoritative boundary (the typeahead filter is the
convenience half). Resources are not workspace-scoped, so "co-member" is
relative to the GRANTOR, not the resource.
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from inqtrix.auth.identity_memory import MemoryIdentityStore
from inqtrix.auth.permissions import PermissionService, WorkspaceRole
from inqtrix.providers.base import ProviderContext
from inqtrix.server.container import build_container
from inqtrix.server.routers.auth import build_auth_router
from inqtrix.server.routers.shares import build_router as build_shares_router
from inqtrix.server.routers.users import build_router as build_users_router
from inqtrix.settings import (
    ServerSettings,
    Settings,
    SharingSettings,
    StorageSettings,
)

from tests.contract._app import StubSearch
from tests.test_knowledge_routes import KnowledgeStubLLM
from tests.test_oidc_bff import ISSUER, FakeIdp, make_provider, run_login

GRANTOR = "user-1234"  # the FakeIdp default login (resource owner)
COMEMBER = "co-member"  # shares a workspace with the grantor
STRANGER = "stranger"  # exists, but in a different workspace


def make_world(*, restrict: bool):
    identity = MemoryIdentityStore()
    idp = FakeIdp()
    provider, users = make_provider(idp)
    container = build_container(
        providers=ProviderContext(llm=KnowledgeStubLLM(), search=StubSearch()),
        strategies=None,
        settings=Settings(
            server=ServerSettings(public_base_url=""),
            storage=StorageSettings(backend="memory", database_url=""),
            sharing=SharingSettings(restrict_to_workspace_members=restrict),
        ),
        semaphore_factory=lambda: asyncio.Semaphore(1),
        auth_provider=provider,
        permissions=PermissionService(
            members=identity, groups=identity, shares=identity, audit=identity
        ),
        workspace_admin=identity,
    )
    assert container.share_service is not None
    app = FastAPI()
    app.include_router(build_auth_router(provider))
    app.include_router(build_shares_router(container))
    app.include_router(build_users_router(container))
    client = TestClient(app, base_url="http://127.0.0.1:5100")
    run_login(client, idp)  # establishes the GRANTOR session

    async def _arrange():
        for sub, email, name in (
            (COMEMBER, "cora@example.com", "Cora Member"),
            (STRANGER, "stan@example.com", "Stan Stranger"),
        ):
            await users.record_login(
                tenant_id="default",
                issuer=ISSUER,
                subject=sub,
                email=email,
                email_verified=True,
                display_name=name,
            )
        # GRANTOR + COMEMBER share one workspace; STRANGER sits in another.
        workspace_id, _ = await identity.create_workspace(
            tenant_id="default", name="Team", created_by_sub=GRANTOR
        )
        await identity.assign_member(
            tenant_id="default",
            workspace_id=workspace_id,
            sub=COMEMBER,
            role=WorkspaceRole.EDITOR,
        )
        await identity.create_workspace(
            tenant_id="default", name="Other", created_by_sub=STRANGER
        )
        return workspace_id

    shared_workspace = asyncio.run(_arrange())
    container.run_store.submit(
        question="meine Recherche",
        stack_name="default",
        work=lambda handle: handle.complete({"answer": "ok"}),
        created_by_sub=GRANTOR,
    )
    owned = container.run_store.list()[0]["run_id"]
    csrf = client.cookies.get("inqtrix_csrf")
    return client, csrf, owned, identity, shared_workspace


def grant(client, csrf, owned, subject, permission="view"):
    return client.post(
        "/v1/shares",
        json={
            "resource_type": "run",
            "resource_id": owned,
            "invitees": [{"subject_id": subject, "permission": permission}],
        },
        headers={"X-CSRF-Token": csrf},
    )


def search(client, q):
    return {
        row["subject"]
        for row in client.get("/v1/users/search", params={"q": q}).json()["data"]
    }


def test_off_keeps_sharing_tenant_wide():
    # The backwards-compat contract rests on this default.
    assert SharingSettings().restrict_to_workspace_members is False
    client, csrf, owned, _identity, _ws = make_world(restrict=False)
    with client:
        # The stranger is searchable and a grant to them succeeds.
        assert STRANGER in search(client, "stan")
        assert grant(client, csrf, owned, STRANGER).status_code == 201


def test_on_grant_allows_comember_denies_stranger():
    client, csrf, owned, identity, _ws = make_world(restrict=True)
    with client:
        assert grant(client, csrf, owned, COMEMBER).status_code == 201
        # A non-co-member is hidden behind 404 (indistinguishable from a
        # foreign resource), and the denial is audited (Designprinzip 1) —
        # not a leaky 400 that would reveal the stranger exists.
        denied = grant(client, csrf, owned, STRANGER)
        assert denied.status_code == 404
        assert any(
            entry.action == "share.denied"
            and entry.detail.get("subject") == STRANGER
            for entry in identity.audit_entries
        )


def test_on_nonexistent_invitee_is_404_like_a_non_member():
    """Existence is not leaked: a non-existent sub and a real non-co-member
    both produce a byte-identical 404 under workspace-scoped sharing."""
    client, csrf, owned, _identity, _ws = make_world(restrict=True)
    with client:
        ghost = grant(client, csrf, owned, "ghost-sub")
        stranger = grant(client, csrf, owned, STRANGER)
        assert ghost.status_code == 404
        assert stranger.status_code == 404
        assert ghost.json() == stranger.json()


def test_on_typeahead_scopes_to_comembers():
    client, _csrf, _owned, _identity, _ws = make_world(restrict=True)
    with client:
        # The co-member is offered; the stranger is filtered out.
        assert search(client, "cora") == {COMEMBER}
        assert search(client, "stan") == set()


def test_on_grant_re_enforced_after_membership_removal():
    """The boundary is the WRITE gate, not a one-time check: once a co-member
    leaves the shared workspace, a later (re-)grant to them is refused."""
    client, csrf, owned, identity, shared_workspace = make_world(restrict=True)
    with client:
        assert grant(client, csrf, owned, COMEMBER).status_code == 201
        # COMEMBER leaves the only workspace they shared with the grantor.
        asyncio.run(
            identity.remove_member(
                tenant_id="default",
                workspace_id=shared_workspace,
                sub=COMEMBER,
            )
        )
        denied = grant(client, csrf, owned, COMEMBER, permission="edit")
        assert denied.status_code == 404


def test_share_workspace_predicate():
    """The membership-boundary predicate behind both halves of the policy."""
    identity = MemoryIdentityStore()
    permissions = PermissionService(
        members=identity, groups=identity, shares=identity, audit=identity
    )

    async def scenario():
        workspace_id, _ = await identity.create_workspace(
            tenant_id="default", name="Team", created_by_sub="a"
        )
        await identity.assign_member(
            tenant_id="default",
            workspace_id=workspace_id,
            sub="b",
            role=WorkspaceRole.VIEWER,
        )
        # A second workspace also shared by a and b, so the intersection runs
        # over a multi-element grantor set and can match on a later element.
        second_id, _ = await identity.create_workspace(
            tenant_id="default", name="Team2", created_by_sub="a"
        )
        await identity.assign_member(
            tenant_id="default",
            workspace_id=second_id,
            sub="b",
            role=WorkspaceRole.VIEWER,
        )
        await identity.create_workspace(
            tenant_id="default", name="Other", created_by_sub="c"
        )
        return (
            # a and b share "Team" (and "Team2").
            await permissions.share_workspace(
                tenant_id="default", sub_a="a", sub_b="b"
            ),
            # a and c are in different workspaces.
            await permissions.share_workspace(
                tenant_id="default", sub_a="a", sub_b="c"
            ),
            # The target has no workspace at all.
            await permissions.share_workspace(
                tenant_id="default", sub_a="a", sub_b="nobody"
            ),
            # The grantor has no workspace at all.
            await permissions.share_workspace(
                tenant_id="default", sub_a="loner", sub_b="a"
            ),
        )

    same, different, no_target, no_grantor = asyncio.run(scenario())
    assert same is True
    assert different is False
    assert no_target is False
    assert no_grantor is False
