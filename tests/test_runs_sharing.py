"""WP-C-C: shared-in run enforcement (store matrix + recipient HTTP path).

The store half pins the ``also_visible`` contract on the in-memory
:class:`~inqtrix.server.runs.RunStore`: the listing union with the
additive ``access`` annotation, the workspace-namespace bypass for
shared-in rows, view-grantees blocked from cancel (deny before
mutate), edit-grantees allowed, and the SSE replay admitted. The HTTP
half drives the real runs + shares routers through ``build_container``
with a header-based oidc test provider — the recipient-side path
``tests/test_share_routes.py`` deliberately defers to here.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from inqtrix.auth.directory import MemoryUserDirectory
from inqtrix.auth.identity_memory import MemoryIdentityStore
from inqtrix.auth.permissions import PermissionService, SharePermission
from inqtrix.auth.principal import (
    ANONYMOUS_PRINCIPAL,
    AuthMode,
    AuthProvider,
    Principal,
    UserContext,
)
from inqtrix.providers.base import ProviderContext
from inqtrix.server.container import build_container
from inqtrix.server.routers import runs as runs_router
from inqtrix.server.routers.shares import build_router as build_shares_router
from inqtrix.server.runs import RunNotFound, RunStore
from inqtrix.settings import Settings, StorageSettings

from tests.contract._app import StubLLM, StubSearch, minimal_agent_result

SUB_HEADER = "X-Test-Sub"
WORKSPACE_HEADER = "X-Inqtrix-Workspace-Id"

OWNER = "user-owner"
RECIPIENT = "user-recipient"
STRANGER = "user-stranger"


# ---------------------------------------------------------------------------
# Store-level matrix (memory RunStore, also_visible passed directly)
# ---------------------------------------------------------------------------


def make_store() -> RunStore:
    return RunStore(
        max_concurrent=1,
        max_queue_size=8,
        completed_ttl_seconds=60,
        event_buffer_size=16,
    )


def scoped(sub: str, *, tenant_id: str = "default") -> UserContext:
    return UserContext(
        principal=Principal(
            sub=sub, kind="oidc_session", tenant_id=tenant_id, role="member"
        )
    )


def submit_completed(
    store: RunStore, *, sub: str = OWNER, workspace_id: str | None = None
) -> str:
    summary = store.submit(
        question="Testfrage?",
        stack_name="default",
        work=lambda handle: handle.complete({"answer": "ok"}),
        created_by_sub=sub,
        created_by_tenant_id="default",
        workspace_id=workspace_id,
    )
    run_id = summary["run_id"]
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if (
            store.get(run_id, visible_to=scoped(sub))["status"]
            == "completed"
        ):
            return run_id
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} did not complete")


def test_view_grantee_reads_but_cannot_cancel():
    store = make_store()
    run_id = submit_completed(store)
    grants = {run_id: SharePermission.VIEW}
    recipient = scoped(RECIPIENT)

    summary = store.get(run_id, visible_to=recipient, also_visible=grants)
    assert summary["access"] == {"via": "share", "permission": "view"}

    result = store.result(run_id, visible_to=recipient, also_visible=grants)
    assert result["answer"] == "ok"

    subscription = store.subscribe(
        run_id, visible_to=recipient, also_visible=grants
    )
    try:
        assert any(
            event["type"] == "inqtrix.run.completed"
            for event in subscription.replay
        )
    finally:
        subscription.close()

    with pytest.raises(RunNotFound):
        store.cancel(run_id, visible_to=recipient, also_visible=grants)


def test_owner_summary_has_no_access_key():
    store = make_store()
    run_id = submit_completed(store)
    summary = store.get(run_id, visible_to=scoped(OWNER))
    assert "access" not in summary


def test_stranger_stays_blind_despite_other_grants():
    """A grant map for OTHER runs must not admit this one."""
    store = make_store()
    run_id = submit_completed(store)
    with pytest.raises(RunNotFound):
        store.get(
            run_id,
            visible_to=scoped(STRANGER),
            also_visible={"run_other": SharePermission.VIEW},
        )


def test_list_union_bypasses_workspace_filter():
    """Shared-in rows join the listing regardless of the caller's
    workspace namespace — they carry the grantor's workspace id."""
    store = make_store()
    shared_run = submit_completed(store, workspace_id="ws-owner")
    own_run = submit_completed(
        store, sub=RECIPIENT, workspace_id="ws-recipient"
    )
    grants = {shared_run: SharePermission.VIEW}

    listed = store.list(
        workspace_id="ws-recipient",
        visible_to=scoped(RECIPIENT),
        also_visible=grants,
    )
    by_id = {item["run_id"]: item for item in listed}
    assert set(by_id) == {own_run, shared_run}
    assert "access" not in by_id[own_run]
    assert by_id[shared_run]["access"] == {
        "via": "share",
        "permission": "view",
    }
    # The shared-in row keeps the GRANTOR's workspace id visible.
    assert by_id[shared_run]["workspace_id"] == "ws-owner"


def test_view_cancel_denies_before_mutating_running_run():
    """A view grantee's cancel must not set the cancel event."""
    store = make_store()
    started = threading.Event()
    release = threading.Event()

    def blocking_work(handle):
        started.set()
        release.wait(timeout=5)
        handle.complete({"answer": "done"})

    run_id = store.submit(
        question="Testfrage?",
        stack_name="default",
        work=blocking_work,
        created_by_sub=OWNER,
        created_by_tenant_id="default",
    )["run_id"]
    assert started.wait(timeout=5)
    try:
        with pytest.raises(RunNotFound):
            store.cancel(
                run_id,
                visible_to=scoped(RECIPIENT),
                also_visible={run_id: SharePermission.VIEW},
            )
        record = store._records[run_id]
        assert not record.cancel_event.is_set()

        edit_summary = store.cancel(
            run_id,
            visible_to=scoped(RECIPIENT),
            also_visible={run_id: SharePermission.EDIT},
        )
        assert record.cancel_event.is_set()
        assert edit_summary["status"] == "running"
    finally:
        release.set()


# ---------------------------------------------------------------------------
# Recipient-side HTTP path (runs + shares routers, real ShareService)
# ---------------------------------------------------------------------------


class OidcHeaderProvider(AuthProvider):
    """Test-only oidc-mode provider: the sub header IS the identity.

    Reports ``mode == "oidc"`` and exposes a users mirror so
    ``build_container`` wires the real :class:`ShareService` with the
    run owner-resolver — the exact production composition minus token
    verification.
    """

    def __init__(self, users: MemoryUserDirectory) -> None:
        self._users = users

    @property
    def mode(self) -> AuthMode:
        return "oidc"

    @property
    def users(self) -> MemoryUserDirectory:
        return self._users

    def resolve_principal(self, request: Request) -> Principal:
        sub = request.headers.get(SUB_HEADER, "")
        if not sub:
            return ANONYMOUS_PRINCIPAL
        return Principal(
            sub=sub, kind="oidc_session", tenant_id="default", role="member"
        )

    def build_principal_dependency(self):
        # The shares/users routers await the dependency — the real
        # oidc provider's is async, so this test double must be too.
        async def _dependency(request: Request) -> Principal:
            return self.resolve_principal(request)

        return _dependency


def make_sharing_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(
        "inqtrix.research.web_research.run_web_graph",
        lambda *args, **kwargs: minimal_agent_result(),
    )
    identity = MemoryIdentityStore()
    users = MemoryUserDirectory()

    async def mirror_all() -> None:
        for sub, name in (
            (OWNER, "Olga Owner"),
            (RECIPIENT, "Rita Recipient"),
            (STRANGER, "Stefan Stranger"),
        ):
            await users.record_login(
                tenant_id="default",
                issuer="http://idp.example",
                subject=sub,
                email=f"{sub}@example.com",
                email_verified=True,
                display_name=name,
            )

    asyncio.run(mirror_all())
    container = build_container(
        providers=ProviderContext(llm=StubLLM(), search=StubSearch()),
        strategies=None,
        settings=Settings(
            storage=StorageSettings(backend="memory", database_url="")
        ),
        semaphore_factory=lambda: asyncio.Semaphore(1),
        auth_provider=OidcHeaderProvider(users),
        permissions=PermissionService(
            members=identity,
            groups=identity,
            shares=identity,
            audit=identity,
        ),
        workspace_admin=identity,
    )
    assert container.share_service is not None
    app = FastAPI()
    app.include_router(runs_router.build_router(container))
    app.include_router(build_shares_router(container))
    from inqtrix.server.routers import agent_runs as agent_runs_router
    from inqtrix.server.routers import agent_sessions as agent_sessions_router

    app.include_router(agent_runs_router.build_router(container))
    app.include_router(agent_sessions_router.build_router(container))
    client = TestClient(app)
    # Store handles for tests that need agent trees / control fixtures
    # (the HTTP API only creates standard runs; agent runs, approvals and
    # artifacts are store-level primitives until the M5 runtime).
    client.run_store = container.run_store  # type: ignore[attr-defined]
    client.agent_control = container.agent_control_service  # type: ignore[attr-defined]
    return client


def create_completed_run(client: TestClient, *, sub: str = OWNER) -> str:
    response = client.post(
        "/v1/runs", json={"question": "Testfrage?"}, headers={SUB_HEADER: sub}
    )
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    deadline = time.time() + 2.0
    while time.time() < deadline:
        summary = client.get(
            f"/v1/runs/{run_id}", headers={SUB_HEADER: sub}
        ).json()
        if summary.get("status") == "completed":
            return run_id
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} did not complete")


def grant_via_http(
    client: TestClient, run_id: str, *, permission: str = "view"
) -> dict[str, Any]:
    response = client.post(
        "/v1/shares",
        json={
            "resource_type": "run",
            "resource_id": run_id,
            "invitees": [
                {"subject_id": RECIPIENT, "permission": permission}
            ],
        },
        headers={SUB_HEADER: OWNER},
    )
    assert response.status_code == 201
    return response.json()["data"][0]


def accept_via_http(client: TestClient, share_id: str) -> None:
    response = client.post(
        f"/v1/shares/{share_id}/accept", headers={SUB_HEADER: RECIPIENT}
    )
    assert response.status_code == 200


def test_http_recipient_sees_shared_run_until_revoked(monkeypatch):
    client = make_sharing_client(monkeypatch)
    with client:
        run_id = create_completed_run(client)

        before_grant = client.get(
            f"/v1/runs/{run_id}", headers={SUB_HEADER: RECIPIENT}
        )
        assert before_grant.status_code == 404

        share = grant_via_http(client, run_id)

        # Consent gate: a pending share grants nothing — still hidden.
        pending = client.get(
            f"/v1/runs/{run_id}", headers={SUB_HEADER: RECIPIENT}
        )
        assert pending.status_code == 404

        accept_via_http(client, share["id"])

        summary = client.get(
            f"/v1/runs/{run_id}", headers={SUB_HEADER: RECIPIENT}
        )
        assert summary.status_code == 200
        assert summary.json()["access"] == {
            "via": "share",
            "permission": "view",
        }

        result = client.get(
            f"/v1/runs/{run_id}/result", headers={SUB_HEADER: RECIPIENT}
        )
        assert result.status_code == 200
        assert "answer" in result.json()

        events = client.get(
            f"/v1/runs/{run_id}/events", headers={SUB_HEADER: RECIPIENT}
        )
        assert events.status_code == 200
        assert '"inqtrix.run.completed"' in events.text

        revoked = client.delete(
            f"/v1/shares/{share['id']}", headers={SUB_HEADER: OWNER}
        )
        assert revoked.status_code == 200
        after_revoke = client.get(
            f"/v1/runs/{run_id}", headers={SUB_HEADER: RECIPIENT}
        )
        assert after_revoke.status_code == 404


def test_http_listing_union_and_shared_with_me(monkeypatch):
    client = make_sharing_client(monkeypatch)
    with client:
        shared_run = create_completed_run(client)
        own_run = create_completed_run(client, sub=RECIPIENT)
        accept_via_http(client, grant_via_http(client, shared_run)["id"])

        listed = client.get(
            "/v1/runs", headers={SUB_HEADER: RECIPIENT}
        ).json()["data"]
        by_id = {item["run_id"]: item for item in listed}
        assert set(by_id) == {own_run, shared_run}
        assert "access" not in by_id[own_run]
        assert by_id[shared_run]["access"]["permission"] == "view"

        # The stranger's listing stays untouched by the grant.
        stranger_listed = client.get(
            "/v1/runs", headers={SUB_HEADER: STRANGER}
        ).json()["data"]
        assert stranger_listed == []

        mine = client.get(
            "/v1/shares/shared-with-me",
            params={"resource_type": "run"},
            headers={SUB_HEADER: RECIPIENT},
        ).json()["data"]
        assert [item["resource_id"] for item in mine] == [shared_run]
        # The grantor's display name rides along for the
        # "Geteilt von <Name>" badge (one batch join, no second call).
        assert mine[0]["granted_by_display_name"] == "Olga Owner"


def test_http_inbox_accept_and_leave(monkeypatch):
    client = make_sharing_client(monkeypatch)
    with client:
        run_id = create_completed_run(client)
        share = grant_via_http(client, run_id)

        # The pending invitation shows in the recipient inbox, title-enriched
        # and carrying the grantor's display name.
        inbox = client.get(
            "/v1/shares/inbox", headers={SUB_HEADER: RECIPIENT}
        ).json()["data"]
        assert inbox["accepted"] == []
        assert len(inbox["pending"]) == 1
        invite = inbox["pending"][0]
        assert invite["id"] == share["id"]
        assert invite["resource_type"] == "run"
        assert invite["resource_id"] == run_id
        assert invite["resource_title"] == "Testfrage?"
        assert invite["permission"] == "view"
        assert invite["granted_by_display_name"] == "Olga Owner"
        assert invite["accepted_at"] is None

        # A stranger cannot accept; the share stays pending.
        assert (
            client.post(
                f"/v1/shares/{share['id']}/accept",
                headers={SUB_HEADER: STRANGER},
            ).status_code
            == 404
        )

        accept_via_http(client, share["id"])

        # After consent it moves to the accepted section.
        accepted_inbox = client.get(
            "/v1/shares/inbox", headers={SUB_HEADER: RECIPIENT}
        ).json()["data"]
        assert accepted_inbox["pending"] == []
        assert len(accepted_inbox["accepted"]) == 1
        assert accepted_inbox["accepted"][0]["accepted_at"] is not None

        # The recipient leaves: their own DELETE drops the share and access.
        left = client.delete(
            f"/v1/shares/{share['id']}", headers={SUB_HEADER: RECIPIENT}
        )
        assert left.status_code == 200
        assert client.get(
            "/v1/shares/inbox", headers={SUB_HEADER: RECIPIENT}
        ).json()["data"] == {"pending": [], "accepted": []}
        assert (
            client.get(
                f"/v1/runs/{run_id}", headers={SUB_HEADER: RECIPIENT}
            ).status_code
            == 404
        )


def test_http_recipient_declines_pending(monkeypatch):
    client = make_sharing_client(monkeypatch)
    with client:
        run_id = create_completed_run(client)
        share = grant_via_http(client, run_id)
        # Decline a pending invitation: the recipient's DELETE before accepting.
        declined = client.delete(
            f"/v1/shares/{share['id']}", headers={SUB_HEADER: RECIPIENT}
        )
        assert declined.status_code == 200
        assert (
            client.get(
                "/v1/shares/inbox", headers={SUB_HEADER: RECIPIENT}
            ).json()["data"]["pending"]
            == []
        )
        # The owner's outgoing view no longer lists the resource.
        assert (
            client.get(
                "/v1/shares/mine", headers={SUB_HEADER: OWNER}
            ).json()["data"]
            == []
        )


def test_http_stranger_cannot_delete_share(monkeypatch):
    client = make_sharing_client(monkeypatch)
    with client:
        run_id = create_completed_run(client)
        share = grant_via_http(client, run_id)
        # Neither owner nor recipient: the DELETE dual-path denies (404), and
        # the invitation survives for the real recipient.
        denied = client.delete(
            f"/v1/shares/{share['id']}", headers={SUB_HEADER: STRANGER}
        )
        assert denied.status_code == 404
        inbox = client.get(
            "/v1/shares/inbox", headers={SUB_HEADER: RECIPIENT}
        ).json()["data"]
        assert len(inbox["pending"]) == 1


def test_http_mine_counts_pending_then_accepted(monkeypatch):
    client = make_sharing_client(monkeypatch)
    with client:
        run_id = create_completed_run(client)
        share = grant_via_http(client, run_id)

        before = client.get(
            "/v1/shares/mine", headers={SUB_HEADER: OWNER}
        ).json()["data"]
        assert len(before) == 1
        assert before[0]["resource_id"] == run_id
        assert before[0]["resource_title"] == "Testfrage?"
        assert before[0]["share_count"] == 1
        assert before[0]["pending_count"] == 1

        accept_via_http(client, share["id"])
        after = client.get(
            "/v1/shares/mine", headers={SUB_HEADER: OWNER}
        ).json()["data"]
        assert after[0]["share_count"] == 1
        assert after[0]["pending_count"] == 0


def test_http_cancel_needs_edit_grant(monkeypatch):
    client = make_sharing_client(monkeypatch)
    with client:
        run_id = create_completed_run(client)
        view_share = grant_via_http(client, run_id, permission="view")
        accept_via_http(client, view_share["id"])

        # Accepted view grant: a view grantee still cannot cancel.
        denied = client.post(
            f"/v1/runs/{run_id}/cancel", headers={SUB_HEADER: RECIPIENT}
        )
        assert denied.status_code == 404

        # The owner upgrades to edit; the re-grant carries the recipient's
        # consent forward, so edit access is live without a second accept.
        grant_via_http(client, run_id, permission="edit")
        allowed = client.post(
            f"/v1/runs/{run_id}/cancel", headers={SUB_HEADER: RECIPIENT}
        )
        assert allowed.status_code == 200


def test_http_children_follow_parent_view_share(monkeypatch):
    """R7: a view share on the PARENT grants the children listing."""
    client = make_sharing_client(monkeypatch)
    with client:
        store = client.run_store  # type: ignore[attr-defined]
        parent = store.submit(
            question="Agentenauftrag",
            stack_name="default",
            work=lambda handle: handle.complete({"answer": "fertig"}),
            kind="agent",
            created_by_sub=OWNER,
            created_by_tenant_id="default",
        )
        child = store.submit(
            question="Teilaufgabe",
            stack_name="default",
            work=lambda handle: handle.complete({"answer": "teil"}),
            kind="agent_child",
            parent_run_id=parent["run_id"],
            root_run_id=parent["run_id"],
            created_by_sub=OWNER,
            created_by_tenant_id="default",
        )
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if (
                store.get(parent["run_id"])["status"] == "completed"
                and store.get(child["run_id"])["status"] == "completed"
            ):
                break
            time.sleep(0.01)

        # Stranger and not-yet-invited recipient: indistinct 404.
        for sub in (RECIPIENT, STRANGER):
            denied = client.get(
                f"/v1/runs/{parent['run_id']}/children",
                headers={SUB_HEADER: sub},
            )
            assert denied.status_code == 404

        accept_via_http(
            client, grant_via_http(client, parent["run_id"])["id"]
        )

        listing = client.get(
            f"/v1/runs/{parent['run_id']}/children",
            headers={SUB_HEADER: RECIPIENT},
        )
        assert listing.status_code == 200
        payload = listing.json()
        assert payload["object"] == "list"
        assert [row["run_id"] for row in payload["data"]] == [child["run_id"]]
        assert payload["data"][0]["parent_run_id"] == parent["run_id"]

        # The child's DIRECT url stays owner-scoped (no grant on it).
        direct = client.get(
            f"/v1/runs/{child['run_id']}", headers={SUB_HEADER: RECIPIENT}
        )
        assert direct.status_code == 404
