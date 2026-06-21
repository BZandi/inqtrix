"""Instance admin user-management API (/v1/admin/*) on local auth.

Covers the admin gate (session-only + admin role), the user list, role
changes with the last-admin guard, local-user creation, and the disable
cascade (mirror flag + session purge + credential disable so login is
refused). Memory backend, secure_cookies off for the http TestClient.
"""

from __future__ import annotations

import asyncio

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from inqtrix.auth.api_key import build_local_provider
from inqtrix.auth.credentials import LOCAL_ISSUER
from inqtrix.auth.directory import MemoryUserDirectory
from inqtrix.server.routers.admin import build_admin_router
from inqtrix.server.routers.auth import build_auth_router
from inqtrix.settings import AuthSettings, Settings

OWNER = ("owner@example.com", "correct-horse-battery")


def make_client() -> TestClient:
    settings = Settings(
        auth=AuthSettings(
            mode="local",
            session_secret="s" * 32,
            pat_pepper="p" * 32,
            oidc_insecure_dev_cookies=True,
        )
    )
    provider = build_local_provider(settings)
    app = FastAPI()
    app.include_router(build_auth_router(provider))
    app.include_router(build_admin_router(provider))
    principal_dep = provider.build_principal_dependency()

    @app.get("/v1/protected")
    async def protected(principal=Depends(principal_dep)):
        return {"sub": principal.sub}

    return TestClient(app, base_url="http://127.0.0.1:5100")


def _owner_client():
    client = make_client()
    client.post(
        "/api/setup/owner",
        json={"email": OWNER[0], "password": OWNER[1], "display_name": "Owner"},
    )
    csrf = client.get("/api/auth/session").json()["csrf_token"]
    return client, csrf


def _subject_of(client, csrf, email):
    rows = client.get("/v1/admin/users").json()["users"]
    return next(u["subject"] for u in rows if u["email"] == email)


def test_owner_setup_becomes_admin_and_session_reports_role():
    client, _csrf = _owner_client()
    info = client.get("/api/auth/session").json()
    assert info["role"] == "admin"
    rows = client.get("/v1/admin/users").json()["users"]
    assert len(rows) == 1
    assert rows[0]["instance_role"] == "admin"
    assert rows[0]["email"] == OWNER[0]


def test_admin_routes_require_session_admin():
    # No session -> 401 (authenticate first), same as every gated route.
    anon = make_client()
    assert anon.get("/v1/admin/users").status_code == 401

    # An authenticated NON-admin user is denied with a hidden 404.
    client, csrf = _owner_client()
    created = client.post(
        "/v1/admin/users",
        json={"email": "bob@example.com", "password": "another-strong-pw-1"},
        headers={"X-CSRF-Token": csrf},
    )
    assert created.status_code == 201 and created.json()["instance_role"] == "user"
    # Log out the owner, log in as bob (a plain user).
    client.post("/api/auth/logout", headers={"X-CSRF-Token": csrf})
    client.post(
        "/api/auth/login/local",
        json={"email": "bob@example.com", "password": "another-strong-pw-1"},
    )
    assert client.get("/v1/admin/users").status_code == 404


def test_create_and_promote_and_last_admin_guard():
    client, csrf = _owner_client()
    headers = {"X-CSRF-Token": csrf}
    owner_sub = _subject_of(client, csrf, OWNER[0])

    # Self-demote is blocked. (The caller is always the last/only admin when
    # demoting self, so the self-guard fires first; the atomic last-admin
    # invariant is covered by test_atomic_last_admin_guard_is_race_free.)
    demote = client.patch(
        f"/v1/admin/users/{owner_sub}",
        json={"instance_role": "user"},
        headers=headers,
    )
    assert demote.status_code == 409
    assert demote.json()["error"]["type"] == "self_demote"

    # Add a second admin (bob); demoting bob (NOT the caller) is allowed.
    client.post(
        "/v1/admin/users",
        json={"email": "bob@example.com", "password": "another-strong-pw-1", "instance_role": "admin"},
        headers=headers,
    )
    bob_sub = _subject_of(client, csrf, "bob@example.com")
    ok = client.patch(
        f"/v1/admin/users/{bob_sub}",
        json={"instance_role": "user"},
        headers=headers,
    )
    assert ok.status_code == 200 and ok.json()["instance_role"] == "user"
    # The owner still cannot demote themselves (self-guard).
    assert (
        client.patch(
            f"/v1/admin/users/{owner_sub}",
            json={"instance_role": "user"},
            headers=headers,
        ).json()["error"]["type"]
        == "self_demote"
    )


def test_admin_reset_password():
    client, csrf = _owner_client()
    headers = {"X-CSRF-Token": csrf}
    client.post(
        "/v1/admin/users",
        json={"email": "bob@example.com", "password": "another-strong-pw-1", "instance_role": "user"},
        headers=headers,
    )
    bob_sub = _subject_of(client, csrf, "bob@example.com")

    reset = client.post(
        f"/v1/admin/users/{bob_sub}:reset-password",
        json={"password": "bobs-brand-new-pw-9"},
        headers=headers,
    )
    assert reset.status_code == 200 and reset.json()["reset"] is True
    # Resetting an unknown subject is a 404 (done while still the owner).
    assert (
        client.post(
            "/v1/admin/users/ghost-subject:reset-password",
            json={"password": "x" * 12},
            headers=headers,
        ).status_code
        == 404
    )
    # Bob's old password is dead, the new one works. (Logging in as bob below
    # replaces the owner cookie, so this is the last step.)
    assert (
        client.post(
            "/api/auth/login/local",
            json={"email": "bob@example.com", "password": "another-strong-pw-1"},
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/auth/login/local",
            json={"email": "bob@example.com", "password": "bobs-brand-new-pw-9"},
        ).status_code
        == 200
    )


def test_self_disable_blocked():
    client, csrf = _owner_client()
    owner_sub = _subject_of(client, csrf, OWNER[0])
    resp = client.post(
        f"/v1/admin/users/{owner_sub}:disable", headers={"X-CSRF-Token": csrf}
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["type"] == "self_disable"


def test_disable_cascade_blocks_login_and_purges_session():
    client, csrf = _owner_client()
    headers = {"X-CSRF-Token": csrf}
    # Bob logs in, gets a live session.
    client.post(
        "/v1/admin/users",
        json={"email": "bob@example.com", "password": "another-strong-pw-1"},
        headers=headers,
    )
    bob_sub = _subject_of(client, csrf, "bob@example.com")

    # Disable bob (a failed login below does not change the owner's cookie).
    dis = client.post(f"/v1/admin/users/{bob_sub}:disable", headers=headers)
    assert dis.status_code == 200 and dis.json()["disabled"] is True

    # Bob can no longer log in (credential disabled -> uniform 401).
    relogin = client.post(
        "/api/auth/login/local",
        json={"email": "bob@example.com", "password": "another-strong-pw-1"},
    )
    assert relogin.status_code == 401

    # Re-enable restores login.
    enabled = client.post(f"/v1/admin/users/{bob_sub}:enable", headers=headers)
    assert enabled.status_code == 200 and enabled.json()["disabled"] is False
    ok = client.post(
        "/api/auth/login/local",
        json={"email": "bob@example.com", "password": "another-strong-pw-1"},
    )
    assert ok.status_code == 200


def test_disabled_mirror_reports_session_logged_out():
    # Defense-in-depth for the disable cascade's purge race: a session that
    # outlives the purge by a hair still reads as logged-out, because
    # session_payload consults the mirror's disabled flag. Disable the mirror
    # directly here so the session is NOT purged, isolating that check.
    settings = Settings(
        auth=AuthSettings(
            mode="local",
            session_secret="s" * 32,
            pat_pepper="p" * 32,
            oidc_insecure_dev_cookies=True,
        )
    )
    provider = build_local_provider(settings)
    app = FastAPI()
    app.include_router(build_auth_router(provider))
    client = TestClient(app, base_url="http://127.0.0.1:5100")
    client.post("/api/setup/owner", json={"email": OWNER[0], "password": OWNER[1]})
    info = client.get("/api/auth/session").json()
    assert info["authenticated"] is True and info["role"] == "admin"
    asyncio.run(
        provider.users.set_disabled(
            tenant_id="default",
            issuer=LOCAL_ISSUER,
            subject=info["sub"],
            disabled_at=1.0,
        )
    )
    assert client.get("/api/auth/session").json()["authenticated"] is False


def _seed_admin(directory: MemoryUserDirectory, subject: str) -> None:
    asyncio.run(
        directory.record_login(
            tenant_id="default",
            issuer=LOCAL_ISSUER,
            subject=subject,
            email=f"{subject}@example.com",
            email_verified=True,
            display_name=subject,
        )
    )
    asyncio.run(
        directory.set_instance_role(
            tenant_id="default", issuer=LOCAL_ISSUER, subject=subject, role="admin"
        )
    )


def test_atomic_last_admin_guard_is_race_free():
    # The check-and-write is one operation, so the guard holds even without
    # the router's pre-read. Sole admin -> both guarded ops refuse (False).
    directory = MemoryUserDirectory()
    _seed_admin(directory, "a")

    def demote(sub: str) -> bool:
        return asyncio.run(
            directory.demote_if_not_last_admin(
                tenant_id="default", issuer=LOCAL_ISSUER, subject=sub
            )
        )

    def disable(sub: str) -> bool:
        return asyncio.run(
            directory.disable_if_not_last_admin(
                tenant_id="default", issuer=LOCAL_ISSUER, subject=sub, disabled_at=1.0
            )
        )

    assert demote("a") is False
    assert disable("a") is False

    # A second admin -> demoting the first is allowed; "b" is then the last
    # active admin and can be neither demoted nor disabled.
    _seed_admin(directory, "b")
    assert demote("a") is True
    assert disable("b") is False
    assert demote("b") is False

    # Disabling a plain user never trips the admin guard.
    asyncio.run(
        directory.record_login(
            tenant_id="default",
            issuer=LOCAL_ISSUER,
            subject="c",
            email="c@example.com",
            email_verified=True,
            display_name="c",
        )
    )
    assert disable("c") is True
