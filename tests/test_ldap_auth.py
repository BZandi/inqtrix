"""Native LDAP bind auth (INQTRIX_AUTH_MODE=ldap).

Drives the real ``ldap3`` MOCK_SYNC strategy through the LdapClient
connection seam (no hand-rolled fake): search-then-bind, attribute
mapping, LDAP-injection escaping, admin-group mapping, and the
provider/route wiring with first-login-owner. Memory backend.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from ldap3 import MOCK_SYNC, Connection, Server

from inqtrix.auth.directory import MemoryUserDirectory
from inqtrix.auth.ldap import (
    LDAP_ISSUER,
    LdapAuthProvider,
    LdapClient,
    LdapError,
    _first,
)
from inqtrix.auth.lifecycle import (
    MemoryUserLifecycleTransaction,
    UserLifecycleService,
)
from inqtrix.auth.pat import MemoryPatStore, PatService, PatVerifier
from inqtrix.auth.principal import resolve_auth_mode
from inqtrix.auth.sessions import MemoryFlowStore, MemorySessionStore
from inqtrix.server.routers.auth import build_auth_router
from inqtrix.settings import AuthSettings, Settings

BASE = "ou=people,dc=example,dc=com"
SERVICE_DN = "cn=svc,dc=example,dc=com"
ADMIN_GROUP = "cn=admins,ou=groups,dc=example,dc=com"
USER_DN = f"cn=bob,{BASE}"
USER_PW = "bob-secret-passphrase"


def _entries(*, admin: bool) -> dict:
    return {
        SERVICE_DN: {
            "objectClass": ["inetOrgPerson"],
            "cn": "svc",
            "userPassword": "svc-pw",
        },
        USER_DN: {
            "objectClass": ["inetOrgPerson"],
            "uid": "bob",
            "userPassword": USER_PW,
            "mail": "bob@example.com",
            "cn": "Bob Builder",
            "entryUUID": "uuid-bob-123",
            "memberOf": [ADMIN_GROUP] if admin else [],
        },
    }


def _client(*, admin_member: bool = False, admin_group_dn: str = "") -> LdapClient:
    entries = _entries(admin=admin_member)

    def factory(user: str, password: str) -> Connection:
        server = Server("fake-ldap")
        conn = Connection(
            server, user=user, password=password, client_strategy=MOCK_SYNC
        )
        for dn, attrs in entries.items():
            conn.strategy.add_entry(dn, attrs)
        return conn

    return LdapClient(
        url="ldap://fake",
        bind_dn=SERVICE_DN,
        bind_password="svc-pw",
        user_search_base=BASE,
        user_search_filter="(uid={username})",
        admin_group_dn=admin_group_dn,
        connection_factory=factory,
    )


def test_search_then_bind_maps_attributes():
    identity = _client().authenticate("bob", USER_PW)
    assert identity.subject == "uuid-bob-123"  # stable id_attr, not the DN
    assert identity.email == "bob@example.com"
    assert identity.display_name == "Bob Builder"
    assert identity.is_admin is False


def test_wrong_password_raises_uniform_error():
    with pytest.raises(LdapError):
        _client().authenticate("bob", "wrong-password")


def test_unknown_user_raises_uniform_error():
    with pytest.raises(LdapError):
        _client().authenticate("nobody", USER_PW)


def test_injection_username_does_not_authenticate():
    # An injection attempt is escaped, so it matches no entry -> reject,
    # instead of widening the filter to match everyone.
    with pytest.raises(LdapError):
        _client().authenticate("*)(uid=*", USER_PW)


def test_admin_group_membership_flags_admin():
    # Member of the configured admin group -> is_admin True.
    flagged = _client(admin_member=True, admin_group_dn=ADMIN_GROUP).authenticate(
        "bob", USER_PW
    )
    assert flagged.is_admin is True
    # Same group exists but not configured -> not admin.
    unflagged = _client(admin_member=True, admin_group_dn="").authenticate(
        "bob", USER_PW
    )
    assert unflagged.is_admin is False


def test_admin_group_matches_despite_dn_format_variance():
    # The directory renders the group DN with AD-style spacing after the
    # commas, mixed case, and an RFC 4514 escaped space; the operator
    # configured the plain canonical form. They must still match -- a
    # string-equality mismatch would silently strip the admin role.
    server_dn = "CN=Admin\\20Group, OU=Groups, DC=Example, DC=Com"
    config_dn = "cn=admin group,ou=groups,dc=example,dc=com"
    entries = {
        SERVICE_DN: {
            "objectClass": ["inetOrgPerson"],
            "cn": "svc",
            "userPassword": "svc-pw",
        },
        USER_DN: {
            "objectClass": ["inetOrgPerson"],
            "uid": "bob",
            "userPassword": USER_PW,
            "mail": "bob@example.com",
            "cn": "Bob Builder",
            "entryUUID": "uuid-bob-123",
            "memberOf": [server_dn],
        },
    }

    def factory(user: str, password: str) -> Connection:
        server = Server("fake-ldap")
        conn = Connection(
            server, user=user, password=password, client_strategy=MOCK_SYNC
        )
        for dn, attrs in entries.items():
            conn.strategy.add_entry(dn, attrs)
        return conn

    client = LdapClient(
        url="ldap://fake",
        bind_dn=SERVICE_DN,
        bind_password="svc-pw",
        user_search_base=BASE,
        user_search_filter="(uid={username})",
        admin_group_dn=config_dn,
        connection_factory=factory,
    )
    assert client.authenticate("bob", USER_PW).is_admin is True


def test_resolve_auth_mode_ldap_requires_connection_settings():
    server = Settings().server
    with pytest.raises(RuntimeError, match="INQTRIX_LDAP_URL"):
        resolve_auth_mode(
            AuthSettings(mode="ldap", session_secret="x" * 32, pat_pepper="y" * 32),
            server,
        )
    ok = resolve_auth_mode(
        AuthSettings(
            mode="ldap",
            session_secret="x" * 32,
            pat_pepper="y" * 32,
            ldap_url="ldap://h",
            ldap_bind_dn="cn=svc",
            ldap_bind_password="pw",
            ldap_user_search_base=BASE,
        ),
        server,
    )
    assert ok == "ldap"


def _provider(*, admin_member: bool = False, admin_group_dn: str = "") -> LdapAuthProvider:
    users = MemoryUserDirectory()
    pat_store = MemoryPatStore()
    sessions = MemorySessionStore()
    lifecycle = UserLifecycleService(
        users=users,
        sessions=sessions,
        pat_service=None,
        transaction=MemoryUserLifecycleTransaction(
            users=users, sessions=sessions, pat_store=pat_store
        ),
    )
    return LdapAuthProvider(
        ldap_client=_client(admin_member=admin_member, admin_group_dn=admin_group_dn),
        first_login_owner=True,
        sessions=sessions,
        flows=MemoryFlowStore(),
        users=users,
        session_secret="s" * 32,
        session_max_age_seconds=3600,
        secure_cookies=False,
        pats=PatVerifier(
            store=pat_store, pepper="p" * 32, user_lookup=users
        ),
        pat_service=PatService(
            store=pat_store, pepper="p" * 32, max_per_user=10, default_ttl_days=0
        ),
        registration_gate=None,
        lifecycle=lifecycle,
    )


def _client_for(provider: LdapAuthProvider) -> TestClient:
    app = FastAPI()
    app.include_router(build_auth_router(provider))
    principal_dep = provider.build_principal_dependency()

    @app.get("/v1/protected")
    async def protected(principal=Depends(principal_dep)):
        return {"sub": principal.user_id, "kind": principal.kind}

    return TestClient(app, base_url="http://127.0.0.1:5100")


def test_login_ldap_route_establishes_session_and_first_login_owner():
    client = _client_for(_provider())
    login = client.post(
        "/api/auth/login/ldap", json={"username": "bob", "password": USER_PW}
    )
    assert login.status_code == 200
    assert login.json()["authenticated"] is True
    protected = client.get("/v1/protected")
    assert protected.status_code == 200
    assert protected.json()["kind"] == "oidc_session"  # shared session kind
    # First LDAP login becomes the instance admin (no admin existed yet).
    assert client.get("/api/auth/session").json()["user"]["role"] == "admin"


def test_login_ldap_wrong_password_is_401():
    client = _client_for(_provider())
    resp = client.post(
        "/api/auth/login/ldap", json={"username": "bob", "password": "nope"}
    )
    assert resp.status_code == 401


def test_disabled_ldap_user_cannot_relogin():
    # The directory has no knowledge of an Inqtrix-side disable, so the LDAP
    # bind still succeeds; the mirror's disabled flag must block the login the
    # same way local (authenticator) and oidc (registration gate) do.
    provider = _provider()
    client = _client_for(provider)
    first = client.post(
        "/api/auth/login/ldap", json={"username": "bob", "password": USER_PW}
    )
    assert first.status_code == 200
    user = asyncio.run(
        provider.users.find_user(
            tenant_id="default", issuer=LDAP_ISSUER, subject="uuid-bob-123"
        )
    )
    assert user is not None
    asyncio.run(
        provider.users.set_disabled(
            tenant_id="default", user_id=user.user_id, disabled_at=1.0
        )
    )
    relogin = client.post(
        "/api/auth/login/ldap", json={"username": "bob", "password": USER_PW}
    )
    assert relogin.status_code == 401


def test_first_coerces_binary_objectguid_to_canonical_guid():
    import uuid

    guid = uuid.UUID("12345678-1234-5678-1234-567812345678")
    # AD returns objectGUID as 16 mixed-endian bytes; render the canonical
    # GUID instead of the b'...' repr that str(bytes) would leak.
    assert _first([guid.bytes_le]) == str(guid)
    # Text attributes and the empty/None cases are unchanged.
    assert _first("plain-uid") == "plain-uid"
    assert _first(["cn-value"]) == "cn-value"
    assert _first([]) is None
    assert _first(None) is None
