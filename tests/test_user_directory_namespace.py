"""Per-user project namespace adoption (cross-device, P2).

The MemoryUserDirectory mirrors the Postgres contract for
``resolve_default_workspace``: the first authenticated boot ADOPTS the browser's
namespace, and every later call (any device/browser) returns that same adopted
value — so a user's project data follows them instead of being stranded under a
per-browser random id. These assertions would go red if the adopt stopped being
idempotent or did not survive a re-login.
"""

import pytest

from inqtrix.auth.directory import MemoryUserDirectory


async def _record(users: MemoryUserDirectory, *, sub: str = "user-1") -> None:
    await users.record_login(
        tenant_id="default",
        issuer="oidc",
        subject=sub,
        email=f"{sub}@example.com",
        email_verified=True,
        display_name=sub,
    )


@pytest.mark.asyncio
async def test_first_call_adopts_then_is_idempotent() -> None:
    users = MemoryUserDirectory()
    await _record(users)
    # First boot adopts this browser's namespace.
    first = await users.resolve_default_workspace(
        tenant_id="default", issuer="oidc", subject="user-1", candidate="ws_browserA",
    )
    assert first == "ws_browserA"
    # A second device sends ITS browser id, but the user already has a canonical
    # namespace — it is returned unchanged (the data follows the user).
    second = await users.resolve_default_workspace(
        tenant_id="default", issuer="oidc", subject="user-1", candidate="ws_browserB",
    )
    assert second == "ws_browserA"


@pytest.mark.asyncio
async def test_adopted_namespace_survives_relogin() -> None:
    users = MemoryUserDirectory()
    await _record(users)
    await users.resolve_default_workspace(
        tenant_id="default", issuer="oidc", subject="user-1", candidate="ws_browserA",
    )
    # A re-login refreshes the profile but must NOT drop the adopted namespace
    # (else the data would orphan on every login).
    await _record(users)
    mirrored = await users.find_user(tenant_id="default", issuer="oidc", subject="user-1")
    assert mirrored is not None
    assert mirrored.default_workspace_id == "ws_browserA"


@pytest.mark.asyncio
async def test_unknown_user_returns_none() -> None:
    users = MemoryUserDirectory()
    result = await users.resolve_default_workspace(
        tenant_id="default", issuer="oidc", subject="ghost", candidate="ws_x",
    )
    # No mirror row → None, so the caller falls back to the browser-local id.
    assert result is None
