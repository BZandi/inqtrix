"""Postgres integration tests for the PAT store (gated suite).

Same gating and conventions as the sibling suites. Pins the behaviors
token security depends on across replicas: the guarded revoke (live
row of the right owner flips exactly once), the single-statement
last-used throttle, and the disable-cascade helper.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest
import pytest_asyncio
from sqlalchemy import text

from inqtrix.auth.pat import PersonalAccessToken
from inqtrix.storage.db import build_engine, build_session_factory
from inqtrix.storage.migrate import run_migrations
from inqtrix.storage.pat_postgres import PostgresPatStore

TEST_DATABASE_URL = os.environ.get("INQTRIX_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="INQTRIX_TEST_DATABASE_URL not set (Postgres integration)",
)

APP_ROLE = "inqtrix_app"


@pytest.fixture(scope="session", autouse=True)
def pat_schema_migrated():
    if TEST_DATABASE_URL:
        run_migrations(TEST_DATABASE_URL)
    yield


@pytest_asyncio.fixture()
async def engine():
    engine = build_engine(TEST_DATABASE_URL)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture()
async def store(engine):
    factory = build_session_factory(engine)
    async with factory() as session:
        async with session.begin():
            bypasses = (
                await session.execute(
                    text(
                        "SELECT rolsuper OR rolbypassrls FROM pg_roles "
                        "WHERE rolname = current_user"
                    )
                )
            ).scalar_one()
            if not bypasses:
                pytest.fail(
                    "INQTRIX_TEST_DATABASE_URL must connect as a "
                    "superuser/BYPASSRLS user for cross-tenant cleanup."
                )
            await session.execute(
                text("DELETE FROM personal_access_tokens")
            )
    return PostgresPatStore(session_factory=factory, app_role=APP_ROLE)


def make_token(
    token_id: str = "tok1",
    *,
    owner_sub: str = "user-1",
    expires_at: float | None = None,
) -> PersonalAccessToken:
    return PersonalAccessToken(
        token_id=token_id,
        tenant_id="default",
        owner_issuer="http://idp.example",
        owner_sub=owner_sub,
        name="ci",
        secret_hmac="ab" * 32,
        created_at=time.time(),
        expires_at=expires_at,
        last_used_at=None,
        revoked_at=None,
        scopes=("runs:read",),
    )


@pytest.mark.asyncio
async def test_roundtrip_and_owner_listing(store):
    await store.create(make_token("tok1"))
    await store.create(make_token("tok2", owner_sub="user-2"))

    loaded = await store.get("tok1")
    assert loaded is not None
    assert loaded.owner_sub == "user-1"
    assert loaded.scopes == ("runs:read",)
    assert loaded.expires_at is None

    listed = await store.list_for_owner(
        tenant_id="default",
        owner_issuer="http://idp.example",
        owner_sub="user-1",
    )
    assert [token.token_id for token in listed] == ["tok1"]


@pytest.mark.asyncio
async def test_revoke_guards_owner_and_liveness(store):
    await store.create(make_token("tok1"))
    now = time.time()
    # Wrong owner never flips the row.
    assert not await store.revoke(
        tenant_id="default",
        token_id="tok1",
        owner_issuer="http://idp.example",
        owner_sub="user-2",
        now=now,
    )
    assert await store.revoke(
        tenant_id="default",
        token_id="tok1",
        owner_issuer="http://idp.example",
        owner_sub="user-1",
        now=now,
    )
    # Idempotent: a second revoke is a no-op.
    assert not await store.revoke(
        tenant_id="default",
        token_id="tok1",
        owner_issuer="http://idp.example",
        owner_sub="user-1",
        now=now,
    )
    assert (await store.get("tok1")).revoked_at is not None


@pytest.mark.asyncio
async def test_concurrent_double_revoke_flips_once(store):
    await store.create(make_token("tok1"))
    now = time.time()
    results = await asyncio.gather(
        *(
            store.revoke(
                tenant_id="default",
                token_id="tok1",
                owner_issuer="http://idp.example",
                owner_sub="user-1",
                now=now,
            )
            for _ in range(2)
        )
    )
    assert sorted(results) == [False, True]


@pytest.mark.asyncio
async def test_last_used_throttle_is_one_guarded_statement(store):
    await store.create(make_token("tok1"))
    await store.touch_last_used("tok1", now=1_000.0, min_interval=300.0)
    assert (await store.get("tok1")).last_used_at == 1_000.0
    # Inside the interval: no write.
    await store.touch_last_used("tok1", now=1_100.0, min_interval=300.0)
    assert (await store.get("tok1")).last_used_at == 1_000.0
    # Past the interval: writes again.
    await store.touch_last_used("tok1", now=1_400.0, min_interval=300.0)
    assert (await store.get("tok1")).last_used_at == 1_400.0


@pytest.mark.asyncio
async def test_disable_cascade_revokes_only_that_owner(store):
    await store.create(make_token("tok1"))
    await store.create(make_token("tok2"))
    await store.create(make_token("tok3", owner_sub="user-2"))
    revoked = await store.revoke_all_for_owner(
        tenant_id="default",
        owner_issuer="http://idp.example",
        owner_sub="user-1",
        now=time.time(),
    )
    assert revoked == 2
    assert (await store.get("tok3")).revoked_at is None
