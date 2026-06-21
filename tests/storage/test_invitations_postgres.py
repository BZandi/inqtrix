"""Postgres integration tests for invitations + disable cascade (gated).

Same gating and conventions as the sibling suites. Pins the
replica-safety contracts: concurrent acceptance consumes exactly once
with the membership landing in the SAME transaction, existing roles
are never downgraded, and the disable cascade flips mirror flag,
sessions, and PATs atomically.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import insert, select, text

from inqtrix.auth.invitations import DuplicateOpenInvitation
from inqtrix.auth.pat import PersonalAccessToken
from inqtrix.auth.permissions import WorkspaceRole
from inqtrix.storage.auth_postgres import (
    PostgresSessionStore,
    PostgresUserDirectory,
)
from inqtrix.storage.db import build_engine, build_session_factory
from inqtrix.storage.identity_orm import workspace_members, workspaces
from inqtrix.storage.invitations_postgres import PostgresInvitationRepository
from inqtrix.storage.migrate import run_migrations
from inqtrix.storage.pat_postgres import PostgresPatStore

from tests.storage.test_auth_postgres import make_session

TEST_DATABASE_URL = os.environ.get("INQTRIX_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="INQTRIX_TEST_DATABASE_URL not set (Postgres integration)",
)

APP_ROLE = "inqtrix_app"
ISSUER = "http://idp.example"


@pytest.fixture(scope="session", autouse=True)
def schema_migrated():
    if TEST_DATABASE_URL:
        run_migrations(TEST_DATABASE_URL)
    yield


@pytest_asyncio.fixture()
async def engine():
    engine = build_engine(TEST_DATABASE_URL)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture()
async def factory(engine):
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
            for table in (
                "invitations",
                "workspace_members",
                "workspaces",
                "personal_access_tokens",
                "auth_sessions",
                "users",
            ):
                await session.execute(text(f"DELETE FROM {table}"))
    return factory


async def seed_workspace(factory) -> str:
    workspace_id = uuid.uuid4()
    async with factory() as session:
        async with session.begin():
            await session.execute(
                insert(workspaces).values(
                    id=workspace_id,
                    tenant_id="default",
                    name="Team",
                    created_by_sub="owner-0",
                )
            )
    return str(workspace_id)


@pytest.mark.asyncio
async def test_concurrent_acceptance_consumes_exactly_once(factory):
    repo = PostgresInvitationRepository(
        session_factory=factory, app_role=APP_ROLE
    )
    workspace_id = await seed_workspace(factory)
    await repo.create(
        tenant_id="default",
        workspace_id=workspace_id,
        email="alice@example.com",
        role=WorkspaceRole.EDITOR,
        invited_by_sub="owner-0",
        expires_at=time.time() + 3600,
    )
    results = await asyncio.gather(
        *(
            repo.accept_open_for_email(
                tenant_id="default",
                email="ALICE@example.com",
                issuer=ISSUER,
                sub="user-1",
                now=time.time(),
            )
            for _ in range(2)
        )
    )
    consumed_counts = sorted(len(result) for result in results)
    assert consumed_counts == [0, 1]
    # The membership landed in the same transaction as the consume.
    async with factory() as session:
        role = (
            await session.execute(
                select(workspace_members.c.role).where(
                    workspace_members.c.sub == "user-1"
                )
            )
        ).scalar_one()
    assert role == "editor"


@pytest.mark.asyncio
async def test_existing_role_is_never_downgraded(factory):
    repo = PostgresInvitationRepository(
        session_factory=factory, app_role=APP_ROLE
    )
    workspace_id = await seed_workspace(factory)
    async with factory() as session:
        async with session.begin():
            await session.execute(
                insert(workspace_members).values(
                    tenant_id="default",
                    workspace_id=uuid.UUID(workspace_id),
                    sub="user-1",
                    role="owner",
                )
            )
    await repo.create(
        tenant_id="default",
        workspace_id=workspace_id,
        email="alice@example.com",
        role=WorkspaceRole.VIEWER,
        invited_by_sub="owner-0",
        expires_at=time.time() + 3600,
    )
    accepted = await repo.accept_open_for_email(
        tenant_id="default",
        email="alice@example.com",
        issuer=ISSUER,
        sub="user-1",
        now=time.time(),
    )
    assert len(accepted) == 1
    async with factory() as session:
        role = (
            await session.execute(
                select(workspace_members.c.role).where(
                    workspace_members.c.sub == "user-1"
                )
            )
        ).scalar_one()
    assert role == "owner"


@pytest.mark.asyncio
async def test_duplicate_open_invitation_hits_the_partial_unique(factory):
    repo = PostgresInvitationRepository(
        session_factory=factory, app_role=APP_ROLE
    )
    workspace_id = await seed_workspace(factory)
    kwargs = dict(
        tenant_id="default",
        workspace_id=workspace_id,
        email="alice@example.com",
        role=WorkspaceRole.VIEWER,
        invited_by_sub="owner-0",
        expires_at=time.time() + 3600,
    )
    await repo.create(**kwargs)
    with pytest.raises(DuplicateOpenInvitation):
        await repo.create(**{**kwargs, "email": "ALICE@example.com"})


@pytest.mark.asyncio
async def test_expired_and_revoked_never_accept(factory):
    repo = PostgresInvitationRepository(
        session_factory=factory, app_role=APP_ROLE
    )
    workspace_id = await seed_workspace(factory)
    expired = await repo.create(
        tenant_id="default",
        workspace_id=workspace_id,
        email="old@example.com",
        role=WorkspaceRole.VIEWER,
        invited_by_sub="owner-0",
        expires_at=time.time() - 1,
    )
    revocable = await repo.create(
        tenant_id="default",
        workspace_id=workspace_id,
        email="gone@example.com",
        role=WorkspaceRole.VIEWER,
        invited_by_sub="owner-0",
        expires_at=time.time() + 3600,
    )
    assert await repo.revoke(
        tenant_id="default",
        workspace_id=workspace_id,
        invitation_id=revocable.id,
        now=time.time(),
    )
    for email in ("old@example.com", "gone@example.com"):
        accepted = await repo.accept_open_for_email(
            tenant_id="default",
            email=email,
            issuer=ISSUER,
            sub="user-1",
            now=time.time(),
        )
        assert accepted == ()
    listed = await repo.list_for_workspace(
        tenant_id="default", workspace_id=workspace_id
    )
    assert len(listed) == 2
    assert expired.id in {invitation.id for invitation in listed}


@pytest.mark.asyncio
async def test_disable_cascade_is_atomic(factory):
    directory = PostgresUserDirectory(
        session_factory=factory, app_role=APP_ROLE
    )
    sessions = PostgresSessionStore(
        session_factory=factory, app_role=APP_ROLE
    )
    pat_store = PostgresPatStore(session_factory=factory, app_role=APP_ROLE)
    await directory.record_login(
        tenant_id="default",
        issuer=ISSUER,
        subject="user-1",
        email="alice@example.com",
        email_verified=True,
        display_name="Alice",
    )
    from inqtrix.auth.sessions import AuthSession

    await sessions.create(
        AuthSession(
            id="sess-1",
            sub="user-1",
            issuer=ISSUER,
            email="alice@example.com",
            display_name="Alice",
            groups=(),
            csrf_random="ab" * 16,
            created_at=time.time(),
            expires_at=time.time() + 3600,
        )
    )
    await pat_store.create(
        PersonalAccessToken(
            token_id="tok1",
            tenant_id="default",
            owner_issuer=ISSUER,
            owner_sub="user-1",
            name="ci",
            secret_hmac="ab" * 32,
            created_at=time.time(),
            expires_at=None,
            last_used_at=None,
            revoked_at=None,
        )
    )
    assert await directory.disable_user(
        tenant_id="default", issuer=ISSUER, subject="user-1", now=time.time()
    )
    found = await directory.find_user(
        tenant_id="default", issuer=ISSUER, subject="user-1"
    )
    assert found is not None and found.disabled_at is not None
    assert (await pat_store.get("tok1")).revoked_at is not None
    # Second disable is a guarded no-op.
    assert not await directory.disable_user(
        tenant_id="default", issuer=ISSUER, subject="user-1", now=time.time()
    )
