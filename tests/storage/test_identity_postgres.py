"""Postgres integration tests: migrations, RLS bypass matrix, repos.

Gated on ``INQTRIX_TEST_DATABASE_URL`` (a *disposable* database — the
session fixture downgrades/upgrades the schema). The default offline
suite never touches these. Start the dev stack and create the test
database as described in ``docs/development/local-infrastructure.md``.

The RLS assertions deliberately run under ``SET LOCAL ROLE
inqtrix_app``: the compose connection user is a superuser, and
superusers bypass row-level security entirely — testing without the
role switch would be false-green.
"""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import func, insert, select, text, update
from sqlalchemy.exc import DBAPIError, IntegrityError

from inqtrix.auth.identity_memory import MemoryIdentityStore
from inqtrix.auth.permissions import (
    PermissionService,
    ResourceNotFound,
    SharePermission,
    SubjectRef,
    WorkspaceNotFound,
    WorkspaceRole,
)
from inqtrix.auth.principal import Principal
from inqtrix.storage.db import build_engine, build_session_factory, tenant_session
from inqtrix.storage.identity_orm import (
    audit_log,
    group_members,
    groups,
    identity_metadata,
    invitations,
    resource_shares,
    users,
    workspace_members,
    workspaces,
)
from inqtrix.storage.identity_postgres import PostgresIdentityBackend
from inqtrix.storage.migrate import downgrade_migrations, run_migrations

TEST_DATABASE_URL = os.environ.get("INQTRIX_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="INQTRIX_TEST_DATABASE_URL not set (Postgres integration)",
)

APP_ROLE = "inqtrix_app"


@pytest.fixture(scope="session", autouse=True)
def migrated_schema():
    """Migration round-trip: upgrade -> downgrade -> upgrade.

    Doubles as the migration regression test — a revision that cannot
    downgrade cleanly fails the whole gated suite immediately.
    """
    if not TEST_DATABASE_URL:
        yield
        return
    run_migrations(TEST_DATABASE_URL)
    downgrade_migrations(TEST_DATABASE_URL, revision="base")
    run_migrations(TEST_DATABASE_URL)
    yield


@pytest_asyncio.fixture()
async def engine():
    engine = build_engine(TEST_DATABASE_URL)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture()
async def session_factory(engine):
    factory = build_session_factory(engine)
    async with factory() as session:
        async with session.begin():
            # FORCE RLS binds even the table owner, so the GUC-less
            # cleanup below only works for a superuser/BYPASSRLS
            # connection user (the compose default). Fail fast with a
            # clear message instead of a confusing 28000 mid-suite.
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
                    "superuser/BYPASSRLS user: the per-test cleanup "
                    "wipes rows across tenants, which FORCE row-level "
                    "security forbids for ordinary owners."
                )
            # Per-test cleanup across all tenants (FK-safe order).
            for table in (
                resource_shares,
                group_members,
                groups,
                invitations,
                workspace_members,
                workspaces,
                users,
                audit_log,
            ):
                await session.execute(table.delete())
    return factory


def scoped(factory, tenant_id: str = "default"):
    return tenant_session(factory, tenant_id=tenant_id, app_role=APP_ROLE)


async def create_workspace(factory, *, tenant_id: str = "default") -> str:
    async with scoped(factory, tenant_id) as session:
        row = await session.execute(
            insert(workspaces)
            .values(tenant_id=tenant_id, name="W", created_by_sub="owner")
            .returning(workspaces.c.id)
        )
        return str(row.scalar_one())


async def insert_minimal_row(session, table, tenant_id: str) -> dict:
    """Insert one valid row into *table*, creating FK parents inline.

    Returns the value dict so WITH-CHECK tests can replay it with a
    foreign tenant_id. Parents always belong to *tenant_id* — foreign
    keys deliberately bypass RLS, so the policy under test is the one
    on *table* itself.
    """
    values: dict = {"tenant_id": tenant_id}
    if table is users:
        values.update(issuer="https://idp.example", subject="alice",
                      email="alice@example.com")
    elif table is workspaces:
        values.update(name="W", created_by_sub="owner")
    elif table is groups:
        values.update(name="legal")
    elif table is workspace_members:
        workspace_id = (
            await session.execute(
                insert(workspaces)
                .values(tenant_id=tenant_id, name="W", created_by_sub="o")
                .returning(workspaces.c.id)
            )
        ).scalar_one()
        values.update(workspace_id=workspace_id, sub="alice", role="viewer")
    elif table is group_members:
        group_id = (
            await session.execute(
                insert(groups)
                .values(tenant_id=tenant_id, name="g")
                .returning(groups.c.id)
            )
        ).scalar_one()
        values.update(group_id=group_id, sub="alice")
    elif table is invitations:
        workspace_id = (
            await session.execute(
                insert(workspaces)
                .values(tenant_id=tenant_id, name="W", created_by_sub="o")
                .returning(workspaces.c.id)
            )
        ).scalar_one()
        values.update(
            workspace_id=workspace_id, email="invitee@example.com",
            role="viewer", invited_by_sub="owner",
            expires_at=text("now() + interval '1 day'"),
        )
    elif table is resource_shares:
        values.update(
            subject_type="user", subject_id="alice",
            resource_type="report", resource_id="r1",
            permission="view", granted_by_sub="owner",
        )
    elif table is audit_log:
        values.update(actor_sub="alice", action="authz.denied",
                      resource_type="report", resource_id="r1", detail={})
    else:
        raise AssertionError(f"no row factory for table {table.name}")
    await session.execute(insert(table).values(**values))
    return values


TENANT_TABLES = (
    users,
    workspaces,
    workspace_members,
    groups,
    group_members,
    invitations,
    resource_shares,
    audit_log,
)


# ------------------------------------------------------------------ #
# RLS bypass matrix (every tenant table)
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
@pytest.mark.parametrize("table", TENANT_TABLES, ids=lambda t: t.name)
async def test_cross_tenant_select_returns_zero_rows(session_factory, table):
    async with scoped(session_factory, "tenant-a") as session:
        await insert_minimal_row(session, table, "tenant-a")

    async with scoped(session_factory, "tenant-a") as session:
        own = (await session.execute(select(table.c.tenant_id))).all()
    async with scoped(session_factory, "tenant-b") as session:
        foreign = (await session.execute(select(table.c.tenant_id))).all()

    assert len(own) >= 1
    assert foreign == []


@pytest.mark.asyncio
@pytest.mark.parametrize("table", TENANT_TABLES, ids=lambda t: t.name)
async def test_cross_tenant_insert_violates_with_check_everywhere(
    session_factory, table
):
    with pytest.raises(DBAPIError, match="row-level security"):
        async with scoped(session_factory, "tenant-a") as session:
            values = await insert_minimal_row(session, table, "tenant-a")
            if table is not audit_log:
                # Clear the tenant-a row so unique indexes cannot fire
                # before the policy does. audit_log is exempt: the app
                # role has no DELETE grant there, and the table carries
                # no unique index anyway.
                await session.execute(table.delete())
            await session.execute(
                insert(table).values(**{**values, "tenant_id": "tenant-b"})
            )


@pytest.mark.asyncio
async def test_cross_tenant_delete_silently_affects_zero_rows(session_factory):
    """The FOR ALL policy's USING clause makes a foreign DELETE a no-op
    — pinned because 'deletes nothing' is the intended fail-safe."""
    await create_workspace(session_factory, tenant_id="tenant-a")

    async with scoped(session_factory, "tenant-b") as session:
        result = await session.execute(workspaces.delete())
    async with scoped(session_factory, "tenant-a") as session:
        remaining = (await session.execute(select(workspaces.c.id))).all()

    assert result.rowcount == 0
    assert len(remaining) == 1


@pytest.mark.asyncio
async def test_rls_catalog_covers_every_metadata_table(session_factory):
    """Catalog parity guard: a table added to the metadata but missed
    in the migration's RLS loop must fail here, not ship unprotected."""
    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT c.relname, c.relrowsecurity, "
                    "c.relforcerowsecurity, "
                    "EXISTS (SELECT 1 FROM pg_policies p "
                    "        WHERE p.tablename = c.relname) AS has_policy "
                    "FROM pg_class c "
                    "WHERE c.relnamespace = 'public'::regnamespace "
                    "AND c.relkind = 'r'"
                )
            )
        ).all()
    from inqtrix.storage.auth_orm import auth_metadata
    from inqtrix.storage.content_orm import content_metadata
    from inqtrix.storage.runs_orm import runs_metadata

    catalog = {name: (rls, forced, policy) for name, rls, forced, policy in rows}
    platform_tables = (
        list(identity_metadata.tables)
        + list(content_metadata.tables)
        + list(runs_metadata.tables)
        + list(auth_metadata.tables)
    )
    for table_name in platform_tables:
        assert table_name in catalog, f"{table_name} missing in database"
        rls, forced, policy = catalog[table_name]
        assert rls and forced and policy, (
            f"{table_name}: ENABLE={rls} FORCE={forced} policy={policy} — "
            "every identity table must carry forced RLS with a policy"
        )


@pytest.mark.asyncio
async def test_query_without_tenant_context_fails_loudly(session_factory):
    async with session_factory() as session:
        async with session.begin():
            await session.execute(text(f'SET LOCAL ROLE "{APP_ROLE}"'))
            with pytest.raises(DBAPIError, match="tenant_id"):
                await session.execute(select(workspaces.c.id))


@pytest.mark.asyncio
async def test_empty_tenant_guc_fails_loudly_not_silently(session_factory):
    """The empty-string GUC state (after a reverted transaction-local
    value) must raise in the helper, never silently match nothing."""
    async with session_factory() as session:
        async with session.begin():
            await session.execute(text(f'SET LOCAL ROLE "{APP_ROLE}"'))
            await session.execute(
                text("SELECT set_config('inqtrix.tenant_id', '', true)")
            )
            with pytest.raises(DBAPIError, match="tenant_id"):
                await session.execute(select(workspaces.c.id))


@pytest.mark.asyncio
async def test_cross_tenant_insert_violates_with_check(session_factory):
    with pytest.raises(DBAPIError, match="row-level security"):
        async with scoped(session_factory, "tenant-a") as session:
            await session.execute(
                insert(workspaces).values(
                    tenant_id="tenant-b", name="X", created_by_sub="evil"
                )
            )


@pytest.mark.asyncio
async def test_audit_log_is_insert_only_for_the_app_role(session_factory):
    async with scoped(session_factory) as session:
        await session.execute(
            insert(audit_log).values(
                tenant_id="default",
                actor_sub="alice",
                action="authz.denied",
                resource_type="report",
                resource_id="r1",
                detail={},
            )
        )

    with pytest.raises(DBAPIError, match="permission denied"):
        async with scoped(session_factory) as session:
            await session.execute(
                update(audit_log).values(action="tampered")
            )

    with pytest.raises(DBAPIError, match="permission denied"):
        async with scoped(session_factory) as session:
            await session.execute(audit_log.delete())


@pytest.mark.asyncio
async def test_duplicate_active_share_is_rejected_but_regrant_works(
    session_factory,
):
    grant = dict(
        tenant_id="default",
        subject_type="user",
        subject_id="alice",
        resource_type="report",
        resource_id="r1",
        permission="view",
        granted_by_sub="owner",
    )
    async with scoped(session_factory) as session:
        await session.execute(insert(resource_shares).values(**grant))

    with pytest.raises(IntegrityError):
        async with scoped(session_factory) as session:
            await session.execute(insert(resource_shares).values(**grant))

    # Revoke-then-regrant: the partial unique index only covers active rows.
    async with scoped(session_factory) as session:
        await session.execute(
            update(resource_shares)
            .where(resource_shares.c.revoked_at.is_(None))
            .values(revoked_at=text("now()"), revoked_by_sub="owner")
        )
    async with scoped(session_factory) as session:
        await session.execute(
            insert(resource_shares).values(**{**grant, "permission": "edit"})
        )


# ------------------------------------------------------------------ #
# Repository parity with the memory backend
# ------------------------------------------------------------------ #


def oidc(sub: str) -> Principal:
    return Principal(sub=sub, kind="oidc_session", role="member")


async def arrange_identity_facts(factory) -> str:
    """One workspace with alice as editor, a group share lifting r1."""
    workspace_id = await create_workspace(factory)
    async with scoped(factory) as session:
        await session.execute(
            insert(workspace_members).values(
                tenant_id="default",
                workspace_id=workspace_id,
                sub="alice",
                role="editor",
            )
        )
        group_id = (
            await session.execute(
                insert(groups)
                .values(tenant_id="default", name="legal")
                .returning(groups.c.id)
            )
        ).scalar_one()
        await session.execute(
            insert(group_members).values(
                tenant_id="default", group_id=group_id, sub="alice"
            )
        )
        await session.execute(
            insert(resource_shares).values(
                tenant_id="default",
                subject_type="group",
                subject_id=str(group_id),
                resource_type="report",
                resource_id="r1",
                permission="manage",
                granted_by_sub="owner",
                # Accepted: these grants stand for active access, and the
                # consent gate (``accepted_at IS NOT NULL``) excludes pending
                # rows from ``permission_for``.
                accepted_at=text("now()"),
            )
        )
        # Competing lower-ranked direct grant on the same resource: the
        # union must pick the group's manage, pinning the max-rank fold
        # against the live SQL row filtering.
        await session.execute(
            insert(resource_shares).values(
                tenant_id="default",
                subject_type="user",
                subject_id="alice",
                resource_type="report",
                resource_id="r1",
                permission="view",
                granted_by_sub="owner",
                accepted_at=text("now()"),
            )
        )
    return workspace_id


@pytest.mark.asyncio
async def test_postgres_backend_matches_memory_semantics(session_factory):
    workspace_id = await arrange_identity_facts(session_factory)
    backend = PostgresIdentityBackend(
        session_factory=session_factory, app_role=APP_ROLE
    )
    service = PermissionService(
        members=backend, groups=backend, shares=backend, audit=backend
    )

    context = await service.resolve_user_context(oidc("alice"))
    assert context is not None
    assert context.workspace_ids == (workspace_id,)
    assert len(context.groups) == 1

    assert (
        await backend.role_in_workspace(
            tenant_id="default", sub="alice", workspace_id=workspace_id
        )
        is WorkspaceRole.EDITOR
    )

    # Group share lifts r1 to manage; the editor role caps r2 at edit.
    assert await service.can(
        oidc("alice"), SharePermission.MANAGE,
        resource_type="report", resource_id="r1", workspace_id=workspace_id,
    )
    assert not await service.can(
        oidc("alice"), SharePermission.MANAGE,
        resource_type="report", resource_id="r2", workspace_id=workspace_id,
    )

    # Foreign sub: workspace hidden, identical to nonexistence.
    with pytest.raises(WorkspaceNotFound):
        await service.resolve_workspace(oidc("mallory"), workspace_id)
    with pytest.raises(WorkspaceNotFound):
        await service.resolve_workspace(oidc("mallory"), "ws-missing")


@pytest.mark.asyncio
async def test_audit_sink_appends_denials(session_factory):
    backend = PostgresIdentityBackend(
        session_factory=session_factory, app_role=APP_ROLE
    )
    service = PermissionService(
        members=backend, groups=backend, shares=backend, audit=backend
    )

    with pytest.raises(ResourceNotFound):
        await service.require(
            oidc("mallory"), SharePermission.VIEW,
            resource_type="report", resource_id="r9",
        )

    async with scoped(session_factory) as session:
        rows = (
            await session.execute(
                select(audit_log.c.action, audit_log.c.actor_sub)
            )
        ).all()
    assert ("authz.denied", "mallory") in rows


@pytest.mark.asyncio
async def test_memory_and_postgres_agree_on_a_scenario(session_factory):
    """Same arrangement, same answers — port parity guard."""
    workspace_id = await arrange_identity_facts(session_factory)
    backend = PostgresIdentityBackend(
        session_factory=session_factory, app_role=APP_ROLE
    )

    memory = MemoryIdentityStore()
    memory.add_workspace(workspace_id)
    memory.add_member(workspace_id, "alice", WorkspaceRole.EDITOR)
    group_ids = await backend.group_ids_for(tenant_id="default", sub="alice")
    memory.add_group(group_ids[0], ["alice"])
    memory.add_share(
        subject_type="group", subject_id=group_ids[0],
        resource_type="report", resource_id="r1",
        permission=SharePermission.MANAGE,
    )
    memory.add_share(
        subject_type="user", subject_id="alice",
        resource_type="report", resource_id="r1",
        permission=SharePermission.VIEW,
    )

    for store in (memory, backend):
        service = PermissionService(
            members=store, groups=store, shares=store,
            audit=MemoryIdentityStore(),
        )
        assert await service.can(
            oidc("alice"), SharePermission.MANAGE,
            resource_type="report", resource_id="r1",
        ), type(store).__name__
        assert not await service.can(
            oidc("alice"), SharePermission.VIEW,
            resource_type="report", resource_id="r-unshared",
        ), type(store).__name__


@pytest.mark.asyncio
async def test_revoke_shares_for_resource_clears_every_grant(session_factory):
    """Deletion cleanup: every active share on a resource flips at once."""
    backend = PostgresIdentityBackend(
        session_factory=session_factory, app_role=APP_ROLE
    )
    for subject in ("alice", "bob"):
        await backend.create_share(
            tenant_id="default",
            subject_type="user",
            subject_id=subject,
            resource_type="knowledge_collection",
            resource_id="kc_doomed",
            permission=SharePermission.VIEW,
            granted_by_sub="owner",
        )
    survivor = await backend.create_share(
        tenant_id="default",
        subject_type="user",
        subject_id="alice",
        resource_type="knowledge_collection",
        resource_id="kc_other",
        permission=SharePermission.VIEW,
        granted_by_sub="owner",
    )

    revoked = await backend.revoke_shares_for_resource(
        tenant_id="default",
        resource_type="knowledge_collection",
        resource_id="kc_doomed",
        revoked_by_sub="owner",
    )
    assert revoked == 2
    assert (
        await backend.list_shares_for_resource(
            tenant_id="default",
            resource_type="knowledge_collection",
            resource_id="kc_doomed",
        )
        == ()
    )
    # The unrelated resource keeps its grant.
    remaining = await backend.list_shares_for_resource(
        tenant_id="default",
        resource_type="knowledge_collection",
        resource_id="kc_other",
    )
    assert [record.id for record in remaining] == [survivor.id]
    # Idempotent: a second sweep finds nothing.
    assert (
        await backend.revoke_shares_for_resource(
            tenant_id="default",
            resource_type="knowledge_collection",
            resource_id="kc_doomed",
            revoked_by_sub="owner",
        )
        == 0
    )


@pytest.mark.asyncio
async def test_consent_gates_access_on_postgres(session_factory):
    """A minted share is pending and grants nothing until the recipient
    accepts; only the recipient may accept, double-accept is a no-op.

    PG-specific: the ``accepted_at IS NOT NULL`` filter in ``permission_for``
    and the guarded ``accept_share_by_id`` UPDATE run only against live SQL.
    """
    backend = PostgresIdentityBackend(
        session_factory=session_factory, app_role=APP_ROLE
    )
    service = PermissionService(
        members=backend, groups=backend, shares=backend, audit=backend
    )
    share = await backend.create_share(
        tenant_id="default",
        subject_type="user",
        subject_id="alice",
        resource_type="report",
        resource_id="r1",
        permission=SharePermission.VIEW,
        granted_by_sub="owner",
    )
    assert share.accepted_at is None

    async def alice_can_view() -> bool:
        return await service.can(
            oidc("alice"),
            SharePermission.VIEW,
            resource_type="report",
            resource_id="r1",
        )

    assert not await alice_can_view()
    # The wrong recipient cannot accept; the share stays pending.
    assert (
        await backend.accept_share_by_id(
            tenant_id="default", share_id=share.id, subject_sub="mallory"
        )
        is None
    )
    assert not await alice_can_view()

    accepted = await backend.accept_share_by_id(
        tenant_id="default", share_id=share.id, subject_sub="alice"
    )
    assert accepted is not None and accepted.accepted_at is not None
    # Double-accept is a benign no-op (already accepted).
    assert (
        await backend.accept_share_by_id(
            tenant_id="default", share_id=share.id, subject_sub="alice"
        )
        is None
    )
    assert await alice_can_view()


@pytest.mark.asyncio
async def test_regrant_preserves_acceptance_on_postgres(session_factory):
    """A permission change on an accepted share keeps access live (the
    re-grant carries ``accepted_at`` forward) — a PG-only RETURNING path."""
    backend = PostgresIdentityBackend(
        session_factory=session_factory, app_role=APP_ROLE
    )
    service = PermissionService(
        members=backend, groups=backend, shares=backend, audit=backend
    )
    share = await backend.create_share(
        tenant_id="default",
        subject_type="user",
        subject_id="alice",
        resource_type="report",
        resource_id="r1",
        permission=SharePermission.VIEW,
        granted_by_sub="owner",
    )
    await backend.accept_share_by_id(
        tenant_id="default", share_id=share.id, subject_sub="alice"
    )
    regranted = await backend.create_share(
        tenant_id="default",
        subject_type="user",
        subject_id="alice",
        resource_type="report",
        resource_id="r1",
        permission=SharePermission.EDIT,
        granted_by_sub="owner",
    )
    assert regranted.accepted_at is not None
    assert await service.can(
        oidc("alice"),
        SharePermission.EDIT,
        resource_type="report",
        resource_id="r1",
    )


@pytest.mark.asyncio
async def test_inbox_and_outgoing_repos_on_postgres(session_factory):
    """inbox_for_subjects spans kinds and keeps pending+accepted; outgoing
    returns the grantor's active shares; both exclude revoked rows.

    PG-specific: these are new SELECTs whose WHERE filters (revoked_at IS NULL,
    the subject-tuple IN, granted_by_sub) only run against live SQL.
    """
    backend = PostgresIdentityBackend(
        session_factory=session_factory, app_role=APP_ROLE
    )
    alice = [SubjectRef(subject_type="user", subject_id="alice")]
    pending = await backend.create_share(
        tenant_id="default", subject_type="user", subject_id="alice",
        resource_type="run", resource_id="run-1",
        permission=SharePermission.VIEW, granted_by_sub="owner",
    )
    accepted = await backend.create_share(
        tenant_id="default", subject_type="user", subject_id="alice",
        resource_type="knowledge_collection", resource_id="kc-1",
        permission=SharePermission.EDIT, granted_by_sub="owner",
    )
    await backend.create_share(
        tenant_id="default", subject_type="user", subject_id="bob",
        resource_type="run", resource_id="run-1",
        permission=SharePermission.VIEW, granted_by_sub="owner",
    )
    await backend.accept_share_by_id(
        tenant_id="default", share_id=accepted.id, subject_sub="alice"
    )

    inbox = await backend.inbox_for_subjects(
        tenant_id="default", subjects=alice
    )
    by_res = {record.resource_id: record for record in inbox}
    assert set(by_res) == {"run-1", "kc-1"}
    assert by_res["run-1"].accepted_at is None
    assert by_res["kc-1"].accepted_at is not None

    outgoing = await backend.outgoing_shares_for_grantor(
        tenant_id="default", grantor_sub="owner"
    )
    assert len(outgoing) == 3  # alice x2 + bob x1

    # Revoking alice's pending run share drops it from both listings.
    await backend.revoke_share_by_id(
        tenant_id="default", share_id=pending.id, revoked_by_sub="alice"
    )
    inbox_after = await backend.inbox_for_subjects(
        tenant_id="default", subjects=alice
    )
    assert {record.resource_id for record in inbox_after} == {"kc-1"}
    outgoing_after = await backend.outgoing_shares_for_grantor(
        tenant_id="default", grantor_sub="owner"
    )
    assert len(outgoing_after) == 2


@pytest.mark.asyncio
async def test_migration_backfill_activates_existing_shares(session_factory):
    """The 0028 backfill keeps PRE-existing active shares accessible on upgrade
    and must NOT touch revoked ones.

    The migration round-trip runs against an empty DB, so the backfill UPDATE
    itself never hits a row there — this exercises it against data: an
    active-but-unaccepted row (an old grant) becomes accepted and grants
    access, while a revoked row stays untouched (pinning the
    ``revoked_at IS NULL`` half of the WHERE clause).
    """
    backend = PostgresIdentityBackend(
        session_factory=session_factory, app_role=APP_ROLE
    )
    service = PermissionService(
        members=backend, groups=backend, shares=backend, audit=backend
    )
    # accepted_at NULL = a share minted before 0028 (or a v1 pending grant).
    alice_share = await backend.create_share(
        tenant_id="default", subject_type="user", subject_id="alice",
        resource_type="report", resource_id="r1",
        permission=SharePermission.VIEW, granted_by_sub="owner",
    )
    assert alice_share.accepted_at is None
    bob_share = await backend.create_share(
        tenant_id="default", subject_type="user", subject_id="bob",
        resource_type="report", resource_id="r1",
        permission=SharePermission.VIEW, granted_by_sub="owner",
    )
    await backend.revoke_share_by_id(
        tenant_id="default", share_id=bob_share.id, revoked_by_sub="owner"
    )

    async def can_view(sub: str) -> bool:
        return await service.can(
            oidc(sub),
            SharePermission.VIEW,
            resource_type="report",
            resource_id="r1",
        )

    # Before the backfill the active row is pending -> no access.
    assert not await can_view("alice")

    # The migration 0028 backfill, verbatim.
    async with scoped(session_factory) as session:
        await session.execute(
            text(
                "UPDATE resource_shares SET accepted_at = created_at "
                "WHERE accepted_at IS NULL AND revoked_at IS NULL"
            )
        )

    # The pre-existing active share is now accepted and grants access...
    assert await can_view("alice")
    # ...and the revoked row was left untouched (WHERE respected revoked_at).
    async with scoped(session_factory) as session:
        bob_accepted_at = (
            await session.execute(
                select(resource_shares.c.accepted_at).where(
                    resource_shares.c.tenant_id == "default",
                    resource_shares.c.subject_id == "bob",
                    resource_shares.c.resource_type == "report",
                    resource_shares.c.resource_id == "r1",
                )
            )
        ).scalar_one()
    assert bob_accepted_at is None
    assert not await can_view("bob")


async def _exercise_membership_admin(store) -> None:
    """One create/assign-upsert/rename/remove/delete-cascade sequence the
    MembershipAdminRepository port must satisfy identically on every backend.

    Run over BOTH memory and Postgres (below) so a divergence in any answer —
    sort order, None-vs-empty, member_count, upsert, cascade, bool returns —
    fails the parity guard rather than only the relevant backend's test.
    """
    workspace_id, _name = await store.create_workspace(
        tenant_id="default", name="Team", created_by_sub="owner"
    )
    # The creator is the sole OWNER member at first.
    assert await store.list_all_workspaces(tenant_id="default") == (
        (workspace_id, "Team", "owner", 1),
    )

    # assign_member upserts: add a member, then change the role in place.
    assert await store.assign_member(
        tenant_id="default",
        workspace_id=workspace_id,
        sub="alice",
        role=WorkspaceRole.EDITOR,
    )
    assert await store.list_members(
        tenant_id="default", workspace_id=workspace_id
    ) == (("alice", WorkspaceRole.EDITOR), ("owner", WorkspaceRole.OWNER))
    assert await store.assign_member(
        tenant_id="default",
        workspace_id=workspace_id,
        sub="alice",
        role=WorkspaceRole.VIEWER,
    )
    members = dict(
        await store.list_members(
            tenant_id="default", workspace_id=workspace_id
        )
    )
    assert members["alice"] is WorkspaceRole.VIEWER
    assert await store.list_all_workspaces(tenant_id="default") == (
        (workspace_id, "Team", "owner", 2),
    )

    assert await store.rename_workspace(
        tenant_id="default", workspace_id=workspace_id, name="Renamed"
    )
    assert (await store.list_all_workspaces(tenant_id="default"))[0][1] == (
        "Renamed"
    )

    assert await store.remove_member(
        tenant_id="default", workspace_id=workspace_id, sub="alice"
    )
    assert not await store.remove_member(
        tenant_id="default", workspace_id=workspace_id, sub="alice"
    )

    # Absent / malformed ids never raise — None (list) / False (mutations).
    assert (
        await store.list_members(
            tenant_id="default", workspace_id="not-a-uuid"
        )
        is None
    )
    assert (
        await store.list_members(
            tenant_id="default", workspace_id=str(uuid.uuid4())
        )
        is None
    )
    assert not await store.assign_member(
        tenant_id="default",
        workspace_id=str(uuid.uuid4()),
        sub="x",
        role=WorkspaceRole.VIEWER,
    )
    assert not await store.rename_workspace(
        tenant_id="default", workspace_id="not-a-uuid", name="X"
    )

    # delete cascades the remaining membership; second delete is False.
    assert await store.delete_workspace(
        tenant_id="default", workspace_id=workspace_id
    )
    assert not await store.delete_workspace(
        tenant_id="default", workspace_id=workspace_id
    )
    assert await store.list_all_workspaces(tenant_id="default") == ()


@pytest.mark.asyncio
async def test_membership_admin_repository_parity(session_factory):
    """Memory and Postgres satisfy ONE shared MembershipAdminRepository spec.

    Same arrangement, same answers — the port parity guard (mirrors
    ``test_memory_and_postgres_agree_on_a_scenario``). A fresh store per
    backend keeps the two runs independent.
    """
    backend = PostgresIdentityBackend(
        session_factory=session_factory, app_role=APP_ROLE
    )
    for store in (MemoryIdentityStore(), backend):
        await _exercise_membership_admin(store)


@pytest.mark.asyncio
async def test_create_share_is_idempotent_under_concurrent_grants(
    session_factory,
):
    """1.6: two concurrent grants of the same tuple collapse to one row.

    Before, the loser hit the active partial-unique index and raised a
    bare IntegrityError (HTTP 500). Now the ON CONFLICT DO UPDATE
    re-points the existing active row instead — both callers succeed and
    exactly ONE active row remains (last-writer-wins on permission), with
    no exception.
    """
    import asyncio

    backend = PostgresIdentityBackend(
        session_factory=session_factory, app_role=APP_ROLE
    )

    def grant(permission: SharePermission):
        return backend.create_share(
            tenant_id="default",
            subject_type="user",
            subject_id="bob",
            resource_type="report",
            resource_id="rc",
            permission=permission,
            granted_by_sub="owner",
        )

    # Run the concurrency a few rounds on FRESH tuples to reliably hit
    # the first-grant INSERT-INSERT race (no prior active row to serialise
    # the two soft-revokes). Every round must stay exception-free and
    # leave exactly one active row.
    for round_index in range(8):
        resource_id = f"rc-{round_index}"

        async def grant_res(permission: SharePermission):
            return await backend.create_share(
                tenant_id="default",
                subject_type="user",
                subject_id="bob",
                resource_type="report",
                resource_id=resource_id,
                permission=permission,
                granted_by_sub="owner",
            )

        results = await asyncio.gather(
            grant_res(SharePermission.VIEW),
            grant_res(SharePermission.EDIT),
            return_exceptions=True,
        )
        for result in results:
            assert not isinstance(result, Exception), result

        # Each returned id must be a row that was actually PERSISTED, not a
        # minted phantom. The race resolves two legitimate ways: the two
        # INSERTs conflict and DO UPDATE the one surviving row (both racers
        # return that id), or one commits first and the other soft-revokes it
        # then inserts a fresh active row (the ids differ, last-writer-wins).
        # Either way both ids come from RETURNING on a real statement. Before
        # the fix the loser echoed a freshly-minted uuid that never hit a row,
        # so a later accept_share on it would 404.
        async with scoped(session_factory) as session:
            for result in results:
                exists = (
                    await session.execute(
                        select(func.count())
                        .select_from(resource_shares)
                        .where(resource_shares.c.id == uuid.UUID(result.id))
                    )
                ).scalar_one()
                assert exists == 1, (
                    f"round {round_index}: returned id {result.id} was never "
                    "persisted (phantom minted uuid)"
                )

        async with scoped(session_factory) as session:
            active = (
                await session.execute(
                    select(func.count())
                    .select_from(resource_shares)
                    .where(
                        resource_shares.c.resource_id == resource_id,
                        resource_shares.c.revoked_at.is_(None),
                    )
                )
            ).scalar_one()
        assert active == 1, f"round {round_index}: {active} active rows"

    # Sequential re-grant still works and stays single-active (the
    # historical soft-revoke path).
    await grant(SharePermission.MANAGE)
    async with scoped(session_factory) as session:
        active = (
            await session.execute(
                select(func.count())
                .select_from(resource_shares)
                .where(
                    resource_shares.c.resource_id == "rc",
                    resource_shares.c.revoked_at.is_(None),
                )
            )
        ).scalar_one()
    assert active == 1


@pytest.mark.asyncio
async def test_concurrent_regrant_preserves_acceptance(session_factory):
    """P2: a concurrent re-grant of an ALREADY-ACCEPTED share keeps access.

    Root cause: the ON CONFLICT DO UPDATE wrote the acceptance captured from
    THIS transaction's own soft-revoke. Under a concurrent re-grant, one
    grant's soft-revoke can match zero rows (the racer already revoked the
    active row) and capture ``None``, then win the conflict on the racer's
    fresh active row and overwrite its live ``accepted_at`` with ``None`` — a
    silent access revocation, since the consent gate treats a pending
    (``accepted_at IS NULL``) share as granting nothing. The COALESCE of the
    EXISTING row's acceptance makes it survive every interleaving. A few
    rounds raise the odds of hitting the losing interleaving; the invariant
    (surviving active row stays accepted) must hold on every one.
    """
    import asyncio

    backend = PostgresIdentityBackend(
        session_factory=session_factory, app_role=APP_ROLE
    )

    for round_index in range(8):
        resource_id = f"rc-accepted-{round_index}"

        granted = await backend.create_share(
            tenant_id="default",
            subject_type="user",
            subject_id="bob",
            resource_type="report",
            resource_id=resource_id,
            permission=SharePermission.VIEW,
            granted_by_sub="owner",
        )
        accepted = await backend.accept_share_by_id(
            tenant_id="default", share_id=granted.id, subject_sub="bob"
        )
        assert accepted is not None and accepted.accepted_at is not None

        async def regrant(permission: SharePermission):
            return await backend.create_share(
                tenant_id="default",
                subject_type="user",
                subject_id="bob",
                resource_type="report",
                resource_id=resource_id,
                permission=permission,
                granted_by_sub="owner",
            )

        results = await asyncio.gather(
            regrant(SharePermission.EDIT),
            regrant(SharePermission.MANAGE),
            return_exceptions=True,
        )
        for result in results:
            assert not isinstance(result, Exception), result
            # The RETURNED record must mirror the persisted acceptance too
            # (accepted_at now comes from RETURNING, not the captured value),
            # so a re-grant of an accepted share never reports itself pending.
            assert result.accepted_at is not None, (
                f"round {round_index}: returned record dropped accepted_at"
            )

        async with scoped(session_factory) as session:
            rows = (
                await session.execute(
                    select(resource_shares.c.accepted_at).where(
                        resource_shares.c.resource_id == resource_id,
                        resource_shares.c.revoked_at.is_(None),
                    )
                )
            ).all()
        assert len(rows) == 1, f"round {round_index}: {len(rows)} active rows"
        assert rows[0].accepted_at is not None, (
            f"round {round_index}: accepted_at was silently reset to NULL "
            "under a concurrent re-grant (access revoked)"
        )
