"""Static checks for Alembic migration graph constraints."""

from __future__ import annotations

import importlib
from pathlib import Path

import sqlalchemy as sa
from alembic.script import ScriptDirectory

from inqtrix.storage.migrate import build_alembic_config
from inqtrix.storage.migration_contract import (
    SCHEMA_HEAD_REVISION,
    TENANT_RLS_TABLES,
    schema_head_revision,
)


def test_alembic_revision_ids_fit_default_version_table() -> None:
    """Alembic's default version table stores revision ids in varchar(32)."""
    script = ScriptDirectory.from_config(
        build_alembic_config("postgresql+asyncpg://example.invalid/inqtrix")
    )

    too_long = [
        revision.revision
        for revision in script.walk_revisions()
        if len(revision.revision) > 32
    ]

    assert too_long == []


def test_tenant_integrity_is_the_single_migration_head() -> None:
    script = ScriptDirectory.from_config(
        build_alembic_config("postgresql+asyncpg://example.invalid/inqtrix")
    )

    assert script.get_current_head() == SCHEMA_HEAD_REVISION
    assert schema_head_revision() == SCHEMA_HEAD_REVISION


def test_execution_authority_migration_installs_import_and_reindex_contracts(
    monkeypatch,
) -> None:
    migration = importlib.import_module(
        "inqtrix.storage.migrations.versions.0046_execution_authority"
    )
    executed: list[str] = []
    monkeypatch.setattr(migration.op, "execute", executed.append)

    migration.upgrade()

    statements = "\n".join(executed)
    assert "IF NOT EXISTS" not in statements
    assert "source_run_id" in statements
    assert "uq_runs_import_owner_source" in statements
    assert "execution_actor_user_id uuid" in statements
    assert "execution_scopes json NOT NULL DEFAULT '[]'::json" in statements
    assert "REFERENCES users(id) ON DELETE RESTRICT" in statements
    assert "'queued', 'running', 'cancelling'" in statements


def test_agent_task_cancellation_migration_keeps_one_task_status_constraint(
    monkeypatch,
) -> None:
    migration = importlib.import_module(
        "inqtrix.storage.migrations.versions."
        "0044_agent_task_cancellation"
    )
    executed: list[str] = []
    monkeypatch.setattr(migration.op, "execute", executed.append)

    migration.upgrade()

    assert any("DROP CONSTRAINT" in sql for sql in executed)
    assert any("cancel_requested" in sql and "cancelled" in sql for sql in executed)


def test_agent_task_migration_backfills_legacy_plan_approval_subjects(
    monkeypatch,
) -> None:
    migration = importlib.import_module(
        "inqtrix.storage.migrations.versions."
        "0043_agent_task_execution_contract"
    )

    statement = migration._APPROVAL_SUBJECT_BACKFILL_SQL
    assert "UPDATE run_approvals AS approval" in statement
    assert "approval.status = 'pending'" not in statement
    assert "approval.payload->>'plan_version'" in statement
    assert "approval.run_id = plan.run_id" in statement
    assert "approval.tenant_id = plan.tenant_id" in statement
    assert "plan.version = CASE" in statement
    budget_statement = migration._LEGACY_CHILD_BUDGET_BACKFILL_SQL
    assert "kind = 'agent_child'" in budget_statement
    assert "- 'token_budget'" in budget_statement
    assert "waiting_for_children" in budget_statement
    root_statement = migration._RUN_ROOT_LINEAGE_BACKFILL_SQL
    assert "WITH RECURSIVE run_tree" in root_statement
    assert "child.tenant_id = run_tree.tenant_id" in root_statement
    assert "target.tenant_id = run_tree.tenant_id" in root_statement
    assert "SET root_run_id = run_tree.canonical_root" in root_statement
    executed: list[str] = []
    monkeypatch.setattr(
        migration.op,
        "execute",
        executed.append,
    )

    migration.upgrade()

    assert executed[0] == migration._TENANT_REFERENCE_PREFLIGHT_SQL
    assert executed[1] == statement
    assert executed[2] == migration._APPROVAL_SUBJECT_POSTCHECK_SQL
    assert budget_statement in executed
    assert root_statement in executed
    assert any("result_payload JSON" in sql for sql in executed)
    assert any("insufficient_evidence" in sql for sql in executed)


def test_tenant_integrity_migration_repairs_and_enforces_scoped_references(
    monkeypatch,
) -> None:
    migration = importlib.import_module(
        "inqtrix.storage.migrations.versions.0049_tenant_integrity"
    )
    executed: list[str] = []
    monkeypatch.setattr(migration.op, "execute", executed.append)

    migration.upgrade()

    assert migration._TENANT_RLS_TABLES == TENANT_RLS_TABLES
    assert executed[:4] == [
        migration._TENANT_REFERENCE_PREFLIGHT_SQL,
        migration._APPROVAL_SUBJECT_BACKFILL_SQL,
        migration._APPROVAL_SUBJECT_POSTCHECK_SQL,
        migration._RUN_ROOT_LINEAGE_BACKFILL_SQL,
    ]
    statements = "\n".join(executed)
    assert "approval.tenant_id = plan.tenant_id" in statements
    assert "plan.version = CASE" in statements
    assert "plan.plan_id IS NULL" in statements
    assert "child.tenant_id = run_tree.tenant_id" in statements
    assert "UNIQUE (tenant_id, run_id)" in statements
    assert "FOREIGN KEY (tenant_id, parent_run_id)" in statements
    assert "UNIQUE (tenant_id, plan_id, run_id)" in statements
    assert "FOREIGN KEY (tenant_id, plan_id, run_id)" in statements
    assert "FOREIGN KEY (tenant_id, child_run_id)" in statements
    assert "ON DELETE SET NULL (child_run_id)" in statements
    assert "FOREIGN KEY (tenant_id, artifact_id)" in statements
    assert (
        "REVOKE ALL PRIVILEGES ON TABLE alembic_version "
        "FROM PUBLIC, inqtrix_app"
        in statements
    )
    assert (
        "REVOKE ALL PRIVILEGES (version_num) ON TABLE alembic_version "
        "FROM PUBLIC, inqtrix_app"
        in statements
    )
    assert "GRANT SELECT ON TABLE alembic_version TO inqtrix_app" in statements
    assert "FROM PUBLIC, inqtrix_app" in statements
    assert "GRANT EXECUTE ON FUNCTION inqtrix_current_tenant_id()" in statements
    assert "GRANT USAGE ON SEQUENCE audit_log_id_seq" in statements
    assert "user_events_id_seq" in statements
    assert (
        "REVOKE ALL PRIVILEGES ON SEQUENCE audit_log_id_seq, "
        "user_events_id_seq FROM PUBLIC, inqtrix_app"
        in statements
    )
    for table_name in TENANT_RLS_TABLES:
        assert (
            f"REVOKE ALL PRIVILEGES ON TABLE {table_name} "
            "FROM PUBLIC, inqtrix_app"
        ) in statements
        expected_grant = (
            "SELECT, INSERT"
            if table_name == "audit_log"
            else "SELECT, INSERT, UPDATE, DELETE"
        )
        assert (
            f"GRANT {expected_grant} ON TABLE {table_name} TO inqtrix_app"
            in statements
        )

    executed.clear()
    migration.downgrade()
    downgrade_statements = "\n".join(executed)
    assert (
        "REVOKE EXECUTE ON FUNCTION inqtrix_current_tenant_id() "
        "FROM inqtrix_app" in downgrade_statements
    )
    assert (
        "GRANT EXECUTE ON FUNCTION inqtrix_current_tenant_id() TO PUBLIC"
        in downgrade_statements
    )


def test_head_tenant_rls_inventory_is_sorted_and_complete() -> None:
    assert TENANT_RLS_TABLES == tuple(sorted(TENANT_RLS_TABLES))
    assert len(TENANT_RLS_TABLES) == len(set(TENANT_RLS_TABLES))
    for expected in (
        "runs",
        "run_approvals",
        "editor_collaboration_updates",
        "tenant_security_state",
    ):
        assert expected in TENANT_RLS_TABLES

    storage_path = Path(__file__).parents[2] / "src" / "inqtrix" / "storage"
    orm_tables: set[str] = set()
    for module_path in storage_path.glob("*_orm.py"):
        module = importlib.import_module(
            f"inqtrix.storage.{module_path.stem}"
        )
        for value in vars(module).values():
            if not isinstance(value, sa.MetaData):
                continue
            orm_tables.update(
                table.name
                for table in value.tables.values()
                if "tenant_id" in table.c
            )
    assert orm_tables == set(TENANT_RLS_TABLES)
