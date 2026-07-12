"""Static checks for Alembic migration graph constraints."""

from __future__ import annotations

from importlib import import_module

from alembic.script import ScriptDirectory

from inqtrix.storage.migrate import build_alembic_config


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


def test_agent_task_cancellation_is_the_single_migration_head() -> None:
    script = ScriptDirectory.from_config(
        build_alembic_config("postgresql+asyncpg://example.invalid/inqtrix")
    )

    assert script.get_current_head() == "0044_agent_task_cancellation"


def test_agent_task_cancellation_migration_keeps_one_task_status_constraint(
    monkeypatch,
) -> None:
    migration = import_module(
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
    migration = import_module(
        "inqtrix.storage.migrations.versions."
        "0043_agent_task_execution_contract"
    )

    statement = migration._APPROVAL_SUBJECT_BACKFILL_SQL
    assert "UPDATE run_approvals AS approval" in statement
    assert "approval.status = 'pending'" not in statement
    assert "approval.payload->>'plan_version'" in statement
    assert "approval.run_id = plan.run_id" in statement
    budget_statement = migration._LEGACY_CHILD_BUDGET_BACKFILL_SQL
    assert "kind = 'agent_child'" in budget_statement
    assert "- 'token_budget'" in budget_statement
    assert "waiting_for_children" in budget_statement
    root_statement = migration._RUN_ROOT_LINEAGE_BACKFILL_SQL
    assert "WITH RECURSIVE run_tree" in root_statement
    assert "SET root_run_id = run_tree.canonical_root" in root_statement
    executed: list[str] = []
    monkeypatch.setattr(
        migration.op,
        "execute",
        executed.append,
    )

    migration.upgrade()

    assert executed[0] == statement
    assert budget_statement in executed
    assert root_statement in executed
    assert any("result_payload JSON" in sql for sql in executed)
    assert any("insufficient_evidence" in sql for sql in executed)
