"""Agent control persistence: plans, approvals, clarifications, artifacts,
sessions.

Revision ID: 0030_agent_control
Revises: 0029_agent_run_tree

Creates the control tables the workspace-agent negotiates through
(``run_plans``/``run_plan_tasks``/``run_approvals``/``run_clarifications``/
``run_artifacts``/``run_artifact_revisions``) plus the ``agent_sessions``
pair (structural clone of knowledge sessions, decision E15). All control
tables are children of ``runs`` via ``ON DELETE CASCADE`` — the durable run
retention window (``run_durable_retention_seconds``, default 90 days) is
the single retention authority; the session memo artifact survives across
turns because every turn re-anchors it onto the newest run.

RLS follows the 0003/0021 precedent: GRANT to the app role, ENABLE + FORCE
ROW LEVEL SECURITY, fail-closed ``tenant_isolation`` policy.

The LangGraph checkpointer tables (M5) are deliberately NOT managed here:
they are library-owned (``AsyncPostgresSaver.setup()`` at container build),
and the checkpoint is a resumability cache, never the source of truth
(rule R5).
"""

from __future__ import annotations

from alembic import op

from inqtrix.storage.agent_control_orm import agent_control_metadata
from inqtrix.storage.agent_sessions_orm import agent_sessions_metadata

revision = "0030_agent_control"
down_revision = "0029_agent_run_tree"
branch_labels = None
depends_on = None

_APP_ROLE = "inqtrix_app"

_TABLES = (
    "run_plans",
    "run_plan_tasks",
    "run_approvals",
    "run_clarifications",
    "run_artifacts",
    "run_artifact_revisions",
    "agent_session_groups",
    "agent_sessions",
)


_RUNS_CASCADES = (
    "run_plans",
    "run_approvals",
    "run_clarifications",
    "run_artifacts",
)


def upgrade() -> None:
    bind = op.get_bind()
    agent_control_metadata.create_all(bind=bind)
    agent_sessions_metadata.create_all(bind=bind)
    # runs(run_id) lives in another MetaData snapshot, so the CASCADE
    # foreign keys are raw DDL here instead of ORM declarations.
    for table in _RUNS_CASCADES:
        op.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT fk_{table}_run "
            "FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE"
        )
    for table in _TABLES:
        op.execute(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO {_APP_ROLE}"
        )
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} FOR ALL "
            "USING (tenant_id = (SELECT inqtrix_current_tenant_id())) "
            "WITH CHECK (tenant_id = (SELECT inqtrix_current_tenant_id()))"
        )


def downgrade() -> None:
    agent_sessions_metadata.drop_all(bind=op.get_bind())
    agent_control_metadata.drop_all(bind=op.get_bind())
