"""Agent feedback history.

Revision ID: 0034_agent_memory_feedback
Revises: 0033_agent_memory
Create Date: 2026-07-03
"""

from __future__ import annotations

from alembic import op

from inqtrix.storage.agent_memory_orm import agent_feedback

revision = "0034_agent_memory_feedback"
down_revision = "0033_agent_memory"
branch_labels = None
depends_on = None

_APP_ROLE = "inqtrix_app"
_TABLES = ("agent_feedback",)


def upgrade() -> None:
    bind = op.get_bind()
    agent_feedback.create(bind=bind, checkfirst=True)
    op.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON agent_feedback TO {_APP_ROLE}"
    )
    op.execute("ALTER TABLE agent_feedback ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE agent_feedback FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_policies
                WHERE schemaname = current_schema()
                  AND tablename = 'agent_feedback'
                  AND policyname = 'tenant_isolation'
            ) THEN
                CREATE POLICY tenant_isolation ON agent_feedback
                    USING (tenant_id = current_setting('app.tenant_id', true))
                    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    for table in reversed(_TABLES):
        op.execute(f"DROP TABLE IF EXISTS {table}")
