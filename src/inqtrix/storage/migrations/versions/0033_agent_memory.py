"""Agent memory candidate queue.

Revision ID: 0033_agent_memory
Revises: 0032_editor_agent_source
Create Date: 2026-07-02
"""

from __future__ import annotations

from alembic import op

from inqtrix.storage.agent_memory_orm import agent_memory_candidates

revision = "0033_agent_memory"
down_revision = "0032_editor_agent_source"
branch_labels = None
depends_on = None

_APP_ROLE = "inqtrix_app"
_TABLES = ("agent_memory_candidates",)


def upgrade() -> None:
    bind = op.get_bind()
    agent_memory_candidates.create(bind=bind, checkfirst=True)
    op.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON agent_memory_candidates TO {_APP_ROLE}"
    )
    op.execute("ALTER TABLE agent_memory_candidates ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE agent_memory_candidates FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_policies
                WHERE schemaname = current_schema()
                  AND tablename = 'agent_memory_candidates'
                  AND policyname = 'tenant_isolation'
            ) THEN
                CREATE POLICY tenant_isolation ON agent_memory_candidates
                    USING (tenant_id = current_setting('app.tenant_id', true))
                    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    for table in reversed(_TABLES):
        op.execute(f"DROP TABLE IF EXISTS {table}")
