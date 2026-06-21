"""Add folders to Knowledge Desk saved sessions.

Revision ID: 0026_knowledge_session_groups
Revises: 0025_vector_index_member_skipped

Adds ``knowledge_session_groups`` plus nullable ``knowledge_sessions.group_id``.
The session foreign key uses ``ON DELETE SET NULL`` so deleting a folder orphans
its sessions to the ungrouped section, matching the frontend reducer and the
chat-history persistence model.

Idempotent: fresh databases may already have these objects because migration
0021 creates the current ORM metadata. Already-migrated databases need this
forward patch.
"""

from __future__ import annotations

from alembic import op

revision = "0026_knowledge_session_groups"
down_revision = "0025_vector_index_member_skipped"
branch_labels = None
depends_on = None

APP_ROLE = "inqtrix_app"


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_session_groups (
            id text PRIMARY KEY,
            tenant_id text NOT NULL DEFAULT 'default',
            created_by_sub text NULL,
            workspace_id text NULL,
            title text NOT NULL DEFAULT '',
            created_at double precision NOT NULL,
            updated_at double precision NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_knowledge_session_groups_owner_created
            ON knowledge_session_groups (
                tenant_id, created_by_sub, workspace_id, created_at, id
            )
        """
    )
    op.execute(
        "ALTER TABLE knowledge_sessions "
        "ADD COLUMN IF NOT EXISTS group_id text "
        "REFERENCES knowledge_session_groups(id) ON DELETE SET NULL"
    )
    op.execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON knowledge_session_groups "
        f"TO {APP_ROLE}"
    )
    op.execute("ALTER TABLE knowledge_session_groups ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE knowledge_session_groups FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_policies
                WHERE schemaname = current_schema()
                  AND tablename = 'knowledge_session_groups'
                  AND policyname = 'tenant_isolation'
            ) THEN
                CREATE POLICY tenant_isolation ON knowledge_session_groups
                    FOR ALL
                    USING (tenant_id = (SELECT inqtrix_current_tenant_id()))
                    WITH CHECK (tenant_id = (SELECT inqtrix_current_tenant_id()));
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE knowledge_sessions DROP COLUMN IF EXISTS group_id")
    op.execute("DROP TABLE IF EXISTS knowledge_session_groups")
