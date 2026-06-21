"""Knowledge-session schema: saved Ask sessions (Wissensmodus).

Revision ID: 0021_knowledge_sessions
Revises: 0020_asset_parser_id

Adds the durable tier for knowledge Ask sessions (previously reducer-only /
ephemeral). ``knowledge_sessions`` holds titled conversations with Q&A items as
a JSON body, scoped per ``(tenant_id, created_by_sub, workspace_id)``. The
current ORM also creates ``knowledge_session_groups`` for folder metadata on
fresh databases; migration 0026 patches already-migrated databases. Same
security layering as 0013/0014/0015: DML grant for ``inqtrix_app`` +
ENABLE/FORCE row-level security with the fail-closed tenant policy.
"""

from __future__ import annotations

from alembic import op

from inqtrix.storage.knowledge_sessions_orm import knowledge_sessions_metadata

revision = "0021_knowledge_sessions"
down_revision = "0020_asset_parser_id"
branch_labels = None
depends_on = None

APP_ROLE = "inqtrix_app"
_TABLES = ("knowledge_session_groups", "knowledge_sessions")


def upgrade() -> None:
    bind = op.get_bind()
    knowledge_sessions_metadata.create_all(bind=bind)

    for table in _TABLES:
        op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO {APP_ROLE}")
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"""
            CREATE POLICY tenant_isolation ON {table}
                FOR ALL
                USING (tenant_id = (SELECT inqtrix_current_tenant_id()))
                WITH CHECK (tenant_id = (SELECT inqtrix_current_tenant_id()))
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    knowledge_sessions_metadata.drop_all(bind=bind)
