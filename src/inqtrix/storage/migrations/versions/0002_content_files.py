"""Content schema: files registry table with tenant RLS.

Revision ID: 0002_content_files
Revises: 0001_identity_schema

Creates the ``files`` table from the content metadata and applies the
same security layering as revision 0001: DML grants for the
``inqtrix_app`` role and ENABLE + FORCE row-level security with the
fail-closed tenant policy (the ``(SELECT ...)`` wrapper keeps the
helper call an InitPlan — once per query, not per row).
"""

from __future__ import annotations

from alembic import op

from inqtrix.storage.content_orm import content_metadata

revision = "0002_content_files"
down_revision = "0001_identity_schema"
branch_labels = None
depends_on = None

APP_ROLE = "inqtrix_app"


def upgrade() -> None:
    bind = op.get_bind()
    content_metadata.create_all(bind=bind)

    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON files TO {APP_ROLE}")
    op.execute("ALTER TABLE files ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE files FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY tenant_isolation ON files
            FOR ALL
            USING (tenant_id = (SELECT inqtrix_current_tenant_id()))
            WITH CHECK (tenant_id = (SELECT inqtrix_current_tenant_id()))
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    content_metadata.drop_all(bind=bind)
