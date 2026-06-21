"""Local email/password credential schema with tenant RLS.

Revision ID: 0008_local_credentials
Revises: 0007_quota

Creates the local-credentials table from its metadata snapshot (incl.
the functional unique index on lower(email)) and applies the established
security layering: DML grants for ``inqtrix_app`` and ENABLE + FORCE
row-level security with the fail-closed tenant policy (InitPlan
``(SELECT ...)`` wrapper). Backs ``INQTRIX_AUTH_MODE=local``.
"""

from __future__ import annotations

from alembic import op

from inqtrix.storage.credentials_orm import credentials_metadata

revision = "0008_local_credentials"
down_revision = "0007_quota"
branch_labels = None
depends_on = None

APP_ROLE = "inqtrix_app"


def upgrade() -> None:
    bind = op.get_bind()
    credentials_metadata.create_all(bind=bind)

    op.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON local_credentials "
        f"TO {APP_ROLE}"
    )
    op.execute("ALTER TABLE local_credentials ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE local_credentials FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY tenant_isolation ON local_credentials
            FOR ALL
            USING (tenant_id = (SELECT inqtrix_current_tenant_id()))
            WITH CHECK (tenant_id = (SELECT inqtrix_current_tenant_id()))
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    credentials_metadata.drop_all(bind=bind)
