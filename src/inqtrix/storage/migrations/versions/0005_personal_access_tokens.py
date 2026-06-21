"""Personal-access-token schema with tenant RLS.

Revision ID: 0005_personal_access_tokens
Revises: 0004_auth_sessions

Creates the PAT table from its metadata snapshot and applies the
established security layering: DML grants for ``inqtrix_app`` and
ENABLE + FORCE row-level security with the fail-closed tenant policy
(InitPlan ``(SELECT ...)`` wrapper).
"""

from __future__ import annotations

from alembic import op

from inqtrix.storage.pat_orm import pat_metadata

revision = "0005_personal_access_tokens"
down_revision = "0004_auth_sessions"
branch_labels = None
depends_on = None

APP_ROLE = "inqtrix_app"


def upgrade() -> None:
    bind = op.get_bind()
    pat_metadata.create_all(bind=bind)

    op.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON personal_access_tokens "
        f"TO {APP_ROLE}"
    )
    op.execute(
        "ALTER TABLE personal_access_tokens ENABLE ROW LEVEL SECURITY"
    )
    op.execute("ALTER TABLE personal_access_tokens FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY tenant_isolation ON personal_access_tokens
            FOR ALL
            USING (tenant_id = (SELECT inqtrix_current_tenant_id()))
            WITH CHECK (tenant_id = (SELECT inqtrix_current_tenant_id()))
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    pat_metadata.drop_all(bind=bind)
