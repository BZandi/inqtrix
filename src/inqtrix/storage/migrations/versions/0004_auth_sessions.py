"""Auth-session schema: auth_sessions and auth_flows with tenant RLS.

Revision ID: 0004_auth_sessions
Revises: 0003_runs_durability

Creates the OIDC BFF's server-side state tables from the auth
metadata and applies the established security layering: DML grants
for ``inqtrix_app`` and ENABLE + FORCE row-level security with the
fail-closed tenant policy (InitPlan ``(SELECT ...)`` wrapper).
"""

from __future__ import annotations

from alembic import op

from inqtrix.storage.auth_orm import auth_metadata

revision = "0004_auth_sessions"
down_revision = "0003_runs_durability"
branch_labels = None
depends_on = None

APP_ROLE = "inqtrix_app"


def upgrade() -> None:
    bind = op.get_bind()
    auth_metadata.create_all(bind=bind)

    for table in ("auth_sessions", "auth_flows"):
        op.execute(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO {APP_ROLE}"
        )
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
    auth_metadata.drop_all(bind=bind)
