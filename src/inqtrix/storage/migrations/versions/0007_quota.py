"""Quota usage counters + limit overrides with tenant RLS.

Revision ID: 0007_quota
Revises: 0006_prompt_templates

Creates the two quota tables from their metadata snapshot and applies
the established security layering: DML grants for ``inqtrix_app`` and
ENABLE + FORCE row-level security with the fail-closed tenant policy
(InitPlan ``(SELECT ...)`` wrapper).
"""

from __future__ import annotations

from alembic import op

from inqtrix.storage.quota_orm import quota_metadata

revision = "0007_quota"
down_revision = "0006_prompt_templates"
branch_labels = None
depends_on = None

APP_ROLE = "inqtrix_app"
_TABLES = ("quota_usage_counters", "quota_limits")


def upgrade() -> None:
    bind = op.get_bind()
    quota_metadata.create_all(bind=bind)

    for table in _TABLES:
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
    quota_metadata.drop_all(bind=bind)
