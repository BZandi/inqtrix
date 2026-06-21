"""Prompt-template schema with tenant RLS.

Revision ID: 0006_prompt_templates
Revises: 0005_personal_access_tokens

Creates the prompt-template table from its metadata snapshot and
applies the established security layering: DML grants for
``inqtrix_app`` and ENABLE + FORCE row-level security with the
fail-closed tenant policy (InitPlan ``(SELECT ...)`` wrapper).
"""

from __future__ import annotations

from alembic import op

from inqtrix.storage.prompt_template_orm import prompt_template_metadata

revision = "0006_prompt_templates"
down_revision = "0005_personal_access_tokens"
branch_labels = None
depends_on = None

APP_ROLE = "inqtrix_app"


def upgrade() -> None:
    bind = op.get_bind()
    prompt_template_metadata.create_all(bind=bind)

    op.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON prompt_templates "
        f"TO {APP_ROLE}"
    )
    op.execute("ALTER TABLE prompt_templates ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE prompt_templates FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY tenant_isolation ON prompt_templates
            FOR ALL
            USING (tenant_id = (SELECT inqtrix_current_tenant_id()))
            WITH CHECK (tenant_id = (SELECT inqtrix_current_tenant_id()))
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    prompt_template_metadata.drop_all(bind=bind)
