"""Skill schema with tenant RLS (plan M3 `3.1`).

Revision ID: 0041_skill_templates
Revises: 0040_kernel_control_kinds

Creates the skill table from its metadata snapshot and applies the
established security layering: DML grants for ``inqtrix_app`` and
ENABLE + FORCE row-level security with the fail-closed tenant policy
(InitPlan ``(SELECT ...)`` wrapper) — the prompt-template recipe
(0006) verbatim.
"""

from __future__ import annotations

from alembic import op

from inqtrix.storage.skill_orm import skill_metadata

revision = "0041_skill_templates"
down_revision = "0040_kernel_control_kinds"
branch_labels = None
depends_on = None

APP_ROLE = "inqtrix_app"


def upgrade() -> None:
    bind = op.get_bind()
    skill_metadata.create_all(bind=bind)

    op.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON skill_templates "
        f"TO {APP_ROLE}"
    )
    op.execute("ALTER TABLE skill_templates ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE skill_templates FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY tenant_isolation ON skill_templates
            FOR ALL
            USING (tenant_id = (SELECT inqtrix_current_tenant_id()))
            WITH CHECK (tenant_id = (SELECT inqtrix_current_tenant_id()))
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    skill_metadata.drop_all(bind=bind)
