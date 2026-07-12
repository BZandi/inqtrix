"""Editor patches: persisted anchored-edit proposals (M7).

Revision ID: 0031_editor_patches
Revises: 0030_agent_control

Creates ``editor_patches``, the persisted proposal-and-decision table for
anchored document edits (editor suggest/instruct and the workspace-agent
patch phase). A patch is a child of its editor DOCUMENT (``ON DELETE
CASCADE`` — deleting the document removes its patch history); the
optional ``run_id`` back-reference uses ``ON DELETE SET NULL`` so run
retention never destroys the applied-edit truth.

RLS follows the 0003/0030 precedent: GRANT to the app role, ENABLE +
FORCE ROW LEVEL SECURITY, fail-closed ``tenant_isolation`` policy.
"""

from __future__ import annotations

from alembic import op

from inqtrix.storage.editor_patch_orm import editor_patch_metadata

revision = "0031_editor_patches"
down_revision = "0030_agent_control"
branch_labels = None
depends_on = None

_APP_ROLE = "inqtrix_app"

_TABLE = "editor_patches"


def upgrade() -> None:
    bind = op.get_bind()
    editor_patch_metadata.create_all(bind=bind)
    # editor_documents(id) and runs(run_id) live in other MetaData
    # snapshots, so the foreign keys are raw DDL here instead of ORM
    # declarations (the 0030 pattern).
    op.execute(
        f"ALTER TABLE {_TABLE} ADD CONSTRAINT fk_{_TABLE}_document "
        "FOREIGN KEY (document_id) REFERENCES editor_documents(id) "
        "ON DELETE CASCADE"
    )
    op.execute(
        f"ALTER TABLE {_TABLE} ADD CONSTRAINT fk_{_TABLE}_run "
        "FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE SET NULL"
    )
    op.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON {_TABLE} TO {_APP_ROLE}"
    )
    op.execute(f"ALTER TABLE {_TABLE} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {_TABLE} FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY tenant_isolation ON {_TABLE} FOR ALL "
        "USING (tenant_id = (SELECT inqtrix_current_tenant_id())) "
        "WITH CHECK (tenant_id = (SELECT inqtrix_current_tenant_id()))"
    )


def downgrade() -> None:
    editor_patch_metadata.drop_all(bind=op.get_bind())
