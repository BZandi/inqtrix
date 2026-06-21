"""Canonical knowledge schema: collections, documents, chunks.

Revision ID: 0010_knowledge
Revises: 0009_instance_role

Creates the three knowledge tables from their metadata snapshot and
applies the established security layering: DML grants for ``inqtrix_app``
and ENABLE + FORCE row-level security with the fail-closed tenant policy
(InitPlan ``(SELECT ...)`` wrapper), identical to ``0007_quota``.

This is the relational source of truth for the knowledge engine in the
``postgres`` storage tier; chunk VECTORS live in Qdrant (or the
in-process vector index), never in these tables.
"""

from __future__ import annotations

from alembic import op

from inqtrix.storage.knowledge_orm import knowledge_metadata

revision = "0010_knowledge"
down_revision = "0009_instance_role"
branch_labels = None
depends_on = None

APP_ROLE = "inqtrix_app"
# Order matters for FK creation/drop: parents before children.
_TABLES = (
    "knowledge_collections",
    "knowledge_documents",
    "knowledge_chunks",
)


def upgrade() -> None:
    bind = op.get_bind()
    knowledge_metadata.create_all(bind=bind)

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
    knowledge_metadata.drop_all(bind=bind)
