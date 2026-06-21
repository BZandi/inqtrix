"""Editor-persistence schema: folders, documents, comments.

Revision ID: 0014_editor_persistence
Revises: 0013_chat_history

Second slice of the project-persistence tier (M6b). Creates the editor
tables from their metadata snapshot and applies the established security
layering: DML grants for ``inqtrix_app`` and ENABLE + FORCE row-level
security with the fail-closed tenant policy (InitPlan ``(SELECT ...)``
wrapper), identical to ``0013_chat_history``.

The CHECK constraints pin source / comment kind / comment status / comment
evidence preset to the frontend unions (``EditorDocumentSource``,
``EditorCommentKind``, ``EditorCommentStatus``, ``EditorEvidencePreset``)
so an out-of-domain write fails loudly at the database boundary (No Silent
Fallbacks) rather than corrupting a round-trip.
"""

from __future__ import annotations

from alembic import op

from inqtrix.storage.editor_orm import editor_metadata

revision = "0014_editor_persistence"
down_revision = "0013_chat_history"
branch_labels = None
depends_on = None

APP_ROLE = "inqtrix_app"
# Parents before children for FK creation/drop:
# editor_folders <- editor_documents <- editor_comments.
_TABLES = ("editor_folders", "editor_documents", "editor_comments")


def upgrade() -> None:
    bind = op.get_bind()
    editor_metadata.create_all(bind=bind)

    op.execute(
        "ALTER TABLE editor_documents ADD CONSTRAINT ck_editor_documents_source "
        "CHECK (source IN ('blank', 'imported-research-report', 'pasted'))"
    )
    op.execute(
        "ALTER TABLE editor_comments ADD CONSTRAINT ck_editor_comments_kind "
        "CHECK (kind IN ('collect', 'inline_edit', 'evidence_review'))"
    )
    op.execute(
        "ALTER TABLE editor_comments ADD CONSTRAINT ck_editor_comments_status "
        "CHECK (status IN ('open', 'resolved', 'stale'))"
    )
    op.execute(
        "ALTER TABLE editor_comments ADD CONSTRAINT ck_editor_comments_preset "
        "CHECK (evidence_preset IS NULL OR evidence_preset IN "
        "('add_sources', 'fact_check', 'verify_citations'))"
    )
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
    editor_metadata.drop_all(bind=bind)
