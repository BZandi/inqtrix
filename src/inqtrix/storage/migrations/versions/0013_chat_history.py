"""Chat-history persistence schema: thread groups, threads, messages.

Revision ID: 0013_chat_history
Revises: 0012_keyset_indexes

First slice of the project-persistence tier (M6a). Creates the chat
tables from their metadata snapshot and applies the established security
layering: DML grants for ``inqtrix_app`` and ENABLE + FORCE row-level
security with the fail-closed tenant policy (InitPlan ``(SELECT ...)``
wrapper), identical to ``0010_knowledge`` and ``0011_indexing_jobs``.

The ``role`` and ``source`` CHECK constraints pin the only legal values
to the frontend ``ChatRole`` / ``ChatThreadRecord.source`` unions so an
out-of-domain write fails loudly at the database boundary (No Silent
Fallbacks) rather than corrupting a round-trip.
"""

from __future__ import annotations

from alembic import op

from inqtrix.storage.chat_orm import chat_metadata

revision = "0013_chat_history"
down_revision = "0012_keyset_indexes"
branch_labels = None
depends_on = None

APP_ROLE = "inqtrix_app"
# Parents before children for FK creation/drop:
# chat_thread_groups <- chat_threads <- chat_messages.
_TABLES = ("chat_thread_groups", "chat_threads", "chat_messages")


def upgrade() -> None:
    bind = op.get_bind()
    chat_metadata.create_all(bind=bind)

    op.execute(
        "ALTER TABLE chat_threads ADD CONSTRAINT ck_chat_threads_source "
        "CHECK (source IN ('api', 'imported', 'mock'))"
    )
    op.execute(
        "ALTER TABLE chat_messages ADD CONSTRAINT ck_chat_messages_role "
        "CHECK (role IN ('user', 'assistant'))"
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
    chat_metadata.drop_all(bind=bind)
