"""Vector-index-record schema: records, members, capped run history.

Revision ID: 0016_vector_index_records
Revises: 0015_asset_records

Fourth slice of the project-persistence tier (M6c). Creates the
vector-index tables from their metadata snapshot and applies the
established security layering: DML grants for ``inqtrix_app`` +
ENABLE/FORCE row-level security with the fail-closed tenant policy,
identical to ``0013``/``0014``/``0015``.

CHECK constraints pin the index status, member state, and run result to
the frontend unions (``VectorIndexStatus`` / ``VectorIndexMemberState`` /
``VectorIndexRunResult``) so an out-of-domain write fails loudly at the
database boundary.
"""

from __future__ import annotations

from alembic import op

from inqtrix.storage.vector_index_orm import vector_index_metadata

revision = "0016_vector_index_records"
down_revision = "0015_asset_records"
branch_labels = None
depends_on = None

APP_ROLE = "inqtrix_app"
# Parent before children: vector_index_records <- members / history.
_TABLES = (
    "vector_index_records",
    "vector_index_members",
    "vector_index_history",
)


def upgrade() -> None:
    bind = op.get_bind()
    vector_index_metadata.create_all(bind=bind)

    op.execute(
        "ALTER TABLE vector_index_records ADD CONSTRAINT ck_vector_index_records_status "
        "CHECK (status IN ('error', 'indexing', 'ready', 'stale'))"
    )
    op.execute(
        "ALTER TABLE vector_index_members ADD CONSTRAINT ck_vector_index_members_state "
        "CHECK (state IN ('pending', 'embedded'))"
    )
    op.execute(
        "ALTER TABLE vector_index_history ADD CONSTRAINT ck_vector_index_history_result "
        "CHECK (result IN ('cancelled', 'error', 'ok'))"
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
    vector_index_metadata.drop_all(bind=bind)
