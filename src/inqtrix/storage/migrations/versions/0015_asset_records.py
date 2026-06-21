"""File-asset-record schema: sections, groups, asset records.

Revision ID: 0015_asset_records
Revises: 0014_editor_persistence

Third slice of the project-persistence tier (M6c). Creates the asset
tables from their metadata snapshot and applies the established security
layering: DML grants for ``inqtrix_app`` + ENABLE/FORCE row-level security
with the fail-closed tenant policy, identical to ``0013``/``0014``.

CHECK constraints pin the section kind, asset origin, and parse status to
the frontend unions (``FileSectionKind`` / ``FileAssetOrigin`` /
``FileParseStatus``) so an out-of-domain write fails loudly at the database
boundary.
"""

from __future__ import annotations

from alembic import op

from inqtrix.storage.asset_records_orm import asset_metadata

revision = "0015_asset_records"
down_revision = "0014_editor_persistence"
branch_labels = None
depends_on = None

APP_ROLE = "inqtrix_app"
# Parents before children: asset_sections <- asset_groups <- asset_records.
_TABLES = ("asset_sections", "asset_groups", "asset_records")


def upgrade() -> None:
    bind = op.get_bind()
    asset_metadata.create_all(bind=bind)

    op.execute(
        "ALTER TABLE asset_sections ADD CONSTRAINT ck_asset_sections_kind "
        "CHECK (kind IN ('temporary', 'custom'))"
    )
    op.execute(
        "ALTER TABLE asset_records ADD CONSTRAINT ck_asset_records_origin "
        "CHECK (origin IN ('chat', 'editor', 'library'))"
    )
    op.execute(
        "ALTER TABLE asset_records ADD CONSTRAINT ck_asset_records_parse_status "
        "CHECK (parse_status IN ('parsed', 'partial', 'unsupported', 'error'))"
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
    asset_metadata.drop_all(bind=bind)
