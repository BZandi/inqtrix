"""Account-preferences schema: per-user UI preferences singleton.

Revision ID: 0017_account_preferences
Revises: 0016_vector_index_records

Final backend slice of M6c. Creates the account-preferences table from its
metadata snapshot and applies the established security layering: DML grants
for ``inqtrix_app`` + ENABLE/FORCE row-level security with the fail-closed
tenant policy, identical to ``0013``..``0016``.

CHECK constraints pin each preference to its frontend union
(``ContrastMode`` / ``Locale`` / ``ThemeMode`` / ``ThemePreset``) so an
out-of-domain write fails loudly at the database boundary.
"""

from __future__ import annotations

from alembic import op

from inqtrix.storage.account_orm import account_metadata

revision = "0017_account_preferences"
down_revision = "0016_vector_index_records"
branch_labels = None
depends_on = None

APP_ROLE = "inqtrix_app"


def upgrade() -> None:
    bind = op.get_bind()
    account_metadata.create_all(bind=bind)

    op.execute(
        "ALTER TABLE account_preferences ADD CONSTRAINT ck_account_preferences_contrast "
        "CHECK (contrast_mode IN ('standard', 'high'))"
    )
    op.execute(
        "ALTER TABLE account_preferences ADD CONSTRAINT ck_account_preferences_locale "
        "CHECK (locale IN ('de', 'en'))"
    )
    op.execute(
        "ALTER TABLE account_preferences ADD CONSTRAINT ck_account_preferences_theme "
        "CHECK (theme IN ('light', 'dark', 'system'))"
    )
    op.execute(
        "ALTER TABLE account_preferences ADD CONSTRAINT ck_account_preferences_preset "
        "CHECK (theme_preset IN ('standard', 'slate', 'graphite', 'sage'))"
    )
    op.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON account_preferences TO {APP_ROLE}"
    )
    op.execute("ALTER TABLE account_preferences ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE account_preferences FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY tenant_isolation ON account_preferences
            FOR ALL
            USING (tenant_id = (SELECT inqtrix_current_tenant_id()))
            WITH CHECK (tenant_id = (SELECT inqtrix_current_tenant_id()))
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    account_metadata.drop_all(bind=bind)
