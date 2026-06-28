"""Static checks for Alembic migration graph constraints."""

from __future__ import annotations

from alembic.script import ScriptDirectory

from inqtrix.storage.migrate import build_alembic_config


def test_alembic_revision_ids_fit_default_version_table() -> None:
    """Alembic's default version table stores revision ids in varchar(32)."""
    script = ScriptDirectory.from_config(
        build_alembic_config("postgresql+asyncpg://example.invalid/inqtrix")
    )

    too_long = [
        revision.revision
        for revision in script.walk_revisions()
        if len(revision.revision) > 32
    ]

    assert too_long == []
