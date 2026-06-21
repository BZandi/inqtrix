"""Programmatic Alembic runner and the ``inqtrix-migrate`` entry point.

Wraps the Alembic command API with the packaged migration scripts so
deployments and tests never depend on the working directory containing
``alembic.ini``. The console script is the operator surface:

    INQTRIX_DATABASE_URL=postgresql+asyncpg://... uv run inqtrix-migrate

The CLI resolves its default URL through the
:class:`~inqtrix.settings.StorageSettings` bridge (so ``.env`` works
exactly like it does for ``python -m inqtrix``); the raw-environment
exception remains confined to the Alembic env script, which the bare
``alembic`` CLI loads without any settings bridge.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from alembic import command
from alembic.config import Config

log = logging.getLogger("inqtrix")

_MIGRATIONS_PATH = Path(__file__).parent / "migrations"


def build_alembic_config(database_url: str) -> Config:
    """Build an Alembic config bound to the packaged migration scripts.

    Args:
        database_url: SQLAlchemy async URL the migrations run against.
            Constructor argument by design; only the CLI entry point
            below reads the environment.
    """
    config = Config()
    config.set_main_option("script_location", str(_MIGRATIONS_PATH))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def run_migrations(database_url: str, *, revision: str = "head") -> None:
    """Upgrade the database schema to *revision* (default: head).

    Synchronous by design — the async migration engine lives inside
    the Alembic env script (``asyncio.run``), so this must be called
    from a context without a running event loop (startup hooks and
    test fixtures use a thread when needed).
    """
    command.upgrade(build_alembic_config(database_url), revision)


def downgrade_migrations(database_url: str, *, revision: str) -> None:
    """Downgrade the database schema to *revision*."""
    command.downgrade(build_alembic_config(database_url), revision)


def main() -> None:
    """Console entry point: migrate the configured database to head."""
    parser = argparse.ArgumentParser(
        prog="inqtrix-migrate",
        description=(
            "Apply the inqtrix platform schema migrations. Reads "
            "INQTRIX_DATABASE_URL unless --database-url is given."
        ),
    )
    parser.add_argument(
        "--database-url",
        default="",
        help="SQLAlchemy async URL (overrides INQTRIX_DATABASE_URL).",
    )
    parser.add_argument(
        "--revision",
        default="head",
        help="Target revision (default: head).",
    )
    args = parser.parse_args()

    from inqtrix.settings import StorageSettings

    database_url = args.database_url or StorageSettings().database_url
    if not database_url:
        raise SystemExit(
            "No database URL: pass --database-url or set "
            "INQTRIX_DATABASE_URL (environment or .env)."
        )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_migrations(database_url, revision=args.revision)
    log.info("Migrations applied (revision=%s).", args.revision)


if __name__ == "__main__":
    main()
