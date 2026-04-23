"""Entry point: ``python -m inqtrix``.

Starts the FastAPI server on port 5100 (override via
``INQTRIX_SERVER_PORT``). Logging is driven by the same environment
variables documented for the webserver-stack examples so both entry
points behave consistently:

- ``INQTRIX_LOG_ENABLED`` — ``true`` to activate file logging.
- ``INQTRIX_LOG_LEVEL``   — ``DEBUG`` / ``INFO`` / ``WARNING`` (default
  ``INFO``).
- ``INQTRIX_LOG_CONSOLE`` — ``true`` to mirror WARNING+ records onto
  stderr.
- ``INQTRIX_LOG_INCLUDE_WEB`` — when file logging is enabled, also
  route uvicorn / FastAPI logs into the same file (default ``true``).
- ``INQTRIX_LOG_WEB_LEVEL`` — level for the uvicorn / FastAPI loggers
  (default ``INFO``).

Regardless of these flags, the server prints a terminal banner at
startup describing the active logging configuration (file path if
any, level, console mirror, web-log mirror) so operators can confirm
the setup at a glance.
"""

from __future__ import annotations

import os
from typing import Any

import uvicorn

from inqtrix.logging_config import build_uvicorn_log_config, configure_logging
from inqtrix.server import create_app


_INQTRIX_LOG_PATH = configure_logging(
    enabled=os.getenv("INQTRIX_LOG_ENABLED", "").lower() == "true",
    level=os.getenv("INQTRIX_LOG_LEVEL", "INFO"),
    console=os.getenv("INQTRIX_LOG_CONSOLE", "").lower() == "true",
)

app = create_app()


def main() -> None:
    """Start uvicorn with env-driven bind + logging configuration."""
    uvicorn_kwargs: dict[str, Any] = dict(
        host=os.getenv("INQTRIX_SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("INQTRIX_SERVER_PORT", "5100")),
        workers=1,
        timeout_keep_alive=300,
    )
    if os.getenv("INQTRIX_LOG_INCLUDE_WEB", "true").lower() != "false":
        uvicorn_kwargs["log_config"] = build_uvicorn_log_config(
            _INQTRIX_LOG_PATH,
            web_level=os.getenv("INQTRIX_LOG_WEB_LEVEL", "INFO"),
        )
    uvicorn.run(app, **uvicorn_kwargs)


if __name__ == "__main__":
    main()
