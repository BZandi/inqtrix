"""Centralized logging configuration for inqtrix.

Provides a single ``configure_logging()`` function that sets up
file-based logging for the ``inqtrix`` logger with automatic secret
redaction.  Designed for both example scripts and the FastAPI server.

Usage in example scripts::

    from inqtrix.logging_config import configure_logging

    log_path = configure_logging(
        enabled=os.getenv("INQTRIX_LOG_ENABLED", "").lower() == "true",
        level=os.getenv("INQTRIX_LOG_LEVEL", "INFO"),
        console=os.getenv("INQTRIX_LOG_CONSOLE", "").lower() == "true",
    )

Environment variables (convention, not enforced here):

- ``INQTRIX_LOG_ENABLED`` — set to ``true`` to activate file logging.
- ``INQTRIX_LOG_LEVEL``   — ``DEBUG``, ``INFO``, ``WARNING`` (default: ``INFO``).
- ``INQTRIX_LOG_CONSOLE``  — set to ``true`` to additionally print
  WARNING+ messages to stderr. Useful for server mode.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from inqtrix.urls import sanitize_error

_FILE_FORMAT = "%(asctime)s | %(levelname)-7s | %(threadName)s | %(name)s | %(message)s"
_CONSOLE_FORMAT = "%(levelname)s | %(message)s"


class _RedactSecretsFilter(logging.Filter):
    """Scrub secrets from log records before they reach any handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = sanitize_error(record.msg)
        if record.args:
            if isinstance(record.args, Mapping):
                record.args = {
                    key: sanitize_error(value) if isinstance(value, (str, Exception)) else value
                    for key, value in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    sanitize_error(arg) if isinstance(arg, (str, Exception)) else arg
                    for arg in record.args
                )
            else:
                record.args = sanitize_error(record.args) if isinstance(
                    record.args, (str, Exception)) else record.args
        return True


def _close_handlers(logger: logging.Logger) -> None:
    """Remove and close all handlers currently attached to *logger*."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def configure_logging(
    *,
    enabled: bool = False,
    level: str = "INFO",
    log_dir: str = "logs",
    console: bool = False,
) -> Path | None:
    """Configure the ``inqtrix`` logger.

    Only touches ``logging.getLogger("inqtrix")`` — never the root
    logger — so third-party libraries like ``botocore``, ``httpx``,
    or ``openai`` do not flood the output.

    Parameters
    ----------
    enabled:
        When *False* (default), logging is silenced (``NullHandler``).
        When *True*, a file handler is created in *log_dir*.
    level:
        Log level name (``DEBUG``, ``INFO``, ``WARNING``, …).
    log_dir:
        Directory for log files.  Created automatically.
    console:
        If *True*, an additional ``StreamHandler`` at ``WARNING`` level
        is attached so critical issues still appear on stderr even when
        file logging is disabled.

    Returns
    -------
    Path | None
        Path to the log file when file logging is active, else *None*.
    """
    logger = logging.getLogger("inqtrix")
    _close_handlers(logger)
    logger.propagate = False

    resolved_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(resolved_level)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
        console_handler.addFilter(_RedactSecretsFilter())
        logger.addHandler(console_handler)

    if not enabled:
        if console:
            return None
        logger.addHandler(logging.NullHandler())
        return None

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"inqtrix_{datetime.now():%Y%m%d_%H%M%S}.log"

    redact_filter = _RedactSecretsFilter()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(resolved_level)
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    file_handler.addFilter(redact_filter)
    logger.addHandler(file_handler)

    return log_file
