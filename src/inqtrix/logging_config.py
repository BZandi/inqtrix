"""Centralized logging configuration for inqtrix.

Provides ``configure_logging()`` for the ``inqtrix`` logger (with
automatic secret redaction) plus ``build_uvicorn_log_config()`` for the
uvicorn/FastAPI loggers. The two helpers are intentionally separate
because uvicorn overwrites its own loggers via ``logging.config.dictConfig``
on every ``uvicorn.run(...)`` call — any handler that was attached to
``uvicorn.*`` before that call gets dropped. The recipe in the
webserver-stack examples is therefore:

1. ``configure_logging(...)`` — sets up the inqtrix logger and returns
   the file path of the active logfile.
2. ``uvicorn.run(app, log_config=build_uvicorn_log_config(log_path), ...)``
   — uvicorn applies the returned dict-config which mirrors its own
   defaults (stderr/stdout) AND adds a FileHandler pointing at the
   inqtrix logfile so request access lines, startup/shutdown notices
   and FastAPI errors land in the same file as the inqtrix records.

Environment variables (convention, not enforced here):

- ``INQTRIX_LOG_ENABLED`` — set to ``true`` to activate file logging.
- ``INQTRIX_LOG_LEVEL``   — ``DEBUG``, ``INFO``, ``WARNING`` (default: ``INFO``).
- ``INQTRIX_LOG_CONSOLE``  — set to ``true`` to additionally print
  WARNING+ messages to stderr. Useful for server mode.
- ``INQTRIX_LOG_INCLUDE_WEB`` — when file logging is enabled, also
  pass a uvicorn ``log_config`` that mirrors web-server logs into the
  same file (default ``true``; set to ``false`` when uvicorn streams
  to a structured-logging sink and the duplication would be noise).
- ``INQTRIX_LOG_WEB_LEVEL`` — log level applied to the uvicorn loggers
  in the generated ``log_config`` (default ``INFO`` so access lines
  ``GET /health 200 OK`` make it into the file).
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from inqtrix.urls import sanitize_log_message

_FILE_FORMAT = "%(asctime)s | %(levelname)-7s | %(threadName)s | %(name)s | %(message)s"
_CONSOLE_FORMAT = "%(levelname)s | %(message)s"

# Web-server loggers that ``build_uvicorn_log_config`` reconfigures so
# uvicorn / FastAPI log records mirror into the inqtrix file. Order
# matters only for documentation clarity; the dict-config wires each
# logger independently.
_WEB_LOGGER_NAMES: tuple[str, ...] = (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "fastapi",
)


class _RedactSecretsFilter(logging.Filter):
    """Scrub secrets from log records before they reach any handler.

    Uses :func:`inqtrix.urls.sanitize_log_message` rather than
    ``sanitize_error`` so harmless URLs in answer text or citation maps are
    preserved (only credential values inside URLs are redacted). This keeps
    the log stream debuggable without leaking API keys.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = sanitize_log_message(record.msg)
        if record.args:
            if isinstance(record.args, Mapping):
                record.args = {
                    key: sanitize_log_message(value) if isinstance(value, (str, Exception)) else value
                    for key, value in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    sanitize_log_message(arg) if isinstance(arg, (str, Exception)) else arg
                    for arg in record.args
                )
            else:
                record.args = sanitize_log_message(record.args) if isinstance(
                    record.args, (str, Exception)) else record.args
        return True


def _close_handlers(logger: logging.Logger) -> None:
    """Remove and close all handlers currently attached to *logger*."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def is_configured() -> bool:
    """Return True iff the ``inqtrix`` logger already has real handlers.

    A ``NullHandler``-only logger is considered *not configured* — it is
    the silent default that example scripts and the server's last-resort
    bootstrap should be allowed to overwrite. Any non-null handler
    (FileHandler, StreamHandler, ...) counts as a real prior
    configuration that callers like :func:`inqtrix.server.app.create_app`
    must respect rather than tear down.

    Returns:
        True when at least one non-``NullHandler`` is attached to the
        ``inqtrix`` logger; False otherwise.
    """
    logger = logging.getLogger("inqtrix")
    return any(
        not isinstance(handler, logging.NullHandler)
        for handler in logger.handlers
    )


def configure_logging(
    *,
    enabled: bool = False,
    level: str = "INFO",
    log_dir: str = "logs",
    console: bool = False,
    force: bool = True,
) -> Path | None:
    """Configure the ``inqtrix`` logger.

    Only touches ``logging.getLogger("inqtrix")`` — never the root
    logger — so third-party libraries like ``botocore``, ``httpx``,
    or ``openai`` do not flood the output. The uvicorn/FastAPI
    loggers are configured separately via
    :func:`build_uvicorn_log_config` because uvicorn applies its own
    ``dictConfig`` on every ``uvicorn.run(...)`` call and would drop
    any handler attached here.

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
    force:
        When *True* (default, backwards-compatible), the existing
        handlers are closed and replaced — useful for tests and for
        an explicit re-configure from a script. When *False*, the
        function becomes a no-op if a real handler is already present
        (see :func:`is_configured`); this is the mode the server's
        last-resort bootstrap uses so a webserver-stack example that
        already wired its own file logger is not torn down by
        :func:`inqtrix.server.app.create_app`. The check looks at
        non-``NullHandler`` handlers only, so the silent default
        installed by ``configure_logging(enabled=False)`` is still
        replaceable.

    Returns
    -------
    Path | None
        Path to the log file when file logging is active, else *None*.
        When ``force=False`` and the logger was already configured the
        return value is also *None* (no new file handler is created).
    """
    logger = logging.getLogger("inqtrix")
    if not force and is_configured():
        return None
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


def build_uvicorn_log_config(
    log_file: Path | str | None,
    *,
    web_level: str = "INFO",
) -> dict[str, object]:
    """Build a uvicorn-compatible ``log_config`` dict that mirrors web logs into *log_file*.

    Designed to be passed to ``uvicorn.run(app, log_config=...)`` so
    uvicorn's own ``dictConfig`` install does not strip the inqtrix
    file handler off the ``uvicorn``/``uvicorn.error``/``uvicorn.access``/
    ``fastapi`` loggers. The returned dict mirrors uvicorn's default
    ``LOGGING_CONFIG`` (stderr stream for ``default``, stdout for
    ``access``) and additionally wires each logger to a ``FileHandler``
    pointing at *log_file*.

    Args:
        log_file: Path to the inqtrix logfile (typically the return
            value of :func:`configure_logging`). When ``None``, the
            returned config is identical to uvicorn's defaults — useful
            for the ``include_web=False`` env path.
        web_level: Log level name applied to every web-server logger.
            Default ``INFO`` so request access lines reach the file.

    Returns:
        A dict-config compatible with ``logging.config.dictConfig`` and
        with uvicorn's ``log_config`` parameter.
    """
    web_level_resolved = getattr(logging, web_level.upper(), logging.INFO)
    web_level_name = logging.getLevelName(web_level_resolved)

    handlers: dict[str, dict[str, object]] = {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    }

    default_handlers = ["default"]
    access_handlers = ["access"]

    if log_file is not None:
        handlers["inqtrix_file"] = {
            "formatter": "file",
            "class": "logging.FileHandler",
            "filename": str(log_file),
            "encoding": "utf-8",
            "filters": ["redact_secrets"],
        }
        default_handlers.append("inqtrix_file")
        access_handlers.append("inqtrix_file")

    config: dict[str, object] = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "redact_secrets": {
                "()": "inqtrix.logging_config._RedactSecretsFilter",
            },
        },
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
            "file": {
                "format": _FILE_FORMAT,
            },
        },
        "handlers": handlers,
        "loggers": {
            "uvicorn": {
                "handlers": default_handlers,
                "level": web_level_name,
                "propagate": False,
            },
            "uvicorn.error": {
                "level": web_level_name,
            },
            "uvicorn.access": {
                "handlers": access_handlers,
                "level": web_level_name,
                "propagate": False,
            },
            "fastapi": {
                "handlers": default_handlers,
                "level": web_level_name,
                "propagate": False,
            },
        },
    }
    return config


def describe_logging_state() -> dict[str, Any]:
    """Introspect the ``inqtrix`` and uvicorn loggers and return a status dict.

    Used by :func:`format_logging_banner` (and by tests) to render a
    terminal-friendly summary of the active logging configuration at
    server startup. The return value is intentionally plain Python
    (no Pydantic) so it stays cheap to call and easy to serialise.

    Returns:
        A dict with the following keys:

        - ``file_enabled`` (bool): whether a real :class:`logging.FileHandler`
          is attached to the ``inqtrix`` logger.
        - ``file_path`` (str | None): absolute path of the first file
          handler, or ``None`` when file logging is off.
        - ``level`` (str): effective level name of the ``inqtrix`` logger
          (``DEBUG``, ``INFO``, ``WARNING``, ...).
        - ``console_enabled`` (bool): whether a :class:`logging.StreamHandler`
          (i.e. stderr mirror) is attached to the ``inqtrix`` logger.
        - ``silent`` (bool): ``True`` when only a ``NullHandler`` is
          attached — useful to tell operators that nothing will be
          emitted anywhere.
        - ``web_mirrored`` (bool): ``True`` when uvicorn / FastAPI
          loggers also write into the inqtrix logfile (i.e. a
          :class:`logging.FileHandler` is wired into ``uvicorn`` or
          ``uvicorn.access``). ``False`` otherwise.
        - ``web_level`` (str | None): uvicorn logger level when web
          loggers have any non-default handler; ``None`` when
          untouched.
    """
    logger = logging.getLogger("inqtrix")

    file_path: str | None = None
    file_enabled = False
    console_enabled = False
    silent_only = True

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            file_enabled = True
            silent_only = False
            if file_path is None:
                try:
                    file_path = str(Path(handler.baseFilename).resolve())
                except Exception:  # noqa: BLE001
                    file_path = handler.baseFilename
        elif isinstance(handler, logging.NullHandler):
            continue
        elif isinstance(handler, logging.StreamHandler):
            console_enabled = True
            silent_only = False
        else:
            silent_only = False

    web_mirrored = False
    web_level: str | None = None
    for name in _WEB_LOGGER_NAMES:
        web_logger = logging.getLogger(name)
        if web_logger.handlers:
            if web_level is None:
                web_level = logging.getLevelName(web_logger.level or logging.INFO)
            for handler in web_logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    web_mirrored = True
                    break
        if web_mirrored:
            break

    return {
        "file_enabled": file_enabled,
        "file_path": file_path,
        "level": logging.getLevelName(logger.level or logging.WARNING),
        "console_enabled": console_enabled,
        "silent": silent_only and not file_enabled and not console_enabled,
        "web_mirrored": web_mirrored,
        "web_level": web_level,
    }


def format_logging_banner(state: dict[str, Any] | None = None) -> str:
    """Render a human-readable banner describing the current logging state.

    Produces a compact multi-line box suitable for ``print()`` to
    stderr at server startup. The banner is intentionally free of ANSI
    colour codes so it stays readable in log collectors (Docker, k8s,
    journalctl) that do not interpret terminal escapes.

    Args:
        state: Pre-computed state dict (see :func:`describe_logging_state`).
            When ``None`` (default), the function calls
            ``describe_logging_state()`` itself. The argument exists
            mainly for tests.

    Returns:
        A newline-terminated multi-line string. The caller decides
        where to print it.
    """
    if state is None:
        state = describe_logging_state()

    lines: list[str] = []
    lines.append("-" * 64)
    lines.append("  Inqtrix Server - Logging Status")
    lines.append("-" * 64)

    if state["file_enabled"]:
        lines.append(f"  File logging:    ENABLED (level={state['level']})")
        lines.append(f"  Log file:        {state['file_path']}")
    else:
        lines.append("  File logging:    DISABLED")

    if state["console_enabled"]:
        lines.append("  Console output:  ENABLED (stderr, WARNING+)")
    else:
        lines.append("  Console output:  disabled")

    if state["web_mirrored"]:
        web_level = state.get("web_level") or "INFO"
        lines.append(
            f"  Web-server logs: mirrored into log file (level={web_level})"
        )
    else:
        lines.append("  Web-server logs: terminal only (uvicorn defaults)")

    if state["silent"]:
        lines.append("")
        lines.append("  Note: Logger is silent. Enable file logging with:")
        lines.append("    INQTRIX_LOG_ENABLED=true INQTRIX_LOG_LEVEL=INFO")
    elif not state["file_enabled"]:
        lines.append("")
        lines.append("  Tip: Activate persistent logs with:")
        lines.append("    INQTRIX_LOG_ENABLED=true INQTRIX_LOG_LEVEL=INFO")

    lines.append("-" * 64)
    return "\n".join(lines) + "\n"


def print_logging_banner(stream: TextIO | None = None) -> dict[str, Any]:
    """Print the logging banner to *stream* (default ``sys.stderr``).

    Convenience wrapper around :func:`describe_logging_state` +
    :func:`format_logging_banner` + ``print``. Returns the state dict
    so callers can react to it (e.g. additionally log it through the
    inqtrix logger).

    Args:
        stream: Output stream. Defaults to ``sys.stderr`` so the banner
            stays visible when stdout is piped to another process.

    Returns:
        The state dict produced by :func:`describe_logging_state`.
    """
    state = describe_logging_state()
    target = stream if stream is not None else sys.stderr
    try:
        target.write(format_logging_banner(state))
        target.flush()
    except Exception:  # noqa: BLE001 — never let a broken stream crash startup
        pass
    return state
