"""Tests for centralized logging configuration."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import io

from inqtrix.logging_config import (
    _WEB_LOGGER_NAMES,
    build_uvicorn_log_config,
    configure_logging,
    describe_logging_state,
    format_logging_banner,
    is_configured,
    print_logging_banner,
)


@pytest.fixture(autouse=True)
def reset_inqtrix_logger():
    """Isolate logger state for each test in this module.

    Restores the inqtrix logger AND every web-server logger that the
    uvicorn log_config tests may dictConfig-reconfigure, so leaks
    don't bleed into later tests (Gotcha #1 territory).
    """
    inqtrix_logger = logging.getLogger("inqtrix")
    previous_handlers = list(inqtrix_logger.handlers)
    previous_level = inqtrix_logger.level
    previous_propagate = inqtrix_logger.propagate

    web_state = {
        name: (
            list(logging.getLogger(name).handlers),
            logging.getLogger(name).level,
            logging.getLogger(name).propagate,
        )
        for name in _WEB_LOGGER_NAMES
    }

    for handler in list(inqtrix_logger.handlers):
        inqtrix_logger.removeHandler(handler)

    yield

    for handler in list(inqtrix_logger.handlers):
        inqtrix_logger.removeHandler(handler)
        handler.close()

    inqtrix_logger.setLevel(previous_level)
    inqtrix_logger.propagate = previous_propagate
    for handler in previous_handlers:
        inqtrix_logger.addHandler(handler)

    for name, (handlers, level, propagate) in web_state.items():
        web_logger = logging.getLogger(name)
        for handler in list(web_logger.handlers):
            web_logger.removeHandler(handler)
            if isinstance(handler, logging.FileHandler):
                handler.close()
        for handler in handlers:
            web_logger.addHandler(handler)
        web_logger.setLevel(level)
        web_logger.propagate = propagate


def test_console_handler_works_without_file_logging(capsys):
    log_path = configure_logging(enabled=False, level="WARNING", console=True)

    assert log_path is None

    logger = logging.getLogger("inqtrix")
    logger.warning("visible warning")

    captured = capsys.readouterr()
    assert "visible warning" in captured.err
    assert all(not isinstance(handler, logging.NullHandler) for handler in logger.handlers)


def test_reconfigure_closes_previous_file_handler(tmp_path):
    log_dir = tmp_path / "logs"

    configure_logging(enabled=True, log_dir=str(log_dir))
    logger = logging.getLogger("inqtrix")
    first_handler = next(
        handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)
    )
    first_stream = first_handler.stream

    configure_logging(enabled=True, log_dir=str(log_dir))

    assert first_stream is not None
    assert first_stream.closed is True


def test_mapping_style_logging_is_preserved_and_redacted(tmp_path, capsys):
    log_path = configure_logging(enabled=True, log_dir=str(tmp_path / "logs"))
    logger = logging.getLogger("inqtrix")

    # Realistic-length API key so the {16,} guard in sanitize_log_message fires.
    logger.info(
        "%(user)s %(token)s",
        {"user": "alice", "token": "sk-secrettoken1234567890abcdef"},
    )
    for handler in logger.handlers:
        if hasattr(handler, "flush"):
            handler.flush()

    captured = capsys.readouterr()
    content = Path(log_path).read_text(encoding="utf-8")

    assert "Logging error" not in captured.err
    assert "alice" in content
    assert "sk-secrettoken1234567890abcdef" not in content
    assert "[KEY]" in content


def test_log_filter_preserves_benign_urls(tmp_path):
    """The log filter must not erase harmless URLs (regression for [URL] redaction)."""
    log_path = configure_logging(enabled=True, log_dir=str(tmp_path / "logs"))
    logger = logging.getLogger("inqtrix")

    logger.info("ANSWER fragment: see [6](https://www.zacks.com/article/abc) for details")
    for handler in logger.handlers:
        if hasattr(handler, "flush"):
            handler.flush()

    content = Path(log_path).read_text(encoding="utf-8")

    assert "https://www.zacks.com/article/abc" in content
    assert "[URL]" not in content


def test_log_filter_strips_credential_inside_url(tmp_path):
    """Credential query parameters inside URLs are still redacted."""
    log_path = configure_logging(enabled=True, log_dir=str(tmp_path / "logs"))
    logger = logging.getLogger("inqtrix")

    logger.info(
        "Outbound call: https://api.example.com/v1?api_key=sk-realLookingApiKey1234567890&page=2"
    )
    for handler in logger.handlers:
        if hasattr(handler, "flush"):
            handler.flush()

    content = Path(log_path).read_text(encoding="utf-8")

    assert "sk-realLookingApiKey1234567890" not in content
    assert "api_key=[REDACTED]" in content
    assert "https://api.example.com/v1" in content
    assert "page=2" in content


def test_log_directory_is_created_automatically(tmp_path):
    log_dir = tmp_path / "nested" / "logs"

    assert not log_dir.exists()

    log_path = configure_logging(enabled=True, log_dir=str(log_dir))

    assert log_dir.exists()
    assert log_path == log_dir / log_path.name


def test_force_false_preserves_existing_handlers(tmp_path):
    """Bug B regression: when a script already configured the inqtrix
    logger (with a FileHandler), a follow-up ``configure_logging(force=False)``
    from ``create_app`` must NOT close and replace those handlers.

    Without this guard, every webserver-stack example silently lost its
    INFO-level file logging because ``create_app`` called
    ``configure_logging`` a second time and tore the script's setup down.
    """
    log_dir = tmp_path / "logs"

    first_path = configure_logging(
        enabled=True, level="INFO", log_dir=str(log_dir), console=True
    )
    logger = logging.getLogger("inqtrix")
    file_handlers_before = [
        h for h in logger.handlers if isinstance(h, logging.FileHandler)
    ]
    assert len(file_handlers_before) == 1
    assert is_configured() is True

    second_path = configure_logging(
        enabled=False, level="WARNING", console=True, force=False
    )

    assert second_path is None
    file_handlers_after = [
        h for h in logger.handlers if isinstance(h, logging.FileHandler)
    ]
    assert file_handlers_after == file_handlers_before, (
        "force=False must not remove the existing FileHandler"
    )
    assert file_handlers_after[0].stream.closed is False
    assert first_path is not None and first_path.exists()


def test_force_true_default_still_replaces_handlers(tmp_path):
    """Backwards compatibility: the default ``force=True`` keeps the
    historical reset-and-replace semantics so existing test fixtures and
    explicit script-driven re-configures still work.
    """
    log_dir = tmp_path / "logs"

    configure_logging(enabled=True, level="INFO", log_dir=str(log_dir))
    logger = logging.getLogger("inqtrix")
    first_handler = next(
        h for h in logger.handlers if isinstance(h, logging.FileHandler)
    )
    first_stream = first_handler.stream

    configure_logging(enabled=True, level="DEBUG", log_dir=str(log_dir))

    assert first_stream.closed is True


def test_build_uvicorn_log_config_with_log_file_mirrors_into_file(tmp_path):
    """When a log_file is provided, the dict-config must wire a
    FileHandler into uvicorn / uvicorn.access / fastapi so request
    access lines land in the inqtrix file once uvicorn applies the
    config.
    """
    import logging.config

    log_file = tmp_path / "logs" / "inqtrix_test.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.touch()

    config = build_uvicorn_log_config(log_file, web_level="INFO")
    logging.config.dictConfig(config)

    logging.getLogger("uvicorn.access").info(
        '127.0.0.1:12345 - "GET /health HTTP/1.1" 200'
    )
    logging.getLogger("fastapi").info("fastapi pre-flight check ok")
    for name in _WEB_LOGGER_NAMES:
        for handler in logging.getLogger(name).handlers:
            if hasattr(handler, "flush"):
                handler.flush()

    content = log_file.read_text(encoding="utf-8")
    assert "GET /health" in content
    assert "fastapi pre-flight check ok" in content


def test_build_uvicorn_log_config_without_file_keeps_defaults(tmp_path):
    """``log_file=None`` must produce a dict-config that preserves
    uvicorn's stderr/stdout defaults but does not write to any file
    (no inqtrix_file handler key).
    """
    config = build_uvicorn_log_config(None)
    assert "inqtrix_file" not in config["handlers"]
    # The two default uvicorn handlers must still be present.
    assert {"default", "access"} <= set(config["handlers"].keys())


def test_build_uvicorn_log_config_respects_web_level(tmp_path):
    """The dict-config must apply ``web_level`` to every uvicorn
    logger so an operator can crank uvicorn down to WARNING without
    losing the inqtrix file logger.
    """
    log_file = tmp_path / "logs" / "inqtrix_test.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.touch()

    config = build_uvicorn_log_config(log_file, web_level="WARNING")

    for logger_name in _WEB_LOGGER_NAMES:
        assert config["loggers"][logger_name]["level"] == "WARNING", (
            f"web_level should propagate to {logger_name}"
        )


def test_describe_logging_state_reports_silent_logger():
    """Before configuration the logger is silent and the banner must say so."""
    state = describe_logging_state()

    assert state["file_enabled"] is False
    assert state["file_path"] is None
    assert state["console_enabled"] is False
    assert state["web_mirrored"] is False


def test_describe_logging_state_reports_file_path(tmp_path):
    """With file logging enabled the state must expose the absolute log path."""
    log_dir = tmp_path / "logs"
    log_path = configure_logging(enabled=True, level="INFO", log_dir=str(log_dir))
    assert log_path is not None

    state = describe_logging_state()

    assert state["file_enabled"] is True
    assert state["file_path"] is not None
    assert Path(state["file_path"]) == log_path.resolve()
    assert state["level"] == "INFO"
    assert state["silent"] is False


def test_describe_logging_state_reports_console_only(tmp_path):
    """Console-only configuration must be reported as such (no file path)."""
    configure_logging(enabled=False, level="WARNING", console=True)

    state = describe_logging_state()

    assert state["file_enabled"] is False
    assert state["file_path"] is None
    assert state["console_enabled"] is True
    assert state["silent"] is False


def test_format_logging_banner_mentions_log_file(tmp_path):
    """The banner must contain the log path when file logging is active."""
    log_path = configure_logging(
        enabled=True, level="DEBUG", log_dir=str(tmp_path / "logs")
    )
    banner = format_logging_banner()

    assert "ENABLED" in banner
    assert "DEBUG" in banner
    assert str(log_path) in banner


def test_format_logging_banner_disabled_has_tip():
    """When logging is off, the banner must tell operators how to turn it on."""
    banner = format_logging_banner()

    assert "DISABLED" in banner
    assert "INQTRIX_LOG_ENABLED=true" in banner


def test_print_logging_banner_writes_to_stream(tmp_path):
    """print_logging_banner writes to the provided stream and returns state."""
    configure_logging(enabled=True, level="INFO", log_dir=str(tmp_path / "logs"))
    buffer = io.StringIO()

    state = print_logging_banner(buffer)

    written = buffer.getvalue()
    assert "Inqtrix Server - Logging Status" in written
    assert state["file_enabled"] is True
    assert state["file_path"] is not None
    assert state["file_path"] in written


def test_is_configured_ignores_null_handler():
    """``is_configured`` must treat a NullHandler-only logger as silent
    so the server's last-resort bootstrap can install real handlers.
    """
    logger = logging.getLogger("inqtrix")
    logger.addHandler(logging.NullHandler())

    assert is_configured() is False

    configure_logging(enabled=False, console=True)

    assert is_configured() is True
