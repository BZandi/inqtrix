"""Tests for centralized logging configuration."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from inqtrix.logging_config import configure_logging


@pytest.fixture(autouse=True)
def reset_inqtrix_logger():
    """Isolate logger state for each test in this module."""
    logger = logging.getLogger("inqtrix")
    previous_handlers = list(logger.handlers)
    previous_level = logger.level
    previous_propagate = logger.propagate

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    yield

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(previous_level)
    logger.propagate = previous_propagate
    for handler in previous_handlers:
        logger.addHandler(handler)


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

    logger.info("%(user)s %(token)s", {"user": "alice", "token": "sk-secret-token"})
    for handler in logger.handlers:
        if hasattr(handler, "flush"):
            handler.flush()

    captured = capsys.readouterr()
    content = Path(log_path).read_text(encoding="utf-8")

    assert "Logging error" not in captured.err
    assert "alice" in content
    assert "sk-secret-token" not in content
    assert "[KEY]" in content


def test_log_directory_is_created_automatically(tmp_path):
    log_dir = tmp_path / "nested" / "logs"

    assert not log_dir.exists()

    log_path = configure_logging(enabled=True, log_dir=str(log_dir))

    assert log_dir.exists()
    assert log_path == log_dir / log_path.name
