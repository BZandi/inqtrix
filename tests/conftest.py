""" Shared test fixtures for inqtrix unit tests."""

from __future__ import annotations

import logging

import pytest

from inqtrix.logging_config import _WEB_LOGGER_NAMES
from inqtrix.providers.base import ProviderContext
from inqtrix.search_result import GroundedSearchResult
from inqtrix.settings import Settings
from inqtrix.strategies import (
    DefaultClaimConsolidator,
    DefaultSourceTiering,
    KeywordRiskScorer,
    MultiSignalStopCriteria,
    StrategyContext,
)


class _StubLLM:
    """Minimal stub satisfying ProviderContext.llm for strategy tests."""

    def complete(self, *a, **kw):
        return ""


    def is_available(self):
        return False


class _StubSearch:
    """Minimal stub satisfying ProviderContext.search for strategy tests."""

    def search(self, *a, **kw):
        return GroundedSearchResult()

    def is_available(self):
        return False


@pytest.fixture(autouse=True)
def reset_inqtrix_logger():
    """Isolate inqtrix logger state around every test.

    ``create_app`` (and the example/server tests that build it) calls
    ``configure_logging``, which sets ``propagate=False`` and attaches
    handlers on ``logging.getLogger("inqtrix")``. Without a reset that leaks
    into whatever test runs next; in default (alphabetical) collection order
    a ``create_app`` test can precede a ``caplog``-based test and silently
    break it because ``caplog`` relies on propagation to the root logger.
    Restoring handlers/level/propagate here keeps the leak from crossing test
    boundaries (Gotcha #1 / Test-Order-Hygiene). Web-server loggers that the
    uvicorn ``dictConfig`` tests reconfigure are restored for the same reason.
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


@pytest.fixture
def settings():
    """A default Settings instance."""
    return Settings()


@pytest.fixture
def tiering():
    """A DefaultSourceTiering strategy."""
    return DefaultSourceTiering()


@pytest.fixture
def consolidator(tiering):
    """A DefaultClaimConsolidator strategy."""
    return DefaultClaimConsolidator(source_tiering=tiering)


@pytest.fixture
def risk_scorer():
    """A KeywordRiskScorer strategy."""
    return KeywordRiskScorer()
