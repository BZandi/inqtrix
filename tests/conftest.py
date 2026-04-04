""" Shared test fixtures for inqtrix unit tests."""

from __future__ import annotations

import pytest

from inqtrix.providers import ProviderContext
from inqtrix.settings import Settings
from inqtrix.strategies import (
    DefaultClaimConsolidator,
    DefaultSourceTiering,
    KeywordRiskScorer,
    MultiSignalStopCriteria,
    RelevanceBasedPruning,
    StrategyContext,
)


class _StubLLM:
    """Minimal stub satisfying ProviderContext.llm for strategy tests."""

    def complete(self, *a, **kw):
        return ""

    def summarize_parallel(self, *a, **kw):
        return ("", 0, 0)

    def is_available(self):
        return False


class _StubSearch:
    """Minimal stub satisfying ProviderContext.search for strategy tests."""

    def search(self, *a, **kw):
        return {"answer": "", "citations": [], "related_questions": [],
                "_prompt_tokens": 0, "_completion_tokens": 0}

    def is_available(self):
        return False


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


@pytest.fixture
def pruning():
    """A RelevanceBasedPruning strategy."""
    return RelevanceBasedPruning()
