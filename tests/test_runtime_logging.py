"""Tests for structured runtime logging helpers and integration points."""

from __future__ import annotations

import logging
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest

from inqtrix.graph import run
from inqtrix.logging_config import configure_logging
from inqtrix.nodes import search
from inqtrix.providers.base import ProviderContext
from inqtrix.runtime_logging import describe_search_provider
from inqtrix.settings import AgentSettings
from inqtrix.state import append_iteration_log, initial_state
from inqtrix.strategies import StrategyContext, create_default_strategies


@pytest.fixture(autouse=True)
def reset_inqtrix_logger():
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


class AzureOpenAILLM:
    def __init__(self) -> None:
        self.models = SimpleNamespace(
            reasoning_model="gpt-4o",
            effective_classify_model="gpt-4o-mini",
            effective_summarize_model="gpt-4o-mini",
            effective_evaluate_model="gpt-4o",
        )
        self._default_max_tokens = 4096

    def complete(self, *args, **kwargs):
        return ""

    def summarize_parallel(self, *args, **kwargs):
        return ("", 0, 0)

    def is_available(self):
        return True


class AzureFoundryWebSearch:
    def __init__(self) -> None:
        self._agent_name = "web-search-agent"
        self._agent_version = "2026-04-01"

    def search(self, *args, **kwargs):
        return {
            "answer": "",
            "citations": [],
            "related_questions": [],
            "_prompt_tokens": 0,
            "_completion_tokens": 0,
        }

    def is_available(self):
        return True


def _flush_inqtrix_handlers() -> None:
    logger = logging.getLogger("inqtrix")
    for handler in logger.handlers:
        if hasattr(handler, "flush"):
            handler.flush()


def test_run_logs_start_metadata_banner(tmp_path, monkeypatch):
    log_path = configure_logging(enabled=True, level="DEBUG", log_dir=str(tmp_path / "logs"))

    class _StubAgent:
        def invoke(self, state):
            return {"answer": "ok", "total_prompt_tokens": 0, "total_completion_tokens": 0}

    monkeypatch.setattr("inqtrix.graph.get_agent", lambda *args, **kwargs: _StubAgent())

    settings = AgentSettings()
    providers = ProviderContext(llm=AzureOpenAILLM(), search=AzureFoundryWebSearch())
    strategies = create_default_strategies(settings)

    run(
        "Was ist der aktuelle Stand?",
        providers=providers,
        strategies=strategies,
        settings=settings,
    )

    _flush_inqtrix_handlers()
    content = Path(log_path).read_text(encoding="utf-8")

    assert "RUN start:" in content
    assert "profile=compact" in content
    assert "llm=AzureOpenAILLM" in content
    assert "reasoning=gpt-4o" in content
    assert "search=AzureFoundryWebSearch" in content
    assert "engine=web-search-agent@2026-04-01" in content
    assert "default_max_tokens=4096" in content
    assert '"event": "run_start"' in content
    assert '"report_profile": "compact"' in content


def test_describe_search_provider_handles_common_engine_labels():
    class PerplexitySearch:
        def __init__(self) -> None:
            self._model = "perplexity-sonar-pro-agent"

    class AzureFoundryBingSearch:
        def __init__(self) -> None:
            self._agent_id = "agent-123"

    class BraveSearch:
        pass

    assert describe_search_provider(PerplexitySearch())[
        "engine"] == "perplexity-sonar-pro-agent"
    assert describe_search_provider(AzureFoundryBingSearch())["engine"] == "agent-123"
    assert describe_search_provider(BraveSearch())["engine"] == "brave-web-search"


def test_append_iteration_log_writes_debug_payload_without_testing_mode(tmp_path):
    log_path = configure_logging(enabled=True, level="DEBUG", log_dir=str(tmp_path / "logs"))
    state = {"iteration_logs": []}

    append_iteration_log(
        state,
        {
            "node": "answer",
            "prompt_citations": ["https://example.com/report"],
            "fallback_attempted": True,
        },
        testing_mode=False,
    )

    _flush_inqtrix_handlers()
    content = Path(log_path).read_text(encoding="utf-8")

    assert state["iteration_logs"] == []
    assert "ITERATION answer:" in content
    # URLs in iteration logs must remain visible (Phase 7 fix: only credential
    # query parameters are redacted, no more pauschal [URL] replacement).
    assert '"prompt_citations": ["https://example.com/report"]' in content
    assert '"fallback_attempted": true' in content


def test_search_logs_per_query_runtime_debug_artifacts(tmp_path):
    log_path = configure_logging(enabled=True, level="DEBUG", log_dir=str(tmp_path / "logs"))

    class _SearchWithNotice:
        def __init__(self) -> None:
            self._notice = None

        def search(self, *args, **kwargs):
            self._notice = "search fallback"
            return {
                "answer": "Gefundener Text",
                "citations": ["https://example.com/report"],
                "related_questions": ["Was ist neu?"],
                "_prompt_tokens": 11,
                "_completion_tokens": 7,
            }

        def consume_nonfatal_notice(self):
            notice = self._notice
            self._notice = None
            return notice

        def is_available(self):
            return True

    class _SummarizeLLM:
        def __init__(self) -> None:
            self._notice = None
            self.models = SimpleNamespace(
                reasoning_model="reasoning-model",
                effective_summarize_model="summary-model",
            )

        def complete(self, *args, **kwargs):
            return ""

        def summarize_parallel(self, *args, **kwargs):
            self._notice = "summarize fallback"
            return ("Fallback-Zusammenfassung", 0, 0)

        def consume_nonfatal_notice(self):
            notice = self._notice
            self._notice = None
            return notice

        def is_available(self):
            return True

    class _ClaimExtractionWithNotice:
        def __init__(self) -> None:
            self._notice = None

        def extract(self, *args, **kwargs):
            self._notice = "claim fallback"
            return ([{"claim_text": "Wichtiger Fakt aus der Quelle", "source_urls": ["https://example.com/report"]}], 0, 0)

        def consume_nonfatal_notice(self):
            notice = self._notice
            self._notice = None
            return notice

    settings = AgentSettings(first_round_queries=1, max_rounds=4)
    llm = _SummarizeLLM()
    defaults = create_default_strategies(settings, llm=llm, summarize_model="summary-model")
    strategies = StrategyContext(
        source_tiering=defaults.source_tiering,
        claim_extraction=_ClaimExtractionWithNotice(),
        claim_consolidation=defaults.claim_consolidation,
        context_pruning=defaults.context_pruning,
        risk_scoring=defaults.risk_scoring,
        stop_criteria=defaults.stop_criteria,
    )
    state = initial_state("Was ist passiert?", progress_queue=Queue(), max_total_seconds=30)
    state["queries"] = ["q1"]

    search(
        state,
        providers=ProviderContext(llm=llm, search=_SearchWithNotice()),
        strategies=strategies,
        settings=settings,
    )

    _flush_inqtrix_handlers()
    content = Path(log_path).read_text(encoding="utf-8")

    assert "ITERATION search:" in content
    assert '"search_fallbacks": 1' in content
    assert '"summarize_fallbacks": 1' in content
    assert '"claim_fallbacks": 1' in content
    assert '"provider_notice": "search fallback"' in content
    assert '"summarize_notice": "summarize fallback"' in content
    assert '"claim_notice": "claim fallback"' in content
    assert '"summary": "Fallback-Zusammenfassung"' in content
