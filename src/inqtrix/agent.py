"""High-level ``ResearchAgent`` — the main public entry point.

Usage
-----
Minimal (uses environment variables or a local ``.env`` file for configuration)::

    from inqtrix import ResearchAgent

    agent = ResearchAgent()
    result = agent.research("Was ist der aktuelle Stand der GKV-Reform?")
    print(result.answer)
    print(result.metrics.confidence)

Custom configuration (Baukasten)::

    from inqtrix import ResearchAgent, AgentConfig

    agent = ResearchAgent(AgentConfig(
        max_rounds=3,
        confidence_stop=7,
    ))
    result = agent.research("Meine Frage")

Fully custom providers::

    from inqtrix import ResearchAgent, AgentConfig
    from inqtrix.providers import LiteLLMProvider, PerplexitySearch

    agent = ResearchAgent(AgentConfig(
        llm=my_custom_llm,
        search=my_custom_search,
    ))
"""

from __future__ import annotations

import logging
import time
from queue import Queue
from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict

from inqtrix.providers import LLMProvider, SearchProvider, ProviderContext
from inqtrix.result import ResearchResult
from inqtrix.strategies import (
    ClaimConsolidationStrategy,
    ClaimExtractionStrategy,
    ContextPruningStrategy,
    RiskScoringStrategy,
    SourceTieringStrategy,
    StopCriteriaStrategy,
    StrategyContext,
)

log = logging.getLogger("inqtrix")


# ------------------------------------------------------------------ #
# AgentConfig — Pydantic model for all configuration
# ------------------------------------------------------------------ #


class AgentConfig(BaseModel):
    """Fully declarative agent configuration.

    Every field has a sensible default.  Pass only what you want to
    override.  Provider and strategy objects are accepted directly,
    enabling the *Baukasten* (building-block) pattern::

        AgentConfig(
            search=BingSearch(api_key="..."),
            max_rounds=2,
        )

    If ``llm`` or ``search`` are left as ``None`` (the default),
    :class:`ResearchAgent` will auto-create them from environment
    variables or a local ``.env`` file on first use (same behaviour as
    the FastAPI server).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # -- Providers (None = auto-create from env vars) --
    llm: LLMProvider | None = None
    search: SearchProvider | None = None

    # -- Strategies (None = use defaults) --
    source_tiering: SourceTieringStrategy | None = None
    claim_extraction: ClaimExtractionStrategy | None = None
    claim_consolidation: ClaimConsolidationStrategy | None = None
    context_pruning: ContextPruningStrategy | None = None
    risk_scoring: RiskScoringStrategy | None = None
    stop_criteria: StopCriteriaStrategy | None = None

    # -- Behaviour --
    max_rounds: int = 4
    confidence_stop: int = 8
    max_context: int = 12
    first_round_queries: int = 6
    answer_prompt_citations_max: int = 60
    max_total_seconds: int = 300
    max_question_length: int = 10_000

    # -- Timeouts --
    reasoning_timeout: int = 120
    search_timeout: int = 60
    summarize_timeout: int = 60

    # -- Risk --
    high_risk_score_threshold: int = 4
    high_risk_classify_escalate: bool = True
    high_risk_evaluate_escalate: bool = True

    # -- Search cache --
    search_cache_maxsize: int = 256
    search_cache_ttl: int = 3600

    # -- Models (only used when auto-creating providers) --
    reasoning_model: str = "claude-opus-4.6-agent"
    search_model: str = "perplexity-sonar-pro-agent"
    classify_model: str = ""
    summarize_model: str = ""
    evaluate_model: str = ""

    # -- Server connection (only used when auto-creating providers) --
    litellm_base_url: str = "http://litellm-proxy:4000/v1"
    litellm_api_key: str = "sk-placeholder"

    # -- Testing --
    testing_mode: bool = False


# ------------------------------------------------------------------ #
# ResearchAgent — the Baukasten entry point
# ------------------------------------------------------------------ #


class ResearchAgent:
    """Iterative AI research agent with pluggable providers and strategies.

    Parameters
    ----------
    config:
        Optional :class:`AgentConfig`.  Defaults are auto-created from
        environment variables when omitted.

    Examples
    --------
    >>> agent = ResearchAgent()
    >>> result = agent.research("What is quantum computing?")
    >>> print(result.answer)
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        self._config = config or AgentConfig()
        self._providers: ProviderContext | None = None
        self._strategies: StrategyContext | None = None
        self._settings: Any = None  # AgentSettings, lazily built

    # -- Public API ------------------------------------------------ #

    def research(
        self,
        question: str,
        *,
        history: str = "",
        prev_session: dict[str, Any] | None = None,
    ) -> ResearchResult:
        """Run iterative research and return a structured result.

        Parameters
        ----------
        question:
            The user question to research.
        history:
            Optional prior conversation history (formatted text).
        prev_session:
            Optional session snapshot for follow-up support.

        Returns
        -------
        ResearchResult
            Typed result with answer, metrics, sources, and claims.
        """
        from inqtrix.graph import run

        providers, strategies, settings = self._ensure_initialised()

        t0 = time.monotonic()
        raw = run(
            question,
            history=history,
            prev_session=prev_session,
            providers=providers,
            strategies=strategies,
            settings=settings,
        )
        elapsed = time.monotonic() - t0

        result = ResearchResult.from_raw(raw)
        result.metrics.elapsed_seconds = round(elapsed, 2)
        return result

    def stream(
        self,
        question: str,
        *,
        history: str = "",
        prev_session: dict[str, Any] | None = None,
        include_progress: bool = True,
    ) -> Iterator[str]:
        """Run research with live progress updates.

        When ``include_progress`` is true, yields progress messages
        (prefixed with ``> ``), then a ``---`` separator, then the final
        answer in word-by-word chunks. When false, yields only the final
        answer chunks.
        """
        from inqtrix.graph import run
        from inqtrix.text import iter_word_chunks

        providers, strategies, settings = self._ensure_initialised()
        progress_queue: Queue | None = Queue() if include_progress else None

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                run,
                question,
                history=history,
                progress_queue=progress_queue,
                prev_session=prev_session,
                providers=providers,
                strategies=strategies,
                settings=settings,
            )

            # Yield progress updates while agent runs
            while include_progress and progress_queue is not None and not future.done():
                try:
                    kind, msg = progress_queue.get(timeout=0.3)
                    if kind == "progress" and msg != "done":
                        yield f"> {msg}\n"
                except Exception:
                    continue

            # Drain remaining progress
            while include_progress and progress_queue is not None and not progress_queue.empty():
                try:
                    kind, msg = progress_queue.get_nowait()
                    if kind == "progress" and msg != "done":
                        yield f"> {msg}\n"
                except Exception:
                    break

            raw = future.result()
            if include_progress:
                yield "---\n"
            for chunk in iter_word_chunks(raw.get("answer", "")):
                yield chunk

    # -- Properties ------------------------------------------------ #

    @property
    def config(self) -> AgentConfig:
        """The active agent configuration."""
        return self._config

    @property
    def providers(self) -> ProviderContext:
        """The active providers (auto-created on first access)."""
        self._ensure_initialised()
        assert self._providers is not None
        return self._providers

    @property
    def strategies(self) -> StrategyContext:
        """The active strategies (auto-created on first access)."""
        self._ensure_initialised()
        assert self._strategies is not None
        return self._strategies

    # -- Internals ------------------------------------------------- #

    def _ensure_initialised(self) -> tuple[ProviderContext, StrategyContext, Any]:
        """Lazily create providers, strategies, and settings from config."""
        if self._providers is not None:
            assert self._strategies is not None
            assert self._settings is not None
            return self._providers, self._strategies, self._settings

        cfg = self._config
        settings = self._build_settings(cfg)
        self._settings = settings

        from inqtrix.settings import Settings, ModelSettings, ServerSettings

        env_settings = Settings()
        full_settings = Settings(
            models=ModelSettings(
                **self._merge_explicit_overrides(
                    env_settings.models,
                    cfg,
                    (
                        "reasoning_model",
                        "search_model",
                        "classify_model",
                        "summarize_model",
                        "evaluate_model",
                    ),
                ),
            ),
            server=ServerSettings(
                **self._merge_explicit_overrides(
                    env_settings.server,
                    cfg,
                    (
                        "litellm_base_url",
                        "litellm_api_key",
                    ),
                ),
            ),
            agent=settings,
        )

        # -- Providers --
        llm = cfg.llm
        search = cfg.search
        if llm is None or search is None:
            from inqtrix.providers import create_providers

            auto = create_providers(full_settings)
            llm = llm or auto.llm
            search = search or auto.search

        if llm is not None and not hasattr(llm, "models"):
            from inqtrix.providers import ConfiguredLLMProvider

            llm = ConfiguredLLMProvider(llm, full_settings.models)

        self._providers = ProviderContext(llm=llm, search=search)

        # -- Strategies --
        from inqtrix.strategies import create_default_strategies
        defaults = create_default_strategies(
            settings,
            llm=llm,
            summarize_model=cfg.summarize_model or cfg.reasoning_model,
            summarize_timeout=cfg.summarize_timeout,
        )
        self._strategies = StrategyContext(
            source_tiering=cfg.source_tiering or defaults.source_tiering,
            claim_extraction=cfg.claim_extraction or defaults.claim_extraction,
            claim_consolidation=cfg.claim_consolidation or defaults.claim_consolidation,
            context_pruning=cfg.context_pruning or defaults.context_pruning,
            risk_scoring=cfg.risk_scoring or defaults.risk_scoring,
            stop_criteria=cfg.stop_criteria or defaults.stop_criteria,
        )

        return self._providers, self._strategies, self._settings

    @staticmethod
    def _merge_explicit_overrides(
        base: Any,
        cfg: AgentConfig,
        field_names: tuple[str, ...],
    ) -> dict[str, Any]:
        """Merge explicit ``AgentConfig`` overrides onto a settings model."""
        data = base.model_dump()
        explicit_fields = cfg.model_fields_set
        for field_name in field_names:
            if field_name in explicit_fields:
                data[field_name] = getattr(cfg, field_name)
        return data

    @staticmethod
    def _build_settings(cfg: AgentConfig) -> Any:
        """Build an AgentSettings instance from the flat AgentConfig."""
        from inqtrix.settings import AgentSettings

        return AgentSettings(
            **ResearchAgent._merge_explicit_overrides(
                AgentSettings(),
                cfg,
                (
                    "max_rounds",
                    "confidence_stop",
                    "max_context",
                    "first_round_queries",
                    "answer_prompt_citations_max",
                    "max_total_seconds",
                    "max_question_length",
                    "reasoning_timeout",
                    "search_timeout",
                    "summarize_timeout",
                    "high_risk_score_threshold",
                    "high_risk_classify_escalate",
                    "high_risk_evaluate_escalate",
                    "search_cache_maxsize",
                    "search_cache_ttl",
                    "testing_mode",
                ),
            )
        )
