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

    from inqtrix import (
        AgentConfig,
        LiteLLM,
        PerplexitySearch,
        ReportProfile,
        ResearchAgent,
    )

    llm = LiteLLM(
        api_key=os.getenv("LITELLM_API_KEY"),
        base_url="http://localhost:4000/v1",
        default_model="gpt-4o",
    )
    search = PerplexitySearch(
        api_key=os.getenv("LITELLM_API_KEY"),
        base_url="http://localhost:4000/v1",
        model="perplexity-sonar-pro-agent",
    )
    agent = ResearchAgent(AgentConfig(
        llm=llm,
        search=search,
        report_profile=ReportProfile.DEEP,
        max_rounds=3,
    ))
"""

from __future__ import annotations

import logging
import time
from queue import Empty, Queue
from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict, Field

from inqtrix.providers.base import LLMProvider, SearchProvider, ProviderContext
from inqtrix.report_profiles import ReportProfile
from inqtrix.result import ResearchResult
from inqtrix.settings import AgentSettings, ModelSettings
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
            llm=LiteLLM(api_key="...", default_model="gpt-4o"),
            search=PerplexitySearch(api_key="...", model="sonar-pro"),
            report_profile=ReportProfile.DEEP,
            max_rounds=2,
        )

    If ``llm`` or ``search`` are left as ``None`` (the default),
    :class:`ResearchAgent` will auto-create them from environment
    variables or a local ``.env`` file on first use (same behaviour as
    the FastAPI server).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # -- Providers (None = auto-create from env vars) --
    llm: LLMProvider | None = Field(
        default=None,
        description=(
            "Concrete LLM provider used for classify, plan, summarize, "
            "evaluate, and answer calls. When ``None`` (the default), "
            "``ResearchAgent`` lazy-creates a ``LiteLLM`` provider from "
            "``Settings`` (env vars / .env) on first ``research()`` call. "
            "Pass an explicit instance (``LiteLLM``, ``AnthropicLLM``, "
            "``BedrockLLM``, ``AzureOpenAILLM``, or any ``LLMProvider`` "
            "subclass) when you want full Baukasten control over auth, "
            "deployment names, and request shaping."
        ),
    )
    """Concrete LLM provider used for classify, plan, summarize, evaluate, and answer calls. When ``None`` (the default), ``ResearchAgent`` lazy-creates a ``LiteLLM`` provider from ``Settings`` (env vars / .env) on first ``research()`` call. Pass an explicit instance (``LiteLLM``, ``AnthropicLLM``, ``BedrockLLM``, ``AzureOpenAILLM``, or any ``LLMProvider`` subclass) when you want full Baukasten control over auth, deployment names, and request shaping."""
    search: SearchProvider | None = Field(
        default=None,
        description=(
            "Concrete search provider used by the search node. When "
            "``None`` (the default), ``ResearchAgent`` lazy-creates a "
            "``PerplexitySearch`` provider from ``Settings``. Pass an "
            "explicit instance (``PerplexitySearch``, ``BraveSearch``, "
            "``AzureFoundryBingSearch``, ``AzureFoundryWebSearch``, "
            "``AzureOpenAIWebSearch``, or any ``SearchProvider`` "
            "subclass) to use a different backend or auth path."
        ),
    )
    """Concrete search provider used by the search node. When ``None`` (the default), ``ResearchAgent`` lazy-creates a ``PerplexitySearch`` provider from ``Settings``. Pass an explicit instance (``PerplexitySearch``, ``BraveSearch``, ``AzureFoundryBingSearch``, ``AzureFoundryWebSearch``, ``AzureOpenAIWebSearch``, or any ``SearchProvider`` subclass) to use a different backend or auth path."""

    # -- Strategies (None = use defaults) --
    source_tiering: SourceTieringStrategy | None = Field(
        default=None,
        description=(
            "Strategy that maps URLs onto quality tiers (``primary``, "
            "``mainstream``, ``stakeholder``, ``unknown``, ``low``) and "
            "computes a per-batch quality score. ``None`` selects "
            "``DefaultSourceTiering`` (uses ``inqtrix.domains`` tables). "
            "Override to inject organisation-specific trust lists or "
            "external taxonomy services."
        ),
    )
    """Strategy that maps URLs onto quality tiers (``primary``, ``mainstream``, ``stakeholder``, ``unknown``, ``low``) and computes a per-batch quality score. ``None`` selects ``DefaultSourceTiering`` (uses ``inqtrix.domains`` tables). Override to inject organisation-specific trust lists or external taxonomy services."""
    claim_extraction: ClaimExtractionStrategy | None = Field(
        default=None,
        description=(
            "Strategy that extracts structured claims from each search "
            "result during the search node. ``None`` selects "
            "``LLMClaimExtractor`` (uses the configured LLM provider). "
            "Override for non-LLM extraction (regex/pipeline) or to "
            "force a different model than the summarize role."
        ),
    )
    """Strategy that extracts structured claims from each search result during the search node. ``None`` selects ``LLMClaimExtractor`` (uses the configured LLM provider). Override for non-LLM extraction (regex/pipeline) or to force a different model than the summarize role."""
    claim_consolidation: ClaimConsolidationStrategy | None = Field(
        default=None,
        description=(
            "Strategy that deduplicates extracted claims, computes "
            "support/contradict counts and assigns ``verified`` / "
            "``contested`` / ``unverified`` status. ``None`` selects "
            "``DefaultClaimConsolidator``. Override to plug in a custom "
            "verification pipeline or external knowledge base."
        ),
    )
    """Strategy that deduplicates extracted claims, computes support/contradict counts and assigns ``verified`` / ``contested`` / ``unverified`` status. ``None`` selects ``DefaultClaimConsolidator``. Override to plug in a custom verification pipeline or external knowledge base."""
    context_pruning: ContextPruningStrategy | None = Field(
        default=None,
        description=(
            "Strategy that prunes accumulated context blocks between "
            "rounds to stay within ``max_context``. ``None`` selects "
            "``RelevanceBasedPruning`` (TF-style score with newest-"
            "evidence protection). Override to change the prune metric "
            "or protection rules."
        ),
    )
    """Strategy that prunes accumulated context blocks between rounds to stay within ``max_context``. ``None`` selects ``RelevanceBasedPruning`` (TF-style score with newest-evidence protection). Override to change the prune metric or protection rules."""
    risk_scoring: RiskScoringStrategy | None = Field(
        default=None,
        description=(
            "Strategy that scores question risk (0-10), derives required "
            "aspects, and may inject quality-site queries. ``None`` "
            "selects ``KeywordRiskScorer``. Override to use external "
            "classifiers or domain-specific risk signals."
        ),
    )
    """Strategy that scores question risk (0-10), derives required aspects, and may inject quality-site queries. ``None`` selects ``KeywordRiskScorer``. Override to use external classifiers or domain-specific risk signals."""
    stop_criteria: StopCriteriaStrategy | None = Field(
        default=None,
        description=(
            "Strategy that runs the 9-signal stop cascade in the "
            "evaluate node (confidence, plateau, utility, falsification, "
            "stagnation, etc.). ``None`` selects ``MultiSignalStopCriteria``. "
            "Override to replace the stop heuristic without touching the "
            "graph wiring."
        ),
    )
    """Strategy that runs the 9-signal stop cascade in the evaluate node (confidence, plateau, utility, falsification, stagnation, etc.). ``None`` selects ``MultiSignalStopCriteria``. Override to replace the stop heuristic without touching the graph wiring."""

    # -- Behaviour --
    report_profile: ReportProfile = Field(
        default=ReportProfile.COMPACT,
        description=(
            "Controls the report style and research-depth preset. "
            "Use ``ReportProfile.COMPACT`` for the current concise "
            "answer style with lower latency. Use ``ReportProfile.DEEP`` "
            "to keep more evidence in the pipeline and produce a denser, "
            "review-style report with broader citation coverage and "
            "higher runtime/token cost. Explicit settings such as "
            "``max_context`` or ``max_rounds`` still override the preset."
        ),
    )
    """Controls the report style and research-depth preset. Use ``ReportProfile.COMPACT`` for the current concise answer style with lower latency. Use ``ReportProfile.DEEP`` to keep more evidence in the pipeline and produce a denser, review-style report with broader citation coverage and higher runtime/token cost. Explicit settings such as ``max_context`` or ``max_rounds`` still override the preset."""
    max_rounds: int = Field(
        default=4,
        description=(
            "Hard upper bound for the research loop. The loop never runs "
            "more search rounds than this, regardless of confidence / "
            "plateau / utility signals. Increase to 5-6 for DEEP-style "
            "coverage on complex topics; lower to 2 to constrain cost."
        ),
    )
    """Hard upper bound for the research loop. The loop never runs more search rounds than this, regardless of confidence / plateau / utility signals. Increase to 5-6 for DEEP-style coverage on complex topics; lower to 2 to constrain cost."""
    min_rounds: int = Field(
        default=1,
        description=(
            "Lower bound for the research loop. Default ``1`` preserves "
            "existing behaviour (an early stop after Round 0 is allowed). "
            "Raise to ``2+`` when the evaluator model tends to over-"
            "confidently signal ``done`` before the STORM diversification "
            "in Round 1+ has had a chance to broaden the source pool. "
            "Typical effect of ``min_rounds=2``: at least one additional "
            "search round runs even if ``confidence_stop`` was already "
            "reached in Round 0. Clamped to ``max_rounds`` at request "
            "time so a misconfiguration never extends the loop beyond "
            "the user-specified hard cap."
        ),
    )
    """Lower bound for the research loop. Default ``1`` preserves existing behaviour (an early stop after Round 0 is allowed). Raise to ``2+`` when the evaluator model tends to over-confidently signal ``done`` before the STORM diversification in Round 1+ has had a chance to broaden the source pool. Typical effect of ``min_rounds=2``: at least one additional search round runs even if ``confidence_stop`` was already reached in Round 0. Clamped to ``max_rounds`` at request time so a misconfiguration never extends the loop beyond the user-specified hard cap."""
    confidence_stop: int = Field(
        default=8,
        description=(
            "Minimum confidence (1-10) at which the loop is allowed to "
            "stop early. The evaluator model assigns the value; once it "
            "reaches this threshold, the stop cascade may emit ``done``. "
            "Default ``8`` matches the COMPACT profile; DEEP raises it "
            "to ``9`` for stricter evidence demands. Lower to ``6-7`` "
            "when latency matters more than evidence breadth."
        ),
    )
    """Minimum confidence (1-10) at which the loop is allowed to stop early. The evaluator model assigns the value; once it reaches this threshold, the stop cascade may emit ``done``. Default ``8`` matches the COMPACT profile; DEEP raises it to ``9`` for stricter evidence demands. Lower to ``6-7`` when latency matters more than evidence breadth."""
    max_context: int = Field(
        default=12,
        description=(
            "Maximum number of context blocks (per-source summaries + "
            "claim digests) retained between rounds. The pruning "
            "strategy enforces this cap after each search round. Higher "
            "values keep more historical evidence but raise prompt cost; "
            "DEEP profile uses ``24``."
        ),
    )
    """Maximum number of context blocks (per-source summaries + claim digests) retained between rounds. The pruning strategy enforces this cap after each search round. Higher values keep more historical evidence but raise prompt cost; DEEP profile uses ``24``."""
    first_round_queries: int = Field(
        default=6,
        description=(
            "Number of broad search queries the plan node generates in "
            "Round 0. Subsequent rounds typically generate 2-3 targeted "
            "gap-filling queries. This is the only round that explores "
            "the question breadth, so setting too low (< 4) starves "
            "later rounds of source diversity. DEEP profile uses ``10``."
        ),
    )
    """Number of broad search queries the plan node generates in Round 0. Subsequent rounds typically generate 2-3 targeted gap-filling queries. This is the only round that explores the question breadth, so setting too low (< 4) starves later rounds of source diversity. DEEP profile uses ``10``."""
    answer_prompt_citations_max: int = Field(
        default=60,
        description=(
            "Hard upper bound on the number of citations the final "
            "answer prompt may reference. Caps prompt size for large-"
            "context models; the answer composer truncates the citation "
            "block to fit. DEEP profile raises this to ``500`` because "
            "it relies on a separate character budget instead."
        ),
    )
    """Hard upper bound on the number of citations the final answer prompt may reference. Caps prompt size for large-context models; the answer composer truncates the citation block to fit. DEEP profile raises this to ``500`` because it relies on a separate character budget instead."""
    max_total_seconds: int = Field(
        default=300,
        description=(
            "Wall-clock deadline for the entire research run, in "
            "seconds. The graph honours this as a soft deadline checked "
            "at node boundaries; in-flight provider calls may run "
            "slightly past it before the next check. Default ``300`` "
            "matches the COMPACT profile; DEEP uses ``540``. Set to a "
            "higher value for slow models or unreliable upstream "
            "search APIs."
        ),
    )
    """Wall-clock deadline for the entire research run, in seconds. The graph honours this as a soft deadline checked at node boundaries; in-flight provider calls may run slightly past it before the next check. Default ``300`` matches the COMPACT profile; DEEP uses ``540``. Set to a higher value for slow models or unreliable upstream search APIs."""
    max_question_length: int = Field(
        default=10_000,
        description=(
            "Maximum input question length in characters. Inputs above "
            "this are rejected before the agent starts to protect "
            "against prompt-flooding accidents. Lower for tighter "
            "input validation in public-facing deployments."
        ),
    )
    """Maximum input question length in characters. Inputs above this are rejected before the agent starts to protect against prompt-flooding accidents. Lower for tighter input validation in public-facing deployments."""

    # -- Timeouts --
    reasoning_timeout: int = Field(
        default=120,
        description=(
            "Per-call timeout (seconds) for reasoning LLM calls "
            "(classify, plan, evaluate, answer). The provider raises "
            "``AgentTimeout`` if a single call exceeds this. Increase "
            "for slow extended-thinking deployments; decrease to fail "
            "fast against unhealthy upstreams."
        ),
    )
    """Per-call timeout (seconds) for reasoning LLM calls (classify, plan, evaluate, answer). The provider raises ``AgentTimeout`` if a single call exceeds this. Increase for slow extended-thinking deployments; decrease to fail fast against unhealthy upstreams."""
    search_timeout: int = Field(
        default=60,
        description=(
            "Per-call timeout (seconds) for search-provider calls. Set "
            "below ``max_total_seconds / first_round_queries`` so a "
            "single slow query cannot consume the entire deadline."
        ),
    )
    """Per-call timeout (seconds) for search-provider calls. Set below ``max_total_seconds / first_round_queries`` so a single slow query cannot consume the entire deadline."""
    summarize_timeout: int = Field(
        default=60,
        description=(
            "Per-call timeout (seconds) for parallel summarize / claim-"
            "extraction LLM calls. Should be tight (60s default) "
            "because many calls run in parallel; a stuck single call "
            "also blocks the round."
        ),
    )
    """Per-call timeout (seconds) for parallel summarize / claim-extraction LLM calls. Should be tight (60s default) because many calls run in parallel; a stuck single call also blocks the round."""

    # -- Risk --
    high_risk_score_threshold: int = Field(
        default=4,
        description=(
            "Risk-score threshold (0-10) above which classify and "
            "evaluate escalate to the reasoning model (instead of the "
            "cheaper classify/evaluate role). Lower the threshold to "
            "use the strong model more often; raise it to favour cost. "
            "Only takes effect when "
            "``high_risk_classify_escalate`` / "
            "``high_risk_evaluate_escalate`` are ``True``."
        ),
    )
    """Risk-score threshold (0-10) above which classify and evaluate escalate to the reasoning model (instead of the cheaper classify/evaluate role). Lower the threshold to use the strong model more often; raise it to favour cost. Only takes effect when ``high_risk_classify_escalate`` / ``high_risk_evaluate_escalate`` are ``True``."""
    high_risk_classify_escalate: bool = Field(
        default=True,
        description=(
            "When ``True``, the classify node uses the reasoning model "
            "for inputs scored ``>= high_risk_score_threshold``. Set "
            "``False`` to always use the dedicated classify model "
            "(cost optimisation; higher risk of mis-classification on "
            "borderline inputs)."
        ),
    )
    """When ``True``, the classify node uses the reasoning model for inputs scored ``>= high_risk_score_threshold``. Set ``False`` to always use the dedicated classify model (cost optimisation; higher risk of mis-classification on borderline inputs)."""
    high_risk_evaluate_escalate: bool = Field(
        default=True,
        description=(
            "When ``True``, the evaluate node uses the reasoning model "
            "for inputs scored ``>= high_risk_score_threshold``. Set "
            "``False`` to always use the dedicated evaluate model. "
            "Disabling this is the most common reason the loop stops "
            "too early on contested topics."
        ),
    )
    """When ``True``, the evaluate node uses the reasoning model for inputs scored ``>= high_risk_score_threshold``. Set ``False`` to always use the dedicated evaluate model. Disabling this is the most common reason the loop stops too early on contested topics."""

    enable_de_policy_bias: bool = Field(
        default=True,
        description=(
            "Enables the German health- and social-policy heuristics in "
            "``KeywordRiskScorer`` (quality-site injection, utility-stop "
            "suppression for DE-political topics). Default ``True`` "
            "preserves the original tuning calibrated against German "
            "policy questions. Set ``False`` for general or non-German "
            "deployments to remove DE-specific bias."
        ),
    )
    """Enables the German health- and social-policy heuristics in ``KeywordRiskScorer`` (quality-site injection, utility-stop suppression for DE-political topics). Default ``True`` preserves the original tuning calibrated against German policy questions. Set ``False`` for general or non-German deployments to remove DE-specific bias."""

    # -- Search cache --
    search_cache_maxsize: int = Field(
        default=256,
        description=(
            "Maximum number of search results retained in the in-memory "
            "TTL cache. Cache hits skip the provider call, which "
            "matters for follow-up questions that re-query overlapping "
            "topics. Set to ``0`` to disable the cache for testing."
        ),
    )
    """Maximum number of search results retained in the in-memory TTL cache. Cache hits skip the provider call, which matters for follow-up questions that re-query overlapping topics. Set to ``0`` to disable the cache for testing."""
    search_cache_ttl: int = Field(
        default=3600,
        description=(
            "Time-to-live (seconds) for cached search results. Default "
            "``3600`` (1 hour) balances staleness against re-query "
            "cost. Lower for fast-moving topics (news), raise for "
            "stable reference questions."
        ),
    )
    """Time-to-live (seconds) for cached search results. Default ``3600`` (1 hour) balances staleness against re-query cost. Lower for fast-moving topics (news), raise for stable reference questions."""

    # -- Testing --
    testing_mode: bool = Field(
        default=False,
        description=(
            "When ``True``, exposes the ``/v1/test/run`` endpoint on "
            "the HTTP server (used by ``inqtrix-parity run``). Has no "
            "effect in library mode. Never enable in production: the "
            "endpoint accepts arbitrary research questions without rate "
            "limiting and returns full iteration logs."
        ),
    )
    """When ``True``, exposes the ``/v1/test/run`` endpoint on the HTTP server (used by ``inqtrix-parity run``). Has no effect in library mode. Never enable in production: the endpoint accepts arbitrary research questions without rate limiting and returns full iteration logs."""


# ------------------------------------------------------------------ #
# ResearchAgent — the Baukasten entry point
# ------------------------------------------------------------------ #


class ResearchAgent:
    """Iterative research agent with pluggable providers and strategies.

    The agent compiles a LangGraph state machine
    (classify → plan → search ↔ evaluate → answer) and orchestrates one
    research run per :meth:`research` or :meth:`stream` call. Providers
    and strategies are pulled from the bound :class:`AgentConfig`; any
    fields left as ``None`` are auto-created from environment variables
    on first use and then cached for the lifetime of the instance.

    The instance is **safe to reuse** across many sequential runs (the
    compiled graph and lazy-created providers are cached). It is **not
    thread-safe** for concurrent runs against the same instance —
    create one instance per worker, or wrap calls in a lock if you must
    share. Streaming via :meth:`stream` uses an internal background
    thread to drain progress; that does not make the rest of the API
    concurrent-safe.

    Attributes:
        config: The bound :class:`AgentConfig`. Read-only after
            construction; mutate by building a new agent instead.
        providers: The lazily-created :class:`ProviderContext`. Auto-
            created on first :meth:`research` / :meth:`stream` call;
            access this property to force eager creation.
        strategies: The lazily-created :class:`StrategyContext`. Same
            lifecycle as ``providers``.

    Example:
        >>> from inqtrix import ResearchAgent
        >>> agent = ResearchAgent()
        >>> result = agent.research("Was ist der Stand der GKV-Reform?")
        >>> print(result.answer)
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Bind a configuration to the agent without performing any I/O.

        Provider and strategy auto-creation is deferred until the first
        :meth:`research` / :meth:`stream` call so that constructing the
        agent never touches the network or environment.

        Args:
            config: Optional :class:`AgentConfig`. When ``None`` (the
                default), an empty ``AgentConfig`` is used and every
                provider/strategy is auto-created from environment
                variables on first call. Pass an explicit config for
                Baukasten-style provider injection or to override
                behaviour fields (timeouts, loop bounds, report
                profile).

        Example:
            >>> from inqtrix import AgentConfig, ResearchAgent
            >>> agent = ResearchAgent(AgentConfig(max_rounds=2))
        """
        self._config = config or AgentConfig()
        self._providers: ProviderContext | None = None
        self._strategies: StrategyContext | None = None
        self._settings: AgentSettings | None = None

    # -- Public API ------------------------------------------------ #

    def research(
        self,
        question: str,
        *,
        history: str = "",
        prev_session: dict[str, Any] | None = None,
    ) -> ResearchResult:
        """Run the full iterative research loop and return a typed result.

        Blocks until the loop terminates (either via the stop cascade
        or the wall-clock deadline). All progress events are discarded;
        use :meth:`stream` if you need live updates.

        Args:
            question: The user question. Must be non-empty and below
                ``AgentConfig.max_question_length`` characters,
                otherwise the agent rejects the input early.
            history: Optional pre-formatted conversation history string
                concatenated into the classify prompt. Use this to give
                follow-up turns context without re-running prior
                research. Empty string (default) is treated as a fresh
                turn.
            prev_session: Optional session snapshot dict produced by an
                earlier run (typically by the HTTP server's session
                store). When provided, selected state fields (claim
                ledger, context blocks) are seeded into the new run so
                follow-up questions build on prior evidence. ``None``
                (default) starts from a clean slate.

        Returns:
            A populated :class:`~inqtrix.result.ResearchResult` with
            answer, metrics, top sources, and top claims.
            ``metrics.elapsed_seconds`` is set from a monotonic clock
            measured around the graph execution.

        Raises:
            inqtrix.exceptions.AgentTimeout: When the run exceeds
                ``AgentConfig.max_total_seconds`` (checked at node
                boundaries).
            inqtrix.exceptions.AgentRateLimited: When a provider
                surfaces a 429 / daily-limit error that the SDK
                retry could not absorb.
            ValueError: For malformed input (empty question, oversize
                question).

        Example:
            >>> agent = ResearchAgent()
            >>> result = agent.research("Was ist Quantencomputing?")
            >>> result.metrics.confidence
            8
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
        """Run research and stream progress + answer chunks incrementally.

        Internally runs the same graph as :meth:`research` in a
        background thread and drains progress events from a queue.
        When the run completes, the final answer is yielded as
        word-aligned chunks for smooth UI rendering.

        Args:
            question: See :meth:`research`.
            history: See :meth:`research`.
            prev_session: See :meth:`research`.
            include_progress: When ``True`` (default), yield progress
                lines (each prefixed with ``"> "`` and terminated with
                ``"\\n"``), then a single ``"---\\n"`` separator, then
                the answer chunks. When ``False``, yield only the
                answer chunks. Use ``False`` when the consumer is
                another program that should only see answer text.

        Yields:
            UTF-8 string chunks. Progress lines are full lines
            (``"> message\\n"``); answer chunks are word-aligned and
            may not end on a newline.

        Raises:
            inqtrix.exceptions.AgentTimeout: Same conditions as
                :meth:`research` — propagated from the background
                thread when ``future.result()`` is awaited.
            inqtrix.exceptions.AgentRateLimited: Same conditions as
                :meth:`research`.

        Example:
            >>> agent = ResearchAgent()
            >>> for chunk in agent.stream("Meine Frage"):
            ...     print(chunk, end="", flush=True)
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
                except Empty:
                    continue
                except Exception as exc:
                    log.warning("Progress-Queue deaktiviert nach unerwartetem Fehler: %s", exc)
                    break

            # Drain remaining progress
            while include_progress and progress_queue is not None and not progress_queue.empty():
                try:
                    kind, msg = progress_queue.get_nowait()
                    if kind == "progress" and msg != "done":
                        yield f"> {msg}\n"
                except Empty:
                    break
                except Exception as exc:
                    log.warning(
                        "Restliche Progress-Meldungen konnten nicht gelesen werden: %s", exc)
                    break

            raw = future.result()
            if include_progress:
                yield "---\n"
            for chunk in iter_word_chunks(raw.get("answer", "")):
                yield chunk

    # -- Properties ------------------------------------------------ #

    @property
    def config(self) -> AgentConfig:
        """Return the immutable :class:`AgentConfig` bound to this agent.

        Returns:
            The exact instance passed to the constructor (or an empty
            ``AgentConfig`` if none was supplied). Mutating this object
            after construction has no effect on already-cached
            providers / strategies; build a new ``ResearchAgent``
            instead.
        """
        return self._config

    @property
    def providers(self) -> ProviderContext:
        """Return the active providers, creating them if necessary.

        Touching this property forces lazy provider auto-creation,
        which may read environment variables and instantiate SDK
        clients. Use this for early-validation of the configuration
        before the first :meth:`research` call.

        Returns:
            The cached :class:`~inqtrix.providers.ProviderContext`. The
            same instance is returned on subsequent calls.
        """
        self._ensure_initialised()
        assert self._providers is not None
        return self._providers

    @property
    def strategies(self) -> StrategyContext:
        """Return the active strategies, creating them if necessary.

        Same eager-init behaviour as :attr:`providers`.

        Returns:
            The cached :class:`~inqtrix.strategies.StrategyContext`.
            The same instance is returned on subsequent calls.
        """
        self._ensure_initialised()
        assert self._strategies is not None
        return self._strategies

    # -- Internals ------------------------------------------------- #

    def _ensure_initialised(
        self,
    ) -> tuple[ProviderContext, StrategyContext, AgentSettings]:
        """Lazily create providers, strategies, and settings from config."""
        if self._providers is not None:
            assert self._strategies is not None
            assert self._settings is not None
            return self._providers, self._strategies, self._settings

        cfg = self._config
        settings = self._build_settings(cfg)
        self._settings = settings

        # -- Providers --
        llm = cfg.llm
        search = cfg.search
        if llm is None or search is None:
            from inqtrix.providers import create_providers
            from inqtrix.settings import Settings

            env_settings = Settings()
            full_settings = Settings(
                models=env_settings.models,
                server=env_settings.server,
                agent=settings,
            )
            auto = create_providers(full_settings)
            llm = llm or auto.llm
            search = search or auto.search

        # Attach model metadata for providers that don't expose it
        # (e.g. AnthropicLLM, custom LLMProvider implementations).
        if llm is not None and not hasattr(llm, "models"):
            from inqtrix.providers.base import ConfiguredLLMProvider

            llm = ConfiguredLLMProvider(
                llm,
                ModelSettings(
                    reasoning_model="",
                    classify_model="",
                    summarize_model="",
                    evaluate_model="",
                ),
            )

        self._providers = ProviderContext(llm=llm, search=search)

        # -- Strategies --
        from inqtrix.strategies import create_default_strategies, resolve_summarize_model

        defaults = create_default_strategies(
            settings,
            llm=llm,
            summarize_model=resolve_summarize_model(llm),
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
    def _build_settings(cfg: AgentConfig) -> AgentSettings:
        """Build an AgentSettings instance from the flat AgentConfig."""
        env_defaults = AgentSettings()
        data = env_defaults.model_dump()
        explicit_fields = set(env_defaults.model_fields_set) | set(cfg.model_fields_set)
        for field_name in (
            "report_profile",
            "max_rounds",
            "min_rounds",
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
            "enable_de_policy_bias",
            "search_cache_maxsize",
            "search_cache_ttl",
            "testing_mode",
        ):
            if field_name in cfg.model_fields_set:
                data[field_name] = getattr(cfg, field_name)
        return AgentSettings(**data).with_report_profile_defaults(
            explicit_fields=explicit_fields,
        )
