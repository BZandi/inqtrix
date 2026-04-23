"""Pydantic Settings stack for type-safe environment-variable configuration.

This module bridges the user-facing ``.env`` / process-env surface to typed
Python objects consumed by :class:`~inqtrix.agent.ResearchAgent` and the HTTP
server. Three concerns are split into dedicated classes (``ModelSettings``,
``AgentSettings``, ``ServerSettings``) and re-aggregated by :class:`Settings`
so providers, agents, and routes can subscribe only to the slice they need.

Precedence (highest wins):

1. Programmatic ``AgentConfig`` overrides (in library mode)
2. Real process environment variables
3. ``.env`` file in the current working directory
4. Built-in defaults defined here
"""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from inqtrix.report_profiles import (
    ReportProfile,
    ReportProfileTuning,
    settings_overrides_for_report_profile,
    tuning_for_report_profile,
)


_SETTINGS_MODEL_CONFIG = {
    "env_prefix": "",
    "extra": "ignore",
    "populate_by_name": True,
    "env_file": ".env",
    "env_file_encoding": "utf-8",
}


class ModelSettings(BaseSettings):
    """Per-role model identifiers loaded from environment variables.

    Inqtrix dispatches each agent role (reasoning, classify, summarize,
    evaluate, search) to a named model. This class collects the env-var
    bindings for those names and provides ``effective_*_model`` properties
    that fall back to ``reasoning_model`` whenever a role-specific name is
    empty. Concrete provider classes such as :class:`~inqtrix.providers.LiteLLM`
    or :class:`~inqtrix.providers.AzureOpenAILLM` translate these names to
    their backend (OpenAI model id, Azure deployment name, Bedrock model id).

    Use this when running in env-driven mode. In Baukasten / explicit-
    provider mode, provider constructors take model names directly and
    ``ModelSettings`` is bypassed.
    """

    model_config = _SETTINGS_MODEL_CONFIG

    reasoning_model: str = Field(
        "claude-opus-4.6-agent",
        alias="REASONING_MODEL",
        description=(
            "Identifier of the primary reasoning model used for plan and "
            "answer synthesis (and as the fallback for unset role-"
            "specific models). Format depends on the provider: an "
            "OpenAI/LiteLLM model id (``gpt-4o``), an Anthropic model "
            "name (``claude-opus-4-20250514``), or an Azure deployment "
            "name when used together with ``AzureOpenAILLM``."
        ),
    )
    """Identifier of the primary reasoning model used for plan and answer synthesis (and as the fallback for unset role-specific models). Format depends on the provider: an OpenAI/LiteLLM model id (``gpt-4o``), an Anthropic model name (``claude-opus-4-20250514``), or an Azure deployment name when used together with ``AzureOpenAILLM``."""
    search_model: str = Field(
        "perplexity-sonar-pro-agent",
        alias="SEARCH_MODEL",
        description=(
            "Identifier of the search model called by ``PerplexitySearch`` "
            "or any LiteLLM-routed search adapter. Ignored when the "
            "configured search provider is non-LLM (e.g. ``BraveSearch`` "
            "or ``AzureFoundryWebSearch`` that uses an agent reference)."
        ),
    )
    """Identifier of the search model called by ``PerplexitySearch`` or any LiteLLM-routed search adapter. Ignored when the configured search provider is non-LLM (e.g. ``BraveSearch`` or ``AzureFoundryWebSearch`` that uses an agent reference)."""
    classify_model: str = Field(
        "",
        alias="CLASSIFY_MODEL",
        description=(
            "Optional cheaper model used by the classify node. Empty "
            "string falls back to ``reasoning_model``. Setting a smaller "
            "model here is the standard cost optimisation: classify runs "
            "on every request and rarely needs frontier reasoning."
        ),
    )
    """Optional cheaper model used by the classify node. Empty string falls back to ``reasoning_model``. Setting a smaller model here is the standard cost optimisation: classify runs on every request and rarely needs frontier reasoning."""
    summarize_model: str = Field(
        "",
        alias="SUMMARIZE_MODEL",
        description=(
            "Optional cheaper model used by the parallel summarize / "
            "claim-extraction calls. Empty string falls back to "
            "``reasoning_model``. This is the highest-volume role in a "
            "research run (one call per search hit), so a smaller model "
            "here typically delivers the largest cost saving."
        ),
    )
    """Optional cheaper model used by the parallel summarize / claim-extraction calls. Empty string falls back to ``reasoning_model``. This is the highest-volume role in a research run (one call per search hit), so a smaller model here typically delivers the largest cost saving."""
    evaluate_model: str = Field(
        "",
        alias="EVALUATE_MODEL",
        description=(
            "Optional cheaper model used by the evaluate node. Empty "
            "string falls back to ``reasoning_model``. Be careful: an "
            "under-powered evaluate model is the most common cause of "
            "premature stops on contested topics — see "
            "``high_risk_evaluate_escalate``."
        ),
    )
    """Optional cheaper model used by the evaluate node. Empty string falls back to ``reasoning_model``. Be careful: an under-powered evaluate model is the most common cause of premature stops on contested topics — see ``high_risk_evaluate_escalate``."""

    @property
    def effective_classify_model(self) -> str:
        """Resolve the classify model, falling back to the reasoning model.

        Returns:
            The configured ``classify_model`` if non-empty, otherwise
            ``reasoning_model``. Used by the classify node to pick the
            actual model id passed to the provider.
        """
        return self.classify_model or self.reasoning_model

    @property
    def effective_summarize_model(self) -> str:
        """Resolve the summarize model, falling back to the reasoning model.

        Returns:
            The configured ``summarize_model`` if non-empty, otherwise
            ``reasoning_model``. Used by the parallel summarize and
            claim-extraction paths.
        """
        return self.summarize_model or self.reasoning_model

    @property
    def effective_evaluate_model(self) -> str:
        """Resolve the evaluate model, falling back to the reasoning model.

        Returns:
            The configured ``evaluate_model`` if non-empty, otherwise
            ``reasoning_model``. Used by the evaluate node and by
            ``high_risk_evaluate_escalate`` decisions.
        """
        return self.evaluate_model or self.reasoning_model


class AgentSettings(BaseSettings):
    """Tunable parameters that shape a single research run.

    These values control the loop bounds, timeouts, risk-scoring
    behaviour, and search-cache sizing. They are sourced from environment
    variables (matching the upper-cased aliases) so that the same
    container image can be re-tuned per environment without code changes.

    The ``report_profile`` field doubles as a preset trigger: assigning
    ``ReportProfile.DEEP`` (via env or constructor) auto-applies the
    DEEP-specific overrides defined in
    :mod:`inqtrix.report_profiles`, but only for fields the user has
    not explicitly set (see :meth:`with_report_profile_defaults`).
    """

    model_config = _SETTINGS_MODEL_CONFIG

    report_profile: ReportProfile = Field(
        ReportProfile.COMPACT,
        alias="REPORT_PROFILE",
        description=(
            "Selects the answer style and depth preset (``compact`` or "
            "``deep``). Assigning ``deep`` triggers a bundle of profile-"
            "specific overrides (``max_rounds=5``, ``confidence_stop=9``, "
            "``max_context=24``, ``first_round_queries=10``, "
            "``answer_prompt_citations_max=500``, "
            "``max_total_seconds=540``) for any field the user has not "
            "set explicitly."
        ),
    )
    """Selects the answer style and depth preset (``compact`` or ``deep``). Assigning ``deep`` triggers a bundle of profile-specific overrides (``max_rounds=5``, ``confidence_stop=9``, ``max_context=24``, ``first_round_queries=10``, ``answer_prompt_citations_max=500``, ``max_total_seconds=540``) for any field the user has not set explicitly."""
    max_rounds: int = Field(
        4,
        alias="MAX_ROUNDS",
        description=(
            "Hard upper bound for the research loop. Mirrors "
            "``AgentConfig.max_rounds`` — see that field for tuning "
            "guidance. Default ``4`` matches COMPACT; DEEP raises "
            "to ``5``."
        ),
    )
    """Hard upper bound for the research loop. Mirrors ``AgentConfig.max_rounds`` — see that field for tuning guidance. Default ``4`` matches COMPACT; DEEP raises to ``5``."""
    min_rounds: int = Field(
        1,
        alias="MIN_ROUNDS",
        description=(
            "Lower bound for the research loop. Default ``1`` preserves "
            "the existing behaviour (an early ``confidence_stop`` / "
            "plateau / utility stop after Round 0 is allowed). Raise "
            "this when the model used as evaluator tends to over-"
            "confidently signal ``done`` before the STORM "
            "diversification in Round 1+ has had a chance to broaden "
            "the source pool. Typical effect of ``min_rounds=2``: at "
            "least one additional search round runs even if the "
            "confidence target was already reached in Round 0. Clamped "
            "to ``max_rounds`` at request time so configuration "
            "mistakes never extend the loop beyond the user-specified "
            "hard cap."
        ),
    )
    """Lower bound for the research loop. Default ``1`` preserves the existing behaviour (an early ``confidence_stop`` / plateau / utility stop after Round 0 is allowed). Raise this when the model used as evaluator tends to over-confidently signal ``done`` before the STORM diversification in Round 1+ has had a chance to broaden the source pool. Typical effect of ``min_rounds=2``: at least one additional search round runs even if the confidence target was already reached in Round 0. Clamped to ``max_rounds`` at request time so configuration mistakes never extend the loop beyond the user-specified hard cap."""
    confidence_stop: int = Field(
        8,
        alias="CONFIDENCE_STOP",
        description=(
            "Minimum evaluator confidence (1-10) at which the stop "
            "cascade may emit ``done``. Default ``8`` for COMPACT, "
            "``9`` for DEEP. Lower for latency-sensitive deployments."
        ),
    )
    """Minimum evaluator confidence (1-10) at which the stop cascade may emit ``done``. Default ``8`` for COMPACT, ``9`` for DEEP. Lower for latency-sensitive deployments."""
    max_context: int = Field(
        12,
        alias="MAX_CONTEXT",
        description=(
            "Maximum number of context blocks retained between rounds "
            "by the pruning strategy. Default ``12`` for COMPACT, "
            "``24`` for DEEP."
        ),
    )
    """Maximum number of context blocks retained between rounds by the pruning strategy. Default ``12`` for COMPACT, ``24`` for DEEP."""
    first_round_queries: int = Field(
        6,
        alias="FIRST_ROUND_QUERIES",
        description=(
            "Number of broad queries generated in Round 0 by the plan "
            "node. Default ``6`` for COMPACT, ``10`` for DEEP. Setting "
            "below ``4`` typically starves later rounds of source "
            "diversity."
        ),
    )
    """Number of broad queries generated in Round 0 by the plan node. Default ``6`` for COMPACT, ``10`` for DEEP. Setting below ``4`` typically starves later rounds of source diversity."""
    answer_prompt_citations_max: int = Field(
        60,
        alias="ANSWER_PROMPT_CITATIONS_MAX",
        description=(
            "Hard upper bound on citations passed to the final answer "
            "prompt. Default ``60`` for COMPACT, ``500`` for DEEP "
            "(combined with the DEEP character budget)."
        ),
    )
    """Hard upper bound on citations passed to the final answer prompt. Default ``60`` for COMPACT, ``500`` for DEEP (combined with the DEEP character budget)."""

    reasoning_timeout: int = Field(
        120,
        alias="REASONING_TIMEOUT",
        description=(
            "Per-call timeout (seconds) for reasoning LLM calls. "
            "Increase for slow extended-thinking deployments; decrease "
            "to fail fast against unhealthy upstreams."
        ),
    )
    """Per-call timeout (seconds) for reasoning LLM calls. Increase for slow extended-thinking deployments; decrease to fail fast against unhealthy upstreams."""
    search_timeout: int = Field(
        60,
        alias="SEARCH_TIMEOUT",
        description=(
            "Per-call timeout (seconds) for search-provider calls. "
            "Should sit below "
            "``max_total_seconds / first_round_queries``."
        ),
    )
    """Per-call timeout (seconds) for search-provider calls. Should sit below ``max_total_seconds / first_round_queries``."""
    summarize_timeout: int = Field(
        60,
        alias="SUMMARIZE_TIMEOUT",
        description=(
            "Per-call timeout (seconds) for parallel summarize / claim-"
            "extraction LLM calls. Tight by design because many calls "
            "run in parallel."
        ),
    )
    """Per-call timeout (seconds) for parallel summarize / claim-extraction LLM calls. Tight by design because many calls run in parallel."""
    max_total_seconds: int = Field(
        300,
        alias="MAX_TOTAL_SECONDS",
        description=(
            "Wall-clock deadline (seconds) for the entire research run. "
            "Default ``300`` for COMPACT, ``540`` for DEEP. Checked at "
            "node boundaries; in-flight provider calls may run slightly "
            "past this before the next check."
        ),
    )
    """Wall-clock deadline (seconds) for the entire research run. Default ``300`` for COMPACT, ``540`` for DEEP. Checked at node boundaries; in-flight provider calls may run slightly past this before the next check."""
    max_question_length: int = Field(
        10_000,
        alias="MAX_QUESTION_LENGTH",
        description=(
            "Maximum input question length in characters. Inputs above "
            "this are rejected before the agent starts to protect "
            "against prompt-flooding accidents. Lower for tighter "
            "input validation in public-facing deployments."
        ),
    )
    """Maximum input question length in characters. Inputs above this are rejected before the agent starts to protect against prompt-flooding accidents. Lower for tighter input validation in public-facing deployments."""

    high_risk_score_threshold: int = Field(
        4,
        alias="HIGH_RISK_SCORE_THRESHOLD",
        description=(
            "Risk-score threshold (0-10) at which classify and "
            "evaluate may escalate to the reasoning model. Lower the "
            "threshold to use the strong model more often; raise it "
            "to favour cost. Only takes effect together with the two "
            "``high_risk_*_escalate`` flags."
        ),
    )
    """Risk-score threshold (0-10) at which classify and evaluate may escalate to the reasoning model. Lower the threshold to use the strong model more often; raise it to favour cost. Only takes effect together with the two ``high_risk_*_escalate`` flags."""
    high_risk_classify_escalate: bool = Field(
        True,
        alias="HIGH_RISK_CLASSIFY_ESCALATE",
        description=(
            "When ``True``, classify uses the reasoning model for "
            "inputs scored ``>= high_risk_score_threshold``. Set "
            "``False`` to always use the cheaper classify model."
        ),
    )
    """When ``True``, classify uses the reasoning model for inputs scored ``>= high_risk_score_threshold``. Set ``False`` to always use the cheaper classify model."""
    high_risk_evaluate_escalate: bool = Field(
        True,
        alias="HIGH_RISK_EVALUATE_ESCALATE",
        description=(
            "When ``True``, evaluate uses the reasoning model for "
            "inputs scored ``>= high_risk_score_threshold``. Disabling "
            "this is the most common cause of premature stops on "
            "contested topics."
        ),
    )
    """When ``True``, evaluate uses the reasoning model for inputs scored ``>= high_risk_score_threshold``. Disabling this is the most common cause of premature stops on contested topics."""

    enable_de_policy_bias: bool = Field(
        True,
        alias="ENABLE_DE_POLICY_BIAS",
        description=(
            "Enables the German health- and social-policy heuristics in "
            "``KeywordRiskScorer`` (quality-site injection, utility-"
            "stop suppression for DE-political topics). Default "
            "``True`` preserves the original tuning calibrated against "
            "German policy questions. Set ``False`` for general or "
            "non-German deployments to remove DE-specific bias."
        ),
    )
    """Enables the German health- and social-policy heuristics in ``KeywordRiskScorer`` (quality-site injection, utility-stop suppression for DE-political topics). Default ``True`` preserves the original tuning calibrated against German policy questions. Set ``False`` for general or non-German deployments to remove DE-specific bias."""

    search_cache_maxsize: int = Field(
        256,
        alias="SEARCH_CACHE_MAXSIZE",
        description=(
            "Maximum number of search results retained in the in-memory "
            "TTL cache. Set to ``0`` to disable the cache for "
            "deterministic test runs."
        ),
    )
    """Maximum number of search results retained in the in-memory TTL cache. Set to ``0`` to disable the cache for deterministic test runs."""
    search_cache_ttl: int = Field(
        3600,
        alias="SEARCH_CACHE_TTL",
        description=(
            "Time-to-live (seconds) for cached search results. Default "
            "``3600`` (1 hour). Lower for fast-moving topics, raise "
            "for stable reference questions."
        ),
    )
    """Time-to-live (seconds) for cached search results. Default ``3600`` (1 hour). Lower for fast-moving topics, raise for stable reference questions."""

    testing_mode: bool = Field(
        False,
        alias="TESTING_MODE",
        description=(
            "When ``True``, the HTTP server exposes the "
            "``/v1/test/run`` endpoint used by ``inqtrix-parity run``. "
            "Never enable in production: the endpoint accepts arbitrary "
            "research questions without rate limiting and returns full "
            "iteration logs."
        ),
    )
    """When ``True``, the HTTP server exposes the ``/v1/test/run`` endpoint used by ``inqtrix-parity run``. Never enable in production: the endpoint accepts arbitrary research questions without rate limiting and returns full iteration logs."""

    skip_search: bool = Field(
        False,
        alias="SKIP_SEARCH",
        description=(
            "When ``True``, bypasses the plan/search/evaluate loop and "
            "routes the request directly to the LLM provider with the "
            "question plus conversation history. Used by UI clients to "
            "offer a 'chat without web search' mode. The answer is "
            "returned without citations and ``round`` stays at ``0``."
        ),
    )
    """When ``True``, bypasses the plan/search/evaluate loop and routes the request directly to the LLM provider with the question plus conversation history. Used by UI clients to offer a 'chat without web search' mode. The answer is returned without citations and ``round`` stays at ``0``."""

    @property
    def report_tuning(self) -> ReportProfileTuning:
        """Return the runtime tuning bundle for the active report profile.

        Returns:
            The :class:`~inqtrix.report_profiles.ReportProfileTuning` for
            the current ``report_profile``. The tuning bundle holds
            char-/token-/citation-cap values consumed by the summarize,
            consolidation and answer-composer paths. Cached structurally
            in ``report_profiles.py``; no per-call computation cost.
        """
        return tuning_for_report_profile(self.report_profile)

    def _report_profile_updates(
        self,
        *,
        explicit_fields: set[str] | None = None,
    ) -> dict[str, object]:
        """Compute profile-derived overrides safe to apply.

        Args:
            explicit_fields: Field names the caller has set explicitly
                (typically ``self.model_fields_set``). These are skipped
                so user intent always wins over the profile preset.

        Returns:
            Mapping ``{field_name: new_value}`` with the subset of the
            profile's ``settings_overrides`` that differs from the
            currently held value and is not in ``explicit_fields``.
            Empty dict when the profile has no overrides or no field
            needs updating.
        """
        overrides = settings_overrides_for_report_profile(self.report_profile)
        if not overrides:
            return {}

        explicit = set(explicit_fields or ())
        updates: dict[str, object] = {}
        for field_name, value in overrides.items():
            if field_name in explicit:
                continue
            if getattr(self, field_name) != value:
                updates[field_name] = value
        return updates

    def with_report_profile_defaults(
        self,
        *,
        explicit_fields: set[str] | None = None,
    ) -> "AgentSettings":
        """Return a copy with profile defaults applied.

        Use this when materialising ``AgentSettings`` from a higher-
        level config (e.g. ``AgentConfig``) where the user-specified
        fields must remain authoritative.

        Args:
            explicit_fields: Field names the user has set explicitly;
                these are preserved exactly. Pass the union of fields
                explicitly set on the source object.

        Returns:
            A new ``AgentSettings`` instance with profile-derived
            defaults filled in for non-explicit fields. Returns ``self``
            unchanged when no updates are needed.
        """
        updates = self._report_profile_updates(explicit_fields=explicit_fields)
        if not updates:
            return self
        return self.model_copy(update=updates)

    @model_validator(mode="after")
    def _apply_report_profile(self) -> "AgentSettings":
        """Apply profile defaults at construction time for non-explicit fields.

        Side effects:
            Mutates ``self`` via ``object.__setattr__`` for any field
            present in the profile's ``settings_overrides`` and not in
            ``self.model_fields_set``. Runs once after Pydantic
            validation completes.

        Returns:
            ``self`` (Pydantic ``model_validator(mode="after")``
            contract).
        """
        updates = self._report_profile_updates(explicit_fields=set(self.model_fields_set))
        for field_name, value in updates.items():
            object.__setattr__(self, field_name, value)
        return self


class ServerSettings(BaseSettings):
    """HTTP-server-only configuration loaded from environment variables.

    These fields steer the FastAPI surface launched by
    ``python -m inqtrix``: upstream LLM-gateway connection, concurrency
    cap, and in-memory session lifecycle. Library-mode users (those
    instantiating :class:`~inqtrix.agent.ResearchAgent` directly) can
    ignore ``ServerSettings`` entirely.
    """

    model_config = _SETTINGS_MODEL_CONFIG

    litellm_base_url: str = Field(
        "http://litellm-proxy:4000/v1",
        alias="LITELLM_BASE_URL",
        description=(
            "Base URL of the LiteLLM proxy (or any OpenAI-compatible "
            "gateway) used by the auto-created ``LiteLLM`` provider in "
            "server mode. Must include the ``/v1`` suffix. Default "
            "matches a typical Docker-Compose service name."
        ),
    )
    """Base URL of the LiteLLM proxy (or any OpenAI-compatible gateway) used by the auto-created ``LiteLLM`` provider in server mode. Must include the ``/v1`` suffix. Default matches a typical Docker-Compose service name."""
    litellm_api_key: str = Field(
        "sk-placeholder",
        alias="LITELLM_API_KEY",
        description=(
            "API key forwarded as ``Authorization: Bearer ...`` to the "
            "LiteLLM proxy. The placeholder default is intentionally "
            "obvious so misconfigured deployments fail loudly during "
            "the first upstream call."
        ),
    )
    """API key forwarded as ``Authorization: Bearer ...`` to the LiteLLM proxy. The placeholder default is intentionally obvious so misconfigured deployments fail loudly during the first upstream call."""
    max_concurrent: int = Field(
        3,
        alias="MAX_CONCURRENT",
        description=(
            "Maximum number of concurrently executing research runs in "
            "the HTTP server. Excess requests queue at the FastAPI "
            "layer. Sized for moderate per-run resource use (LLM tokens, "
            "search-API quota); raise carefully when upstream providers "
            "support higher parallelism."
        ),
    )
    """Maximum number of concurrently executing research runs in the HTTP server. Excess requests queue at the FastAPI layer. Sized for moderate per-run resource use (LLM tokens, search-API quota); raise carefully when upstream providers support higher parallelism."""
    max_messages_history: int = Field(
        20,
        alias="MAX_MESSAGES_HISTORY",
        description=(
            "Maximum number of OpenAI-compatible chat messages "
            "extracted from a single ``/v1/chat/completions`` request "
            "for history reconstruction. Older messages are truncated. "
            "Caps prompt cost on long-running conversations."
        ),
    )
    """Maximum number of OpenAI-compatible chat messages extracted from a single ``/v1/chat/completions`` request for history reconstruction. Older messages are truncated. Caps prompt cost on long-running conversations."""
    session_ttl_seconds: int = Field(
        1800,
        alias="SESSION_TTL_SECONDS",
        description=(
            "TTL (seconds) for in-memory follow-up sessions. After this "
            "interval without activity, the session and its preserved "
            "state fields (claim ledger, context blocks) are evicted. "
            "Default ``1800`` (30 min)."
        ),
    )
    """TTL (seconds) for in-memory follow-up sessions. After this interval without activity, the session and its preserved state fields (claim ledger, context blocks) are evicted. Default ``1800`` (30 min)."""
    session_max_count: int = Field(
        20,
        alias="SESSION_MAX_COUNT",
        description=(
            "Maximum number of concurrent sessions kept in memory. When "
            "exceeded, the least-recently-used session is evicted. "
            "Sized for a moderate per-server user count; the in-memory "
            "store is not designed for thousands of concurrent users."
        ),
    )
    """Maximum number of concurrent sessions kept in memory. When exceeded, the least-recently-used session is evicted. Sized for a moderate per-server user count; the in-memory store is not designed for thousands of concurrent users."""
    session_max_context_blocks: int = Field(
        8,
        alias="SESSION_MAX_CONTEXT_BLOCKS",
        description=(
            "Maximum number of context blocks preserved across follow-"
            "up turns within one session. Bounded to prevent unbounded "
            "growth on long conversations."
        ),
    )
    """Maximum number of context blocks preserved across follow-up turns within one session. Bounded to prevent unbounded growth on long conversations."""
    session_max_claim_ledger: int = Field(
        50,
        alias="SESSION_MAX_CLAIM_LEDGER",
        description=(
            "Maximum number of consolidated claim ledger entries "
            "preserved across follow-up turns within one session. "
            "Older entries are dropped when this limit is exceeded."
        ),
    )
    """Maximum number of consolidated claim ledger entries preserved across follow-up turns within one session. Older entries are dropped when this limit is exceeded."""

    tls_keyfile: str = Field(
        "",
        alias="INQTRIX_SERVER_TLS_KEYFILE",
        description=(
            "Path to the PEM-encoded TLS private key file. When set "
            "together with ``tls_certfile``, the example webserver "
            "scripts hand both paths to ``uvicorn.run(...)`` so the "
            "server speaks HTTPS instead of HTTP. Setting only one of "
            "the two raises ``RuntimeError`` at startup (no silent "
            "fallback). Empty string (default) keeps the server on "
            "plain HTTP. TLS is opt-in and intended as a minimum "
            "viable hardening layer for the experimental phase; "
            "production deployments should still terminate TLS at a "
            "dedicated reverse proxy (nginx / Traefik / Caddy) for "
            "richer cipher policies."
        ),
    )
    """Path to the PEM-encoded TLS private key file. When set together with ``tls_certfile``, the example webserver scripts hand both paths to ``uvicorn.run(...)`` so the server speaks HTTPS instead of HTTP. Setting only one of the two raises ``RuntimeError`` at startup (no silent fallback). Empty string (default) keeps the server on plain HTTP. TLS is opt-in and intended as a minimum viable hardening layer for the experimental phase; production deployments should still terminate TLS at a dedicated reverse proxy (nginx / Traefik / Caddy) for richer cipher policies."""
    tls_certfile: str = Field(
        "",
        alias="INQTRIX_SERVER_TLS_CERTFILE",
        description=(
            "Path to the PEM-encoded TLS certificate file. Companion "
            "to ``tls_keyfile`` — both must be set together or both "
            "empty. See ``tls_keyfile`` for the broader rationale."
        ),
    )
    """Path to the PEM-encoded TLS certificate file. Companion to ``tls_keyfile`` — both must be set together or both empty. See ``tls_keyfile`` for the broader rationale."""
    api_key: str = Field(
        "",
        alias="INQTRIX_SERVER_API_KEY",
        description=(
            "Static Bearer API key. When set, the server installs a "
            "FastAPI dependency on ``/v1/chat/completions`` and "
            "``/v1/test/run`` that requires "
            "``Authorization: Bearer <api_key>`` and compares with "
            "``hmac.compare_digest`` for constant-time safety. "
            "``/health`` and ``/v1/models`` deliberately stay "
            "unauthenticated so Kubernetes liveness probes and model "
            "discovery clients keep working without credentials. "
            "Empty string (default) disables the gate, matching the "
            "historical behaviour. Rotation requires a server restart "
            "in this iteration; multi-key support is a follow-up task."
        ),
    )
    """Static Bearer API key. When set, the server installs a FastAPI dependency on ``/v1/chat/completions`` and ``/v1/test/run`` that requires ``Authorization: Bearer <api_key>`` and compares with ``hmac.compare_digest`` for constant-time safety. ``/health`` and ``/v1/models`` deliberately stay unauthenticated so Kubernetes liveness probes and model discovery clients keep working without credentials. Empty string (default) disables the gate, matching the historical behaviour. Rotation requires a server restart in this iteration; multi-key support is a follow-up task."""
    cors_origins: str = Field(
        "",
        alias="INQTRIX_SERVER_CORS_ORIGINS",
        description=(
            "Comma-separated list of allowed CORS origins (e.g. "
            "``\"https://app1.example,https://app2.example\"``). When "
            "non-empty, the server installs ``CORSMiddleware`` with "
            "those origins, ``allow_methods=['GET','POST','OPTIONS']``, "
            "``allow_headers=['Authorization','Content-Type']`` and "
            "``allow_credentials=True``. Wildcard (``\"*\"``) is "
            "accepted but logged with a WARNING because browsers "
            "ignore wildcard origins when credentials are sent — use "
            "explicit origins for any browser-based UI. Empty string "
            "(default) installs no middleware (no CORS headers; same "
            "as before this feature)."
        ),
    )
    """Comma-separated list of allowed CORS origins (e.g. ``"https://app1.example,https://app2.example"``). When non-empty, the server installs ``CORSMiddleware`` with those origins, ``allow_methods=['GET','POST','OPTIONS']``, ``allow_headers=['Authorization','Content-Type']`` and ``allow_credentials=True``. Wildcard (``"*"``) is accepted but logged with a WARNING because browsers ignore wildcard origins when credentials are sent — use explicit origins for any browser-based UI. Empty string (default) installs no middleware (no CORS headers; same as before this feature)."""


class Settings(BaseSettings):
    """Root container that aggregates the three Settings groups.

    This is the convenience entry point when a single object needs to
    expose all configuration at once (HTTP-server bootstrap, parity
    tooling, integration tests). Library-mode users typically work
    directly with :class:`AgentConfig` and never instantiate this
    class explicitly.
    """

    models: ModelSettings = Field(
        default_factory=ModelSettings,
        description=(
            "Per-role model identifiers. Initialised from environment "
            "variables on instantiation. Replace with a custom "
            "instance to inject test fixtures."
        ),
    )
    """Per-role model identifiers. Initialised from environment variables on instantiation. Replace with a custom instance to inject test fixtures."""
    agent: AgentSettings = Field(
        default_factory=AgentSettings,
        description=(
            "Behavioural tuning (loop bounds, timeouts, risk scoring, "
            "search cache). Initialised from environment variables and "
            "auto-applies the configured ``report_profile`` overrides."
        ),
    )
    """Behavioural tuning (loop bounds, timeouts, risk scoring, search cache). Initialised from environment variables and auto-applies the configured ``report_profile`` overrides."""
    server: ServerSettings = Field(
        default_factory=ServerSettings,
        description=(
            "HTTP-server-only settings. Ignored in pure library mode."
        ),
    )
    """HTTP-server-only settings. Ignored in pure library mode."""
