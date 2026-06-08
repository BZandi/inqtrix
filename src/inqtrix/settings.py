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

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from inqtrix.constants import REQUIRED_CONTEXT_WINDOW_TOKENS
from inqtrix.model_routing import resolve_model, validate_model_tier
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

    Inqtrix dispatches each agent role (reasoning, classify, claim extraction,
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
        "",
        alias="SEARCH_MODEL",
        description=(
            "Identifier of the search model called by ``PerplexitySearch`` "
            "or any LiteLLM-routed search adapter. Ignored when the "
            "configured search provider is non-LLM (e.g. "
            "``AzureFoundryWebSearch`` that uses an agent reference)."
        ),
    )
    """Identifier of the search model called by ``PerplexitySearch`` or any LiteLLM-routed search adapter. Ignored when the configured search provider is non-LLM (e.g. ``AzureFoundryWebSearch`` that uses an agent reference)."""
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
    claim_extract_model: str = Field(
        "",
        alias="CLAIM_EXTRACT_MODEL",
        description=(
            "Optional cheaper model used by claim-extraction calls. "
            "Empty string falls back to "
            "``reasoning_model``. This is the highest-volume role in a "
            "research run (one call per search hit), so a smaller model "
            "here typically delivers the largest cost saving."
        ),
    )
    """Optional cheaper model used by claim-extraction calls. Empty string falls back to ``reasoning_model``. This is the highest-volume role in a research run (one call per search hit), so a smaller model here typically delivers the largest cost saving."""
    evaluate_model: str = Field(
        "",
        alias="EVALUATE_MODEL",
        description=(
            "Optional per-node override for the evaluate node. Empty string "
            "falls back to the node's tier model and then to "
            "``reasoning_model``. Be careful: an under-powered evaluate model "
            "is a common cause of premature stops on contested topics."
        ),
    )
    """Optional per-node override for the evaluate node. Empty string falls back to the node's tier model and then to ``reasoning_model``. Be careful: an under-powered evaluate model is a common cause of premature stops on contested topics."""
    plan_model: str = Field(
        "",
        alias="PLAN_MODEL",
        description=(
            "Optional per-node override for the plan node (search-query "
            "generation). Empty string falls back to the node's tier model "
            "and then to ``reasoning_model``."
        ),
    )
    """Optional per-node override for the plan node (search-query generation). Empty string falls back to the node's tier model and then to ``reasoning_model``."""
    answer_model: str = Field(
        "",
        alias="ANSWER_MODEL",
        description=(
            "Optional per-node override for the answer node (final report "
            "synthesis). Empty string falls back to the node's tier model "
            "and then to ``reasoning_model``."
        ),
    )
    """Optional per-node override for the answer node (final report synthesis). Empty string falls back to the node's tier model and then to ``reasoning_model``."""
    direct_chat_model: str = Field(
        "",
        alias="DIRECT_CHAT_MODEL",
        description=(
            "Optional per-node override for the direct-chat node (skip-search "
            "conversational answer). Empty string falls back to the node's "
            "tier model and then to ``reasoning_model``."
        ),
    )
    """Optional per-node override for the direct-chat node (skip-search conversational answer). Empty string falls back to the node's tier model and then to ``reasoning_model``."""

    tier_high_model: str = Field(
        "",
        alias="TIER_HIGH_MODEL",
        description=(
            "Model for the 'high' tier (strongest). Assigned to the answer "
            "node by default. Empty string falls back to ``reasoning_model``."
        ),
    )
    """Model for the 'high' tier (strongest). Assigned to the answer node by default. Empty string falls back to ``reasoning_model``."""
    tier_mid_model: str = Field(
        "",
        alias="TIER_MID_MODEL",
        description=(
            "Model for the 'mid' tier. Assigned to the plan, evaluate, and "
            "direct_chat nodes by default. Empty string falls back to "
            "``reasoning_model``."
        ),
    )
    """Model for the 'mid' tier. Assigned to the plan, evaluate, and direct_chat nodes by default. Empty string falls back to ``reasoning_model``."""
    tier_fast_model: str = Field(
        "",
        alias="TIER_FAST_MODEL",
        description=(
            "Model for the 'fast' tier (cheapest/quickest). Assigned to the "
            "classify and claim_extract nodes by default. Empty string falls "
            "back to ``reasoning_model``."
        ),
    )
    """Model for the 'fast' tier (cheapest/quickest). Assigned to the classify and claim_extract nodes by default. Empty string falls back to ``reasoning_model``."""

    tier_high_effort: str = Field(
        "",
        alias="TIER_HIGH_EFFORT",
        description=(
            "Reasoning effort for the 'high' tier. One of ``none``, "
            "``minimal``, ``low``, ``medium``, ``high``, ``xhigh``. Empty "
            "string inherits the provider's constructor default (no implicit "
            "reasoning). For Anthropic any non-empty/non-none value turns on "
            "adaptive thinking; the provider maps the level to its backend."
        ),
    )
    """Reasoning effort for the 'high' tier. One of ``none``, ``minimal``, ``low``, ``medium``, ``high``, ``xhigh``. Empty string inherits the provider's constructor default (no implicit reasoning). For Anthropic any non-empty/non-none value turns on adaptive thinking; the provider maps the level to its backend."""
    tier_mid_effort: str = Field(
        "",
        alias="TIER_MID_EFFORT",
        description=(
            "Reasoning effort for the 'mid' tier. See ``tier_high_effort`` "
            "for accepted values and semantics."
        ),
    )
    """Reasoning effort for the 'mid' tier. See ``tier_high_effort`` for accepted values and semantics."""
    tier_fast_effort: str = Field(
        "",
        alias="TIER_FAST_EFFORT",
        description=(
            "Reasoning effort for the 'fast' tier. See ``tier_high_effort`` "
            "for accepted values and semantics."
        ),
    )
    """Reasoning effort for the 'fast' tier. See ``tier_high_effort`` for accepted values and semantics."""

    @property
    def effective_classify_model(self) -> str:
        """Resolve the classify model via the central tier router.

        Returns:
            ``classify_model`` if set, else the classify tier model, else
            ``reasoning_model``. Equivalent to ``resolve_model("classify",
            self)``; kept as a property for backward compatibility with
            existing callers and the ``/health`` / ``/v1/stacks`` payloads.
        """
        return resolve_model("classify", self)

    @property
    def effective_claim_extract_model(self) -> str:
        """Resolve the claim-extraction model via the central tier router.

        Returns:
            ``claim_extract_model`` if set, else the claim_extract tier
            model, else ``reasoning_model``.
        """
        return resolve_model("claim_extract", self)

    @property
    def effective_evaluate_model(self) -> str:
        """Resolve the evaluate model via the central tier router.

        Returns:
            ``evaluate_model`` if set, else the evaluate tier model, else
            ``reasoning_model``.
        """
        return resolve_model("evaluate", self)


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
            "specific overrides (``max_rounds=5``, ``min_rounds=2``, "
            "``confidence_stop=9``, ``first_round_queries=10``, "
            "``answer_prompt_citations_max=500``, "
            "``reasoning_timeout=900``, ``claim_extract_timeout=600``, "
            "``search_timeout=300``, ``max_total_seconds=1800``) for any field the user has not "
            "set explicitly."
        ),
    )
    """Selects the answer style and depth preset (``compact`` or ``deep``). Assigning ``deep`` triggers a bundle of profile-specific overrides (``max_rounds=5``, ``min_rounds=2``, ``confidence_stop=9``, ``first_round_queries=10``, ``answer_prompt_citations_max=500``, ``reasoning_timeout=900``, ``claim_extract_timeout=600``, ``search_timeout=300``, ``max_total_seconds=1800``) for any field the user has not set explicitly."""
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
            "prompt. Default ``60`` for COMPACT, ``500`` for DEEP."
        ),
    )
    """Hard upper bound on citations passed to the final answer prompt. Default ``60`` for COMPACT, ``500`` for DEEP."""
    required_context_window_tokens: int = Field(
        REQUIRED_CONTEXT_WINDOW_TOKENS,
        alias="REQUIRED_CONTEXT_WINDOW_TOKENS",
        description=(
            "Minimum model context-window size expected for DEEP / forensic "
            "runs. Default ``128_000`` tracks the common 128k model tier; "
            "the answer composer also checks the concrete prompt "
            "estimate plus requested output budget and safety margin. "
            "Unknown provider capacity emits a visible warning; known "
            "capacity below the requirement blocks normal report synthesis."
        ),
    )
    """Minimum model context-window size expected for DEEP / forensic runs. Default ``128_000`` tracks the common 128k model tier. Unknown provider capacity emits a visible warning; known capacity below the requirement blocks normal report synthesis."""

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
    claim_extract_timeout: int = Field(
        60,
        alias="CLAIM_EXTRACT_TIMEOUT",
        description=(
            "Per-call timeout (seconds) for claim-extraction LLM calls. "
            "Tight by design because one call runs per search hit."
        ),
    )
    """Per-call timeout (seconds) for claim-extraction LLM calls. Tight by design because one call runs per search hit."""
    max_total_seconds: int = Field(
        300,
        alias="MAX_TOTAL_SECONDS",
        description=(
            "Wall-clock deadline (seconds) for the entire research run. "
            "Default ``300`` for COMPACT, ``1800`` for DEEP. Checked at "
            "node boundaries; in-flight provider calls may run slightly "
            "past this before the next check."
        ),
    )
    """Wall-clock deadline (seconds) for the entire research run. Default ``300`` for COMPACT, ``1800`` for DEEP. Checked at node boundaries; in-flight provider calls may run slightly past this before the next check."""
    max_question_length: int = Field(
        60_000,
        alias="MAX_QUESTION_LENGTH",
        description=(
            "Maximum input question length in characters. Inputs above "
            "this are rejected before the agent starts to protect "
            "against prompt-flooding accidents. The default is generous "
            "because the chat composer inlines attached file content into "
            "the message; lower for tighter input validation in "
            "public-facing deployments."
        ),
    )
    """Maximum input question length in characters. Inputs above this are rejected before the agent starts to protect against prompt-flooding accidents. The default is generous because the chat composer inlines attached file content into the message; lower for tighter input validation in public-facing deployments."""

    high_risk_score_threshold: int = Field(
        4,
        alias="HIGH_RISK_SCORE_THRESHOLD",
        description=(
            "Risk-score threshold (0-10) at and above which a question is "
            "flagged ``high_risk``. The flag is an observability signal only: "
            "it is surfaced in forensic events and ``/health`` and preserved "
            "across follow-up questions. It does not change model selection "
            "(use the model tiers or a per-node model override for that) and "
            "no longer drives any query/answer heuristic. Lower the threshold "
            "to flag more questions as high-risk; raise it to flag fewer."
        ),
    )
    """Risk-score threshold (0-10) at and above which a question is flagged ``high_risk``. The flag is an observability signal only: it is surfaced in forensic events and ``/health`` and preserved across follow-up questions. It does not change model selection (use the model tiers or a per-node model override for that) and no longer drives any query/answer heuristic. Lower the threshold to flag more questions as high-risk; raise it to flag fewer."""
    model_tier: str = Field(
        "",
        alias="MODEL_TIER",
        description=(
            "Optional per-run tier selection ('high', 'mid', or 'fast'). When "
            "set, it replaces the default per-node tier assignment for this "
            "run, for every LLM call site; an explicit per-node model "
            "override still wins. Empty string uses the default assignment. "
            "Primary use: the chat endpoint's ``model_tier`` override lets a "
            "caller pick the model class for a direct-chat answer."
        ),
    )
    """Optional per-run tier selection ('high', 'mid', or 'fast'). When set, it replaces the default per-node tier assignment for this run, for every LLM call site; an explicit per-node model override still wins. Empty string uses the default assignment. Primary use: the chat endpoint's ``model_tier`` override lets a caller pick the model class for a direct-chat answer."""
    model: str = Field(
        "",
        alias="MODEL_OVERRIDE",
        description=(
            "Optional explicit model id for the direct-chat answer, set by the "
            "chat/editor model picker selecting a concrete model instead of a "
            "tier. When non-empty it bypasses tier routing for the direct-chat "
            "call only; the research pipeline keeps tier routing. Empty string "
            "uses the tier. Pair with ``effort`` for reasoning depth."
        ),
    )
    """Optional explicit model id for the direct-chat answer, set by the chat/editor model picker selecting a concrete model instead of a tier. When non-empty it bypasses tier routing for the direct-chat call only; the research pipeline keeps tier routing. Empty string uses the tier. Pair with ``effort`` for reasoning depth."""
    effort: str = Field(
        "",
        alias="EFFORT_OVERRIDE",
        description=(
            "Optional reasoning effort for the direct-chat answer, paired with "
            "``model``. One of ``none``, ``minimal``, ``low``, ``medium``, "
            "``high``, ``xhigh``, ``max`` (model-dependent). Empty string "
            "inherits the provider default. Only consulted on the explicit-"
            "model direct-chat path."
        ),
    )
    """Optional reasoning effort for the direct-chat answer, paired with ``model``. One of ``none``, ``minimal``, ``low``, ``medium``, ``high``, ``xhigh``, ``max`` (model-dependent). Empty string inherits the provider default. Only consulted on the explicit-model direct-chat path."""

    @field_validator("model_tier")
    @classmethod
    def _validate_model_tier(cls, value: str) -> str:
        """Reject an unknown ``model_tier`` loudly at construction (Designprinzip 1)."""
        return validate_model_tier(value)

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

    observability_profile: str = Field(
        "summary",
        alias="OBSERVABILITY_PROFILE",
        description=(
            "Controls structured runtime event detail. ``summary`` keeps "
            "the compact INFO/DEBUG behavior, ``debug`` is reserved for "
            "future mid-level diagnostics, and ``forensic`` emits "
            "provider-neutral source, citation, claim, stop, and answer "
            "lineage events through the existing sanitized logger. "
            "Forensic mode is semantically complete but does not log raw "
            "provider request bodies, headers, SDK responses, or secrets."
        ),
    )
    """Controls structured runtime event detail. ``summary`` keeps the compact INFO/DEBUG behavior, ``debug`` is reserved for future mid-level diagnostics, and ``forensic`` emits provider-neutral source, citation, claim, stop, and answer lineage events through the existing sanitized logger. Forensic mode is semantically complete but does not log raw provider request bodies, headers, SDK responses, or secrets."""

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
            char-/token-/citation-cap values consumed by claim extraction,
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
        profile = str(self.observability_profile or "summary").strip().lower()
        if profile not in {"summary", "debug", "forensic"}:
            raise ValueError(
                "observability_profile must be one of: summary, debug, forensic"
            )
        object.__setattr__(self, "observability_profile", profile)
        return self


class ServerSettings(BaseSettings):
    """HTTP-server-only configuration loaded from environment variables.

    These fields steer the FastAPI surface launched by
    ``python -m inqtrix``: upstream LLM-gateway connection, request
    concurrency, chat-history extraction, native run retention, and
    transport/security settings. Library-mode users (those instantiating
    :class:`~inqtrix.agent.ResearchAgent` directly) can ignore
    ``ServerSettings`` entirely.
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
    perplexity_api_key: str = Field(
        "",
        alias="PERPLEXITY_API_KEY",
        description=(
            "API key for the native Perplexity Agent API used by the "
            "auto-created ``PerplexitySearch`` provider in server mode. "
            "Perplexity search runs against its own endpoint, not the "
            "LiteLLM proxy, so it needs a dedicated key. Empty default "
            "leaves the auto-created search provider unconfigured until set."
        ),
    )
    """API key for the native Perplexity Agent API used by the auto-created ``PerplexitySearch`` provider in server mode. Perplexity search runs against its own endpoint, not the LiteLLM proxy, so it needs a dedicated key. Empty default leaves the auto-created search provider unconfigured until set."""
    perplexity_base_url: str = Field(
        "",
        alias="PERPLEXITY_BASE_URL",
        description=(
            "Optional base-URL override for the auto-created "
            "``PerplexitySearch`` provider. Empty (default) uses the "
            "Perplexity SDK default (``https://api.perplexity.ai``); set "
            "only to target a Perplexity-compatible proxy."
        ),
    )
    """Optional base-URL override for the auto-created ``PerplexitySearch`` provider. Empty (default) uses the Perplexity SDK default (``https://api.perplexity.ai``); set only to target a Perplexity-compatible proxy."""
    max_concurrent: int = Field(
        3,
        alias="MAX_CONCURRENT",
        ge=1,
        description=(
            "Maximum number of concurrently executing OpenAI-compatible "
            "chat-completion requests in the HTTP server. Native "
            "``/v1/runs`` uses ``run_max_concurrent`` when set, or this "
            "same value otherwise. Sized for moderate per-run resource "
            "use (LLM tokens, search-API quota); raise carefully when "
            "upstream providers support higher parallelism."
        ),
    )
    """Maximum number of concurrently executing OpenAI-compatible chat-completion requests in the HTTP server. Native ``/v1/runs`` uses ``run_max_concurrent`` when set, or this same value otherwise. Sized for moderate per-run resource use (LLM tokens, search-API quota); raise carefully when upstream providers support higher parallelism."""
    run_max_concurrent: int | None = Field(
        None,
        alias="RUN_MAX_CONCURRENT",
        ge=1,
        description=(
            "Optional active-worker cap for native ``/v1/runs`` jobs. "
            "When unset, native runs reuse ``max_concurrent``; when set, "
            "chat completions and native runs have explicit per-surface "
            "caps. This does not create a global cross-endpoint cap."
        ),
    )
    """Optional active-worker cap for native ``/v1/runs`` jobs. When unset, native runs reuse ``max_concurrent``; when set, chat completions and native runs have explicit per-surface caps. This does not create a global cross-endpoint cap."""
    run_queue_max_size: int = Field(
        50,
        alias="RUN_QUEUE_MAX_SIZE",
        ge=0,
        description=(
            "Maximum number of native ``/v1/runs`` jobs waiting in the "
            "FIFO queue. Active native jobs are governed by "
            "``run_max_concurrent`` or ``max_concurrent`` and do not "
            "count against this limit. When the queue is full, ``POST "
            "/v1/runs`` returns HTTP 429."
        ),
    )
    """Maximum number of native ``/v1/runs`` jobs waiting in the FIFO queue. Active native jobs are governed by ``run_max_concurrent`` or ``max_concurrent`` and do not count against this limit. When the queue is full, ``POST /v1/runs`` returns HTTP 429."""
    run_completed_ttl_seconds: int = Field(
        300,
        alias="RUN_COMPLETED_TTL_SECONDS",
        ge=0,
        description=(
            "TTL (seconds) for completed, failed, or cancelled native "
            "run records. During this window the UI can still fetch "
            "``/v1/runs/{run_id}``, replay buffered events, or load the "
            "result payload. Persistence across refreshes beyond this "
            "short window is intentionally left to a future database "
            "adapter."
        ),
    )
    """TTL (seconds) for completed, failed, or cancelled native run records. During this window the UI can still fetch ``/v1/runs/{run_id}``, replay buffered events, or load the result payload. Persistence across refreshes beyond this short window is intentionally left to a future database adapter."""
    run_event_buffer_size: int = Field(
        200,
        alias="RUN_EVENT_BUFFER_SIZE",
        ge=1,
        description=(
            "Number of recent structured events retained per native run "
            "for late SSE subscribers. The buffer is per-run and bounded "
            "to avoid unbounded memory growth during long research jobs."
        ),
    )
    """Number of recent structured events retained per native run for late SSE subscribers. The buffer is per-run and bounded to avoid unbounded memory growth during long research jobs."""
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
            "FastAPI dependency on chat, text-improvement, test-run, "
            "and native run routes that requires "
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
    """Static Bearer API key. When set, the server installs a FastAPI dependency on chat, text-improvement, test-run, and native run routes that requires ``Authorization: Bearer <api_key>`` and compares with ``hmac.compare_digest`` for constant-time safety. ``/health`` and ``/v1/models`` deliberately stay unauthenticated so Kubernetes liveness probes and model discovery clients keep working without credentials. Empty string (default) disables the gate, matching the historical behaviour. Rotation requires a server restart in this iteration; multi-key support is a follow-up task."""
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

    max_total_input_tokens: int = Field(
        500_000,
        alias="INQTRIX_MAX_TOTAL_INPUT_TOKENS",
        ge=10_000,
        description=(
            "Approximate-token cap on the combined size of ``question`` + "
            "``messages[]`` in ``/v1/chat/completions``. Token count is "
            "estimated as ``len(text) // 4`` (provider-agnostic heuristic "
            "that overshoots slightly, which is the safe side for a DoS "
            "guard). Default ``500_000`` is generous — real research "
            "requests stay well below this. Raise only when serving "
            "providers with 1M+ effective context windows. Lower bound "
            "``10_000`` prevents accidental misconfiguration that would "
            "block typical conversational use."
        ),
    )
    """Approximate-token cap on the combined size of ``question`` + ``messages[]`` in ``/v1/chat/completions``. Token count is estimated as ``len(text) // 4`` (provider-agnostic heuristic that overshoots slightly, which is the safe side for a DoS guard). Default ``500_000`` is generous — real research requests stay well below this. Raise only when serving providers with 1M+ effective context windows. Lower bound ``10_000`` prevents accidental misconfiguration that would block typical conversational use."""
    max_message_count: int = Field(
        200,
        alias="INQTRIX_MAX_MESSAGE_COUNT",
        ge=1,
        description=(
            "Hard cap on the number of entries in ``messages[]`` for "
            "``/v1/chat/completions``. Defends against array-bomb "
            "payloads without limiting normal multi-turn flows (a typical "
            "long conversation is 20-40 messages). Returns HTTP 413 when "
            "exceeded."
        ),
    )
    """Hard cap on the number of entries in ``messages[]`` for ``/v1/chat/completions``. Defends against array-bomb payloads without limiting normal multi-turn flows (a typical long conversation is 20-40 messages). Returns HTTP 413 when exceeded."""


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
