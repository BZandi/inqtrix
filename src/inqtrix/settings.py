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

from typing import Annotated, Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode

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
            "``reasoning_timeout=900``, ``editor_assistant_timeout=900``, "
            "``claim_extract_timeout=600``, "
            "``search_timeout=300``, ``max_total_seconds=1800``) for any field the user has not "
            "set explicitly."
        ),
    )
    """Selects the answer style and depth preset (``compact`` or ``deep``). Assigning ``deep`` triggers a bundle of profile-specific overrides (``max_rounds=5``, ``min_rounds=2``, ``confidence_stop=9``, ``first_round_queries=10``, ``answer_prompt_citations_max=500``, ``reasoning_timeout=900``, ``editor_assistant_timeout=900``, ``claim_extract_timeout=600``, ``search_timeout=300``, ``max_total_seconds=1800``) for any field the user has not set explicitly."""
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
    editor_assistant_timeout: int = Field(
        120,
        alias="EDITOR_ASSISTANT_TIMEOUT",
        description=(
            "Per-call timeout (seconds) for editor suggest/instruct calls. "
            "Decoupled from ``reasoning_timeout`` so editor work (a full "
            "generation over large attached context) can get a longer budget "
            "without lengthening every research reasoning call. Defaults to "
            "the reasoning-timeout default; ``900`` under DEEP."
        ),
    )
    """Per-call timeout (seconds) for editor suggest/instruct calls. Decoupled from ``reasoning_timeout`` so editor work (a full generation over large attached context) can get a longer budget without lengthening every research reasoning call. Defaults to the reasoning-timeout default (``120``); ``900`` under DEEP."""
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
        6,
        alias="MAX_CONCURRENT",
        ge=1,
        description=(
            "Maximum number of concurrently executing OpenAI-compatible "
            "chat-completion requests in the HTTP server. Native "
            "``/v1/runs`` uses ``run_max_concurrent`` when set, or this "
            "same value otherwise. Each active run holds one thread for "
            "its full duration (the research graph is synchronous), so "
            "this also bounds the in-process thread pool. Sized for "
            "moderate per-run resource use (LLM tokens, search-API "
            "quota); the ceiling in practice is the upstream provider's "
            "rate limit and the host's CPU/RAM, not this number — raise "
            "it when the provider supports higher parallelism, lower it "
            "if you hit provider 429s."
        ),
    )
    """Maximum number of concurrently executing OpenAI-compatible chat-completion requests in the HTTP server. Native ``/v1/runs`` uses ``run_max_concurrent`` when set, or this same value otherwise. Each active run holds one thread for its full duration (the research graph is synchronous), so this also bounds the in-process thread pool. The real ceiling is the upstream provider's rate limit and the host's CPU/RAM; raise when the provider supports more parallelism, lower on provider 429s."""
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
    run_durable_retention_seconds: int = Field(
        7_776_000,
        alias="RUN_DURABLE_RETENTION_SECONDS",
        ge=0,
        description=(
            "Retention (seconds) for terminal native run records in the "
            "DURABLE (Postgres) store; default 90 days. Distinct from "
            "``run_completed_ttl_seconds``, which bounds the in-memory "
            "store's replay buffer to a short window: this governs how long "
            "completed research reports stay fetchable from the database, so "
            "they survive page reloads, re-logins, and other devices instead "
            "of vanishing after the in-memory TTL. The in-memory store "
            "ignores this value. Lazy cleanup deletes terminal rows older "
            "than the window (their events and result payloads cascade); "
            "raise it to keep reports longer at the cost of bounded table "
            "growth, or set 0 to evict on the next cleanup."
        ),
    )
    """Retention (seconds) for terminal native run records in the durable (Postgres) store; default 90 days. Distinct from ``run_completed_ttl_seconds``, which bounds the in-memory store's replay buffer to a short window: this governs how long completed research reports stay fetchable from the database, so they survive page reloads, re-logins, and other devices instead of vanishing after the in-memory TTL. The in-memory store ignores this value. Lazy cleanup deletes terminal rows older than the window (their events and result payloads cascade); raise it to keep reports longer at the cost of bounded table growth, or set 0 to evict on the next cleanup."""
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
    enable_openapi: bool = Field(
        False,
        alias="INQTRIX_ENABLE_OPENAPI",
        description=(
            "Serve the FastAPI OpenAPI schema and interactive docs "
            "(``/openapi.json``, ``/docs``, ``/redoc``). Disabled by "
            "default so production deployments keep the historical "
            "no-schema surface (the schema enumerates every route and "
            "request shape, which operators may not want public). "
            "Enable for development, SDK generation, and contract "
            "testing; the toggle only adds documentation routes and "
            "never changes API behaviour."
        ),
    )
    """Serve the FastAPI OpenAPI schema and interactive docs (``/openapi.json``, ``/docs``, ``/redoc``). Disabled by default so production deployments keep the historical no-schema surface (the schema enumerates every route and request shape, which operators may not want public). Enable for development, SDK generation, and contract testing; the toggle only adds documentation routes and never changes API behaviour."""

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
    public_base_url: str = Field(
        "",
        alias="INQTRIX_PUBLIC_BASE_URL",
        description=(
            "Externally reachable base URL of this server, e.g. "
            "``https://inqtrix.example.com``. When set, knowledge "
            "citations become clickable HTTP links into "
            "``/v1/sources/...``; when empty (default) citations keep "
            "the internal ``inqtrix://`` URI scheme — a deliberate, "
            "visible degradation, never a guessed hostname."
        ),
    )
    """Externally reachable base URL of this server, e.g. ``https://inqtrix.example.com``. When set, knowledge citations become clickable HTTP links into ``/v1/sources/...``; when empty (default) citations keep the internal ``inqtrix://`` URI scheme — a deliberate, visible degradation, never a guessed hostname."""


class StorageSettings(BaseSettings):
    """Persistence-backend configuration for the platform layer.

    ``memory`` (the default) keeps the historical no-infrastructure
    deployment: identity facts live in process memory and vanish on
    restart. ``postgres`` activates the SQLAlchemy/asyncpg-backed
    identity repositories and requires ``database_url``. Run/event
    persistence stays in-memory in either mode until the durable run
    ports land.
    """

    model_config = _SETTINGS_MODEL_CONFIG

    backend: Literal["memory", "postgres"] = Field(
        "memory",
        alias="INQTRIX_STORAGE_BACKEND",
        description=(
            "Persistence backend selector. ``memory`` (default) needs "
            "no external services and is the pytest default; "
            "``postgres`` requires ``INQTRIX_DATABASE_URL`` and a "
            "migrated database (``inqtrix-migrate``). There is "
            "deliberately no SQLite option — the schema relies on "
            "Postgres row-level security."
        ),
    )
    """Persistence backend selector. ``memory`` (default) needs no external services and is the pytest default; ``postgres`` requires ``INQTRIX_DATABASE_URL`` and a migrated database (``inqtrix-migrate``). There is deliberately no SQLite option — the schema relies on Postgres row-level security."""
    database_url: str = Field(
        "",
        alias="INQTRIX_DATABASE_URL",
        description=(
            "SQLAlchemy async database URL, e.g. "
            "``postgresql+asyncpg://inqtrix:...@127.0.0.1:5432/inqtrix``. "
            "Required (and only read) when ``backend`` is "
            "``postgres``; the empty default combined with "
            "``backend=postgres`` fails loudly at startup."
        ),
    )
    """SQLAlchemy async database URL, e.g. ``postgresql+asyncpg://inqtrix:...@127.0.0.1:5432/inqtrix``. Required (and only read) when ``backend`` is ``postgres``; the empty default combined with ``backend=postgres`` fails loudly at startup."""
    app_role: str = Field(
        "inqtrix_app",
        alias="INQTRIX_DATABASE_APP_ROLE",
        description=(
            "Postgres role every application transaction switches to "
            "via ``SET LOCAL ROLE`` before touching tenant tables. The "
            "role is created NOLOGIN/NOSUPERUSER/NOBYPASSRLS by the "
            "migrations so row-level security applies even when the "
            "connection user is the table owner or a superuser "
            "(dev-compose convenience). Empty disables the switch — "
            "only sensible when the connection user itself is a "
            "restricted role."
        ),
    )
    """Postgres role every application transaction switches to via ``SET LOCAL ROLE`` before touching tenant tables. The role is created NOLOGIN/NOSUPERUSER/NOBYPASSRLS by the migrations so row-level security applies even when the connection user is the table owner or a superuser (dev-compose convenience). Empty disables the switch — only sensible when the connection user itself is a restricted role."""
    object_store_backend: Literal["local", "s3"] = Field(
        "local",
        alias="INQTRIX_OBJECT_STORE_BACKEND",
        description=(
            "Binary-blob storage backend for uploaded files. ``local`` "
            "(default) writes content-addressed blobs below "
            "``object_store_path`` — no external services, suitable "
            "for dev and single-node deployments. ``s3`` targets any "
            "S3-compatible endpoint (SeaweedFS is the reference dev "
            "stack) and requires the ``s3_*`` fields. File METADATA "
            "(owner, hash, permissions) always lives in the file "
            "registry, never in the blob store."
        ),
    )
    """Binary-blob storage backend for uploaded files. ``local`` (default) writes content-addressed blobs below ``object_store_path``; ``s3`` targets any S3-compatible endpoint (SeaweedFS is the reference dev stack) and requires the ``s3_*`` fields. File METADATA (owner, hash, permissions) always lives in the file registry, never in the blob store."""
    object_store_path: str = Field(
        "data/object-store",
        alias="INQTRIX_OBJECT_STORE_PATH",
        description=(
            "Root directory of the ``local`` object-store backend, "
            "relative paths resolve against the working directory. "
            "Created on first write."
        ),
    )
    """Root directory of the ``local`` object-store backend, relative paths resolve against the working directory. Created on first write."""
    s3_endpoint_url: str = Field(
        "",
        alias="INQTRIX_S3_ENDPOINT_URL",
        description=(
            "Endpoint URL of the S3-compatible service, e.g. "
            "``http://127.0.0.1:8333`` for the SeaweedFS dev stack. "
            "Required (and only read) when ``object_store_backend`` "
            "is ``s3``. Path-style addressing is always used — "
            "virtual-host buckets do not exist on self-hosted stores."
        ),
    )
    """Endpoint URL of the S3-compatible service, e.g. ``http://127.0.0.1:8333`` for the SeaweedFS dev stack. Required (and only read) when ``object_store_backend`` is ``s3``. Path-style addressing is always used."""
    s3_bucket: str = Field(
        "inqtrix-files",
        alias="INQTRIX_S3_BUCKET",
        description=(
            "Bucket holding every uploaded blob. Object keys are "
            "namespaced ``tenants/<tenant>/files/<uuid>`` so one "
            "bucket serves all tenants; the bucket is created on "
            "startup when missing."
        ),
    )
    """Bucket holding every uploaded blob. Object keys are namespaced ``tenants/<tenant>/files/<uuid>``; the bucket is created on startup when missing."""
    s3_access_key: str = Field(
        "",
        alias="INQTRIX_S3_ACCESS_KEY",
        description=(
            "Access key for the S3-compatible service. Credentials "
            "never leave the backend — clients receive streamed "
            "content, not store URLs."
        ),
    )
    """Access key for the S3-compatible service. Credentials never leave the backend — clients receive streamed content, not store URLs."""
    s3_secret_key: str = Field(
        "",
        alias="INQTRIX_S3_SECRET_KEY",
        description="Secret key for the S3-compatible service.",
    )
    """Secret key for the S3-compatible service."""
    s3_region: str = Field(
        "us-east-1",
        alias="INQTRIX_S3_REGION",
        description=(
            "Region name sent to the S3 client. Self-hosted stores "
            "ignore it but boto3 requires a value; the default is the "
            "conventional placeholder."
        ),
    )
    """Region name sent to the S3 client. Self-hosted stores ignore it but boto3 requires a value; the default is the conventional placeholder."""
    max_file_bytes: int = Field(
        104_857_600,
        alias="INQTRIX_MAX_FILE_BYTES",
        ge=1,
        description=(
            "Upper bound for one uploaded file (default 100 MiB). "
            "Uploads stream through a spool with running hash/size "
            "accounting, so the limit is enforced without buffering "
            "the file in memory; exceeding it aborts with HTTP 413."
        ),
    )
    """Upper bound for one uploaded file (default 100 MiB). Enforced while spooling; exceeding it aborts with HTTP 413."""


class AuthSettings(BaseSettings):
    """Authentication configuration for the HTTP server.

    The mode selector plus the GENERIC OIDC contract surface
    (ADR-AUTH-1): Inqtrix speaks only standard OIDC — discovery,
    authorization code + PKCE — and hardwires no identity provider.
    The claim-mapping and fallback fields exist because issuer +
    client id + secret alone are not enough for real IdP
    exchangeability (verified against Grafana/ArgoCD/Outline/Gitea
    surfaces): Okta needs an extra groups scope, Entra omits
    ``email_verified``, Keycloak nests roles under a path.
    """

    model_config = _SETTINGS_MODEL_CONFIG

    mode: Literal["infer", "none", "apikey", "oidc", "local", "ldap"] = Field(
        "infer",
        alias="INQTRIX_AUTH_MODE",
        description=(
            "Authentication-mode selector. ``infer`` (the default) "
            "derives the mode for backwards compatibility: a "
            "non-empty ``INQTRIX_SERVER_API_KEY`` means ``apikey``, "
            "an empty one means ``none`` — existing deployments "
            "behave bit-for-bit without touching their environment. "
            "An explicit value wins over inference and survives "
            "settings serialization round-trips (the sentinel is a "
            "first-class value, not field-set introspection). "
            "``apikey`` without a configured key and ``oidc`` without "
            "its connection settings are rejected loudly at startup; "
            "``none`` "
            "with a configured key disables the gate deliberately and "
            "logs a WARNING. This raw field never reports the ACTIVE "
            "mode — consult "
            ":func:`inqtrix.auth.principal.resolve_auth_mode` or the "
            "auth provider's ``mode`` property for that."
        ),
    )
    """Authentication-mode selector. ``infer`` (the default) derives the mode for backwards compatibility: a non-empty ``INQTRIX_SERVER_API_KEY`` means ``apikey``, an empty one means ``none``. An explicit value wins over inference and survives settings serialization round-trips. ``apikey`` without a configured key and ``oidc`` without its connection settings are rejected loudly at startup; ``none`` with a configured key disables the gate deliberately and logs a WARNING. This raw field never reports the ACTIVE mode — consult :func:`inqtrix.auth.principal.resolve_auth_mode` or the auth provider's ``mode`` property for that."""
    oidc_issuer: str = Field(
        "",
        alias="INQTRIX_OIDC_ISSUER",
        description=(
            "OIDC issuer URL (e.g. ``https://login.example.com/realms/x`` "
            "or the Dex dev issuer ``http://127.0.0.1:5556/dex``). "
            "Discovery is fetched from "
            "``{issuer}/.well-known/openid-configuration`` and the "
            "document's ``issuer`` must match this value exactly. "
            "Required for ``INQTRIX_AUTH_MODE=oidc``."
        ),
    )
    """OIDC issuer URL; discovery is fetched from ``{issuer}/.well-known/openid-configuration`` and must echo this exact value. Required for mode ``oidc``."""
    oidc_client_id: str = Field(
        "",
        alias="INQTRIX_OIDC_CLIENT_ID",
        description=(
            "OAuth client id registered at the identity provider. "
            "Required for ``INQTRIX_AUTH_MODE=oidc``."
        ),
    )
    """OAuth client id registered at the identity provider. Required for mode ``oidc``."""
    oidc_client_secret: str = Field(
        "",
        alias="INQTRIX_OIDC_CLIENT_SECRET",
        description=(
            "Confidential-client secret (the BFF is a confidential "
            "client per the browser-apps BCP; tokens never reach the "
            "browser). Required for ``INQTRIX_AUTH_MODE=oidc``."
        ),
    )
    """Confidential-client secret; the BFF keeps all tokens server-side. Required for mode ``oidc``."""
    oidc_redirect_url: str = Field(
        "",
        alias="INQTRIX_OIDC_REDIRECT_URL",
        description=(
            "Absolute redirect/callback URL registered at the IdP "
            "(must match byte-for-byte — providers compare exact "
            "strings). Empty derives "
            "``{INQTRIX_PUBLIC_BASE_URL}/api/auth/callback``."
        ),
    )
    """Absolute callback URL registered at the IdP; empty derives ``{public_base_url}/api/auth/callback``."""
    oidc_scopes: str = Field(
        "openid profile email",
        alias="INQTRIX_OIDC_SCOPES",
        description=(
            "Space-separated scopes for the authorization request. "
            "Okta requires an additional ``groups`` scope for group "
            "claims; Dex uses ``groups`` too."
        ),
    )
    """Space-separated authorization scopes; add ``groups`` for IdPs that gate group claims behind a scope (Okta, Dex)."""
    oidc_username_claim: str = Field(
        "preferred_username",
        alias="INQTRIX_OIDC_USERNAME_CLAIM",
        description=(
            "Claim used as the display username; falls back to "
            "``email`` then ``sub`` when absent. Dot-separated paths "
            "descend into nested claims (Keycloak's "
            "``realm_access.roles`` style)."
        ),
    )
    """Display-username claim with ``email``/``sub`` fallback; dot paths descend into nested claims."""
    oidc_email_claim: str = Field(
        "email",
        alias="INQTRIX_OIDC_EMAIL_CLAIM",
        description="Claim carrying the user's email address.",
    )
    """Claim carrying the user's email address."""
    oidc_groups_claim: str = Field(
        "groups",
        alias="INQTRIX_OIDC_GROUPS_CLAIM",
        description=(
            "Claim carrying group memberships (list of strings). "
            "Dot-separated paths descend into nested claims."
        ),
    )
    """Group-membership claim (list of strings); dot paths descend into nested claims."""
    oidc_allowed_groups: str = Field(
        "",
        alias="INQTRIX_OIDC_ALLOWED_GROUPS",
        description=(
            "Optional comma-separated allowlist: when non-empty, a "
            "login whose group claim shares no entry with this list "
            "is rejected (visible 403 at the callback, audit-logged). "
            "Empty admits every authenticated user."
        ),
    )
    """Optional comma-separated group allowlist; non-empty rejects logins without a matching group."""
    oidc_roles_claim: str = Field(
        "roles",
        alias="INQTRIX_OIDC_ROLES_CLAIM",
        description=(
            "Claim carrying role assignments used for admin elevation. "
            "Dot-separated paths descend into nested claims (Keycloak "
            "``realm_access.roles`` or ``resource_access.<client>.roles``). "
            "May equal ``INQTRIX_OIDC_GROUPS_CLAIM`` for tenants that emit "
            "groups-as-roles."
        ),
    )
    """Role claim for admin elevation; dot paths descend into nested claims."""
    oidc_admin_roles: str = Field(
        "",
        alias="INQTRIX_OIDC_ADMIN_ROLES",
        description=(
            "Comma-separated role values that grant instance-admin on "
            "login (grant-only: a match promotes, a non-match never "
            "demotes). Matched literally against the roles claim."
        ),
    )
    """Comma-separated roles that grant instance-admin (grant-only)."""
    oidc_admin_groups: str = Field(
        "",
        alias="INQTRIX_OIDC_ADMIN_GROUPS",
        description=(
            "Comma-separated group values that grant instance-admin on "
            "login (grant-only). Matched literally against the groups "
            "claim — use the exact emitted value (Entra group GUIDs, "
            "Keycloak ``/path`` strings)."
        ),
    )
    """Comma-separated groups that grant instance-admin (grant-only)."""
    oidc_allowed_domains: str = Field(
        "",
        alias="INQTRIX_OIDC_ALLOWED_DOMAINS",
        description=(
            "Optional comma-separated email-domain allowlist, orthogonal "
            "to the group allowlist: when non-empty, a login whose email "
            "domain is not listed is rejected (visible 403). A login "
            "without an email is rejected (fail-closed). Matched "
            "case-insensitively."
        ),
    )
    """Optional comma-separated email-domain allowlist (orthogonal to groups)."""
    oidc_claim_separators: str = Field(
        " ,",
        alias="INQTRIX_OIDC_CLAIM_SEPARATORS",
        description=(
            "Characters a STRING-valued group/role claim is split on (a "
            "JSON array is used as-is). Default whitespace and comma, so "
            "``admin, staff`` and ``admin staff`` both parse."
        ),
    )
    """Separators for string-valued group/role claims (default whitespace + comma)."""
    oidc_groups_strip_path_prefix: bool = Field(
        False,
        alias="INQTRIX_OIDC_GROUPS_STRIP_PATH_PREFIX",
        description=(
            "Strip a single leading ``/`` from each group value "
            "(Keycloak full-path groups like ``/Engineering/Backend``). "
            "Explicit rather than a silent normalisation; match the "
            "allowlist to whichever form you keep."
        ),
    )
    """Strip one leading ``/`` from group values (Keycloak full-path groups)."""
    oidc_provider_name: str = Field(
        "",
        alias="INQTRIX_OIDC_PROVIDER_NAME",
        description=(
            "Display name for the SSO login button, surfaced by the "
            "public auth-config endpoint (e.g. ``Okta``, ``Entra ID``). "
            "Empty leaves the frontend's generic SSO label."
        ),
    )
    """SSO login-button display name surfaced by the auth-config endpoint."""
    oidc_skip_email_verified: bool = Field(
        False,
        alias="INQTRIX_OIDC_SKIP_EMAIL_VERIFIED",
        description=(
            "Accept logins whose token lacks ``email_verified=true``. "
            "Required for Entra ID, which omits the claim entirely; "
            "leave off for IdPs that emit it."
        ),
    )
    """Accept logins without ``email_verified=true`` (Entra ID omits the claim)."""
    oidc_discovery_url: str = Field(
        "",
        alias="INQTRIX_OIDC_DISCOVERY_URL",
        description=(
            "Override for the discovery-document URL when the IdP "
            "serves metadata somewhere other than "
            "``{issuer}/.well-known/openid-configuration``. The "
            "document's ``issuer`` must still match "
            "``INQTRIX_OIDC_ISSUER``."
        ),
    )
    """Discovery-document URL override; the document's ``issuer`` must still match the configured issuer."""
    oidc_userinfo_fallback: bool = Field(
        True,
        alias="INQTRIX_OIDC_USERINFO_FALLBACK",
        description=(
            "Fetch the userinfo endpoint when the id_token lacks the "
            "mapped claims (Okta thin tokens, Entra group overflow). "
            "Off saves one IdP round-trip per login."
        ),
    )
    """Fetch userinfo when the id_token lacks mapped claims (Okta thin tokens)."""
    oidc_ca_cert: str = Field(
        "",
        alias="INQTRIX_OIDC_CA_CERT",
        description=(
            "Path to a PEM CA bundle for the IdP's TLS endpoints "
            "(on-prem IdPs behind a private CA). Empty uses the "
            "system trust store."
        ),
    )
    """PEM CA bundle path for IdPs behind a private CA; empty uses the system trust store."""
    oidc_insecure_dev_cookies: bool = Field(
        False,
        alias="INQTRIX_OIDC_INSECURE_DEV_COOKIES",
        description=(
            "Development escape hatch: drop the ``Secure`` flag and "
            "the ``__Host-`` prefix so login works over plain "
            "``http://127.0.0.1`` in every browser (Safari rejects "
            "Secure cookies on loopback HTTP). NEVER in production; "
            "activation logs a WARNING at startup."
        ),
    )
    """Development escape hatch: drop ``Secure``/``__Host-`` cookie hardening for plain-HTTP loopback; logs a WARNING."""
    session_secret: str = Field(
        "",
        alias="INQTRIX_SESSION_SECRET",
        description=(
            "Server-side secret for CSRF-token derivation (OWASP "
            "signed double-submit: HMAC over the session id). "
            "Required for ``INQTRIX_AUTH_MODE=oidc``; rotate to "
            "invalidate outstanding CSRF tokens (sessions survive)."
        ),
    )
    """Server-side secret for CSRF-token derivation (signed double-submit). Required for mode ``oidc``."""
    session_max_age_seconds: int = Field(
        28_800,
        alias="INQTRIX_SESSION_MAX_AGE_SECONDS",
        ge=300,
        description=(
            "Absolute server-side session lifetime (default 8 hours). "
            "Expired sessions resolve to 401 and the SPA re-runs the "
            "login redirect; no silent refresh in this iteration."
        ),
    )
    """Absolute server-side session lifetime in seconds (default 8 hours)."""
    pat_pepper: str = Field(
        "",
        alias="INQTRIX_PAT_PEPPER",
        description=(
            "Server-side secret mixed into the HMAC of every personal "
            "access token. REQUIRED in oidc mode (fail-loud at "
            "startup): without a pepper a database leak alone would "
            "let stolen hashes be brute-forced offline against short "
            "secrets. Rotating the pepper invalidates EVERY issued "
            "token — clients see uniform 401s; plan rotations as "
            "deliberate maintenance."
        ),
    )
    """Server-side secret mixed into the HMAC of every personal access token. Required in oidc mode (fail-loud at startup); a database leak alone must not suffice to verify guesses against stolen hashes. Rotating the pepper invalidates every issued token at once — clients receive uniform 401s, so treat rotation as deliberate maintenance with re-issue."""
    pat_max_per_user: int = Field(
        10,
        alias="INQTRIX_PAT_MAX_PER_USER",
        ge=1,
        description=(
            "Cap on ACTIVE (non-revoked, non-expired) personal access "
            "tokens per user. A guardrail against unbounded token "
            "sprawl, not a security boundary — concurrent creates "
            "across replicas may briefly exceed it."
        ),
    )
    """Cap on active (non-revoked, non-expired) personal access tokens per user. A sprawl guardrail, not a security boundary: the count-then-insert check may briefly overshoot under concurrent creates across replicas."""
    pat_default_ttl_days: int = Field(
        0,
        alias="INQTRIX_PAT_DEFAULT_TTL_DAYS",
        ge=0,
        description=(
            "Default lifetime applied when a token is created WITHOUT "
            "an explicit expiry. 0 (default) means such tokens never "
            "expire; a positive value sets the default in days. An "
            "explicit per-token expiry always wins — this is a "
            "default, not a cap."
        ),
    )
    """Default lifetime in days for personal access tokens created without an explicit expiry; 0 (default) keeps them non-expiring. An explicit per-token expiry always wins — this field is a default, deliberately not a cap."""
    registration: Literal["open", "invite"] = Field(
        "open",
        alias="INQTRIX_REGISTRATION",
        description=(
            "Admission policy for FIRST-time OIDC logins. ``open`` "
            "(default, the historical behaviour) admits every "
            "IdP-authenticated user; ``invite`` admits unknown users "
            "only when an open invitation matches their email and "
            "rejects everyone else with 403 BEFORE any user record is "
            "created. ``invite`` requires oidc mode AND the postgres "
            "storage backend (memory invitations would evaporate on "
            "restart and lock everyone out) — both contradictions "
            "fail loudly at startup."
        ),
    )
    """Admission policy for first-time OIDC logins. ``open`` (default) keeps the historical admit-everyone behaviour; ``invite`` requires a matching open invitation for unknown users (403 otherwise, before any user record exists) and is only valid with oidc mode plus the postgres storage backend — both contradictions are rejected loudly at startup."""
    local_registration: Literal["closed", "open"] = Field(
        "closed",
        alias="INQTRIX_LOCAL_REGISTRATION",
        description=(
            "Self-signup policy for ``INQTRIX_AUTH_MODE=local``. "
            "``closed`` (default, secure): the first-run setup creates "
            "the owner, and every further account is created by an admin "
            "or via an invitation — there is no public registration "
            "route. ``open``: a public self-signup route is mounted "
            "(logged loudly at startup) so anyone can create an account. "
            "Independent of the OIDC ``registration`` field above."
        ),
    )
    """Self-signup policy for ``local`` mode. ``closed`` (default): owner via first-run setup, further accounts admin-created/invited, no public registration. ``open``: a public self-signup route is mounted (logged loudly). Independent of the OIDC ``registration`` field."""

    ldap_url: str = Field(
        "",
        alias="INQTRIX_LDAP_URL",
        description=(
            "LDAP server URL for ``INQTRIX_AUTH_MODE=ldap``: "
            "``ldap://host:389`` or ``ldaps://host:636``. Required for ldap "
            "mode."
        ),
    )
    """LDAP server URL (``ldap://``/``ldaps://``). Required for ldap mode."""
    ldap_bind_dn: str = Field(
        "",
        alias="INQTRIX_LDAP_BIND_DN",
        description=(
            "Service-account DN used to search for users before re-binding "
            "as them (search-then-bind). Use a least-privilege read-only "
            "account. Required for ldap mode."
        ),
    )
    """Service-account DN for the user search (least-privilege). Required for ldap mode."""
    ldap_bind_password: str = Field(
        "",
        alias="INQTRIX_LDAP_BIND_PASSWORD",
        description="Password for the LDAP service account. Required for ldap mode.",
    )
    """Password for the LDAP service account. Required for ldap mode."""
    ldap_user_search_base: str = Field(
        "",
        alias="INQTRIX_LDAP_USER_SEARCH_BASE",
        description=(
            "Base DN for the user search, e.g. "
            "``ou=people,dc=example,dc=com``. Required for ldap mode."
        ),
    )
    """Base DN for the user search. Required for ldap mode."""
    ldap_user_search_filter: str = Field(
        "(uid={username})",
        alias="INQTRIX_LDAP_USER_SEARCH_FILTER",
        description=(
            "User search filter with a ``{username}`` placeholder (the "
            "login name, escaped against LDAP injection before formatting). "
            "Defaults to ``(uid={username})``; Active Directory commonly uses "
            "``(sAMAccountName={username})`` or ``(mail={username})``."
        ),
    )
    """User search filter with a ``{username}`` placeholder (escaped). Default ``(uid={username})``."""
    ldap_email_attr: str = Field(
        "mail",
        alias="INQTRIX_LDAP_EMAIL_ATTR",
        description="LDAP attribute mapped to email. Falls back to the login username.",
    )
    """LDAP attribute mapped to email (default ``mail``); falls back to the username."""
    ldap_display_name_attr: str = Field(
        "cn",
        alias="INQTRIX_LDAP_DISPLAY_NAME_ATTR",
        description="LDAP attribute mapped to the display name. Falls back to email.",
    )
    """LDAP attribute mapped to the display name (default ``cn``); falls back to email."""
    ldap_id_attr: str = Field(
        "entryUUID",
        alias="INQTRIX_LDAP_ID_ATTR",
        description=(
            "Stable LDAP attribute mapped to the subject identity anchor. "
            "Falls back to the user DN. ``entryUUID`` (OpenLDAP) or "
            "``objectGUID`` (AD) are stable across renames; ``uid`` is not."
        ),
    )
    """Stable LDAP attribute mapped to the subject (default ``entryUUID``); falls back to the DN."""
    ldap_admin_group_dn: str = Field(
        "",
        alias="INQTRIX_LDAP_ADMIN_GROUP_DN",
        description=(
            "Optional group DN whose members receive the instance-admin "
            "role on login (matched case-insensitively against the user's "
            "``memberOf``). Empty disables group-based admin mapping."
        ),
    )
    """Optional group DN whose members become instance-admin (via ``memberOf``)."""
    ldap_start_tls: bool = Field(
        False,
        alias="INQTRIX_LDAP_START_TLS",
        description="Issue StartTLS on an ``ldap://`` connection before binding.",
    )
    """Issue StartTLS on an ``ldap://`` connection before binding."""
    ldap_ca_cert: str = Field(
        "",
        alias="INQTRIX_LDAP_CA_CERT",
        description="Optional PEM CA-bundle path for ldaps/StartTLS verification.",
    )
    """Optional PEM CA-bundle path for ldaps/StartTLS verification."""
    ldap_tls_validate: bool = Field(
        True,
        alias="INQTRIX_LDAP_TLS_VALIDATE",
        description=(
            "Verify the LDAP server certificate (ldaps/StartTLS). ``false`` "
            "(trusted-network dev only) logs a WARNING."
        ),
    )
    """Verify the LDAP server certificate. ``false`` (dev only) logs a WARNING."""
    ldap_first_login_owner: bool = Field(
        True,
        alias="INQTRIX_LDAP_FIRST_LOGIN_OWNER",
        description=(
            "When True (default), the first user to authenticate via LDAP "
            "becomes the instance admin if none exists yet. Combine with "
            "``INQTRIX_LDAP_ADMIN_GROUP_DN`` for group-driven admin."
        ),
    )
    """First LDAP login becomes instance admin if none exists yet (default True)."""
    login_rate_limit_enabled: bool = Field(
        True,
        alias="INQTRIX_LOGIN_RATE_LIMIT_ENABLED",
        description=(
            "Throttle failed password-mode logins (local/ldap) per "
            "``(identifier, source_ip)``. On by default (secure default); a "
            "multi-replica deployment should ALSO throttle per-IP at the "
            "reverse proxy, since the counters are process-local."
        ),
    )
    """Throttle failed local/ldap logins per (identifier, ip). On by default."""
    login_rate_limit_max_attempts: int = Field(
        10,
        alias="INQTRIX_LOGIN_RATE_LIMIT_MAX_ATTEMPTS",
        ge=1,
        description=(
            "Failed attempts within the window that trip a lockout. The "
            "default 10 tolerates fat-fingering while stopping dictionary runs."
        ),
    )
    """Failed attempts within the window that trip a lockout (default 10)."""
    login_rate_limit_window_seconds: int = Field(
        300,
        alias="INQTRIX_LOGIN_RATE_LIMIT_WINDOW_SECONDS",
        ge=1,
        description="Rolling window (seconds) over which failed attempts accumulate.",
    )
    """Rolling window (seconds) over which failed attempts accumulate (default 300)."""
    login_rate_limit_lockout_seconds: int = Field(
        60,
        alias="INQTRIX_LOGIN_RATE_LIMIT_LOCKOUT_SECONDS",
        ge=1,
        description="How long a tripped ``(identifier, ip)`` stays locked (seconds).",
    )
    """How long a tripped (identifier, ip) stays locked, in seconds (default 60)."""


class KnowledgeSettings(BaseSettings):
    """Knowledge-engine (internal document retrieval) configuration.

    Disabled by default — the knowledge surface, the embedding
    provider, and the ``mode=knowledge`` algorithm only exist when
    ``enabled`` is true, keeping the historical deployment shape
    untouched. Like every Settings group, this is the only env-coupled
    surface; the providers it configures receive everything via
    constructor arguments.
    """

    model_config = _SETTINGS_MODEL_CONFIG

    enabled: bool = Field(
        False,
        alias="INQTRIX_KNOWLEDGE_ENABLED",
        description=(
            "Master switch for the knowledge engine. When false "
            "(default) no knowledge routes are registered, no "
            "embedding provider is constructed, and ``mode=knowledge`` "
            "is not a registered algorithm — requests naming it get "
            "the standard mode-validation 400."
        ),
    )
    """Master switch for the knowledge engine. When false (default) no knowledge routes are registered, no embedding provider is constructed, and ``mode=knowledge`` is not a registered algorithm — requests naming it get the standard mode-validation 400."""
    vector_backend: Literal["memory", "qdrant"] = Field(
        "memory",
        alias="INQTRIX_VECTOR_BACKEND",
        description=(
            "Vector/document store backend. ``memory`` (in-process, "
            "lost on restart, default) needs no services; ``qdrant`` "
            "persists collections/documents/chunks in a Qdrant "
            "instance and enables hybrid (dense + BM25) retrieval — "
            "requires the ``knowledge-qdrant`` extra and "
            "``qdrant_url``. ``pgvector`` joins the Literal when its "
            "store lands."
        ),
    )
    """Vector/document store backend. ``memory`` (in-process, lost on restart, default) needs no services; ``qdrant`` persists in a Qdrant instance and enables hybrid (dense + BM25) retrieval — requires the ``knowledge-qdrant`` extra and ``qdrant_url``."""
    qdrant_url: str = Field(
        "http://127.0.0.1:6333",
        alias="INQTRIX_QDRANT_URL",
        description=(
            "Qdrant REST endpoint for ``vector_backend=qdrant``. The "
            "default matches the loopback dev compose stack."
        ),
    )
    """Qdrant REST endpoint for ``vector_backend=qdrant``. The default matches the loopback dev compose stack."""
    qdrant_api_key: str = Field(
        "",
        alias="INQTRIX_QDRANT_API_KEY",
        description=(
            "Qdrant API key. Self-hosted Qdrant is UNAUTHENTICATED by "
            "default — set this everywhere except pure loopback dev "
            "(an empty key is accepted but logged loudly)."
        ),
    )
    """Qdrant API key. Self-hosted Qdrant is UNAUTHENTICATED by default — set this everywhere except pure loopback dev (an empty key is accepted but logged loudly)."""
    sparse: Literal["bm25_german", "off"] = Field(
        "bm25_german",
        alias="INQTRIX_KNOWLEDGE_SPARSE",
        description=(
            "Lexical retrieval branch for the qdrant backend. "
            "``bm25_german`` (default) computes BM25 sparse vectors "
            "client-side — a tokenizer/stemmer ALGORITHM, no hosted "
            "model — and fuses them with the dense branch via RRF. "
            "``off`` runs dense-only. Ignored by the memory backend."
        ),
    )
    """Lexical retrieval branch for the qdrant backend. ``bm25_german`` (default) computes BM25 sparse vectors client-side and fuses via RRF; ``off`` runs dense-only. Ignored by the memory backend."""
    reranker_provider: Literal["none", "cohere", "llm"] = Field(
        "none",
        alias="INQTRIX_RERANKER_PROVIDER",
        description=(
            "Optional rerank stage after retrieval. ``none`` (default) "
            "skips the stage — a visible capability flag, never a "
            "silent downgrade. ``cohere`` calls a Cohere-rerank-schema "
            "endpoint (native or Azure AI Foundry serverless) and "
            "requires the ``reranker_*`` fields. ``llm`` ranks "
            "listwise through the deployment's own LLM — a fallback "
            "for deployments without a rerank API contract: roughly "
            "an order of magnitude more expensive and slower than a "
            "specialized cross-encoder and hard-capped at 20 "
            "candidates per query, so deeper candidate pools are "
            "truncated (visibly logged)."
        ),
    )
    """Optional rerank stage after retrieval. ``none`` (default) skips the stage; ``cohere`` calls a Cohere-rerank-schema endpoint (native or Azure AI Foundry serverless) and requires the ``reranker_*`` fields; ``llm`` ranks listwise through the deployment's own LLM — a fallback without extra infrastructure, roughly an order of magnitude costlier/slower than a cross-encoder and hard-capped at 20 candidates (deeper pools are truncated with a visible log line)."""
    reranker_base_url: str = Field(
        "",
        alias="INQTRIX_RERANKER_BASE_URL",
        description=(
            "Rerank endpoint base, e.g. ``https://api.cohere.com`` or "
            "an Azure serverless deployment URL. Required when "
            "``reranker_provider`` is not ``none``."
        ),
    )
    """Rerank endpoint base, e.g. ``https://api.cohere.com`` or an Azure serverless deployment URL. Required when ``reranker_provider`` is not ``none``."""
    reranker_api_key: str = Field(
        "",
        alias="INQTRIX_RERANKER_API_KEY",
        description="API key for the rerank endpoint.",
    )
    """API key for the rerank endpoint."""
    reranker_model: str = Field(
        "",
        alias="INQTRIX_RERANKER_MODEL",
        description=(
            "Rerank model/deployment id, e.g. ``rerank-v3.5`` or the "
            "Azure deployment name. Required when ``reranker_provider`` "
            "is not ``none``."
        ),
    )
    """Rerank model/deployment id, e.g. ``rerank-v3.5`` or the Azure deployment name. Required when ``reranker_provider`` is not ``none``."""
    document_parser: Literal["markitdown", "none"] = Field(
        "markitdown",
        alias="INQTRIX_DOCUMENT_PARSER",
        description=(
            "Parser for ingesting uploaded FILES into knowledge "
            "collections. ``markitdown`` (default) converts "
            "PDF/DOCX/PPTX/XLSX/HTML to Markdown in pure Python — no "
            "ML models, no cloud dependency. ``none`` disables file "
            "ingestion (text-only API stays available). Azure Document "
            "Intelligence joins the Literal as the optional premium "
            "tier for scans/complex tables when its adapter lands."
        ),
    )
    """Parser for ingesting uploaded FILES into knowledge collections. ``markitdown`` (default) converts PDF/DOCX/PPTX/XLSX/HTML to Markdown in pure Python; ``none`` disables file ingestion. Azure Document Intelligence joins as the optional premium tier when its adapter lands."""
    contextualize: Literal["on", "off"] = Field(
        "off",
        alias="INQTRIX_KNOWLEDGE_CONTEXTUALIZE",
        description=(
            "Contextual retrieval: at ingestion, one batched fast-tier "
            "LLM call per document generates a short situating context "
            "per chunk, prepended before embedding and BM25 indexing. "
            "Improves retrieval of context-dependent chunks at a "
            "one-time ingestion cost; off by default until the "
            "deployment opts into the extra LLM usage. Existing "
            "documents are unaffected — re-ingest to contextualize."
        ),
    )
    """Contextual retrieval: one batched fast-tier LLM call per document at ingestion generates situating contexts prepended before embedding and BM25 indexing. Off by default; re-ingest existing documents to apply."""
    gate: Literal["on", "off"] = Field(
        "on",
        alias="INQTRIX_KNOWLEDGE_GATE",
        description=(
            "Sufficiency gate for mode=knowledge: one fast-tier LLM "
            "call judges the retrieved evidence and may trigger one "
            "second retrieval pass; insufficient evidence yields the "
            "honest no-evidence answer instead of a fabricated one. "
            "``off`` restores the always-answer-from-top-k behaviour "
            "(saves one small LLM call per question)."
        ),
    )
    """Sufficiency gate for mode=knowledge: one fast-tier LLM call judges the retrieved evidence and may trigger one second retrieval pass; insufficient evidence yields the honest no-evidence answer. ``off`` restores always-answer-from-top-k."""
    grounding: Literal["on", "off"] = Field(
        "on",
        alias="INQTRIX_KNOWLEDGE_GROUNDING",
        description=(
            "Quote-then-answer grounding for mode=knowledge: the "
            "answer prompt requires a block of verbatim, labelled "
            "quotes before the answer; the quotes are verified "
            "deterministically against the evidence (no extra LLM "
            "call), reported visibly via markers and a run event, and "
            "stripped from the user-facing answer. ``off`` restores "
            "the plain single-section answer prompt."
        ),
    )
    """Quote-then-answer grounding for mode=knowledge: the answer prompt requires a block of verbatim, labelled quotes before the answer; quotes are verified deterministically against the evidence (no extra LLM call), reported visibly, and stripped from the user-facing answer. ``off`` restores the plain single-section answer prompt."""
    gate_max_rounds: int = Field(
        3,
        alias="INQTRIX_KNOWLEDGE_GATE_MAX_ROUNDS",
        ge=1,
        le=5,
        description=(
            "Hard operator cap on gate rewrite-and-retrieve rounds for "
            "EVERY retrieval profile (the deep profile requests up to "
            "this many; standard always uses one). Each round costs "
            "one fast-tier gate call plus one retrieval pass, so the "
            "cap bounds the worst-case cost of an agentic run. "
            "``ge=1`` keeps the standard profile's single rewrite "
            "always possible while the gate itself is enabled."
        ),
    )
    """Hard operator cap on gate rewrite-and-retrieve rounds for every retrieval profile. The deep profile requests up to this many rounds, standard exactly one; each round costs one fast-tier gate call plus one retrieval pass, so this bounds the worst-case cost of an agentic run. Lower bound 1 keeps standard's single rewrite possible while the gate is enabled."""
    rerank_candidate_depth: int = Field(
        40,
        alias="INQTRIX_RERANK_CANDIDATE_DEPTH",
        ge=5,
        le=200,
        description=(
            "Candidate pool retrieved before the rerank stage reduces "
            "it to the requested top_k. Larger pools improve recall at "
            "the cost of rerank latency/cost per query."
        ),
    )
    """Candidate pool retrieved before the rerank stage reduces it to the requested top_k. Larger pools improve recall at the cost of rerank latency/cost per query."""
    embedding_provider: Literal["openai_compatible", "azure"] = Field(
        "openai_compatible",
        alias="INQTRIX_EMBEDDING_PROVIDER",
        description=(
            "How the embedding endpoint authenticates. "
            "``openai_compatible`` (default) posts to "
            "``{base_url}/embeddings`` with a Bearer key (LiteLLM "
            "proxy, OpenAI, vLLM, Azure v1 surface). ``azure`` uses "
            "Azure deployment-based auth (api-key header + "
            "api-version) for resources without the OpenAI-compatible "
            "surface; it reads the ``embedding_azure_*`` fields, which "
            "fall back to the established ``AZURE_AI_PROJECT_*`` / "
            "``AZURE_OPENAI_API_KEY`` variables so an existing Azure "
            "deployment needs exactly this one extra setting."
        ),
    )
    """How the embedding endpoint authenticates. ``openai_compatible`` (default) posts to ``{base_url}/embeddings`` with a Bearer key. ``azure`` uses Azure deployment-based auth (api-key header + api-version) and reads the ``embedding_azure_*`` fields, which fall back to ``AZURE_AI_PROJECT_*`` / ``AZURE_OPENAI_API_KEY``."""
    embedding_azure_endpoint: str = Field(
        "",
        validation_alias=AliasChoices(
            "INQTRIX_EMBEDDING_AZURE_ENDPOINT",
            "AZURE_AI_PROJECT_ENDPOINT",
        ),
        description=(
            "Azure resource endpoint for ``embedding_provider=azure``. "
            "An AI-Foundry PROJECT endpoint is accepted — the provider "
            "reduces it to the resource root where deployments live. "
            "Falls back to ``AZURE_AI_PROJECT_ENDPOINT``."
        ),
    )
    """Azure resource endpoint for ``embedding_provider=azure``. An AI-Foundry PROJECT endpoint is accepted — the provider reduces it to the resource root. Falls back to ``AZURE_AI_PROJECT_ENDPOINT``."""
    embedding_azure_api_key: str = Field(
        "",
        validation_alias=AliasChoices(
            "INQTRIX_EMBEDDING_AZURE_API_KEY",
            "AZURE_AI_PROJECT_API_KEY",
            "AZURE_OPENAI_API_KEY",
        ),
        description=(
            "API key for ``embedding_provider=azure``. Falls back to "
            "``AZURE_AI_PROJECT_API_KEY`` and then "
            "``AZURE_OPENAI_API_KEY``."
        ),
    )
    """API key for ``embedding_provider=azure``. Falls back to ``AZURE_AI_PROJECT_API_KEY`` and then ``AZURE_OPENAI_API_KEY``."""
    embedding_azure_api_version: str = Field(
        "2024-10-21",
        alias="INQTRIX_EMBEDDING_AZURE_API_VERSION",
        description=(
            "Azure OpenAI data-plane API version for the embeddings "
            "deployment. The default is the 2024-10-21 GA version."
        ),
    )
    """Azure OpenAI data-plane API version for the embeddings deployment. The default is the 2024-10-21 GA version."""
    embedding_base_url: str = Field(
        "",
        alias="INQTRIX_EMBEDDING_BASE_URL",
        description=(
            "OpenAI-compatible ``/embeddings`` endpoint base URL. "
            "Empty (default) reuses ``LITELLM_BASE_URL`` so a standard "
            "LiteLLM-proxy deployment needs no extra configuration."
        ),
    )
    """OpenAI-compatible ``/embeddings`` endpoint base URL. Empty (default) reuses ``LITELLM_BASE_URL`` so a standard LiteLLM-proxy deployment needs no extra configuration."""
    embedding_api_key: str = Field(
        "",
        alias="INQTRIX_EMBEDDING_API_KEY",
        description=(
            "API key for the embeddings endpoint. Empty (default) "
            "reuses ``LITELLM_API_KEY``."
        ),
    )
    """API key for the embeddings endpoint. Empty (default) reuses ``LITELLM_API_KEY``."""
    embedding_model: str = Field(
        "text-embedding-3-small",
        alias="INQTRIX_EMBEDDING_MODEL",
        description=(
            "Default embedding model for NEW collections. Each "
            "collection stores its model immutably at creation; "
            "changing this default never affects existing collections."
        ),
    )
    """Default embedding model for NEW collections. Each collection stores its model immutably at creation; changing this default never affects existing collections."""
    selectable_embedding_models: str = Field(
        "",
        alias="INQTRIX_SELECTABLE_EMBEDDING_MODELS",
        description=(
            "Comma-separated embedding-model ids the UI may offer in "
            "the collection-creation picker (annotated via the "
            "embedding catalog). Empty hides the picker — collections "
            "then always use ``embedding_model``."
        ),
    )
    """Comma-separated embedding-model ids the UI may offer in the collection-creation picker (annotated via the embedding catalog). Empty hides the picker — collections then always use ``embedding_model``."""
    default_top_k: int = Field(
        8,
        alias="INQTRIX_KNOWLEDGE_TOP_K",
        ge=1,
        le=50,
        description=(
            "Default number of evidence chunks retrieved per question. "
            "Requests may override per call via "
            "``knowledge_filters.top_k``."
        ),
    )
    """Default number of evidence chunks retrieved per question. Requests may override per call via ``knowledge_filters.top_k``."""
    chunk_max_chars: int = Field(
        2_000,
        alias="INQTRIX_KNOWLEDGE_CHUNK_MAX_CHARS",
        ge=200,
        le=20_000,
        description=(
            "Character budget per document chunk at ingestion. Must "
            "stay below the embedding model's input limit; the "
            "default (~500 tokens) fits every catalogued model."
        ),
    )
    """Character budget per document chunk at ingestion. Must stay below the embedding model's input limit; the default (~500 tokens) fits every catalogued model."""
    max_document_chars: int = Field(
        2_000_000,
        alias="INQTRIX_KNOWLEDGE_MAX_DOCUMENT_CHARS",
        ge=10_000,
        description=(
            "Upper bound on one uploaded document's text size. "
            "Protects the synchronous in-process ingestion path from "
            "unbounded embedding work; raise when the worker-based "
            "pipeline lands."
        ),
    )
    """Upper bound on one uploaded document's text size. Protects the synchronous in-process ingestion path from unbounded embedding work; raise when the worker-based pipeline lands."""
    reindex_max_concurrent: int = Field(
        6,
        alias="INQTRIX_REINDEX_MAX_CONCURRENT",
        ge=1,
        description=(
            "Maximum number of reindex (re-embed) jobs executing at "
            "once — i.e. how many DIFFERENT collections can re-embed "
            "simultaneously. Within a single job the documents are "
            "re-embedded one after another, and reindex is serialized "
            "per collection (one active job per collection, enforced at "
            "the database), so this value never parallelizes a single "
            "collection — it only sets how many separate collections "
            "reindex in parallel. Each running job streams embedding "
            "calls, so concurrent jobs add up on the embedding endpoint "
            "and compete with live query-embedding for retrieval; raise "
            "this for faster bulk reindexing when the embedding provider "
            "has headroom, lower it if background reindex starves "
            "interactive search or hits the provider's rate limit. In "
            "worker mode (INQTRIX_QUEUE_BACKEND=valkey) the actual "
            "execution parallelism is INQTRIX_WORKER_CONCURRENCY per "
            "worker process; this value then governs admission only. "
            "Additional jobs wait in the FIFO queue."
        ),
    )
    """Maximum number of reindex (re-embed) jobs executing at once = how many different collections re-embed simultaneously (a single collection is always serialized, one active job per collection). Concurrent jobs add up on the embedding endpoint and compete with live query-embedding; raise for faster bulk reindex when the provider has headroom, lower if it starves interactive search. In worker mode the actual parallelism is INQTRIX_WORKER_CONCURRENCY; this then governs admission only."""
    reindex_queue_max_size: int = Field(
        50,
        alias="INQTRIX_REINDEX_QUEUE_MAX_SIZE",
        ge=0,
        description=(
            "Maximum number of reindex jobs waiting in the FIFO queue. "
            "Active jobs governed by ``reindex_max_concurrent`` do not "
            "count. When the queue is full, the reindex endpoint returns "
            "HTTP 429."
        ),
    )
    """Maximum number of reindex jobs waiting in the FIFO queue. Active jobs do not count against this limit. When the queue is full, the reindex endpoint returns HTTP 429."""
    reindex_completed_ttl_seconds: int = Field(
        3_600,
        alias="INQTRIX_REINDEX_COMPLETED_TTL_SECONDS",
        ge=0,
        description=(
            "TTL (seconds) for terminal (completed/failed/cancelled) "
            "reindex job records. During this window the UI can still "
            "fetch the job, replay events, and read its history entry. "
            "Longer than the run TTL because a returning browser shows "
            "recent reindex history. Governs retention in both the "
            "in-memory store and the durable Postgres store "
            "(INQTRIX_STORAGE_BACKEND=postgres); the per-collection "
            "INQTRIX_REINDEX_HISTORY_LIMIT caps history independently."
        ),
    )
    """TTL (seconds) for terminal reindex job records. During this window the UI can still fetch the job, replay events, and read its history entry. Longer than the run TTL so a returning browser sees recent history. Governs retention in both the in-memory store and the durable Postgres store; the per-collection history limit caps history independently."""
    reindex_event_buffer_size: int = Field(
        200,
        alias="INQTRIX_REINDEX_EVENT_BUFFER_SIZE",
        ge=1,
        description=(
            "Number of recent structured events retained per reindex "
            "job for late SSE subscribers. Bounded to avoid unbounded "
            "memory growth during a long re-embed over many documents."
        ),
    )
    """Number of recent structured events retained per reindex job for late SSE subscribers. Bounded to avoid unbounded memory growth during a long re-embed over many documents."""
    reindex_history_limit: int = Field(
        10,
        alias="INQTRIX_REINDEX_HISTORY_LIMIT",
        ge=0,
        description=(
            "Maximum number of terminal reindex records retained per "
            "collection (the inline 'last N runs' history the UI "
            "shows). Older terminal records for a collection are evicted "
            "beyond this count even before their TTL expires."
        ),
    )
    """Maximum number of terminal reindex records retained per collection (the inline 'last N runs' history the UI shows). Older terminal records for a collection are evicted beyond this count even before their TTL expires."""

    def selectable_embedding_model_list(self) -> list[str]:
        """Parse the comma-separated selectable-models field."""
        return [
            item.strip()
            for item in (self.selectable_embedding_models or "").split(",")
            if item.strip()
        ]


class QueueSettings(BaseSettings):
    """Run-queue backend selection and worker tuning.

    Two orthogonal switches keep the zero-infrastructure default
    intact: run *records* become durable via
    ``INQTRIX_STORAGE_BACKEND=postgres`` (execution stays in-process),
    and run *execution* moves to separate worker processes via
    ``INQTRIX_QUEUE_BACKEND=valkey`` (which requires the Postgres run
    store, validated loudly at startup). Both unset = today's
    in-memory, in-process behaviour bit-for-bit.
    """

    model_config = _SETTINGS_MODEL_CONFIG

    backend: Literal["memory", "valkey"] = Field(
        "memory",
        alias="INQTRIX_QUEUE_BACKEND",
        description=(
            "Job-queue backend for native runs. ``memory`` (default) "
            "executes runs in-process exactly as before. ``valkey`` "
            "dispatches accepted runs to a Valkey Stream consumed by "
            "``inqtrix-worker`` processes; requires "
            "``INQTRIX_STORAGE_BACKEND=postgres`` (the run row is the "
            "source of truth, the stream only carries dispatch "
            "messages) and a non-empty ``INQTRIX_VALKEY_URL``."
        ),
    )
    """Job-queue backend for native runs. ``memory`` (default) executes runs in-process exactly as before. ``valkey`` dispatches accepted runs to a Valkey Stream consumed by ``inqtrix-worker`` processes; requires ``INQTRIX_STORAGE_BACKEND=postgres`` and a non-empty ``INQTRIX_VALKEY_URL``."""
    valkey_url: str = Field(
        "",
        alias="INQTRIX_VALKEY_URL",
        description=(
            "Valkey connection URL (``redis://`` scheme, e.g. "
            "``redis://127.0.0.1:6379/0``) for the job queue. Required "
            "when the queue backend is ``valkey``; ignored otherwise."
        ),
    )
    """Valkey connection URL (``redis://`` scheme, e.g. ``redis://127.0.0.1:6379/0``) for the job queue. Required when the queue backend is ``valkey``; ignored otherwise."""
    worker_concurrency: int = Field(
        2,
        alias="INQTRIX_WORKER_CONCURRENCY",
        ge=1,
        description=(
            "Maximum number of runs one worker process executes "
            "concurrently. Each run blocks one thread for its full "
            "duration (the research graph is synchronous), so this "
            "bounds provider load per worker replica."
        ),
    )
    """Maximum number of runs one worker process executes concurrently. Each run blocks one thread for its full duration, so this bounds provider load per worker replica."""
    worker_max_attempts: int = Field(
        3,
        alias="INQTRIX_WORKER_MAX_ATTEMPTS",
        ge=1,
        description=(
            "Delivery attempts per run before the worker dead-letters "
            "the job and marks the run failed. At-least-once delivery "
            "means crashes redeliver; the run-row state machine makes "
            "redelivery of finished runs a no-op, so this bound only "
            "fires for runs that repeatedly crash a worker."
        ),
    )
    """Delivery attempts per run before the worker dead-letters the job and marks the run failed. Redelivery of finished runs is a no-op; this bound only fires for runs that repeatedly crash a worker."""
    worker_heartbeat_seconds: float = Field(
        15.0,
        alias="INQTRIX_WORKER_HEARTBEAT_SECONDS",
        gt=0,
        description=(
            "Interval at which a worker re-claims its own in-flight "
            "stream entries (XCLAIM JUSTID) to reset their idle time. "
            "Keeps long research runs from being stolen while still "
            "letting crashed workers be detected quickly."
        ),
    )
    """Interval at which a worker re-claims its own in-flight stream entries (XCLAIM JUSTID) to reset their idle time."""
    worker_claim_idle_seconds: float = Field(
        90.0,
        alias="INQTRIX_WORKER_CLAIM_IDLE_SECONDS",
        gt=0,
        description=(
            "Idle threshold after which another worker may reclaim a "
            "pending stream entry (XAUTOCLAIM). Sized to heartbeat "
            "loss (several missed heartbeats), NOT to the maximum run "
            "duration — the heartbeat keeps legitimate long runs "
            "below this idle time."
        ),
    )
    """Idle threshold after which another worker may reclaim a pending stream entry (XAUTOCLAIM). Sized to heartbeat loss, not to maximum run duration."""


class QuotaSettings(BaseSettings):
    """Per-user usage quotas for multi-user (cookie-session) deployments.

    The operator ceiling layer of the two-level rule (the admin UI sets
    the middle layer — a tenant default and per-user overrides — within
    these bounds; see :mod:`inqtrix.quota`). Each dimension has a
    ``*_default`` (the out-of-box per-user allowance) and a ``*_max``
    (the hard ceiling no admin-set value may exceed). A value of ``0``
    means UNLIMITED for that field — so the all-zero default leaves
    every deployment byte-identical until an operator sets real
    numbers. Quotas apply ONLY when ``enabled`` and the active auth mode
    is one of the cookie-session modes (``oidc``/``local``/``ldap``); the
    single-operator anonymous/static principals (``none``/``apikey``) are
    never metered.
    """

    model_config = _SETTINGS_MODEL_CONFIG

    enabled: bool = Field(
        False,
        alias="INQTRIX_QUOTA_ENABLED",
        description=(
            "Master switch for per-user usage quotas. Off by default. "
            "Even when on, quotas bind only in the cookie-session modes "
            "(``oidc``/``local``/``ldap``) — the single-operator "
            "``none``/``apikey`` modes are never metered. The capability "
            "manifest advertises ``features.quota`` accordingly."
        ),
    )
    """Master switch for per-user usage quotas. Off by default; binds only in the cookie-session modes (oidc/local/ldap)."""

    runs_default: int = Field(
        0,
        alias="INQTRIX_QUOTA_RUNS_PER_MONTH",
        ge=0,
        description=(
            "Default native research runs a user may start per calendar "
            "month. ``0`` = unlimited. Checked exactly at submission "
            "(the count is known before the run starts)."
        ),
    )
    """Default native research runs per user per calendar month. ``0`` = unlimited."""
    runs_max: int = Field(
        0,
        alias="INQTRIX_QUOTA_RUNS_PER_MONTH_MAX",
        ge=0,
        description=(
            "Hard ceiling for the per-user monthly run allowance — no "
            "admin-set default or override may exceed it. ``0`` = no "
            "ceiling."
        ),
    )
    """Hard ceiling for the per-user monthly run allowance. ``0`` = no ceiling."""

    llm_tokens_default: int = Field(
        0,
        alias="INQTRIX_QUOTA_LLM_TOKENS_PER_MONTH",
        ge=0,
        description=(
            "Default LLM tokens (prompt + completion, summed across "
            "runs, chat and editor) a user may consume per calendar "
            "month. ``0`` = unlimited. Recorded post-hoc from the "
            "usage each call reports; the current run finishes and the "
            "NEXT submission is blocked once the budget is reached."
        ),
    )
    """Default LLM tokens per user per calendar month (runs + chat + editor). ``0`` = unlimited."""
    llm_tokens_max: int = Field(
        0,
        alias="INQTRIX_QUOTA_LLM_TOKENS_PER_MONTH_MAX",
        ge=0,
        description="Hard ceiling for the per-user monthly LLM-token allowance. ``0`` = no ceiling.",
    )
    """Hard ceiling for the per-user monthly LLM-token allowance. ``0`` = no ceiling."""

    embedding_tokens_default: int = Field(
        0,
        alias="INQTRIX_QUOTA_EMBEDDING_TOKENS_PER_MONTH",
        ge=0,
        description=(
            "Default embedding input tokens a user may consume per "
            "calendar month through document ingestion. ``0`` = "
            "unlimited. Same block-next model as the other flow "
            "dimensions: the current ingestion finishes and the NEXT is "
            "blocked once the budget is reached. A single ingestion "
            "cannot run away because per-document size is already "
            "bounded (INQTRIX_KNOWLEDGE_MAX_DOCUMENT_CHARS / the file "
            "size limit); the exact embedded-text tokens are recorded "
            "after ingestion."
        ),
    )
    """Default embedding input tokens per user per calendar month. ``0`` = unlimited."""
    embedding_tokens_max: int = Field(
        0,
        alias="INQTRIX_QUOTA_EMBEDDING_TOKENS_PER_MONTH_MAX",
        ge=0,
        description="Hard ceiling for the per-user monthly embedding-token allowance. ``0`` = no ceiling.",
    )
    """Hard ceiling for the per-user monthly embedding-token allowance. ``0`` = no ceiling."""

    stored_bytes_default: int = Field(
        0,
        alias="INQTRIX_QUOTA_STORED_BYTES",
        ge=0,
        description=(
            "Default object-store occupancy a user may hold, in bytes. "
            "A STOCK quota (not a per-period flow): it rises on upload "
            "and falls on delete, never resets. ``0`` = unlimited."
        ),
    )
    """Default per-user object-store occupancy in bytes (a stock quota, freed by deletion). ``0`` = unlimited."""
    stored_bytes_max: int = Field(
        0,
        alias="INQTRIX_QUOTA_STORED_BYTES_MAX",
        ge=0,
        description="Hard ceiling for per-user object-store occupancy in bytes. ``0`` = no ceiling.",
    )
    """Hard ceiling for per-user object-store occupancy in bytes. ``0`` = no ceiling."""

    max_tokens_per_run: int = Field(
        0,
        alias="INQTRIX_QUOTA_MAX_TOKENS_PER_RUN",
        ge=0,
        description=(
            "Optional HARD per-run token budget. ``0`` (default) = off "
            "— the per-period quota lets the current bounded run finish "
            "and blocks the next. When set, a single run that crosses "
            "this budget mid-flight is cancelled at the next graph-node "
            "boundary via the existing cancel token (a graceful stop "
            "that returns the partial result, never a mid-call kill). "
            "Independent of the monthly token quota."
        ),
    )
    """Optional hard per-run token budget; ``0`` = off. Cancels a run gracefully at the next node boundary via the existing cancel token."""


class ProviderSettings(BaseSettings):
    """Server-mode provider selection plus the credentials and inference knobs
    that have no other Settings home.

    This is the env-driven, mix-and-match front door for the HTTP server: the
    two independent axes ``llm_provider`` and ``search_provider`` pick which
    concrete providers :func:`~inqtrix.providers.create_providers` builds, and
    the remaining fields supply the per-provider credentials and the
    construction knobs (selectable model catalogue, temperature, token-budget
    parameter, search preset/instructions) that are not already covered by
    :class:`ModelSettings` (model names/tiers) or :class:`ServerSettings`
    (LiteLLM/Perplexity creds).

    Backward compatibility: the defaults ``litellm``/``perplexity`` reproduce
    the historical auto-create stack byte for byte. Library-mode users
    (explicit ``AgentConfig`` providers) bypass this class entirely. Per the
    constructor-first rule, providers never read these env vars themselves —
    only the factory translates them into constructor arguments.
    """

    model_config = _SETTINGS_MODEL_CONFIG

    llm_provider: Literal["litellm", "anthropic", "azure", "bedrock"] = Field(
        "litellm",
        alias="INQTRIX_LLM_PROVIDER",
        description=(
            "Selects the LLM axis of the server-mode provider stack. One of "
            "``litellm`` (OpenAI-compatible gateway, default and historical "
            "behaviour), ``anthropic`` (direct Messages API), ``azure`` "
            "(Azure OpenAI), ``bedrock`` (AWS Bedrock). Independent of "
            "``search_provider`` — any LLM pairs with any search provider. "
            "An unknown value is rejected loudly at ``Settings()`` construction."
        ),
    )
    """Selects the LLM axis of the server-mode provider stack (``litellm`` default). Independent of ``search_provider``; an unknown value fails loudly at construction."""
    search_provider: Literal["perplexity", "azure_foundry"] = Field(
        "perplexity",
        alias="INQTRIX_SEARCH_PROVIDER",
        description=(
            "Selects the search axis of the server-mode provider stack. One "
            "of ``perplexity`` (native Perplexity Agent API, default) or "
            "``azure_foundry`` (Azure AI Foundry Web Search agent). "
            "Independent of ``llm_provider``."
        ),
    )
    """Selects the search axis of the server-mode provider stack (``perplexity`` default). Independent of ``llm_provider``."""

    anthropic_api_key: str = Field(
        "",
        alias="ANTHROPIC_API_KEY",
        description=(
            "API key for the direct Anthropic Messages API. Required when "
            "``llm_provider=anthropic``; the factory fails loudly at startup "
            "if it is empty in that case."
        ),
    )
    """API key for the direct Anthropic Messages API. Required when ``llm_provider=anthropic``."""
    anthropic_base_url: str = Field(
        "",
        alias="ANTHROPIC_BASE_URL",
        description=(
            "Optional Messages-endpoint override for the Anthropic provider. "
            "Empty (default) uses the provider's built-in endpoint."
        ),
    )
    """Optional Messages-endpoint override for the Anthropic provider. Empty uses the provider default."""

    azure_openai_endpoint: str = Field(
        "",
        alias="AZURE_OPENAI_ENDPOINT",
        description=(
            "Azure OpenAI resource endpoint such as "
            "``https://<resource>.openai.azure.com/``. Required when "
            "``llm_provider=azure``. Do not append ``/openai/v1`` — the "
            "provider does that internally."
        ),
    )
    """Azure OpenAI resource endpoint. Required when ``llm_provider=azure``; the provider appends the ``/openai/v1`` path itself."""
    azure_openai_api_key: str = Field(
        "",
        alias="AZURE_OPENAI_API_KEY",
        description=(
            "Azure OpenAI API key (key-auth path). Either this or the full "
            "Service-Principal trio (``AZURE_TENANT_ID``/``AZURE_CLIENT_ID``/"
            "``AZURE_CLIENT_SECRET``) is required when ``llm_provider=azure``."
        ),
    )
    """Azure OpenAI API key (key-auth path). Alternative to the Service-Principal trio for ``llm_provider=azure``."""
    azure_tenant_id: str = Field(
        "",
        alias="AZURE_TENANT_ID",
        description=(
            "Entra tenant id for Service-Principal auth, shared by the Azure "
            "OpenAI LLM and the Azure Foundry search provider. Supply all "
            "three SP vars together or none."
        ),
    )
    """Entra tenant id for Service-Principal auth (shared by Azure LLM + Foundry search)."""
    azure_client_id: str = Field(
        "",
        alias="AZURE_CLIENT_ID",
        description="Entra client id for Service-Principal auth (see ``azure_tenant_id``).",
    )
    """Entra client id for Service-Principal auth (see ``azure_tenant_id``)."""
    azure_client_secret: str = Field(
        "",
        alias="AZURE_CLIENT_SECRET",
        description="Entra client secret for Service-Principal auth (see ``azure_tenant_id``).",
    )
    """Entra client secret for Service-Principal auth (see ``azure_tenant_id``)."""

    aws_profile: str = Field(
        "",
        alias="AWS_PROFILE",
        description=(
            "Optional AWS named profile for the Bedrock provider. Empty "
            "(default) lets boto3 resolve credentials from the standard AWS "
            "chain (env vars, instance role, etc.)."
        ),
    )
    """Optional AWS named profile for Bedrock. Empty lets boto3 use the default credential chain."""
    aws_region: str = Field(
        "eu-central-1",
        alias="AWS_REGION",
        description="AWS region for the Bedrock endpoint when ``llm_provider=bedrock``.",
    )
    """AWS region for the Bedrock endpoint when ``llm_provider=bedrock``."""

    azure_ai_project_endpoint: str = Field(
        "",
        alias="AZURE_AI_PROJECT_ENDPOINT",
        description=(
            "Azure AI Foundry project endpoint such as "
            "``https://<project>.services.ai.azure.com/api/projects/<project>``. "
            "Required when ``search_provider=azure_foundry``."
        ),
    )
    """Azure AI Foundry project endpoint. Required when ``search_provider=azure_foundry``."""
    azure_ai_project_api_key: str = Field(
        "",
        alias="AZURE_AI_PROJECT_API_KEY",
        description=(
            "Optional Foundry project API key for the Azure Foundry search "
            "provider. Either this or the shared Service-Principal trio "
            "authenticates the agent."
        ),
    )
    """Optional Foundry project API key for Azure Foundry search (alternative to the Service-Principal trio)."""
    web_search_agent_name: str = Field(
        "",
        alias="WEB_SEARCH_AGENT_NAME",
        description=(
            "Name of the pre-created Azure Foundry Web Search agent. Required "
            "when ``search_provider=azure_foundry``."
        ),
    )
    """Name of the pre-created Azure Foundry Web Search agent. Required when ``search_provider=azure_foundry``."""
    web_search_agent_version: str = Field(
        "",
        alias="WEB_SEARCH_AGENT_VERSION",
        description="Optional pinned version label for the Azure Foundry Web Search agent.",
    )
    """Optional pinned version label for the Azure Foundry Web Search agent."""

    selectable_chat_models: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        alias="INQTRIX_SELECTABLE_CHAT_MODELS",
        description=(
            "Comma-separated model ids offered in the UI model picker. Passed "
            "to every LLM provider as ``selectable_models``; the model is "
            "resolved against the model-card catalogue and surfaced in "
            "``/health.models_catalog``. Empty (default) leaves the explicit "
            "picker unfed (tier routing still applies)."
        ),
    )
    """Comma-separated model ids for the UI picker; passed as ``selectable_models`` and surfaced in ``/health.models_catalog``."""
    temperature: float | None = Field(
        None,
        alias="INQTRIX_TEMPERATURE",
        description=(
            "Optional sampling temperature for the Anthropic/Azure/Bedrock "
            "LLM providers. ``None`` (default) leaves it unset. LiteLLM is a "
            "generic gateway and ignores this; setting it under "
            "``llm_provider=litellm`` logs a visible warning."
        ),
    )
    """Optional sampling temperature (Anthropic/Azure/Bedrock). Ignored by LiteLLM (logged when set)."""
    token_budget_parameter: str = Field(
        "",
        alias="INQTRIX_TOKEN_BUDGET_PARAMETER",
        description=(
            "Output-budget request field for the LiteLLM/Azure providers: "
            "``max_tokens`` or ``max_completion_tokens`` (the latter is "
            "required by OpenAI o-series). Empty (default) uses each "
            "provider's own default. Anthropic/Bedrock ignore this; setting "
            "it there logs a visible warning."
        ),
    )
    """Output-budget request field (``max_tokens``/``max_completion_tokens``) for LiteLLM/Azure. Empty uses the provider default."""
    search_preset: str = Field(
        "",
        alias="INQTRIX_SEARCH_PRESET",
        description=(
            "Perplexity Agent preset: ``fast-search``, ``pro-search`` or "
            "``deep-research``. Empty (default) uses the provider default "
            "(``fast-search``). Web search has no high/mid/fast tiers like the "
            "LLM — this preset plus ``search_model`` are the search knobs. "
            "Ignored by ``azure_foundry`` (logged when set)."
        ),
    )
    """Perplexity Agent preset (``fast-search``/``pro-search``/``deep-research``). Empty uses the provider default; ignored by ``azure_foundry``."""
    search_instructions: str = Field(
        "",
        alias="INQTRIX_SEARCH_INSTRUCTIONS",
        description=(
            "Optional system instructions for the Perplexity search agent. "
            "Empty (default) leaves the agent's behaviour at its built-in "
            "default. Ignored by ``azure_foundry`` (logged when set)."
        ),
    )
    """Optional system instructions for the Perplexity search agent. Ignored by ``azure_foundry``."""

    @field_validator("llm_provider", "search_provider", mode="before")
    @classmethod
    def _normalize_selector(cls, value: object) -> object:
        """Lower-case and strip the axis selectors so ``Anthropic`` resolves.

        Keeps the env surface forgiving (case/whitespace) while the ``Literal``
        type still rejects genuinely unknown values loudly at construction.
        """
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("selectable_chat_models", mode="before")
    @classmethod
    def _split_selectable(cls, value: object) -> object:
        """Accept a comma-separated env string or an explicit list.

        Splits on commas, strips each entry, and drops empties so a trailing
        comma or stray whitespace never produces a phantom model id. A present
        but empty value collapses to ``[]`` (treated as unset).

        The field is annotated ``NoDecode`` for exactly this reason: without it,
        pydantic-settings JSON-decodes ``list[str]`` env values before any
        validator runs, so a comma string (the documented format) would raise a
        ``SettingsError`` and this splitter would never be reached. Keep both
        together.
        """
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("temperature", mode="before")
    @classmethod
    def _blank_temperature_to_none(cls, value: object) -> object:
        """Treat a blank ``INQTRIX_TEMPERATURE`` as unset rather than an error.

        A common ``.env`` pattern is to leave a variable present but empty
        (``INQTRIX_TEMPERATURE=``). Without this, Pydantic would reject the
        empty string as an invalid float and fail ``Settings()`` construction;
        mapping blank to ``None`` keeps "present but empty" equivalent to unset.
        """
        if isinstance(value, str) and value.strip() == "":
            return None
        return value


class SharingSettings(BaseSettings):
    """Resource-sharing policy for the cookie-session multi-user modes.

    Sharing is otherwise tenant-wide: any authenticated user may be a share
    target, and the share typeahead searches the whole tenant. This knob lets
    an operator confine collaboration to workspace boundaries without changing
    the permission chokepoint's read semantics — it is a grant-time (write)
    restriction only, so flipping it never revokes an existing grant.
    """

    model_config = _SETTINGS_MODEL_CONFIG

    restrict_to_workspace_members: bool = Field(
        False,
        alias="INQTRIX_SHARING_RESTRICT_TO_WORKSPACE_MEMBERS",
        description=(
            "When true, a user may only share a resource with people they "
            "share at least one workspace with, and the share typeahead is "
            "scoped the same way. Default false keeps sharing tenant-wide, "
            "byte-identical to deployments before this setting existed. Only "
            "meaningful in the cookie-session modes (oidc/local/ldap) that "
            "mount the sharing surface; the single-operator none/apikey modes "
            "never mount it."
        ),
    )
    """When true, sharing (grant + typeahead) is limited to the grantor's
    workspace co-members. Default false = tenant-wide, byte-identical."""


class Settings(BaseSettings):
    """Root container that aggregates the Settings groups.

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
    providers: ProviderSettings = Field(
        default_factory=ProviderSettings,
        description=(
            "Server-mode provider selection (LLM and search axes) plus the "
            "credentials and inference knobs not covered by ``models``/"
            "``server``. Defaults reproduce the historical LiteLLM+Perplexity "
            "stack. Ignored in pure library mode."
        ),
    )
    """Server-mode provider selection + credentials/inference knobs. Defaults to the historical LiteLLM+Perplexity stack."""
    auth: AuthSettings = Field(
        default_factory=AuthSettings,
        description=(
            "Authentication-mode settings for the HTTP server. "
            "Ignored in pure library mode."
        ),
    )
    """Authentication-mode settings for the HTTP server. Ignored in pure library mode."""
    knowledge: KnowledgeSettings = Field(
        default_factory=KnowledgeSettings,
        description=(
            "Knowledge-engine settings (internal document retrieval). "
            "Disabled by default."
        ),
    )
    """Knowledge-engine settings (internal document retrieval). Disabled by default."""
    storage: StorageSettings = Field(
        default_factory=StorageSettings,
        description=(
            "Persistence-backend settings for the platform layer "
            "(identity schema). Memory-backed by default."
        ),
    )
    """Persistence-backend settings for the platform layer (identity schema). Memory-backed by default."""
    queue: QueueSettings = Field(
        default_factory=QueueSettings,
        description=(
            "Run-queue backend and worker tuning. Memory-backed "
            "(in-process execution) by default."
        ),
    )
    """Run-queue backend and worker tuning. Memory-backed (in-process execution) by default."""
    quota: QuotaSettings = Field(
        default_factory=QuotaSettings,
        description=(
            "Per-user usage quotas (oidc multi-user). Disabled by "
            "default; the operator-ceiling layer of the two-level rule."
        ),
    )
    """Per-user usage quotas (oidc multi-user). Disabled by default."""
    sharing: SharingSettings = Field(
        default_factory=SharingSettings,
        description=(
            "Resource-sharing policy. Tenant-wide by default; can confine "
            "sharing to workspace co-members (cookie-session modes)."
        ),
    )
    """Resource-sharing policy (tenant-wide by default)."""
