"""Direct Amazon Bedrock adapter for the LLMProvider interface.

This module calls the Bedrock **Converse** API via ``boto3`` — no
LiteLLM proxy.  The Converse API is AWS's recommended model-agnostic
inference endpoint for Bedrock; it provides structured request/response
formats, native reasoning (thinking) support, and consistent error
handling across all supported models.

Authentication is handled through **AWS named profiles** (configured
in ``~/.aws/config`` / ``~/.aws/credentials``).  Pass the profile
name to the constructor or let boto3 resolve credentials from the
standard AWS credential chain (env vars, instance role, etc.).

Key design decisions mirror :mod:`inqtrix.providers.anthropic`:

* **Retry with jitter** — Bedrock returns ``ThrottlingException``
  under sustained load.  The provider wraps boto3 calls in its own
  retry loop with jittered exponential backoff.  boto3 built-in
  retries are **disabled** to avoid double-retry and to allow
  deadline-aware backoff.

* **Per-call reasoning control** — Extended thinking is valuable for
  complex reasoning calls but wastes tokens on short helper calls.  Like
  :class:`AnthropicLLM`, each ``complete*`` call takes a ``reasoning_effort``
  token (``""`` inherits the constructor default, ``"none"`` forces it off,
  a graded level turns it on).

* **Token-budget auto-raise** — Bedrock (Claude models) counts
  thinking tokens inside ``maxTokens``.  The provider auto-raises
  the budget to a safe minimum when thinking is enabled.

Requires ``boto3``::

    uv sync
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from inqtrix.constants import (
    DEFAULT_LLM_MAX_OUTPUT_TOKENS,
    REASONING_TIMEOUT,
)
from inqtrix.exceptions import (
    AgentModelCapacityError,
    AgentRateLimited,
    AgentTimeout,
    BedrockAPIError,
)
from inqtrix.providers.base import (
    LLMProvider,
    LLMResponse,
    StructuredLLMResponse,
    _NonFatalNoticeMixin,
    _RetryNoticeMixin,
    _THINKING_MIN_MAX_TOKENS,
    _check_deadline,
    is_model_capacity_error,
    _retry_delay_seconds,
    _sleep_before_retry,
    extract_usage_tokens,
    normalize_reasoning_effort,
    parse_structured_response_content,
    validate_reasoning_effort,
)
from inqtrix.settings import ModelSettings
from inqtrix.state import track_tokens

import boto3  # type: ignore[import-untyped]
from botocore.config import Config as BotoConfig  # type: ignore[import-untyped]
from botocore.exceptions import (  # type: ignore[import-untyped]
    ClientError,
    ConnectTimeoutError,
    ConnectionClosedError,
    ConnectionError as BotoConnectionError,
    EndpointConnectionError,
    ReadTimeoutError,
)

log = logging.getLogger("inqtrix")

# ---------------------------------------------------------------------------
# Retry & backoff constants
#
# Bedrock returns ThrottlingException under sustained load — especially
# when many parallel claim-extraction threads hit the API at once. The
# following constants control the retry loop in
# _converse_with_retry.
#
# Shared backoff constants (_BACKOFF_BASE_SECONDS, _BACKOFF_MAX_SECONDS,
# _JITTER_RANGE, _THINKING_MIN_MAX_TOKENS) are imported from base.py.
# ---------------------------------------------------------------------------
_RETRYABLE_BEDROCK_ERRORS = frozenset({
    "ThrottlingException",
    "ModelTimeoutException",
    "InternalServerException",
    "ServiceUnavailableException",
    "ModelNotReadyException",
})
_MAX_BEDROCK_ATTEMPTS = 5
_MAX_BEDROCK_TRANSPORT_ATTEMPTS = 2
_RETRYABLE_BEDROCK_TRANSPORT_ERRORS = (
    BotoConnectionError,
    ConnectionClosedError,
    EndpointConnectionError,
    ReadTimeoutError,
    ConnectTimeoutError,
)

# ---------------------------------------------------------------------------
# Effort capability blacklist
#
# Mirrors the Anthropic-direct adapter: ``output_config.effort`` is supported
# on Opus 4.5+ / Sonnet 4.6+ / Mythos, but rejected by Haiku-class models.
# The Bedrock model id format is e.g. ``eu.anthropic.claude-haiku-4-5-...``,
# so a substring match on "haiku" still works as a low-maintenance check.
# ---------------------------------------------------------------------------
_EFFORT_UNSUPPORTED_FRAGMENTS: tuple[str, ...] = ("haiku",)
_STRUCTURED_OUTPUT_MODEL_FRAGMENTS: tuple[str, ...] = (
    "claude-haiku-4-5",
    "claude-mythos",
    "claude-opus-4-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
)


def _model_supports_effort(model: str) -> bool:
    """Return True when the given Bedrock model id accepts output_config.effort."""
    if not model:
        return True
    lowered = model.lower()
    return not any(fragment in lowered for fragment in _EFFORT_UNSUPPORTED_FRAGMENTS)


# Effort levels Bedrock Converse accepts in
# ``additionalModelRequestFields["output_config"]["effort"]`` for Claude models.
# Matches the Anthropic set: no ``minimal``, but ``max`` is available. ``none``
# (off) and ``""`` (inherit) are handled separately.
_BEDROCK_EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")


def _model_supports_structured_output(model: str) -> bool:
    """Return True when the Bedrock model id advertises JSON-schema output."""
    if not model:
        return False
    lowered = model.lower()
    return any(
        fragment in lowered for fragment in _STRUCTURED_OUTPUT_MODEL_FRAGMENTS
    )


class BedrockLLM(
    _RetryNoticeMixin,
    _NonFatalNoticeMixin,
    LLMProvider,
):
    """Call Amazon Bedrock Converse directly via boto3.

    Use this provider when Claude or other supported models should run on
    Amazon Bedrock rather than on direct Anthropic or LiteLLM-backed
    infrastructure. It is the right choice when AWS regions, named
    profiles, or Bedrock-specific access controls are operational
    requirements.

    Attributes:
        _default_model (str): Primary Bedrock model ID for reasoning
            calls.
        _claim_extract_model (str): Bedrock model ID used for claim extraction.
        _default_max_tokens (int): Output-token budget for reasoning
            requests before thinking-related auto-raise.
        _temperature (float | None): Optional sampling temperature.
        _thinking (dict[str, Any] | None): Extended-thinking config applied
            on reasoning calls when the per-call ``reasoning_effort`` inherits
            the constructor default.
        _models (ModelSettings): Effective role-to-model mapping exposed
            to the runtime.
        _client (Any): boto3 Bedrock Runtime client configured without
            built-in retries.
    """

    def __init__(
        self,
        *,
        profile_name: str | None = None,
        region_name: str = "eu-central-1",
        default_model: str = "",
        classify_model: str = "",
        claim_extract_model: str = "",
        evaluate_model: str = "",
        plan_model: str = "",
        answer_model: str = "",
        direct_chat_model: str = "",
        tier_high_model: str = "",
        tier_mid_model: str = "",
        tier_fast_model: str = "",
        tier_high_effort: str = "",
        tier_mid_effort: str = "",
        tier_fast_effort: str = "",
        default_max_tokens: int = DEFAULT_LLM_MAX_OUTPUT_TOKENS,
        context_window_tokens: int | None = None,
        temperature: float | None = None,
        thinking: dict[str, Any] | None = None,
        effort: str | None = None,
        selectable_models: list[str] | None = None,
    ) -> None:
        """Initialize the Bedrock provider.

        Use the constructor when Bedrock should handle reasoning calls via
        the Converse API. The provider disables boto3 retries and runs its
        own deadline-aware retry loop, so constructor-level choices mainly
        decide credentials, region, model roles, and token budgets.

        Args:
            profile_name: Optional AWS named profile from
                ``~/.aws/config`` or ``~/.aws/credentials``. When omitted,
                boto3 falls back to the default AWS credential chain.
            region_name: AWS region for the Bedrock endpoint. The default
                is ``"eu-central-1"``.
            default_model: Primary Bedrock model ID for classify, plan,
                evaluate fallback, and answer calls.
            classify_model: Optional cheaper override for classification.
                When omitted, classification falls back to
                ``default_model``.
            claim_extract_model: Bedrock model ID used for claim extraction.
            evaluate_model: Optional override for evidence evaluation.
                When omitted, evaluation falls back to ``default_model``.
            plan_model: Optional per-node model override for the plan node.
            answer_model: Optional per-node model override for the answer node.
            direct_chat_model: Optional per-node model override for the
                skip-search direct-chat node.
            tier_high_model: Model for the high tier (answer by default).
            tier_mid_model: Model for the mid tier (plan/evaluate/direct_chat).
            tier_fast_model: Model for the fast tier (classify/claim_extract).
                Nodes map to a tier via
                ``inqtrix.model_routing.NODE_TIER_ASSIGNMENT``; empty tiers and
                per-node overrides both fall back to ``default_model``.
            tier_high_effort: Per-tier reasoning effort for the high tier.
            tier_mid_effort: Per-tier reasoning effort for the mid tier.
            tier_fast_effort: Per-tier reasoning effort for the fast tier.
                ``""`` inherits the constructor ``effort``/``thinking``,
                ``"none"`` forces reasoning off, and a graded level enables it.
            default_max_tokens: Output-token budget for reasoning calls
                before any thinking-related auto-raise.
            context_window_tokens: Known context-window size for the
                Bedrock model. ``None`` means unknown capacity.
            temperature: Optional sampling temperature. Do not set this
                together with ``thinking`` because Bedrock Claude models
                reject that combination.
            thinking: Optional Bedrock thinking configuration forwarded via
                ``additionalModelRequestFields``.
            effort: Optional ``effort`` value (``"low"``, ``"medium"``,
                ``"high"``, ``"xhigh"``, ``"max"``). Forwarded to Bedrock
                via ``additionalModelRequestFields["output_config"]``.
                Controls overall token spend (text + tool calls + thinking).
                Works with or without ``thinking`` enabled. ``"xhigh"`` is
                only supported on Claude Opus 4.7+. The Anthropic API
                default is ``"high"`` (= same as omitting the parameter).

        Raises:
            ValueError: If both ``temperature`` and ``thinking`` are set,
                or if ``effort`` is not one of the accepted values.

        Example:
            >>> from inqtrix import BedrockLLM
            >>> llm = BedrockLLM(
            ...     profile_name="default",
            ...     region_name="eu-central-1",
            ...     default_model="eu.anthropic.claude-sonnet-4-6",
            ... )
            >>> llm.models.reasoning_model
            'eu.anthropic.claude-sonnet-4-6'
        """
        if temperature is not None and thinking is not None:
            raise ValueError(
                "temperature and thinking are mutually exclusive — "
                "the Bedrock API (Claude models) rejects requests that set both."
            )
        if effort is not None and effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError(
                "effort must be one of 'low', 'medium', 'high', 'xhigh', or 'max'."
            )
        self._default_model = default_model
        # Store the raw arg (empty when unset) so the central tier router can
        # fall through claim_extract -> fast tier -> reasoning_model. Eagerly
        # pinning it to default_model here would shadow the fast tier and force
        # the expensive reasoning model on the highest-volume node.
        self._claim_extract_model = claim_extract_model
        self._default_max_tokens = default_max_tokens
        self._context_window_tokens = context_window_tokens
        self._selectable_models = list(selectable_models or [])
        self._temperature = temperature
        self._thinking = thinking
        self._effort = effort
        self._models = ModelSettings(
            reasoning_model=default_model,
            search_model="",
            classify_model=classify_model,
            claim_extract_model=self._claim_extract_model,
            evaluate_model=evaluate_model,
            plan_model=plan_model,
            answer_model=answer_model,
            direct_chat_model=direct_chat_model,
            tier_high_model=tier_high_model,
            tier_mid_model=tier_mid_model,
            tier_fast_model=tier_fast_model,
            tier_high_effort=tier_high_effort,
            tier_mid_effort=tier_mid_effort,
            tier_fast_effort=tier_fast_effort,
        )

        # Disable boto3 built-in retries — we run our own retry loop
        # with deadline enforcement and jittered backoff.
        session = boto3.Session(
            profile_name=profile_name,
            region_name=region_name,
        )
        self._client = session.client(
            "bedrock-runtime",
            config=BotoConfig(
                retries={"max_attempts": 0},
                read_timeout=180,
            ),
        )

        # Phase 12: surface effort/model incompatibilities up front (mirrors
        # AnthropicLLM). Drained by classify on the first run.
        self._effort_config_warnings: list[str] = self._collect_effort_warnings(
            default_model=default_model,
            classify_model=classify_model,
            claim_extract_model=self._claim_extract_model,
            evaluate_model=evaluate_model,
        )
        for warning in self._effort_config_warnings:
            log.warning(warning)

    def _collect_effort_warnings(
        self,
        *,
        default_model: str,
        classify_model: str,
        claim_extract_model: str,
        evaluate_model: str,
    ) -> list[str]:
        if self._effort is None:
            return []
        warnings: list[str] = []
        roles: list[tuple[str, str]] = [
            ("default_model (reasoning/answer)", default_model),
            ("classify_model", classify_model),
            ("claim_extract_model", claim_extract_model),
            ("evaluate_model", evaluate_model),
        ]
        for role, model in roles:
            if not model:
                continue
            if _model_supports_effort(model):
                continue
            warnings.append(
                f"BedrockLLM: effort='{self._effort}' wird auf {role}='{model}' "
                f"NICHT gesendet, da dieses Bedrock-Modell output_config.effort "
                f"nicht akzeptiert (Haiku-Klasse). Andere Rollen mit effort-faehigem "
                f"Modell (Opus/Sonnet 4.6+) erhalten effort weiterhin. "
                f"Setze effort=None oder waehle ein effort-faehiges Modell, um diese "
                f"Inkonsistenz aufzuloesen."
            )
        return warnings

    def consume_effort_config_warnings(self) -> list[str]:
        """Return and clear the buffered effort-config warnings.

        Side effects:
            Clears the internal warning buffer so each warning is
            surfaced exactly once. Subsequent calls return an empty
            list until new ``effort=`` mismatches are detected (e.g.
            after rotating in a Haiku-class claim-extraction model).

        Returns:
            The list of buffered German-language warning strings,
            in the order they were detected. Each entry describes a
            specific role-to-model combination where the configured
            ``effort`` parameter is silently dropped because the
            target model does not accept ``output_config.effort``
            (Haiku-class). Empty list when no mismatch has been
            detected yet.
        """
        out = list(self._effort_config_warnings)
        self._effort_config_warnings = []
        return out

    @property
    def models(self) -> ModelSettings:
        """Return the effective role-to-model mapping for the runtime.

        Returns:
            ModelSettings: Resolved Bedrock model IDs used by graph nodes.
        """
        return self._models

    @property
    def selectable_models(self) -> list[str]:
        """Return the operator-curated model ids offered for direct selection."""
        return self._selectable_models

    @property
    def context_window_tokens(self) -> int | None:
        """Return the configured context window for capacity checks."""
        return self._context_window_tokens

    @staticmethod
    def _extract_error_details(exc: ClientError) -> dict[str, Any]:
        """Extract structured error information from a boto3 ClientError."""
        response = getattr(exc, "response", None) or {}
        error = response.get("Error", {})
        metadata = response.get("ResponseMetadata", {})
        return {
            "error_code": error.get("Code", ""),
            "message": error.get("Message", ""),
            "status_code": metadata.get("HTTPStatusCode"),
            "request_id": metadata.get("RequestId"),
        }

    @staticmethod
    def _build_api_error(
        *,
        model: str,
        details: dict[str, Any] | None = None,
        message: str = "",
        original: Exception,
    ) -> BedrockAPIError:
        details = details or {}
        sc = details.get("status_code")
        return BedrockAPIError(
            model=model,
            error_code=str(details.get("error_code") or "").strip(),
            status_code=sc if isinstance(sc, int) else None,
            message=message.strip() or str(details.get("message") or "").strip() or str(original),
            request_id=str(details.get("request_id") or "").strip() or None,
            original=original,
        )

    def _converse_with_retry(
        self,
        *,
        params: dict[str, Any],
        deadline: float | None = None,
        operation: str = "converse",
    ) -> dict[str, Any]:
        """Call Bedrock Converse with retry and deadline enforcement.

        Implements jittered exponential backoff for transient errors
        (ThrottlingException, ServiceUnavailableException, etc.).
        The client-level ``read_timeout`` (set at creation) prevents
        individual calls from hanging; deadline enforcement handles
        the tighter agent time budget.
        """
        self._clear_retry_notices()
        use_model = str(params.get("modelId") or self._default_model)

        for attempt in range(_MAX_BEDROCK_ATTEMPTS):
            if deadline is not None:
                _check_deadline(deadline)

            try:
                response = self._client.converse(**params)
                return response if isinstance(response, dict) else {}
            except ClientError as exc:
                details = self._extract_error_details(exc)
                error_code = details.get("error_code", "")

                if error_code == "ThrottlingException":
                    if attempt >= (_MAX_BEDROCK_ATTEMPTS - 1):
                        raise AgentRateLimited(use_model, exc) from exc
                    # Retryable — fall through to backoff below.

                api_error = self._build_api_error(
                    model=use_model, details=details, original=exc)
                if is_model_capacity_error(api_error):
                    log.warning("ALGO-FAIL model_capacity (%s): %s", use_model, api_error)
                    raise AgentModelCapacityError(
                        use_model,
                        "llm_complete",
                        str(api_error),
                        original=api_error,
                    ) from exc

                if error_code in _RETRYABLE_BEDROCK_ERRORS and attempt < (_MAX_BEDROCK_ATTEMPTS - 1):
                    delay = _retry_delay_seconds(attempt)
                    self._append_retry_notice({
                        "provider": "Bedrock",
                        "model": use_model,
                        "operation": operation,
                        "error_code": error_code,
                        "attempt": attempt + 1,
                        "max_attempts": _MAX_BEDROCK_ATTEMPTS,
                        "delay_seconds": round(delay, 3),
                    })
                    log.warning(
                        "Bedrock transient error (%s, code=%s, request-id=%s, attempt=%d/%d). Retrying in %.2fs.",
                        use_model,
                        error_code,
                        details.get("request_id") or "-",
                        attempt + 1,
                        _MAX_BEDROCK_ATTEMPTS,
                        delay,
                    )
                    _sleep_before_retry(delay, deadline)
                    continue

                raise api_error from exc
            except _RETRYABLE_BEDROCK_TRANSPORT_ERRORS as exc:
                details = {"error_code": type(exc).__name__}
                api_error = self._build_api_error(
                    model=use_model,
                    details=details,
                    original=exc,
                )
                if attempt < (_MAX_BEDROCK_TRANSPORT_ATTEMPTS - 1):
                    delay = _retry_delay_seconds(attempt)
                    self._append_retry_notice({
                        "provider": "Bedrock",
                        "model": use_model,
                        "operation": operation,
                        "error_code": type(exc).__name__,
                        "attempt": attempt + 1,
                        "max_attempts": _MAX_BEDROCK_TRANSPORT_ATTEMPTS,
                        "delay_seconds": round(delay, 3),
                    })
                    log.warning(
                        "Bedrock transport error (%s, attempt=%d/%d). Retrying in %.2fs: %s",
                        use_model,
                        attempt + 1,
                        _MAX_BEDROCK_TRANSPORT_ATTEMPTS,
                        delay,
                        exc,
                    )
                    _sleep_before_retry(delay, deadline)
                    continue
                raise api_error from exc

        raise self._build_api_error(  # pragma: no cover
            model=use_model,
            message="Bedrock request exhausted retries without a final response.",
            original=RuntimeError("retries exhausted"),
        )

    # ------------------------------------------------------------------ #
    # Response extraction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_text(response: dict[str, Any]) -> str:
        """Extract visible text from a Bedrock Converse response.

        Collects only ``{"text": ...}`` content blocks — reasoning
        (thinking) content blocks are intentionally skipped.
        """
        parts: list[str] = []
        output = response.get("output", {})
        message = output.get("message", {}) if isinstance(output, dict) else {}
        content = message.get("content", []) if isinstance(message, dict) else []
        for block in content or []:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    @staticmethod
    def _extract_usage(response: dict[str, Any]) -> tuple[int, int]:
        return extract_usage_tokens(
            response,
            input_keys=("inputTokens",),
            output_keys=("outputTokens",),
        )

    # ------------------------------------------------------------------ #
    # LLMProvider interface
    # ------------------------------------------------------------------ #

    def _call_thinking_and_effort(
        self, reasoning_effort: str | None, *, use_model: str
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Resolve the (thinking, effort) to apply for a single Converse call.

        Same contract as the Anthropic provider: ``None``/``""`` inherits the
        constructor ``self._thinking`` / ``self._effort``; ``"none"`` forces
        reasoning off; a graded level turns on adaptive thinking and sets
        ``output_config`` effort where the model accepts it (Haiku class gets
        adaptive thinking without an effort cap and a visible warning).

        Args:
            reasoning_effort: The per-call effort override.
            use_model: The effective Bedrock model id for this call.

        Returns:
            A ``(thinking, effort)`` tuple destined for
            ``additionalModelRequestFields``.
        """
        token = normalize_reasoning_effort(reasoning_effort)
        if token == "":
            thinking = self._thinking if self._thinking is not None else None
            effort = (
                self._effort
                if (self._effort is not None and _model_supports_effort(use_model))
                else None
            )
            return thinking, effort
        if token == "none":
            return None, None
        effort, warnings = validate_reasoning_effort(
            token,
            supported_levels=_BEDROCK_EFFORT_LEVELS,
            label=f"BedrockLLM({use_model})",
        )
        for warning in warnings:
            log.warning("CONFIG: %s", warning)
        if effort in ("", "none"):
            return None, None
        if _model_supports_effort(use_model):
            return {"type": "adaptive"}, effort
        log.warning(
            "CONFIG: BedrockLLM: reasoning_effort=%r requested on model=%s, "
            "which does not accept output_config.effort (Haiku class); using "
            "adaptive thinking without an effort cap.",
            token,
            use_model,
        )
        return {"type": "adaptive"}, None

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        timeout: float = REASONING_TIMEOUT,
        state: dict | None = None,
        deadline: float | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        """Generate text through Bedrock and discard token metadata.

        Args:
            prompt: User-facing input text.
            system: Optional system instruction.
            model: Optional per-call Bedrock model override. When omitted,
                the provider uses ``self._default_model``.
            timeout: Per-call timeout budget in seconds.
            state: Optional mutable agent state for token tracking.
            deadline: Optional absolute monotonic deadline for the full
                run.

        Returns:
            str: Visible assistant text extracted from the Converse
            response.

        Raises:
            AgentTimeout: If the full run deadline has elapsed.
            AgentRateLimited: If Bedrock throttles the request fatally.
            BedrockAPIError: If a non-retryable Bedrock error occurs.
        """
        return self.complete_with_metadata(
            prompt,
            system=system,
            model=model,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
            state=state,
            deadline=deadline,
            reasoning_effort=reasoning_effort,
        ).content

    def complete_with_metadata(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        timeout: float = REASONING_TIMEOUT,
        state: dict | None = None,
        deadline: float | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """Generate text and token metadata through Bedrock Converse.

        Use this method for reasoning calls when the runtime wants both
        visible content and token usage. When thinking is enabled, the
        method raises ``maxTokens`` so hidden reasoning does not crowd out
        the visible answer. Request execution itself is delegated to
        ``_converse_with_retry()``, which owns deadline-aware retry logic
        for throttling and transient Bedrock failures.

        Args:
            prompt: User-facing input text.
            system: Optional system instruction.
            model: Optional per-call model override. When omitted, the
                provider uses ``self._default_model``.
            timeout: Per-call timeout budget in seconds. Bedrock uses the
                provider's retry loop instead of a dedicated per-request
                timeout field.
            state: Optional mutable agent state that receives token counts
                through ``track_tokens()`` when provided.
            deadline: Optional absolute monotonic deadline for the full
                run.

        Returns:
            LLMResponse: Structured response containing visible content,
            token counts, and the effective model label.

        Raises:
            AgentTimeout: If the full run deadline has already elapsed.
            AgentRateLimited: If Bedrock throttles the request after retry
                exhaustion.
            BedrockAPIError: If a non-retryable Bedrock or transport error
                occurs.
        """
        if deadline is not None:
            _check_deadline(deadline)

        use_model = model or self._default_model
        max_tokens = max_output_tokens or self._default_max_tokens
        thinking_cfg, effort_value = self._call_thinking_and_effort(
            reasoning_effort, use_model=use_model
        )
        use_thinking = thinking_cfg is not None
        if use_thinking:
            budget = thinking_cfg.get("budget_tokens")
            if isinstance(budget, int) and budget >= max_tokens:
                max_tokens = budget + 1024
            if max_tokens < _THINKING_MIN_MAX_TOKENS:
                log.debug(
                    "maxTokens auto-raised from %d to %d (thinking enabled)",
                    max_tokens,
                    _THINKING_MIN_MAX_TOKENS,
                )
                max_tokens = _THINKING_MIN_MAX_TOKENS

        params: dict[str, Any] = {
            "modelId": use_model,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if system:
            params["system"] = [{"text": system}]
        if self._temperature is not None:
            params["inferenceConfig"]["temperature"] = self._temperature
        # Bedrock Converse routes Anthropic-specific fields through
        # additionalModelRequestFields; both `thinking` and `output_config`
        # (which carries `effort`) live in the same dict. The per-call resolver
        # already dropped effort for models that reject it (Haiku class) and for
        # none / inherit-without-effort calls.
        # See https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        additional_fields: dict[str, Any] = {}
        if use_thinking:
            additional_fields["thinking"] = thinking_cfg
        if effort_value is not None:
            additional_fields["output_config"] = {"effort": effort_value}
        if additional_fields:
            params["additionalModelRequestFields"] = additional_fields

        raw = self._converse_with_retry(
            params=params,
            deadline=deadline,
            operation="complete",
        )

        prompt_tokens, completion_tokens = self._extract_usage(raw)
        response = LLMResponse(
            content=self._extract_text(raw),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=use_model,
            finish_reason=str(raw.get("stopReason") or raw.get("stop_reason") or ""),
            raw=raw,
            request_max_tokens=int(
                params.get("inferenceConfig", {}).get("maxTokens") or 0
            ),
        )
        if state is not None:
            track_tokens(state, response)
        return response

    def supports_structured_output(self, *, model: str | None = None) -> bool:
        """Return whether the selected Bedrock model supports JSON schemas.

        Args:
            model: Optional model override. When omitted, the provider's
                default reasoning model is checked.

        Returns:
            ``True`` for Bedrock Claude model ids documented to support
            Converse ``outputConfig.textFormat`` structured outputs.
        """
        return _model_supports_structured_output(model or self._default_model)

    def complete_structured(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        schema_name: str,
        schema_description: str = "",
        system: str | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        timeout: float = REASONING_TIMEOUT,
        state: dict | None = None,
        deadline: float | None = None,
        reasoning_effort: str | None = None,
    ) -> StructuredLLMResponse:
        """Generate a JSON-schema-constrained Bedrock Converse response.

        Args:
            prompt: User-facing input text.
            schema: JSON Schema object sent through
                ``outputConfig.textFormat``.
            schema_name: Stable schema name used by Bedrock grammar
                caching and local diagnostics.
            schema_description: Optional schema purpose shown to Bedrock.
            system: Optional system instruction.
            model: Optional Bedrock model override.
            max_output_tokens: Optional output-token budget.
            timeout: Per-call timeout accepted for interface parity; the
                Bedrock client uses its configured read timeout and the
                shared retry loop.
            state: Optional mutable token-accounting state.
            deadline: Optional absolute monotonic deadline.

        Returns:
            StructuredLLMResponse with parsed top-level JSON object.

        Raises:
            AgentTimeout: If the full run deadline has elapsed.
            AgentRateLimited: If Bedrock throttles the request fatally.
            BedrockAPIError: If a non-retryable Bedrock error occurs.
            AgentStructuredOutputError: If the visible structured JSON
                cannot be parsed into a dictionary.
        """
        del timeout
        if deadline is not None:
            _check_deadline(deadline)

        use_model = model or self._default_model
        max_tokens = max_output_tokens or self._default_max_tokens
        thinking_cfg, effort_value = self._call_thinking_and_effort(
            reasoning_effort, use_model=use_model
        )
        use_thinking = thinking_cfg is not None
        if use_thinking:
            budget = thinking_cfg.get("budget_tokens")
            if isinstance(budget, int) and budget >= max_tokens:
                max_tokens = budget + 1024
            if max_tokens < _THINKING_MIN_MAX_TOKENS:
                log.debug(
                    "maxTokens auto-raised from %d to %d (thinking enabled)",
                    max_tokens,
                    _THINKING_MIN_MAX_TOKENS,
                )
                max_tokens = _THINKING_MIN_MAX_TOKENS

        params: dict[str, Any] = {
            "modelId": use_model,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens},
            "outputConfig": {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "schema": json.dumps(schema, ensure_ascii=False),
                            "name": schema_name,
                            "description": schema_description or schema_name,
                        }
                    },
                }
            },
        }
        if system:
            params["system"] = [{"text": system}]
        if self._temperature is not None:
            params["inferenceConfig"]["temperature"] = self._temperature

        additional_fields: dict[str, Any] = {}
        if use_thinking:
            additional_fields["thinking"] = thinking_cfg
        if effort_value is not None:
            additional_fields["output_config"] = {"effort": effort_value}
        if additional_fields:
            params["additionalModelRequestFields"] = additional_fields

        raw = self._converse_with_retry(
            params=params,
            deadline=deadline,
            operation="structured_response",
        )

        prompt_tokens, completion_tokens = self._extract_usage(raw)
        content = self._extract_text(raw)
        response = StructuredLLMResponse(
            parsed=parse_structured_response_content(
                content,
                model=use_model,
                schema_name=schema_name,
            ),
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=use_model,
            finish_reason=str(raw.get("stopReason") or raw.get("stop_reason") or ""),
            raw=raw,
            request_max_tokens=int(
                params.get("inferenceConfig", {}).get("maxTokens") or 0
            ),
            schema_name=schema_name,
        )
        if state is not None:
            track_tokens(state, response)
        return response

    def is_available(self) -> bool:
        """Report whether the provider is configured to attempt requests.

        Configuration here means: a boto3 ``bedrock-runtime`` client
        was successfully constructed. AWS credential resolution
        (profile, IAM role, env vars) happens lazily inside boto3 on
        the first request, so a successful constructor does not
        guarantee that credentials are valid — invalid credentials
        surface as :class:`BedrockAPIError` with
        ``error_code="UnrecognizedClientException"`` on the first
        ``complete()`` call.

        Returns:
            ``True`` when the internal Bedrock Runtime client was
            constructed, otherwise ``False``.
        """
        return self._client is not None
