"""Provider abstractions, response types, and shared infrastructure.

Defines the abstract base classes :class:`LLMProvider` and
:class:`SearchProvider`, frozen response dataclasses, deadline helpers,
the :class:`_NonFatalNoticeMixin` for thread-safe fallback notices, the
:class:`ConfiguredLLMProvider` adapter, and the :class:`ProviderContext`
container.

All concrete provider implementations (LiteLLM, Anthropic, Bedrock, etc.)
import their base contracts and shared utilities from this module.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from inqtrix.exceptions import AgentTimeout

if TYPE_CHECKING:
    from inqtrix.settings import ModelSettings

log = logging.getLogger("inqtrix")

# =========================================================
# Retry budget for OpenAI-SDK-based providers
# =========================================================
# The OpenAI Python SDK retries 408/409/429/500/502/503/504 with
# exponential backoff + jitter.  Its default of 2 retries (3 total
# attempts) is too low for Bedrock / proxy throttling ("Too many
# connections").  4 retries = 5 total attempts, matching the direct
# Anthropic provider's _MAX_ANTHROPIC_ATTEMPTS.
_SDK_MAX_RETRIES = 4

# =========================================================
# Shared retry / backoff constants (Anthropic & Bedrock)
# =========================================================
# Exponential backoff — delay doubles each attempt (1, 2, 4, 8, 8, …)
# but is capped at _BACKOFF_MAX_SECONDS.  Jitter spreads concurrent
# retries to avoid thundering-herd bursts against rate-limited APIs.
_BACKOFF_BASE_SECONDS: float = 1.0
_BACKOFF_MAX_SECONDS: float = 8.0
_JITTER_RANGE: tuple[float, float] = (0.5, 1.5)

# Floor for max_tokens when extended thinking is enabled.  Anthropic
# counts thinking tokens *inside* max_tokens, so a low budget leaves
# almost nothing for the visible answer.
_THINKING_MIN_MAX_TOKENS: int = 16_384

_SEARCH_PARAMETER_NAMES = frozenset({
    "search_context_size",
    "recency_filter",
    "language_filter",
    "domain_filter",
    "search_mode",
    "return_related",
})

# =========================================================
# Internal deadline helpers
# =========================================================


def _check_deadline(deadline: float) -> None:
    """Raise AgentTimeout when the deadline has passed."""
    if time.monotonic() > deadline:
        raise AgentTimeout(
            "Agent-Zeitlimit ueberschritten. "
            "Antwort wird mit bisherigem Kontext generiert."
        )


def _bounded_timeout(
    default_timeout: int | float, deadline: float | None = None
) -> float:
    """Clamp API timeout to the remaining agent time budget."""
    if deadline is None:
        return float(default_timeout)
    _check_deadline(deadline)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AgentTimeout(
            "Agent-Zeitlimit ueberschritten. "
            "Antwort wird mit bisherigem Kontext generiert."
        )
    return min(float(default_timeout), remaining)


class _NonFatalNoticeMixin:
    """Thread-local helper for surfacing provider fallback notices."""

    _nonfatal_init_lock = threading.Lock()

    def _notice_state(self) -> threading.local:
        """Return the thread-local state object, creating it on first access."""
        state = getattr(self, "_nonfatal_notice_state", None)
        if state is None:
            with self._nonfatal_init_lock:
                state = getattr(self, "_nonfatal_notice_state", None)
                if state is None:
                    state = threading.local()
                    self._nonfatal_notice_state = state
        return state

    def _set_nonfatal_notice(self, message: str) -> None:
        """Store a fallback notice for the current thread."""
        if message:
            self._notice_state().message = message

    def _clear_nonfatal_notice(self) -> None:
        """Clear any pending notice for the current thread."""
        state = self._notice_state()
        if hasattr(state, "message"):
            delattr(state, "message")

    def consume_nonfatal_notice(self) -> str | None:
        """Return and clear the pending notice, or ``None`` if none exists."""
        state = self._notice_state()
        message = getattr(state, "message", None)
        if hasattr(state, "message"):
            delattr(state, "message")
        return str(message) if message else None


@dataclass(frozen=True, slots=True)
class SearchProviderCapabilities:
    """Describe which generic search hints a provider supports.

    This is an additive helper for the Baukastensystem: providers may
    expose a capability object without changing the stable
    :class:`SearchProvider` contract. Callers should treat omitted
    capability metadata as "supports everything" to preserve backward
    compatibility with existing providers and custom user adapters.

    Attributes:
        supported_parameters: Search-hint names accepted by the backend.
            Valid values correspond to the optional keyword parameters of
            :meth:`SearchProvider.search` except ``deadline``, which is
            always handled separately.
    """

    supported_parameters: frozenset[str] = _SEARCH_PARAMETER_NAMES

    def supports(self, parameter_name: str) -> bool:
        """Return whether the provider accepts a generic search hint."""
        return parameter_name in self.supported_parameters


def get_search_provider_capabilities(provider: object) -> SearchProviderCapabilities:
    """Resolve optional capability metadata from a search provider.

    Providers may expose either a ``search_capabilities`` attribute that
    returns a :class:`SearchProviderCapabilities` instance or a simpler
    ``supported_search_parameters`` attribute returning an iterable of
    accepted parameter names. When neither is present, the default is to
    preserve today's behavior and assume all generic hints are allowed.
    """

    raw_capabilities = getattr(provider, "search_capabilities", None)
    if callable(raw_capabilities):
        raw_capabilities = raw_capabilities()
    if isinstance(raw_capabilities, SearchProviderCapabilities):
        return raw_capabilities

    raw_supported = getattr(provider, "supported_search_parameters", None)
    if callable(raw_supported):
        raw_supported = raw_supported()
    if raw_supported is None:
        return SearchProviderCapabilities()

    try:
        supported = frozenset(
            str(name).strip()
            for name in raw_supported
            if str(name).strip()
        )
    except TypeError:
        return SearchProviderCapabilities()

    return SearchProviderCapabilities(
        supported_parameters=supported or _SEARCH_PARAMETER_NAMES
    )


class ThinkingSuppressionMixin:
    """Re-entrant, thread-safe suppression of extended thinking AND effort.

    Subclasses must set ``self._thinking`` (the thinking config dict or
    ``None``) and ``self._thread_state`` (a ``threading.local()``) in
    their ``__init__``. They may optionally set ``self._effort`` (a string
    or ``None``).

    The ``without_thinking`` context manager increments a thread-local
    depth counter.  While depth > 0, ``_thinking_enabled()`` AND
    ``_effort_enabled()`` both return ``False``, so the provider omits
    BOTH the ``thinking`` and the ``effort``/``output_config`` fields
    from API payloads.

    Both fields are suppressed together because:

    1. They have the same intent (token-spend control for high-cost
       reasoning calls). Helper paths — claim extraction, parallel
       summarization — should stay lean and fast.
    2. ``effort`` is not supported on every model. For example, Claude
       Haiku 4.5 (a typical summarize/claim-extract model) **rejects**
       ``output_config.effort`` with HTTP 400, while Claude Opus and
       Sonnet 4.6+ accept it. Sending ``effort`` through helper threads
       therefore triggers per-source 400-failures that look like
       transient errors but are actually parameter incompatibility.

    This is re-entrant and thread-safe: parallel helper threads each
    have their own counter.
    """

    _thinking: dict | None
    _thread_state: threading.local

    @contextmanager
    def without_thinking(self):
        """Suppress extended thinking for the duration of the ``with`` block.

        Use this around classify or evaluate calls when the configured
        reasoning model has extended thinking enabled but the helper
        path does not benefit from it (and would just spend tokens).
        Re-entrant: nested ``with`` blocks share a per-thread counter
        so suppression survives until the outermost block exits.

        Yields:
            The provider instance itself, so the context manager can
            be used as ``with provider.without_thinking() as p:`` when
            the caller wants to bind the provider to a local name.

        Side effects:
            Mutates a per-thread suppression counter on
            ``self._thread_state``. The counter is removed entirely
            when the outermost block exits so a new thread sees a
            fresh state.
        """
        depth = int(getattr(self._thread_state, "suppress_thinking_depth", 0))
        self._thread_state.suppress_thinking_depth = depth + 1
        try:
            yield self
        finally:
            if depth:
                self._thread_state.suppress_thinking_depth = depth
            else:
                try:
                    delattr(self._thread_state, "suppress_thinking_depth")
                except AttributeError:
                    pass

    def _thinking_enabled(self) -> bool:
        return self._thinking is not None and int(
            getattr(self._thread_state, "suppress_thinking_depth", 0)
        ) == 0

    def _effort_enabled(self) -> bool:
        """Whether ``effort`` should be forwarded to the backend right now.

        Returns ``True`` only when the provider has a non-``None``
        ``self._effort`` AND no ``without_thinking`` block is currently
        active on this thread. Suppressing effort in helper paths avoids
        per-call 400-failures on models that do not support it (e.g.
        Claude Haiku 4.5).
        """
        return (
            getattr(self, "_effort", None) is not None
            and int(
                getattr(self._thread_state, "suppress_thinking_depth", 0)
            ) == 0
        )


# =========================================================
# Shared retry helpers (Anthropic & Bedrock)
# =========================================================


def _retry_delay_seconds(
    attempt: int, retry_after: str | None = None
) -> float:
    """Compute retry delay with exponential backoff and jitter.

    If the server sent a ``Retry-After`` header, honour it.
    Otherwise use exponential backoff (base * 2^attempt, capped)
    multiplied by a random jitter factor to desynchronise parallel
    threads.
    """
    if retry_after:
        try:
            parsed = float(retry_after)
        except ValueError:
            parsed = 0.0
        if parsed > 0:
            return parsed
    base = min(_BACKOFF_BASE_SECONDS * (2 ** attempt), _BACKOFF_MAX_SECONDS)
    return base * random.uniform(*_JITTER_RANGE)


def _sleep_before_retry(
    delay: float, deadline: float | None = None
) -> None:
    """Sleep for *delay* seconds, clamped to the remaining deadline."""
    if delay <= 0:
        return
    if deadline is not None:
        _check_deadline(deadline)
        delay = min(delay, max(0.0, deadline - time.monotonic()))
    if delay > 0:
        time.sleep(delay)


# =========================================================
# Response dataclasses
# =========================================================


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Structured wrapper around an LLM completion result.

    Attributes:
        content: Visible assistant text.
        prompt_tokens: Input tokens billed by the backend.
        completion_tokens: Output tokens billed by the backend. For Anthropic
            with extended thinking enabled this includes both visible and
            hidden (thinking) tokens — there is no separate breakdown.
        model: Effective model identifier reported by the backend.
        finish_reason: Provider-specific stop signal (``stop``, ``end_turn``,
            ``length``, ...).
        raw: Original payload for debugging.
        request_max_tokens: The max-output budget the provider actually sent
            to the backend, **after** any clamping (e.g. Anthropic's
            thinking auto-raise to 16384). Use this rather than the caller's
            requested ``max_output_tokens`` when computing token-utilization
            ratios — otherwise thinking models appear to run "over budget"
            even though they finished freely.  ``0`` means the caller did
            not pass a budget and the provider default (or backend default)
            was used.
    """

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    finish_reason: str = ""
    raw: dict[str, Any] | None = None
    request_max_tokens: int = 0


@dataclass(frozen=True, slots=True)
class SummarizeOptions:
    """Optional tuning values for helper summarization calls."""

    prompt_template: str | None = None
    input_char_limit: int | None = None
    fallback_char_limit: int | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class _NormalizedCompletion:
    """Internal normalized representation of a chat completion."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    raw: dict[str, Any] | None = None


def _extract_content_value(content: Any) -> str:
    """Extract text from a content field that may be a string, list, or object."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


def _extract_choice_content(choice: Any) -> str:
    """Extract text content from a chat completion choice (message or delta)."""
    for field_name in ("message", "delta"):
        if isinstance(choice, dict):
            payload = choice.get(field_name)
        else:
            payload = getattr(choice, field_name, None)
        if payload is None:
            continue
        if isinstance(payload, dict):
            content = payload.get("content")
        else:
            content = getattr(payload, "content", None)
        text = _extract_content_value(content)
        if text:
            return text
    return ""


def _extract_choice_finish_reason(choice: Any) -> str:
    """Extract the finish reason from a chat completion choice."""
    if isinstance(choice, dict):
        finish_reason = choice.get("finish_reason")
    else:
        finish_reason = getattr(choice, "finish_reason", None)
    if isinstance(finish_reason, str):
        return finish_reason
    return ""


def _extract_usage_from_payload(payload: dict[str, Any]) -> tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from a dict payload."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return (0, 0)
    return (
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
    )


def extract_usage_tokens(
    payload: dict[str, Any] | None,
    *,
    input_keys: tuple[str, ...] = ("prompt_tokens", "input_tokens", "inputTokens"),
    output_keys: tuple[str, ...] = ("completion_tokens", "output_tokens", "outputTokens"),
    usage_key: str = "usage",
) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from a payload ``usage`` block.

    Supports OpenAI-style (``prompt_tokens``/``completion_tokens``),
    Anthropic-style (``input_tokens``/``output_tokens``), and Bedrock-style
    (``inputTokens``/``outputTokens``) key variants. Returns the first key
    in each tuple that yields a positive integer.
    """
    if not isinstance(payload, dict):
        return (0, 0)
    usage = payload.get(usage_key)
    if not isinstance(usage, dict):
        return (0, 0)

    def _first_int(keys: tuple[str, ...]) -> int:
        for key in keys:
            val = usage.get(key)
            if val:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    continue
        return 0

    return (_first_int(input_keys), _first_int(output_keys))


@dataclass(slots=True)
class SummarizePreamble:
    """Result of the shared summarize pre-flight sanity checks."""
    prompt_template: str
    input_char_limit: int
    fallback_char_limit: int
    max_output_tokens: int | None
    truncated_text: str
    ready: bool


def prepare_summarize_call(
    text: str,
    options: "SummarizeOptions | None",
    *,
    default_prompt: str,
    default_input_char_limit: int = 6000,
    default_fallback_char_limit: int = 800,
    default_max_output_tokens: int | None = None,
    deadline: float | None = None,
    notice_mixin: "_NonFatalNoticeMixin | None" = None,
) -> SummarizePreamble:
    """Shared summarize preamble: clear notice, derive options, check deadline.

    Returns a ``SummarizePreamble`` whose ``ready`` flag indicates whether
    the caller should proceed with the API request. When text is empty the
    preamble returns ``ready=False`` so the caller can short-circuit with a
    (``""``, 0, 0) tuple without duplicating the option-derivation code.
    """
    if notice_mixin is not None:
        notice_mixin._clear_nonfatal_notice()

    prompt_template = (
        options.prompt_template if options and options.prompt_template else default_prompt
    )
    input_char_limit = (
        options.input_char_limit if options and options.input_char_limit
        else default_input_char_limit
    )
    fallback_char_limit = (
        options.fallback_char_limit if options and options.fallback_char_limit
        else default_fallback_char_limit
    )
    max_output_tokens = (
        options.max_output_tokens if options and options.max_output_tokens is not None
        else default_max_output_tokens
    )

    if not text.strip():
        return SummarizePreamble(
            prompt_template=prompt_template,
            input_char_limit=input_char_limit,
            fallback_char_limit=fallback_char_limit,
            max_output_tokens=max_output_tokens,
            truncated_text="",
            ready=False,
        )
    if deadline is not None:
        _check_deadline(deadline)

    return SummarizePreamble(
        prompt_template=prompt_template,
        input_char_limit=input_char_limit,
        fallback_char_limit=fallback_char_limit,
        max_output_tokens=max_output_tokens,
        truncated_text=text[:input_char_limit],
        ready=True,
    )


def summarize_fallback_on_error(
    notice_mixin: "_NonFatalNoticeMixin",
    *,
    provider_label: str,
    model: str,
    fallback_char_limit: int,
    raw_text: str,
    exc: Exception,
) -> tuple[str, int, int]:
    """Standard summarize fallback: record notice, log, return truncated text."""
    notice_mixin._set_nonfatal_notice(
        f"{provider_label}-Summarize fehlgeschlagen ({model}); Fallback auf Rohtext."
    )
    log.error("%s-Summarize fehlgeschlagen (%s): %s", provider_label, model, exc)
    return (raw_text[:fallback_char_limit], 0, 0)


def _extract_finish_reason_from_payload(payload: dict[str, Any]) -> str:
    """Extract the first non-empty finish reason from a completion payload."""
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return ""
    for choice in choices:
        finish_reason = _extract_choice_finish_reason(choice)
        if finish_reason:
            return finish_reason
    return ""


def _extract_usage_from_response(response: Any) -> tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from an SDK response object."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return (0, 0)
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


def _extract_finish_reason_from_response(response: Any) -> str:
    """Extract the first non-empty finish reason from an SDK response object."""
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    for choice in choices:
        finish_reason = _extract_choice_finish_reason(choice)
        if finish_reason:
            return finish_reason
    return ""


def _normalize_completion_payload(payload: dict[str, Any]) -> _NormalizedCompletion:
    """Normalize a JSON chat completion payload into a ``_NormalizedCompletion``."""
    content_parts: list[str] = []
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            text = _extract_choice_content(choice)
            if text:
                content_parts.append(text)
    prompt_tokens, completion_tokens = _extract_usage_from_payload(payload)
    finish_reason = _extract_finish_reason_from_payload(payload)
    return _NormalizedCompletion(
        content="".join(content_parts),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        raw=payload,
    )


def _normalize_text_completion(response: str) -> _NormalizedCompletion:
    """Normalize a raw text response (plain JSON or SSE stream) into a ``_NormalizedCompletion``."""
    response_text = response.strip()
    if not response_text:
        return _NormalizedCompletion(content="", raw={})

    if not response_text.startswith("data:"):
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            return _NormalizedCompletion(content=response_text, raw={})
        if isinstance(payload, dict):
            return _normalize_completion_payload(payload)
        return _NormalizedCompletion(content=response_text, raw={})

    content_parts: list[str] = []
    citations: list[str] = []
    related_questions: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    finish_reason = ""
    model = ""

    for line in response.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload_text = line[5:].strip()
        if not payload_text or payload_text == "[DONE]":
            continue
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        normalized = _normalize_completion_payload(payload)
        if normalized.content:
            content_parts.append(normalized.content)
        if normalized.prompt_tokens or normalized.completion_tokens:
            prompt_tokens = normalized.prompt_tokens
            completion_tokens = normalized.completion_tokens
        if normalized.finish_reason:
            finish_reason = normalized.finish_reason

        payload_citations = payload.get("citations")
        if isinstance(payload_citations, list):
            for citation in payload_citations:
                text = str(citation)
                if text and text not in citations:
                    citations.append(text)

        payload_related = payload.get("related_questions")
        if isinstance(payload_related, list):
            for question in payload_related:
                text = str(question)
                if text and text not in related_questions:
                    related_questions.append(text)

        payload_model = payload.get("model")
        if isinstance(payload_model, str) and payload_model:
            model = payload_model

    raw: dict[str, Any] = {}
    if citations:
        raw["citations"] = citations
    if related_questions:
        raw["related_questions"] = related_questions
    if model:
        raw["model"] = model
    if prompt_tokens or completion_tokens:
        raw["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    if finish_reason:
        raw["finish_reason"] = finish_reason

    return _NormalizedCompletion(
        content="".join(content_parts),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        raw=raw,
    )


def _normalize_completion_response(response: Any) -> _NormalizedCompletion:
    """Normalize any completion response (str, SDK object, or dict) into a ``_NormalizedCompletion``."""
    if isinstance(response, str):
        return _normalize_text_completion(response)

    content_parts: list[str] = []
    choices = getattr(response, "choices", None)
    if choices:
        for choice in choices:
            text = _extract_choice_content(choice)
            if text:
                content_parts.append(text)

    prompt_tokens, completion_tokens = _extract_usage_from_response(response)
    finish_reason = _extract_finish_reason_from_response(response)
    raw: dict[str, Any] = {}
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except TypeError:
            dumped = None
        if isinstance(dumped, dict):
            raw = dumped
            if not content_parts:
                normalized = _normalize_completion_payload(dumped)
                if normalized.content:
                    content_parts.append(normalized.content)
                if normalized.finish_reason and not finish_reason:
                    finish_reason = normalized.finish_reason
            if not prompt_tokens and not completion_tokens:
                prompt_tokens, completion_tokens = _extract_usage_from_payload(dumped)
            if not finish_reason:
                finish_reason = _extract_finish_reason_from_payload(dumped)

    return _NormalizedCompletion(
        content="".join(content_parts),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        raw=raw,
    )


# =========================================================
# Shared search helpers
# =========================================================


def _apply_domain_filters(query: str, domain_filter: list[str] | None) -> str:
    """Inject ``site:`` / ``-site:`` operators into the query string."""
    if not domain_filter:
        return query

    suffix_parts: list[str] = []
    for raw_domain in domain_filter[:10]:
        domain = str(raw_domain or "").strip()
        if not domain:
            continue
        is_exclusion = domain.startswith("-")
        if is_exclusion:
            domain = domain[1:].strip()
        if not domain:
            continue
        token = f"site:{domain}"
        if is_exclusion:
            token = f"-site:{domain}"
        suffix_parts.append(token)

    if not suffix_parts:
        return query
    return f"{query} {' '.join(suffix_parts)}"


def _build_recency_language_hints(
    recency_filter: str | None,
    language_filter: list[str] | None,
) -> str | None:
    """Build best-effort recency/language hints for search agents."""
    parts: list[str] = []

    recency = (recency_filter or "").strip().lower()
    if recency == "hour":
        parts.append("Fokussiere dich ausschliesslich auf Ergebnisse der letzten Stunde.")
    elif recency == "day":
        parts.append("Fokussiere dich auf Ergebnisse der letzten 24 Stunden.")
    elif recency == "week":
        parts.append("Fokussiere dich auf Ergebnisse der letzten Woche.")
    elif recency == "month":
        parts.append("Fokussiere dich auf Ergebnisse des letzten Monats.")
    elif recency == "year":
        parts.append("Fokussiere dich auf Ergebnisse des letzten Jahres.")

    if language_filter:
        lang = language_filter[0]
        parts.append(
            f"Antworte auf {lang} und bevorzuge Quellen in dieser Sprache."
        )

    return " ".join(parts) if parts else None


# =========================================================
# Abstract base classes
# =========================================================


class LLMProvider(ABC):
    """Define the contract for LLM completions and helper summarization.

    Use this abstract base class when implementing a custom language
    model backend for Inqtrix. Concrete providers are expected to offer
    a reasoning call path for graph nodes such as classify, plan,
    evaluate, and answer, plus a thread-safe summarization path for the
    search node's parallel helper work.
    """

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        timeout: float = 120.0,
        state: dict | None = None,
        deadline: float | None = None,
    ) -> str:
        """Generate a completion and return only the visible text.

        Call this method for standard reasoning work when the caller only
        needs the model's text output and does not care about explicit
        token accounting. Implementations typically route ``model=None``
        to the provider's default reasoning model and may internally call
        :meth:`complete_with_metadata` before discarding token metadata.

        Args:
            prompt: User-facing input text to send to the model.
            system: Optional system instruction. Omit this when the
                provider or backend does not need a separate system role.
                The default is ``None``.
            model: Optional per-call model override. When omitted, the
                provider should use its default reasoning model or the
                role-specific fallback selected by the runtime.
            max_output_tokens: Optional output-token budget for the visible
                answer. Providers may ignore it when unsupported.
            timeout: Per-call timeout budget in seconds. Providers may
                shorten this further when ``deadline`` leaves less time.
                The default is ``120.0`` seconds.
            state: Optional mutable agent state used for token tracking
                in non-parallel code paths. Omit this in helper threads or
                whenever shared state would be unsafe to mutate.
            deadline: Optional absolute monotonic deadline for the whole
                agent run. When present, providers should clamp the call
                timeout to the remaining budget and raise ``AgentTimeout``
                once the budget is exhausted.

        Returns:
            str: The visible assistant text. Providers should return an
            empty string when the backend responded without user-visible
            content rather than fabricating placeholder output.

        Raises:
            AgentTimeout: If the provider detects that the absolute agent
                deadline has been reached before or during the request.
            AgentRateLimited: If the backend explicitly rate-limits the
                request and the provider chooses to surface that as a
                fatal graph-level error.
            Exception: Backend-specific errors may propagate when the
                provider cannot degrade safely.
        """

    def complete_with_metadata(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        timeout: float = 120.0,
        state: dict | None = None,
        deadline: float | None = None,
    ) -> LLMResponse:
        """Generate a completion together with token metadata.

        Override this method when the backend can report token counts,
        model identity, or other response metadata that the runtime wants
        to preserve for logging, diagnostics, and cost tracking. Custom
        providers may implement only :meth:`complete`; in that case the
        default implementation wraps the returned text in ``LLMResponse``
        without token counts.

        Args:
            prompt: User-facing input text to send to the model.
            system: Optional system instruction. The default is ``None``.
            model: Optional per-call model override. The default is
                ``None``, which signals the provider to use its default
                reasoning model.
            max_output_tokens: Optional output-token budget for the visible
                answer. Providers may ignore it when unsupported.
            timeout: Per-call timeout budget in seconds before deadline
                clamping. The default is ``120.0`` seconds.
            state: Optional mutable agent state for token aggregation in
                non-parallel code paths. Omit it when no shared token
                accounting is needed.
            deadline: Optional absolute monotonic deadline for the whole
                run. Providers should treat it as the hard ceiling for all
                retries and request backoff.

        Returns:
            LLMResponse: Structured response containing visible content,
            token counts when available, and the effective model label.

        Raises:
            AgentTimeout: If the absolute run deadline is exceeded.
            AgentRateLimited: If the backend returns a fatal rate-limit
                condition that should abort the run.
            Exception: Backend-specific provider errors may propagate.
        """
        return LLMResponse(
            content=self.complete(
                prompt,
                system=system,
                model=model,
                max_output_tokens=max_output_tokens,
                timeout=timeout,
                state=state,
                deadline=deadline,
            ),
            model=model or "",
        )

    @abstractmethod
    def summarize_parallel(
        self,
        text: str,
        deadline: float | None = None,
        options: SummarizeOptions | None = None,
    ) -> tuple[str, int, int]:
        """Summarize a search result in a thread-safe helper path.

        The search node calls this method from worker threads to turn raw
        search snippets into compact factual text before claim extraction.
        Implementations must not rely on shared mutable state here; that
        is why the contract does not include a ``state`` parameter.

        Args:
            text: Raw search-result text to condense. Providers should
                typically return an empty summary immediately when ``text``
                is blank.
            deadline: Optional absolute monotonic deadline for the agent
                run. Providers should clamp any backend timeout to the
                remaining budget and stop retrying once the deadline has
                passed.
            options: Optional summarize tuning such as prompt template,
                input trimming, fallback trimming, or helper output budget.

        Returns:
            tuple[str, int, int]: A tuple of ``(facts_text,
            prompt_tokens, completion_tokens)``. Providers that must
            degrade to raw text should still return zero token counts.

        Raises:
            AgentTimeout: If the global run deadline is already exceeded.
            AgentRateLimited: If a provider chooses to escalate a helper
                rate-limit instead of degrading locally.
            Exception: Provider-specific fatal errors may propagate when no
                safe fallback exists.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Report whether the provider is configured to attempt requests.

        Use this method for readiness checks such as health endpoints or
        auto-create diagnostics. ``True`` means the provider has enough
        configuration to make a request; it does not guarantee that the
        remote backend is reachable or healthy at this exact moment.

        Returns:
            bool: ``True`` when the provider is configured and ready to
            attempt requests, otherwise ``False``.
        """


class SearchProvider(ABC):
    """Define the contract for structured web-search providers.

    Use this abstract base class when connecting a search backend such as
    Perplexity, Brave, or Azure Foundry to the research graph. Concrete
    providers are responsible for translating Inqtrix search hints into
    backend-specific request parameters and normalizing the result into a
    stable dictionary shape.
    """

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        search_context_size: str = "high",
        recency_filter: str | None = None,
        language_filter: list[str] | None = None,
        domain_filter: list[str] | None = None,
        search_mode: str | None = None,
        return_related: bool = False,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """Execute a search request and normalize the backend response.

        Concrete implementations should map the generic Inqtrix search
        hints onto whatever the backend supports natively, best-effort, or
        not at all. The returned dictionary must keep the same shape across
        providers so later graph nodes can consume results without special
        cases.

        Args:
            query: User-facing search query text.
            search_context_size: Backend-independent hint for how much web
                context to request. Common values are ``"low"``,
                ``"medium"``, and ``"high"``. Providers may map this to
                result counts, search depth, or ignore it when unsupported.
            recency_filter: Optional freshness hint such as ``"day"``,
                ``"week"``, ``"month"``, or ``"year"``. Providers may map
                it natively or approximate it through prompt hints.
            language_filter: Optional language hints, usually ISO 639-1
                codes such as ``["de"]``. Most providers use only the
                first value when the backend accepts a single language.
            domain_filter: Optional allow/deny list of domains. Entries
                starting with ``"-"`` mean exclusion. Providers may pass
                this natively or inject ``site:`` operators into the query.
            search_mode: Optional backend-specific mode such as
                ``"academic"``. Omit it when the backend does not expose a
                matching concept.
            return_related: Whether the caller wants related questions or
                query suggestions when the backend supports them. The
                default is ``False``.
            deadline: Optional absolute monotonic deadline for the whole
                run. Providers should clamp their timeout budget and stop
                retrying once it is exceeded.

        Returns:
            dict[str, Any]: Normalized result with the keys ``answer``,
            ``citations``, ``related_questions``, ``_prompt_tokens``, and
            ``_completion_tokens``. Providers that do not receive token
            usage from the backend should return ``0`` for the token keys.

        Raises:
            AgentTimeout: If the global run deadline has already been
                exhausted.
            AgentRateLimited: If the backend signals a fatal rate-limit
                condition that should abort the run.
            Exception: Provider-specific fatal errors may propagate when no
                safe empty-result fallback exists.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Report whether the search provider is configured to run.

        Returns:
            bool: ``True`` when the provider has enough local
            configuration to attempt a request, otherwise ``False``.
        """

    @property
    def search_model(self) -> str:
        """Operator-friendly identifier of the search backend in use.

        Surfaces in the ``GET /health`` payload and in the
        ``GET /v1/stacks`` discovery response so an operator (or a UI
        rendering stack chips) can verify which engine each request
        actually routes through. Subclasses MUST override; the default
        returns ``"<ClassName>(unknown)"`` to make a missing override
        loud rather than silently leaking the global
        ``Settings.models.search_model`` default. Loud defaults follow
        Designprinzip 1 ("No Silent Fallbacks") and prevent a repeat
        of the AzureOpenAIWebSearch live-test surprise where the
        health endpoint showed a LiteLLM-flavoured Perplexity model
        name on an Azure-only stack.

        Returns:
            str: Stable identifier of the underlying search engine —
            e.g. ``"sonar-pro"`` for PerplexitySearch,
            ``"foundry-bing:my-agent@v1"`` for the Foundry agents.
            Empty strings are not returned; subclasses that genuinely
            have no model concept should return a constant like
            ``"brave-search-api"``.
        """
        return f"{type(self).__name__}(unknown)"


class ConfiguredLLMProvider(LLMProvider):
    """Attach explicit model metadata to a custom LLM provider.

    Use this adapter when a custom provider implements the runtime call
    methods but does not expose a ``models`` property with the effective
    reasoning, classify, summarize, and evaluate model names. Wrapping
    the provider here preserves the lightweight custom-provider contract
    while still giving the graph stable role-to-model metadata.

    Attributes:
        _provider (LLMProvider): Wrapped provider that performs the real
            backend calls.
        _models (ModelSettings): Effective role-to-model mapping exposed to
            the runtime.
    """

    def __init__(self, provider: LLMProvider, models: ModelSettings) -> None:
        """Initialize the adapter with a provider and explicit models.

        Args:
            provider: Wrapped provider that already implements the
                ``LLMProvider`` call methods.
            models: Effective model settings to expose through the
                adapter's ``models`` property.
        """
        self._provider = provider
        self._models = models

    @property
    def models(self) -> ModelSettings:
        """Return the explicit role-to-model mapping injected at construction.

        Returns:
            The :class:`~inqtrix.settings.ModelSettings` instance the
            adapter was built with. The graph reads ``reasoning_model``,
            ``effective_classify_model`` etc. from this object to
            select per-call model ids.
        """
        return self._models

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        timeout: float = 120.0,
        state: dict | None = None,
        deadline: float | None = None,
    ) -> str:
        """Forward to the wrapped provider, defaulting ``model`` from settings.

        Args:
            prompt: User-facing input text.
            system: Optional system instruction. Forwarded unchanged.
            model: Optional per-call model override. When ``None``,
                falls back to ``self.models.reasoning_model``; the
                effective value is forwarded only if it is non-empty
                (so a wrapped provider with its own default model
                can still receive ``None``).
            max_output_tokens: Optional output-token budget. Forwarded
                unchanged.
            timeout: Per-call timeout in seconds. Forwarded unchanged.
            state: Optional mutable agent state for token tracking.
                Forwarded unchanged.
            deadline: Optional absolute monotonic deadline. Forwarded
                unchanged.

        Returns:
            The visible assistant text from the wrapped provider.

        Raises:
            AgentTimeout: Surfaced from the wrapped provider.
            AgentRateLimited: Surfaced from the wrapped provider.
        """
        call_kwargs: dict[str, object] = {
            "system": system,
            "max_output_tokens": max_output_tokens,
            "timeout": timeout,
            "state": state,
            "deadline": deadline,
        }
        effective_model = model or self._models.reasoning_model
        if effective_model:
            call_kwargs["model"] = effective_model
        return self._provider.complete(prompt, **call_kwargs)

    def complete_with_metadata(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        timeout: float = 120.0,
        state: dict | None = None,
        deadline: float | None = None,
    ) -> LLMResponse:
        """Forward to the wrapped provider, defaulting ``model`` from settings.

        Behaves identically to :meth:`complete` except it returns the
        full :class:`LLMResponse` (text + token counts + effective
        model label).

        Args:
            prompt: See :meth:`complete`.
            system: See :meth:`complete`.
            model: See :meth:`complete`.
            max_output_tokens: See :meth:`complete`.
            timeout: See :meth:`complete`.
            state: See :meth:`complete`.
            deadline: See :meth:`complete`.

        Returns:
            :class:`LLMResponse` from the wrapped provider.

        Raises:
            AgentTimeout: Surfaced from the wrapped provider.
            AgentRateLimited: Surfaced from the wrapped provider.
        """
        call_kwargs: dict[str, object] = {
            "system": system,
            "max_output_tokens": max_output_tokens,
            "timeout": timeout,
            "state": state,
            "deadline": deadline,
        }
        effective_model = model or self._models.reasoning_model
        if effective_model:
            call_kwargs["model"] = effective_model
        return self._provider.complete_with_metadata(prompt, **call_kwargs)

    def summarize_parallel(
        self,
        text: str,
        deadline: float | None = None,
        options: SummarizeOptions | None = None,
    ) -> tuple[str, int, int]:
        """Forward summarize work to the wrapped provider unchanged.

        Args:
            text: Raw search-result text to condense.
            deadline: Optional absolute monotonic deadline. Forwarded
                unchanged.
            options: Optional summarize tuning. Forwarded unchanged so
                the wrapped provider sees the same prompt template,
                input-trim, fallback-trim and output-budget hints the
                caller intended.

        Returns:
            Tuple ``(facts_text, prompt_tokens, completion_tokens)``
            from the wrapped provider.
        """
        return self._provider.summarize_parallel(text, deadline=deadline, options=options)

    def is_available(self) -> bool:
        """Forward the availability check to the wrapped provider.

        Returns:
            ``True`` when the wrapped provider reports itself ready
            to attempt requests; ``False`` otherwise. The adapter
            adds no availability logic of its own.
        """
        return self._provider.is_available()

    def consume_nonfatal_notice(self) -> str | None:
        """Forward an optional non-fatal notice from the wrapped provider.

        Returns:
            The wrapped provider's ``consume_nonfatal_notice()`` value
            when that method exists (typically a one-shot warning
            string used by the runtime to surface degraded-but-
            successful conditions). ``None`` when the wrapped provider
            does not implement this hook.
        """
        consumer = getattr(self._provider, "consume_nonfatal_notice", None)
        if callable(consumer):
            return consumer()
        return None

    def without_thinking(self):
        """Forward an optional thinking-suppression context to the wrapped provider.

        Returns:
            The wrapped provider's ``without_thinking()`` context
            manager when defined (typically used by classify/evaluate
            paths that must disable extended thinking on Anthropic-
            family models). When the wrapped provider does not
            implement this hook, returns a ``contextlib.nullcontext``
            yielding ``self`` so callers can use the same ``with``
            block unconditionally.
        """
        ctx = getattr(self._provider, "without_thinking", None)
        if callable(ctx):
            return ctx()
        from contextlib import nullcontext
        return nullcontext(self)


# =========================================================
# ProviderContext
# =========================================================


@dataclass(frozen=True, slots=True)
class ProviderContext:
    """Group the active LLM and search providers for runtime injection.

    Attributes:
        llm (LLMProvider): Active language-model provider used by the
            graph's reasoning and summarization paths.
        search (SearchProvider): Active search provider used by the
            graph's search node.
    """

    llm: LLMProvider
    search: SearchProvider
