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


class ThinkingSuppressionMixin:
    """Re-entrant, thread-safe suppression of extended thinking.

    Subclasses must set ``self._thinking`` (the thinking config dict or
    ``None``) and ``self._thread_state`` (a ``threading.local()``) in
    their ``__init__``.

    The ``without_thinking`` context manager increments a thread-local
    depth counter.  While depth > 0, ``_thinking_enabled()`` returns
    ``False``, so the provider omits the ``thinking`` key from API
    payloads.  This is re-entrant and thread-safe: parallel summarise
    threads each have their own counter.
    """

    _thinking: dict | None
    _thread_state: threading.local

    @contextmanager
    def without_thinking(self):
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
    """Structured wrapper around an LLM completion result."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


@dataclass(frozen=True, slots=True)
class _NormalizedCompletion:
    """Internal normalized representation of a chat completion."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
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


def _extract_usage_from_payload(payload: dict[str, Any]) -> tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from a dict payload."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return (0, 0)
    return (
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
    )


def _extract_usage_from_response(response: Any) -> tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from an SDK response object."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return (0, 0)
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


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
    return _NormalizedCompletion(
        content="".join(content_parts),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
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

    return _NormalizedCompletion(
        content="".join(content_parts),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
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
            if not prompt_tokens and not completion_tokens:
                prompt_tokens, completion_tokens = _extract_usage_from_payload(dumped)

    return _NormalizedCompletion(
        content="".join(content_parts),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
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
    if recency == "day":
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
                timeout=timeout,
                state=state,
                deadline=deadline,
            ),
            model=model or "",
        )

    @abstractmethod
    def summarize_parallel(
        self, text: str, deadline: float | None = None
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
        return self._models

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        state: dict | None = None,
        deadline: float | None = None,
    ) -> str:
        call_kwargs: dict[str, object] = {
            "system": system,
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
        timeout: float = 120.0,
        state: dict | None = None,
        deadline: float | None = None,
    ) -> LLMResponse:
        call_kwargs: dict[str, object] = {
            "system": system,
            "timeout": timeout,
            "state": state,
            "deadline": deadline,
        }
        effective_model = model or self._models.reasoning_model
        if effective_model:
            call_kwargs["model"] = effective_model
        return self._provider.complete_with_metadata(prompt, **call_kwargs)

    def summarize_parallel(
        self, text: str, deadline: float | None = None
    ) -> tuple[str, int, int]:
        return self._provider.summarize_parallel(text, deadline=deadline)

    def is_available(self) -> bool:
        return self._provider.is_available()

    def consume_nonfatal_notice(self) -> str | None:
        consumer = getattr(self._provider, "consume_nonfatal_notice", None)
        if callable(consumer):
            return consumer()
        return None

    def without_thinking(self):
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
