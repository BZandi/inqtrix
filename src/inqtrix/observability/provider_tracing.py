"""Measuring wrappers around the provider chokepoints (Paket C0).

ONE wrap point per capability covers every call site in the codebase:
``TracingLLMProvider`` around the :class:`~inqtrix.providers.base.LLMProvider`
ABC (all ``complete*``/``chat`` callers — research nodes, kernel via
``_DepsChatProvider``, knowledge, chat, editor), ``TracingSearchProvider``
around :class:`~inqtrix.providers.base.SearchProvider`, and
``TracingEmbeddingProvider`` around the knowledge embedding client.

Each call gets: wall-clock duration (``duration_ms`` on the response
DTOs — measured for the first time anywhere in the codebase), a
``gen_ai``-semconv span (metadata always; CONTENT only through the
:mod:`~inqtrix.observability.content` policy), and error visibility
(exception recorded on the span, status ERROR). Without the
``observability`` extra or with tracing off, the wrappers degrade to
pure duration measurement — no OpenTelemetry import is required.

Transparency contract: the wrappers subclass the ABCs, delegate every
non-instrumented member (including private attributes) to the wrapped
provider via ``__getattr__``, and expose it as ``_provider`` so the
existing unwrap helpers (``provider_label``,
``runtime_logging.unwrap_provider``) reach the real backend class.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Iterator

from inqtrix.constants import REASONING_TIMEOUT
from inqtrix.observability import semconv
from inqtrix.observability.content import ContentCapturePolicy
from inqtrix.providers.base import LLMProvider, SearchProvider
from inqtrix.providers.embeddings import EmbeddingProvider

try:  # Optional dependency: the wrappers work without OpenTelemetry.
    from opentelemetry import trace as _otel_trace
except Exception:  # noqa: BLE001 — extra not installed
    _otel_trace = None

if TYPE_CHECKING:
    from inqtrix.providers.base import (
        ChatTurn,
        LLMResponse,
        StructuredLLMResponse,
    )
    from inqtrix.search_result import GroundedSearchResult


def _set_content(
    span: Any, key: str, value: Any, policy: ContentCapturePolicy
) -> None:
    """Set one CONTENT attribute through the policy (redact + cap).

    Every truncation raises the ``inqtrix.truncation`` span event so an
    accidentally capped prompt is findable instead of silently thin.
    """
    clipped = (
        policy.clip_text(value)
        if isinstance(value, str)
        else policy.clip_payload(value)
    )
    span.set_attribute(key, clipped.text)
    if clipped.truncated:
        span.add_event(
            semconv.TRUNCATION_EVENT,
            {
                semconv.TRUNCATION_LIMIT_NAME: key,
                semconv.TRUNCATION_ORIGINAL_SIZE: clipped.original_size,
                semconv.TRUNCATION_CAPPED_SIZE: len(
                    clipped.text.encode("utf-8")
                ),
            },
        )


class _TracingBase:
    """Shared span plumbing for the three wrappers."""

    def __init__(
        self,
        provider: Any,
        *,
        policy: ContentCapturePolicy,
        tracer_provider: Any | None = None,
    ) -> None:
        self._provider = provider
        self._policy = policy
        self._tracer_provider = tracer_provider

    def __getattr__(self, name: str) -> Any:
        # Reached only when normal lookup fails. Guard the delegate
        # attribute itself so a half-constructed instance cannot recurse.
        if name == "_provider":
            raise AttributeError(name)
        return getattr(self._provider, name)

    @contextmanager
    def _span(
        self, name: str, attributes: dict[str, Any]
    ) -> Iterator[Any | None]:
        if _otel_trace is None:
            yield None
            return
        tracer = _otel_trace.get_tracer(
            "inqtrix.providers", tracer_provider=self._tracer_provider
        )
        with tracer.start_as_current_span(
            name, attributes=attributes
        ) as span:
            # A non-recording span (tracing off, or sampled out) exports
            # nothing — yield None so the whole wrapper collapses to the
            # real call plus duration, and no content redaction/
            # serialization runs on the hot path.
            yield span if span.is_recording() else None


class _CallMeter:
    """Mutable per-call carrier the metered block fills in (model/usage)."""

    __slots__ = ("model", "prompt_tokens", "completion_tokens")

    def __init__(self, model: str | None) -> None:
        self.model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0


@contextmanager
def _metered_llm(
    provider_name: str, operation: str, model: str | None
) -> Iterator[_CallMeter]:
    """Prometheus feed for one LLM call — success AND error.

    Independent of tracing state on purpose: metrics must not vanish
    when spans are off or sampled out. No-op without an active holder;
    recording failures warn once and never touch the call.
    """
    from inqtrix.observability.metrics_defs import (
        active_metrics,
        metric_model_label,
    )

    meter = _CallMeter(model)
    metrics = active_metrics()
    started = time.monotonic()
    try:
        yield meter
    except Exception as exc:
        from inqtrix.observability.metrics_defs import (
            outcome_from_exception,
        )
        from inqtrix.usage.recorder import record_provider_call

        error_outcome = outcome_from_exception(exc)
        error_duration = time.monotonic() - started
        if metrics is not None:
            from inqtrix.observability.context import current_feature

            metrics.observe_llm(
                provider=provider_name,
                model=metric_model_label(meter.model),
                operation=operation,
                outcome=error_outcome,
                duration_seconds=error_duration,
                feature=current_feature(),
            )
        record_provider_call(
            operation=operation,
            model=meter.model or "unknown",
            outcome=error_outcome,
            duration_seconds=error_duration,
        )
        raise
    from inqtrix.usage.recorder import record_provider_call

    duration = time.monotonic() - started
    if metrics is not None:
        from inqtrix.observability.context import current_feature

        metrics.observe_llm(
            provider=provider_name,
            model=metric_model_label(meter.model),
            operation=operation,
            outcome="success",
            duration_seconds=duration,
            feature=current_feature(),
            prompt_tokens=meter.prompt_tokens,
            completion_tokens=meter.completion_tokens,
        )
    record_provider_call(
        operation=operation,
        model=meter.model or "unknown",
        outcome="success",
        duration_seconds=duration,
        input_tokens=meter.prompt_tokens,
        output_tokens=meter.completion_tokens,
    )


@contextmanager
def _metered_search(provider_name: str, engine: str) -> Iterator["_CallMeter"]:
    """Prometheus + ledger feed for one web-search call.

    Grounded-search calls carry their OWN provider token usage; the
    caller sets it on the yielded meter so the ledger books the same
    numbers the run folds into its totals (Ledger↔Quota-Konsistenz).
    """
    from inqtrix.observability.metrics_defs import active_metrics
    from inqtrix.usage.recorder import record_provider_call

    meter = _CallMeter(engine)
    metrics = active_metrics()
    started = time.monotonic()
    try:
        yield meter
    except Exception as exc:
        from inqtrix.observability.metrics_defs import (
            outcome_from_exception,
        )

        error_outcome = outcome_from_exception(exc)
        error_duration = time.monotonic() - started
        if metrics is not None:
            metrics.observe_search(
                provider=provider_name,
                engine=engine,
                outcome=error_outcome,
                duration_seconds=error_duration,
            )
        record_provider_call(
            operation="web_search",
            model=engine,
            outcome=error_outcome,
            duration_seconds=error_duration,
        )
        raise
    duration = time.monotonic() - started
    if metrics is not None:
        metrics.observe_search(
            provider=provider_name,
            engine=engine,
            outcome="success",
            duration_seconds=duration,
        )
    record_provider_call(
        operation="web_search",
        model=engine,
        outcome="success",
        duration_seconds=duration,
        input_tokens=meter.prompt_tokens,
        output_tokens=meter.completion_tokens,
    )


class TracingLLMProvider(_TracingBase, LLMProvider):
    """Instrument every LLM call of the wrapped provider."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        provider_name: str,
        policy: ContentCapturePolicy,
        tracer_provider: Any | None = None,
    ) -> None:
        _TracingBase.__init__(
            self, provider, policy=policy, tracer_provider=tracer_provider
        )
        self._gen_ai_provider = provider_name

    # -- delegated capability surface (base-class defaults must never
    # -- shadow the wrapped provider's answers) -------------------------
    @property
    def selectable_models(self) -> list[str]:
        return self._provider.selectable_models

    @property
    def context_window_tokens(self) -> int | None:
        return self._provider.context_window_tokens

    def supports_structured_output(self, *, model: str | None = None) -> bool:
        return self._provider.supports_structured_output(model=model)

    def supports_tool_calls(self, *, model: str | None = None) -> bool:
        return self._provider.supports_tool_calls(model=model)

    def is_available(self) -> bool:
        return self._provider.is_available()

    # -- instrumentation helpers ----------------------------------------
    def _base_attributes(
        self,
        operation: str,
        *,
        model: str | None,
        reasoning_effort: str | None,
    ) -> dict[str, Any]:
        attributes: dict[str, Any] = {
            semconv.GEN_AI_OPERATION_NAME: operation,
            semconv.GEN_AI_PROVIDER_NAME: self._gen_ai_provider,
        }
        if model:
            attributes[semconv.GEN_AI_REQUEST_MODEL] = model
        if reasoning_effort:
            attributes[semconv.INQTRIX_REASONING_EFFORT] = reasoning_effort
        return attributes

    def _record_response(
        self,
        span: Any | None,
        *,
        response_model: str,
        prompt_tokens: int,
        completion_tokens: int,
        finish_reason: str,
        request_max_tokens: int,
    ) -> None:
        if span is None:
            return
        if response_model:
            span.set_attribute(
                semconv.GEN_AI_RESPONSE_MODEL, response_model
            )
        span.set_attribute(
            semconv.GEN_AI_USAGE_INPUT_TOKENS, int(prompt_tokens)
        )
        span.set_attribute(
            semconv.GEN_AI_USAGE_OUTPUT_TOKENS, int(completion_tokens)
        )
        if finish_reason:
            span.set_attribute(
                semconv.GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason]
            )
        if request_max_tokens:
            span.set_attribute(
                semconv.GEN_AI_REQUEST_MAX_TOKENS, int(request_max_tokens)
            )

    def _record_prompt_content(
        self,
        span: Any | None,
        *,
        prompt: str,
        system: str | None,
        output_text: str,
        raw: dict[str, Any] | None,
    ) -> None:
        if span is None or not self._policy.capture_content:
            return
        if system:
            _set_content(
                span,
                semconv.GEN_AI_SYSTEM_INSTRUCTIONS,
                system,
                self._policy,
            )
        _set_content(
            span,
            semconv.GEN_AI_INPUT_MESSAGES,
            [{"role": "user", "content": prompt}],
            self._policy,
        )
        _set_content(
            span,
            semconv.GEN_AI_OUTPUT_MESSAGES,
            [{"role": "assistant", "content": output_text}],
            self._policy,
        )
        if raw:
            _set_content(
                span, semconv.INQTRIX_RESPONSE_RAW, raw, self._policy
            )

    # -- instrumented calls ----------------------------------------------
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
        attributes = self._base_attributes(
            semconv.OPERATION_TEXT_COMPLETION,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        with self._span("text_completion", attributes) as span:
            with _metered_llm(
                self._gen_ai_provider,
                semconv.OPERATION_TEXT_COMPLETION,
                model,
            ) as meter:
                # complete() returns bare text; its token usage only ever
                # reaches the caller through the state accumulator (the
                # SAME numbers quota books). Read the delta so the ledger
                # cannot systematically undercount this path.
                before_prompt = int((state or {}).get("total_prompt_tokens", 0) or 0)
                before_completion = int(
                    (state or {}).get("total_completion_tokens", 0) or 0
                )
                text = self._provider.complete(
                    prompt,
                    system=system,
                    model=model,
                    max_output_tokens=max_output_tokens,
                    timeout=timeout,
                    state=state,
                    deadline=deadline,
                    reasoning_effort=reasoning_effort,
                )
                if state is not None:
                    meter.prompt_tokens = max(
                        0,
                        int(state.get("total_prompt_tokens", 0) or 0)
                        - before_prompt,
                    )
                    meter.completion_tokens = max(
                        0,
                        int(state.get("total_completion_tokens", 0) or 0)
                        - before_completion,
                    )
            # Export the SAME numbers the ledger and Prometheus book, from
            # the same meter. Without this the span reaches the trace
            # backend carrying no usage at all, and Langfuse then infers
            # token counts from the message text — an invented number that
            # disagrees with the ledger for exactly this call path.
            if state is not None:
                self._record_response(
                    span,
                    response_model="",
                    prompt_tokens=meter.prompt_tokens,
                    completion_tokens=meter.completion_tokens,
                    finish_reason="",
                    request_max_tokens=int(max_output_tokens or 0),
                )
            elif span is not None:
                # No accumulator, so no honest number exists here. Exporting
                # zero would read as a free call; staying silent would invite
                # the same inference this fix removes.
                span.set_attribute(semconv.INQTRIX_USAGE_UNAVAILABLE, True)
            self._record_prompt_content(
                span, prompt=prompt, system=system, output_text=text, raw=None
            )
            if span is not None and self._policy.capture_content:
                # Structural limit, made VISIBLE instead of looking like
                # an empty raw response: complete() returns bare text, so
                # the provider payload — including extended-thinking
                # blocks that were produced and billed — never reaches
                # this wrapper. Callers that need it use
                # complete_with_metadata (see the tracing legend).
                span.set_attribute(semconv.INQTRIX_RAW_UNAVAILABLE, True)
            return text

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
    ) -> "LLMResponse":
        attributes = self._base_attributes(
            semconv.OPERATION_TEXT_COMPLETION,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        started = time.monotonic()
        with self._span("text_completion", attributes) as span:
            with _metered_llm(
                self._gen_ai_provider,
                semconv.OPERATION_TEXT_COMPLETION,
                model,
            ) as meter:
                response = self._provider.complete_with_metadata(
                    prompt,
                    system=system,
                    model=model,
                    max_output_tokens=max_output_tokens,
                    timeout=timeout,
                    state=state,
                    deadline=deadline,
                    reasoning_effort=reasoning_effort,
                )
                meter.model = response.model or model
                meter.prompt_tokens = int(response.prompt_tokens or 0)
                meter.completion_tokens = int(
                    response.completion_tokens or 0
                )
            response = replace(
                response,
                duration_ms=round((time.monotonic() - started) * 1000.0, 3),
            )
            self._record_response(
                span,
                response_model=response.model,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                finish_reason=response.finish_reason,
                request_max_tokens=response.request_max_tokens,
            )
            self._record_prompt_content(
                span,
                prompt=prompt,
                system=system,
                output_text=response.content,
                raw=response.raw,
            )
            return response

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
    ) -> "StructuredLLMResponse":
        attributes = self._base_attributes(
            semconv.OPERATION_TEXT_COMPLETION,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        attributes[semconv.INQTRIX_SCHEMA_NAME] = schema_name
        started = time.monotonic()
        with self._span("text_completion", attributes) as span:
            with _metered_llm(
                self._gen_ai_provider,
                semconv.OPERATION_TEXT_COMPLETION,
                model,
            ) as meter:
                response = self._provider.complete_structured(
                    prompt,
                    schema=schema,
                    schema_name=schema_name,
                    schema_description=schema_description,
                    system=system,
                    model=model,
                    max_output_tokens=max_output_tokens,
                    timeout=timeout,
                    state=state,
                    deadline=deadline,
                    reasoning_effort=reasoning_effort,
                )
                meter.model = response.model or model
                meter.prompt_tokens = int(response.prompt_tokens or 0)
                meter.completion_tokens = int(
                    response.completion_tokens or 0
                )
            response = replace(
                response,
                duration_ms=round((time.monotonic() - started) * 1000.0, 3),
            )
            self._record_response(
                span,
                response_model=response.model,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                finish_reason=response.finish_reason,
                request_max_tokens=response.request_max_tokens,
            )
            self._record_prompt_content(
                span,
                prompt=prompt,
                system=system,
                output_text=response.content,
                raw=response.raw,
            )
            return response

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        timeout: float = REASONING_TIMEOUT,
        deadline: float | None = None,
        reasoning_effort: str | None = None,
    ) -> "ChatTurn":
        attributes = self._base_attributes(
            semconv.OPERATION_CHAT,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        started = time.monotonic()
        with self._span("chat", attributes) as span:
            with _metered_llm(
                self._gen_ai_provider, semconv.OPERATION_CHAT, model
            ) as meter:
                turn = self._provider.chat(
                    messages,
                    tools=tools,
                    model=model,
                    max_output_tokens=max_output_tokens,
                    timeout=timeout,
                    deadline=deadline,
                    reasoning_effort=reasoning_effort,
                )
                meter.model = turn.model or model
                meter.prompt_tokens = int(turn.prompt_tokens or 0)
                meter.completion_tokens = int(turn.completion_tokens or 0)
            turn = replace(
                turn,
                duration_ms=round((time.monotonic() - started) * 1000.0, 3),
            )
            self._record_response(
                span,
                response_model=turn.model,
                prompt_tokens=turn.prompt_tokens,
                completion_tokens=turn.completion_tokens,
                finish_reason=turn.finish_reason,
                request_max_tokens=0,
            )
            if span is not None:
                span.set_attribute(
                    semconv.INQTRIX_TOOL_CALL_COUNT, len(turn.tool_calls)
                )
                if self._policy.capture_content:
                    _set_content(
                        span,
                        semconv.GEN_AI_INPUT_MESSAGES,
                        messages,
                        self._policy,
                    )
                    output_message: dict[str, Any] = {
                        "role": "assistant",
                        "content": turn.text,
                    }
                    if turn.tool_calls:
                        output_message["tool_calls"] = [
                            {
                                "id": call.id,
                                "name": call.name,
                                "arguments": call.arguments,
                            }
                            for call in turn.tool_calls
                        ]
                    _set_content(
                        span,
                        semconv.GEN_AI_OUTPUT_MESSAGES,
                        [output_message],
                        self._policy,
                    )
                    if turn.raw:
                        _set_content(
                            span,
                            semconv.INQTRIX_RESPONSE_RAW,
                            turn.raw,
                            self._policy,
                        )
            return turn


class TracingSearchProvider(_TracingBase, SearchProvider):
    """Instrument every web-search call of the wrapped provider."""

    def __init__(
        self,
        provider: SearchProvider,
        *,
        policy: ContentCapturePolicy,
        tracer_provider: Any | None = None,
    ) -> None:
        _TracingBase.__init__(
            self, provider, policy=policy, tracer_provider=tracer_provider
        )

    @property
    def search_model(self) -> str:
        return self._provider.search_model

    def is_available(self) -> bool:
        return self._provider.is_available()

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
    ) -> "GroundedSearchResult":
        # The query text is user content and can carry PII, so it is
        # attached ONLY under content capture (below), like every other
        # content attribute — never as always-on metadata. Provider,
        # engine, source count and latency keep the span useful without
        # it.
        attributes: dict[str, Any] = {
            semconv.INQTRIX_SEARCH_PROVIDER: type(self._provider).__name__,
            semconv.INQTRIX_SEARCH_ENGINE: self._provider.search_model,
        }
        if search_mode:
            attributes[semconv.INQTRIX_SEARCH_MODE] = search_mode
        if recency_filter:
            attributes[semconv.INQTRIX_SEARCH_RECENCY] = recency_filter
        if domain_filter:
            attributes[semconv.INQTRIX_SEARCH_DOMAIN_FILTER_COUNT] = len(
                domain_filter
            )
        with self._span("web_search", attributes) as span:
            with _metered_search(
                type(self._provider).__name__,
                str(self._provider.search_model or "unknown"),
            ) as search_meter:
                result = self._provider.search(
                    query,
                    search_context_size=search_context_size,
                    recency_filter=recency_filter,
                    language_filter=language_filter,
                    domain_filter=domain_filter,
                    search_mode=search_mode,
                    return_related=return_related,
                    deadline=deadline,
                )
                search_meter.prompt_tokens = int(result.prompt_tokens or 0)
                search_meter.completion_tokens = int(
                    result.completion_tokens or 0
                )
            if span is not None:
                span.set_attribute(
                    semconv.INQTRIX_SEARCH_SOURCE_COUNT, len(result.sources)
                )
                span.set_attribute(
                    semconv.INQTRIX_SEARCH_ANSWER_LENGTH,
                    len(result.answer or ""),
                )
                span.set_attribute(
                    semconv.INQTRIX_SEARCH_INPUT_TOKENS,
                    int(result.prompt_tokens or 0),
                )
                span.set_attribute(
                    semconv.INQTRIX_SEARCH_OUTPUT_TOKENS,
                    int(result.completion_tokens or 0),
                )
                if self._policy.capture_content:
                    _set_content(
                        span,
                        semconv.INQTRIX_SEARCH_QUERY,
                        query,
                        self._policy,
                    )
                if self._policy.capture_content and result.answer:
                    _set_content(
                        span,
                        semconv.INQTRIX_SEARCH_ANSWER,
                        result.answer,
                        self._policy,
                    )
                if self._policy.capture_content and result.sources:
                    # The per-source records ARE the evidence a bad
                    # report has to be explained from — without them the
                    # trace shows a provider answer whose provenance
                    # cannot be reconstructed. Content, so the same
                    # redaction + cap + truncation-event path applies.
                    _set_content(
                        span,
                        semconv.INQTRIX_SEARCH_SOURCES,
                        [
                            {
                                "url": source.url,
                                "title": source.title,
                                "snippet": source.snippet,
                                "date": source.date,
                                "rank": source.rank,
                            }
                            for source in result.sources
                        ],
                        self._policy,
                    )
            return result


class TracingEmbeddingProvider(_TracingBase, EmbeddingProvider):
    """Instrument embedding calls (ingestion batches and queries)."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        policy: ContentCapturePolicy,
        tracer_provider: Any | None = None,
    ) -> None:
        _TracingBase.__init__(
            self, provider, policy=policy, tracer_provider=tracer_provider
        )

    @property
    def selectable_embedding_models(self) -> list[str]:
        return self._provider.selectable_embedding_models

    @property
    def default_model(self) -> str:
        return self._provider.default_model

    def _attributes(self, model: str | None, count: int) -> dict[str, Any]:
        return {
            semconv.GEN_AI_OPERATION_NAME: semconv.OPERATION_EMBEDDINGS,
            semconv.GEN_AI_PROVIDER_NAME: type(self._provider).__name__,
            semconv.GEN_AI_REQUEST_MODEL: model or self.default_model,
            semconv.INQTRIX_EMBED_TEXT_COUNT: count,
        }

    def embed_documents(
        self, texts: list[str], *, model: str | None = None
    ) -> list[list[float]]:
        with self._span(
            "embeddings", self._attributes(model, len(texts))
        ):
            with _metered_llm(
                "embeddings",
                semconv.OPERATION_EMBEDDINGS,
                model or self.default_model,
            ) as meter:
                from inqtrix.quota.models import estimate_tokens

                vectors = self._provider.embed_documents(texts, model=model)
                # Same estimator the quota path books against the same texts,
                # so ledger and quota agree by construction. A failed call
                # keeps the zero row, exactly like the search wrapper.
                meter.prompt_tokens = sum(estimate_tokens(text) for text in texts)
                return vectors

    def embed_query(
        self, text: str, *, model: str | None = None
    ) -> list[float]:
        with self._span("embeddings", self._attributes(model, 1)):
            with _metered_llm(
                "embeddings",
                semconv.OPERATION_EMBEDDINGS,
                model or self.default_model,
            ) as meter:
                from inqtrix.quota.models import estimate_tokens

                vector = self._provider.embed_query(text, model=model)
                meter.prompt_tokens = estimate_tokens(text)
                return vector

def instrument_llm(
    provider: LLMProvider,
    *,
    provider_name: str,
    policy: ContentCapturePolicy,
    tracer_provider: Any | None = None,
) -> LLMProvider:
    """Wrap one LLM provider (idempotent — never double-wraps)."""
    if isinstance(provider, TracingLLMProvider):
        return provider
    return TracingLLMProvider(
        provider,
        provider_name=provider_name,
        policy=policy,
        tracer_provider=tracer_provider,
    )


def instrument_search(
    provider: SearchProvider,
    *,
    policy: ContentCapturePolicy,
    tracer_provider: Any | None = None,
) -> SearchProvider:
    """Wrap one search provider (idempotent)."""
    if isinstance(provider, TracingSearchProvider):
        return provider
    return TracingSearchProvider(
        provider, policy=policy, tracer_provider=tracer_provider
    )


def instrument_embeddings(
    provider: EmbeddingProvider,
    *,
    policy: ContentCapturePolicy,
    tracer_provider: Any | None = None,
) -> EmbeddingProvider:
    """Wrap one embedding provider (idempotent)."""
    if isinstance(provider, TracingEmbeddingProvider):
        return provider
    return TracingEmbeddingProvider(
        provider, policy=policy, tracer_provider=tracer_provider
    )
