"""OpenTelemetry setup (``INQTRIX_TRACING``) — optional and no-op-safe.

Follows the ``server/metrics.py`` contract for optional subsystems: the
feature is off by default, a missing extra degrades LOUDLY (one
WARNING) but never crashes, and with ``off`` nothing OpenTelemetry is
even imported.

Modes (see ``ObservabilitySettings``):

* ``local`` — a real ``TracerProvider`` with an always-drop sampler:
  spans stay NON-recording but carry valid ids, so JSON log lines gain
  ``trace_id``/``span_id`` while content capture and span-event bridges
  stay on their cheap early-return paths; nothing leaves the process.
* ``file`` — spans spool as OTLP-JSON lines (see
  :mod:`inqtrix.observability.spool`), replayable into Langfuse later.
* ``otlp`` — live ``OTLPSpanExporter`` (http/protobuf). Endpoint and
  headers come from the standard ``OTEL_EXPORTER_OTLP_ENDPOINT`` /
  ``OTEL_EXPORTER_OTLP_HEADERS`` variables; for Langfuse the headers
  MUST include ``Authorization=Basic <base64(pk:sk)>`` and
  ``x-langfuse-ingestion-version=4``.

The provider is installed process-globally exactly once; repeated
``setup_tracing`` calls (tests build many apps) return the existing
provider instead of fighting over the global.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator, Mapping

if TYPE_CHECKING:  # pragma: no cover — typing only
    from opentelemetry.sdk.trace import TracerProvider

    from inqtrix.settings import Settings

log = logging.getLogger("inqtrix")

try:  # Optional dependency: resolved ONCE at import time so the first
    # run thread never pays the OpenTelemetry import latency.
    from opentelemetry import trace as _otel_trace
except Exception:  # noqa: BLE001 — extra not installed
    _otel_trace = None

_state: dict[str, Any] = {"provider": None, "installed": False}


def build_tracer_provider(
    settings: "Settings", *, service_role: str
) -> "TracerProvider | None":
    """Build (but do not install) the provider for the configured mode.

    Returns ``None`` for mode ``off`` and for a missing extra (with one
    WARNING). Pure factory — tests use it without touching the global.
    """
    mode = settings.observability.tracing
    if mode == "off":
        return None
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import SpanLimits, TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import (
            ALWAYS_OFF,
            ParentBased,
            TraceIdRatioBased,
        )
    except ImportError:
        log.warning(
            "INQTRIX_TRACING=%s gesetzt, aber das `observability`-Extra "
            "ist nicht installiert (pip install inqtrix[observability]) "
            "- Tracing bleibt aus.",
            mode,
        )
        return None

    from inqtrix import __version__

    resource = Resource.create(
        {
            "service.name": f"inqtrix-{service_role}",
            "service.version": __version__,
        }
    )
    if mode == "local":
        # local = correlation ids only. A DROP-sampled span still gets a
        # valid SpanContext (trace_id/span_id for JSON log lines) but is
        # non-recording — every recording-gated code path (content
        # capture, forensic span-event bridge, attribute serialization)
        # collapses to its cheap early return. A recording span without
        # an exporter would pay all of that for data nobody ever sees.
        sampler = ALWAYS_OFF
    else:
        sampler = ParentBased(
            TraceIdRatioBased(settings.observability.trace_sample_rate)
        )
    provider = TracerProvider(
        resource=resource,
        sampler=sampler,
        # The SDK default of 128 events per span silently drops events
        # (the bridge attaches one per runtime event to the enclosing
        # node span). 1024 was NOT enough: a deep research run with 8
        # first-round queries x ~40 sources hit exactly 1024 on the
        # search node and lost the OLDEST 476 — the head of the query
        # lineage. 8192 covers that with headroom, and add_span_event
        # now WARNS whenever the ceiling still evicts anything.
        span_limits=SpanLimits(max_events=8192),
    )
    if mode == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        # An unreachable backend must be VISIBLE. The SDK reports export
        # failures and queue-full drops only on the `opentelemetry`
        # logger, which configure_logging deliberately never wires (it
        # touches `inqtrix` only) — so without this bridge a wrong
        # endpoint or a dead Langfuse looks exactly like "no traffic".
        _bridge_otel_diagnostics()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    elif mode == "file":
        from inqtrix.observability.spool import build_spool_exporter

        try:
            exporter = build_spool_exporter(
                settings.observability.trace_spool_dir,
                settings.observability.trace_spool_max_mb,
            )
        except OSError as exc:
            # An unwritable spool directory (read-only mount, missing
            # volume, wrong ownership) must NOT take the API or worker
            # down — telemetry is an add-on. Degrade to correlation-only
            # ids, loudly: the operator sees exactly what is missing.
            log.error(
                "INQTRIX_TRACING=file: Spool-Verzeichnis %s ist nicht "
                "beschreibbar (%s) - Spans werden NICHT aufgezeichnet, "
                "der Dienst startet mit reiner Trace-ID-Korrelation.",
                settings.observability.trace_spool_dir,
                exc,
            )
        else:
            provider.add_span_processor(BatchSpanProcessor(exporter))
    # mode == "local": provider without processor — valid span ids for
    # log correlation, zero persistence, zero network.
    return provider


def _bridge_otel_diagnostics() -> None:
    """Route the SDK's export diagnostics into the inqtrix log stream.

    Attaches the `inqtrix` logger's handlers to the `opentelemetry`
    exporter loggers at WARNING and stops propagation, so a failing
    OTLP export surfaces exactly once per SDK message in the operator's
    normal log — without letting third-party INFO chatter flood it.
    Idempotent: repeated setup_tracing calls do not stack handlers.
    """
    inqtrix_logger = logging.getLogger("inqtrix")
    for name in (
        "opentelemetry.exporter.otlp.proto.http.trace_exporter",
        "opentelemetry.sdk.trace.export",
    ):
        sdk_logger = logging.getLogger(name)
        if getattr(sdk_logger, "_inqtrix_bridged", False):
            continue
        sdk_logger.setLevel(logging.WARNING)
        for handler in inqtrix_logger.handlers:
            sdk_logger.addHandler(handler)
        sdk_logger.propagate = False
        sdk_logger._inqtrix_bridged = True  # type: ignore[attr-defined]


def setup_tracing(
    settings: "Settings", *, service_role: str
) -> "TracerProvider | None":
    """Install the tracer provider process-globally (idempotent)."""
    if _state["installed"]:
        return _state["provider"]
    provider = build_tracer_provider(settings, service_role=service_role)
    if provider is None:
        return None
    _otel_trace.set_tracer_provider(provider)
    _state["provider"] = provider
    _state["installed"] = True
    log.info(
        "Tracing aktiv: mode=%s service=inqtrix-%s sample_rate=%s",
        settings.observability.tracing,
        service_role,
        settings.observability.trace_sample_rate,
    )
    return provider


def tracing_installed() -> bool:
    """Whether a real tracer provider is installed in this process.

    Lets status surfaces distinguish CONFIGURED (settings say local/
    file/otlp) from EFFECTIVE (the observability extra was present and
    setup actually installed a provider).
    """
    return bool(_state.get("installed"))


def shutdown_tracing() -> None:
    """Flush pending spans (SIGTERM/lifespan) WITHOUT tearing the provider down.

    Uses ``force_flush``, not ``shutdown``: a full shutdown would leave
    the tracer provider dead while the OpenTelemetry global still points
    at it (``set_tracer_provider`` refuses a second install), so a later
    ``create_app`` in the same process — the multi-app test topology —
    would silently drop every span. force_flush drains the
    BatchSpanProcessor's last batch (the real span-loss risk on worker
    exit) and is safe to call repeatedly; the process teardown itself
    reclaims the daemon export thread.
    """
    provider = _state.get("provider")
    if provider is None:
        return
    try:
        provider.force_flush()
    except Exception:  # noqa: BLE001 — flush must never mask the real exit
        log.warning("Tracing-Flush fehlgeschlagen.", exc_info=True)


@contextmanager
def run_execute_span(
    *,
    run_id: str,
    tenant_id: str,
    attempt: int,
    payload: Mapping[str, Any] | None,
) -> Iterator[Any]:
    """The ``inqtrix.run`` span around one run segment (worker or in-process).

    Parented via the traceparent persisted in the run payload — the 1:1
    job relationship makes parent-child the documented messaging-semconv
    exception (more readable than links). ``execute_run_request``
    enriches this span with the Langfuse trace fields and the outcome.
    Degrades to a no-op without the extra; with tracing off the span is
    simply non-recording.
    """
    if _otel_trace is None:
        yield None
        return
    from inqtrix.observability import semconv
    from inqtrix.observability.propagation import extract_context

    tracer = _otel_trace.get_tracer("inqtrix.worker")
    # The run algorithm catches its own failures (terminate_native_run)
    # so the exception never reaches this block — the failure path marks
    # the span explicitly via mark_current_span_error() in run_service.
    with tracer.start_as_current_span(
        "inqtrix.run",
        context=extract_context(payload),
        attributes={
            semconv.INQTRIX_RUN_ID: run_id,
            semconv.INQTRIX_TENANT: tenant_id,
            semconv.INQTRIX_ATTEMPT: attempt,
            # Langfuse maps invoke_agent to the "agent" observation type.
            semconv.GEN_AI_OPERATION_NAME: semconv.OPERATION_INVOKE_AGENT,
        },
    ) as span:
        yield span


@contextmanager
def operation_span(
    name: str,
    attributes: Mapping[str, Any] | None = None,
    *,
    tracer_provider: Any | None = None,
) -> Iterator[Any]:
    """Generic span for a service-start boundary (chat, indexing, ...).

    No-op without the extra; non-recording with tracing off.
    ``tracer_provider`` defaults to the process-global provider and is
    injectable for tests.
    """
    if _otel_trace is None:
        yield None
        return
    tracer = _otel_trace.get_tracer(
        "inqtrix.runs", tracer_provider=tracer_provider
    )
    with tracer.start_as_current_span(
        name, attributes=dict(attributes or {})
    ) as span:
        yield span


def enrich_current_span(attributes: Mapping[str, Any]) -> None:
    """Set attributes on the active span (no-op-safe).

    The run root span is opened by the EXECUTION boundary (worker
    ``_execute`` / in-process ``_run_worker``); the shared
    ``execute_run_request`` body enriches it here with the Langfuse
    trace fields and the outcome — one enrichment path for both.
    """
    if _otel_trace is None:
        return
    span = _otel_trace.get_current_span()
    if not span.is_recording():
        return
    for key, value in attributes.items():
        if value is None or value == "":
            continue
        span.set_attribute(key, value)


def span_is_recording() -> bool:
    """Whether a recording span is active (cheap gate for bridges)."""
    if _otel_trace is None:
        return False
    return _otel_trace.get_current_span().is_recording()


_NESTED_ATTR_MAX_BYTES = 262_144
"""Byte BACKSTOP for a flattened nested span-event value.

Sized so it does not fire in normal operation: a cap that trims real
lineage would defeat the point of recording it. The original 2048 cut
id lists mid-token (300 ids survived as ~93 entries of invalid JSON);
256 KiB holds the largest observed evidence/citation payloads whole.
It exists only to stop a pathological value from breaking the export —
and when it does fire, an inqtrix.truncation event plus a WARNING name
the field and both sizes."""

_event_truncation_warned = False
_event_ceiling_warned = False


def _clip_nested(value: Any) -> tuple[str, int]:
    """Serialize a nested value, returning (text, original_byte_size)."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    raw = text.encode("utf-8")
    if len(raw) <= _NESTED_ATTR_MAX_BYTES:
        return text, len(raw)
    clipped = raw[:_NESTED_ATTR_MAX_BYTES].decode("utf-8", "ignore")
    return clipped, len(raw)


def add_span_event(name: str, payload: Mapping[str, Any]) -> None:
    """Attach one event to the active span (no-op-safe).

    Span-event attributes must stay flat: scalars pass through, nested
    containers become compact JSON strings. Every cap is REPORTED —
    both the per-value byte cap (a truncation event names the field and
    both sizes) and the per-span event ceiling, whose OTel BoundedList
    evicts the OLDEST events, i.e. the head of a run's lineage.
    """
    global _event_truncation_warned, _event_ceiling_warned
    if not span_is_recording():
        return
    attributes: dict[str, Any] = {}
    truncated: list[tuple[str, int, int]] = []
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            attributes[str(key)] = value
            continue
        text, original = _clip_nested(value)
        attributes[str(key)] = text
        if original > _NESTED_ATTR_MAX_BYTES:
            truncated.append((str(key), original, len(text.encode("utf-8"))))
    span = _otel_trace.get_current_span()
    span.add_event(name, attributes)
    for field, original_size, capped_size in truncated:
        # The documented invariant: a capped value is findable, never
        # silently thin (same contract as the content-capture path).
        from inqtrix.observability import semconv

        span.add_event(
            semconv.TRUNCATION_EVENT,
            {
                semconv.TRUNCATION_LIMIT_NAME: f"{name}.{field}",
                semconv.TRUNCATION_ORIGINAL_SIZE: original_size,
                semconv.TRUNCATION_CAPPED_SIZE: capped_size,
            },
        )
        if not _event_truncation_warned:
            _event_truncation_warned = True
            log.warning(
                "Span-Event-Feld %s.%s war %d Bytes und wurde auf %d "
                "gekappt - die Trace-Ansicht zeigt es unvollstaendig "
                "(INQTRIX_TRACE_MAX_ATTR_BYTES betrifft NUR Content-"
                "Attribute, dies ist die Ereignis-Grenze).",
                name,
                field,
                original_size,
                capped_size,
            )
    dropped = getattr(span, "dropped_events", 0) or 0
    if dropped and not _event_ceiling_warned:
        _event_ceiling_warned = True
        log.warning(
            "Span-Ereignisgrenze erreicht: %d Ereignisse wurden verworfen "
            "- OpenTelemetry entfernt die AELTESTEN zuerst, also den "
            "Anfang der Lauf-Lineage. Tiefe Laeufe brauchen ein hoeheres "
            "SpanLimits(max_events).",
            dropped,
        )


def mark_current_span_error(reason: str) -> None:
    """Set the active span's status to ERROR (no-op-safe).

    A failed or cancelled run segment must not look like a clean run in
    the waterfall; the algorithm swallows its own exception, so the
    outcome path marks the span here.
    """
    if _otel_trace is None:
        return
    span = _otel_trace.get_current_span()
    if not span.is_recording():
        return
    try:
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR, reason))
    except Exception:  # noqa: BLE001 — telemetry must never raise
        # Fail-safe, never fail-SILENT: an unmarkable span means failed
        # runs would render as clean traces.
        log.warning(
            "Span-Fehlerstatus konnte nicht gesetzt werden (%s).",
            reason,
            exc_info=True,
        )


def current_trace_id_hex() -> str | None:
    """Hex trace id of the active span context, or ``None``."""
    if _otel_trace is None:
        return None
    span_context = _otel_trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return None
    return format(span_context.trace_id, "032x")


def traced_thread_call(
    name: str,
    attributes: Mapping[str, Any] | None,
    fn: Any,
    enrich: Any | None = None,
    *,
    tracer_provider: Any | None = None,
    feature: str | None = None,
    usage_subject: tuple[Any, Any, Any] | None = None,
) -> Any:
    """Wrap an executor callable in a span OPENED INSIDE the thread.

    ``loop.run_in_executor`` does not copy contextvars, so a span opened
    in the async caller would never reach the worker thread; opening it
    inside the callable makes every provider span of the execution nest
    under it. ``enrich(span, result)`` runs before the span closes
    (usage/outcome attributes); enrichment failures never mask the
    result. ``usage_subject`` is the raw ledger booking identity
    ``(tenant_id, user_id, workspace_id)`` — bound inside the thread for
    the same contextvar reason as the feature label.
    """

    # Snapshot the caller's correlation context HERE: this runs in the
    # async task where the request middleware bound it; the pool thread
    # would otherwise start blank.
    from inqtrix.observability.context import current_log_context

    log_context = {
        key: value
        for key, value in current_log_context().items()
        if value
    }

    def runner() -> Any:
        # Executor threads copy no contextvars, so the metrics feature
        # label must be bound INSIDE the thread — and cleared, because
        # the default executor reuses its threads.
        feature_token = None
        if feature:
            from inqtrix.observability.context import bind_feature

            feature_token = bind_feature(feature)
        if usage_subject is not None:
            from inqtrix.observability.context import bind_usage_subject

            bind_usage_subject(*usage_subject)
        # The correlation context (request_id/user/tenant/workspace) was
        # bound in the ASYNC task by the request middleware; the pool
        # thread inherits none of it. Without this re-bind every log
        # line and forensic event of the executed algorithm loses its
        # request_id — the very envelope package B exists for.
        from inqtrix.observability.context import (
            bind_log_context,
            reset_log_context,
        )

        log_tokens = bind_log_context(**(log_context or {}))
        try:
            with operation_span(
                name, attributes, tracer_provider=tracer_provider
            ) as span:
                result = fn()
                if span is not None and enrich is not None:
                    try:
                        enrich(span, result)
                    except Exception:  # noqa: BLE001 — telemetry stays non-fatal
                        # WARNING, not debug: a failing enricher silently
                        # strips usage/outcome from every span of this
                        # kind.
                        log.warning(
                            "Span-Enrichment fuer %s fehlgeschlagen.",
                            name,
                            exc_info=True,
                        )
                return result
        finally:
            if feature_token is not None:
                from inqtrix.observability.context import reset_feature

                reset_feature(feature_token)
            if usage_subject is not None:
                from inqtrix.observability.context import (
                    clear_usage_subject,
                )

                clear_usage_subject()
            reset_log_context(log_tokens)

    return runner


def chat_thread_call(
    fn: Any,
    *,
    mode: str,
    principal: Any,
    streamed: bool = False,
    tracer_provider: Any | None = None,
) -> Any:
    """The ONE chat root-span wrapper for both chat mouths.

    Native (`chat_service`) and OpenAI-compatible streaming
    (`server/streaming.py`) produce the identical span: same name, same
    attribute set, same usage enrichment, same feature label and ledger
    subject — they differ only in ``inqtrix.streamed``. Keeping that in
    one place is what stops the two paths from drifting apart (they
    already had two hand-maintained copies of it).
    """
    from inqtrix.auth.log_redaction import stable_pseudonym
    from inqtrix.observability import semconv

    user_id = getattr(principal, "user_id", None)
    attributes: dict[str, Any] = {
        semconv.GEN_AI_OPERATION_NAME: semconv.OPERATION_INVOKE_AGENT,
        semconv.LANGFUSE_TRACE_NAME: f"chat:{mode}",
        "inqtrix.mode": mode,
    }
    if streamed:
        attributes["inqtrix.streamed"] = True
    if user_id is not None:
        attributes[semconv.LANGFUSE_USER_ID] = stable_pseudonym(
            "usr", user_id
        )

    def _enrich(span: Any, agent_result: Any) -> None:
        usage = (agent_result.raw or {}).get("usage", {}) or {}
        span.set_attribute(
            semconv.GEN_AI_USAGE_INPUT_TOKENS,
            int(usage.get("prompt_tokens", 0) or 0),
        )
        span.set_attribute(
            semconv.GEN_AI_USAGE_OUTPUT_TOKENS,
            int(usage.get("completion_tokens", 0) or 0),
        )

    return traced_thread_call(
        "inqtrix.chat",
        attributes,
        fn,
        enrich=_enrich,
        tracer_provider=tracer_provider,
        feature="chat",
        usage_subject=(
            (getattr(principal, "tenant_id", None), user_id, None)
            if user_id is not None
            else None
        ),
    )


def operation_root_span(name: str, **attributes: Any):
    """Decorator: open a root-level operation span around a SYNC core.

    For entry points that ARE the service start but run as a plain
    function (the editor assist cores). Without it their provider spans
    become orphan roots — the same gap the chat wrapper closes for the
    executor-thread path.
    """

    def decorate(fn):
        from functools import wraps

        @wraps(fn)
        def wrapper(*args, **kwargs):
            from inqtrix.observability import semconv

            span_attributes = {
                semconv.GEN_AI_OPERATION_NAME: semconv.OPERATION_INVOKE_AGENT,
                semconv.LANGFUSE_TRACE_NAME: name,
                **attributes,
            }
            with operation_span(name, span_attributes):
                return fn(*args, **kwargs)

        return wrapper

    return decorate
