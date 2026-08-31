"""OTel fundament (INQTRIX_TRACING): setup modes, spool, propagation, envelope.

Tests build their OWN TracerProvider instances (never installing the
process-global one) so the suite stays free of global tracing state;
``setup_tracing``'s install-once contract is verified against a stubbed
global setter.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
)

from inqtrix.observability import otel as otel_module
from inqtrix.observability.context import bind_log_context, reset_log_context
from inqtrix.observability.otel import (
    build_tracer_provider,
    run_execute_span,
    setup_tracing,
)
from inqtrix.observability.propagation import (
    TELEMETRY_FIELD,
    extract_context,
    inject_traceparent,
)
from inqtrix.observability.spool import build_spool_exporter
from inqtrix.runtime_logging import emit_runtime_event, sanitize_event_payload
from inqtrix.server.runs import RunStore
from inqtrix.settings import ObservabilitySettings


def _settings(mode: str, **overrides) -> SimpleNamespace:
    return SimpleNamespace(
        observability=ObservabilitySettings(tracing=mode, **overrides)
    )


def test_mode_off_builds_nothing():
    assert build_tracer_provider(_settings("off"), service_role="api") is None


def test_mode_local_yields_valid_span_ids_without_exporters():
    provider = build_tracer_provider(_settings("local"), service_role="api")
    try:
        assert provider is not None
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("probe") as span:
            context = span.get_span_context()
            assert context.is_valid
        # local mode: no span processor may export anything.
        active = provider._active_span_processor._span_processors
        assert not [
            p
            for p in active
            if isinstance(p, (BatchSpanProcessor, SimpleSpanProcessor))
        ]
    finally:
        provider.shutdown()


def test_mode_file_spools_otlp_json_lines(tmp_path):
    provider = build_tracer_provider(
        _settings(
            "file",
            trace_spool_dir=str(tmp_path),
            trace_spool_max_mb=16,
        ),
        service_role="worker",
    )
    try:
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("spooled"):
            pass
        assert provider.force_flush()
        files = list(tmp_path.glob("trace-spool-*.otlp.jsonl"))
        assert len(files) == 1
        line = json.loads(files[0].read_text().strip())
        resource_spans = line["resourceSpans"]
        span_names = [
            span["name"]
            for rs in resource_spans
            for scope in rs["scopeSpans"]
            for span in scope["spans"]
        ]
        assert "spooled" in span_names
        attributes = {
            entry["key"]: entry["value"]
            for rs in resource_spans
            for entry in rs["resource"]["attributes"]
        }
        assert attributes["service.name"] == {
            "stringValue": "inqtrix-worker"
        }
    finally:
        provider.shutdown()


def test_spool_total_cap_reclaims_stale_but_protects_fresh_foreign(tmp_path):
    import os as _os
    import time as _time

    # A FRESH foreign spool file may still be actively written by
    # another process — the cap must never touch it.
    foreign_fresh = tmp_path / "trace-spool-1-aa11bb22-000001.otlp.jsonl"
    foreign_fresh.write_text("x" * 4096)
    # A STALE file (writer exited or idle) is reclaimable regardless of
    # who created it — otherwise dead processes leak disk forever.
    stale = tmp_path / "trace-spool-999-cc33dd44-000001.otlp.jsonl"
    stale.write_text("y" * 4096)
    backdated = _time.time() - 3600
    _os.utime(stale, (backdated, backdated))
    exporter = build_spool_exporter(str(tmp_path), 0)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        with provider.get_tracer("test").start_as_current_span("fresh"):
            pass
        assert foreign_fresh.exists()
        assert not stale.exists()
        assert list(tmp_path.glob("trace-spool-*.otlp.jsonl"))
    finally:
        provider.shutdown()


def test_spool_never_appends_into_a_preexisting_lax_mode_file(tmp_path):
    """PID reuse must not route fresh forensic content into a legacy
    world-readable file: the writer creates its slices O_EXCL and skips
    any name that already exists."""
    import os as _os
    import stat as _stat

    exporter = build_spool_exporter(str(tmp_path), 16)
    # Pre-create the EXACT first slice name this writer would pick,
    # with lax permissions (simulates a pre-fix leftover after pid
    # reuse).
    legacy = tmp_path / (
        f"trace-spool-{exporter._pid}-{exporter._token}-000001.otlp.jsonl"
    )
    legacy.write_text("legacy")
    _os.chmod(legacy, 0o644)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        with provider.get_tracer("test").start_as_current_span("s"):
            pass
        assert legacy.read_text() == "legacy"
        written = [
            f
            for f in tmp_path.glob("trace-spool-*.otlp.jsonl")
            if f != legacy and f.stat().st_size > 0
        ]
        (slice_file,) = written
        # The point of this test is that the LEGACY world-readable file
        # was not appended to; the fresh slice carries the writer's own
        # mode (owner + group, never other).
        assert _stat.S_IMODE(_os.stat(slice_file).st_mode) == 0o640
    finally:
        provider.shutdown()


def test_spool_is_never_readable_by_other(tmp_path):
    """Spans can carry forensic prompts, so NOTHING outside the owner and
    the group may read them.

    Group access is deliberate, not laxity: containers run as an
    arbitrary non-root UID in group 0 and write through the group bit,
    and the API process reads the worker's slices through the shared
    spool volume. Forcing owner-only would lock the process out of its
    own spool and break the trace export.
    """
    import os as _os
    import stat as _stat

    spool_dir = tmp_path / "traces"
    exporter = build_spool_exporter(str(spool_dir), 16)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        with provider.get_tracer("test").start_as_current_span("secret"):
            pass
        dir_mode = _stat.S_IMODE(_os.stat(spool_dir).st_mode)
        assert dir_mode & 0o007 == 0, "spool directory reachable by other"
        assert dir_mode & _stat.S_IRWXU == _stat.S_IRWXU
        (slice_file,) = spool_dir.glob("trace-spool-*.otlp.jsonl")
        file_mode = _stat.S_IMODE(_os.stat(slice_file).st_mode)
        assert file_mode & 0o007 == 0, "spool slice readable by other"
        assert file_mode == 0o640
    finally:
        provider.shutdown()


def test_setup_tracing_installs_exactly_once(monkeypatch):
    installed: list[object] = []
    monkeypatch.setattr(
        otel_module, "_state", {"provider": None, "installed": False}
    )
    monkeypatch.setattr(
        otel_trace, "set_tracer_provider", installed.append
    )
    first = setup_tracing(_settings("local"), service_role="api")
    second = setup_tracing(_settings("local"), service_role="api")
    try:
        assert first is not None
        assert second is first
        assert len(installed) == 1
    finally:
        first.shutdown()


def test_setup_tracing_off_touches_nothing(monkeypatch):
    monkeypatch.setattr(
        otel_module, "_state", {"provider": None, "installed": False}
    )
    assert setup_tracing(_settings("off"), service_role="api") is None
    assert otel_module._state["installed"] is False


def test_traceparent_roundtrip_through_payload():
    provider = build_tracer_provider(_settings("local"), service_role="api")
    try:
        tracer = provider.get_tracer("test")
        payload: dict = {"question": "q"}
        with tracer.start_as_current_span("submit") as span:
            inject_traceparent(payload)
            expected_trace_id = span.get_span_context().trace_id
        carrier = payload[TELEMETRY_FIELD]
        assert "traceparent" in carrier
        context = extract_context(payload)
        assert context is not None
        extracted = otel_trace.get_current_span(context).get_span_context()
        assert extracted.trace_id == expected_trace_id
    finally:
        provider.shutdown()


def test_extract_context_tolerates_legacy_payloads():
    assert extract_context(None) is None
    assert extract_context({}) is None
    assert extract_context({"question": "old row"}) is None
    assert extract_context({TELEMETRY_FIELD: "not-a-mapping"}) is None


def test_run_execute_span_is_safe_without_tracing():
    with run_execute_span(
        run_id="run_x", tenant_id="default", attempt=1, payload=None
    ) as span:
        # Non-recording (no global provider installed by the suite),
        # but the context manager itself must never fail.
        assert span is not None


def test_memory_store_submit_persists_traceparent():
    provider = build_tracer_provider(_settings("local"), service_role="api")
    store = RunStore(
        max_concurrent=1,
        max_queue_size=2,
        completed_ttl_seconds=30,
        event_buffer_size=20,
    )
    try:
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("api-request"):
            summary = store.submit(
                question="traced",
                stack_name="default",
                work=lambda handle: None,
                request_payload={"body": {}},
            )
        record = store._records[summary["run_id"]]
        assert "traceparent" in record.request_payload[TELEMETRY_FIELD]
    finally:
        provider.shutdown()


def test_memory_store_trace_id_survives_event_ring_eviction():
    """The admin trace lookup must NOT depend on the bounded SSE replay
    ring: long runs evict old events, the captured field survives."""
    store = RunStore(
        max_concurrent=1,
        max_queue_size=2,
        completed_ttl_seconds=30,
        event_buffer_size=5,
    )

    def work(handle):
        handle.emit("inqtrix.run.trace", {"trace_id": "a" * 32})
        for index in range(20):  # evicts the trace event from the ring
            handle.emit("phase", {"status": f"step-{index}"})
        handle.emit_answer("done")

    summary = store.submit(
        question="ring", stack_name="default", work=work
    )
    run_id = summary["run_id"]
    import time as _time

    from inqtrix.server.runs import RunStatus

    deadline = _time.time() + 10
    while (
        store._records[run_id].status is not RunStatus.COMPLETED
        and _time.time() < deadline
    ):
        _time.sleep(0.05)
    record = store._records[run_id]
    assert record.status is RunStatus.COMPLETED
    assert all(
        event["type"] != "inqtrix.run.trace" for event in record.events
    )
    assert store.trace_id(run_id) == "a" * 32


def test_sanitize_event_payload_keeps_correlation_envelope():
    sanitized = sanitize_event_payload(
        "run_end",
        {
            "event": "run_end",
            "run_id": "run_1",
            "status": "completed",
            "trace_id": "a" * 32,
            "span_id": "b" * 16,
            "request_id": "req-1",
        },
    )
    assert sanitized["trace_id"] == "a" * 32
    assert sanitized["span_id"] == "b" * 16
    assert sanitized["request_id"] == "req-1"


def test_runtime_event_carries_the_request_id():
    """The event's join key back to the HTTP request.

    Trace and span ids are NOT repeated in the payload — the event
    hangs on its span, which already has both. The request id is the
    one correlation field the span does not carry.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from inqtrix.observability import otel as otel_module

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    class _ProxyTrace:
        def get_tracer(self, name, tracer_provider=None):
            return (tracer_provider or provider).get_tracer(name)

        def get_current_span(self):
            return otel_trace.get_current_span()

    original = otel_module._otel_trace
    otel_module._otel_trace = _ProxyTrace()
    tokens = bind_log_context(request_id="req-99")
    try:
        with provider.get_tracer("test").start_as_current_span("run-segment"):
            emit_runtime_event(
                "run_end", {"run_id": "run_1", "status": "completed"}
            )
    finally:
        otel_module._otel_trace = original
        reset_log_context(tokens)
        provider.shutdown()

    (span,) = exporter.get_finished_spans()
    (event,) = span.events
    assert event.attributes["request_id"] == "req-99"
    assert "trace_id" not in event.attributes
    assert "span_id" not in event.attributes


def test_span_event_reports_value_truncation(monkeypatch):
    """A capped nested value must raise the
    truncation event — the old 2048-byte cut silently produced invalid
    JSON with no marker anywhere."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from inqtrix.observability import otel as otel_module

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel_module, "_event_truncation_warned", False)

    class _ProxyTrace:
        def get_tracer(self, *args, **kwargs):
            return provider.get_tracer("test")

        def get_current_span(self):
            import opentelemetry.trace as real

            return real.get_current_span()

        def set_span_in_context(self, *args, **kwargs):
            import opentelemetry.trace as real

            return real.set_span_in_context(*args, **kwargs)

    monkeypatch.setattr(otel_module, "_otel_trace", _ProxyTrace())
    # Deliberately past the 256 KiB backstop: normal payloads must NOT
    # truncate (that is the contract), so the test has to be pathological
    # to exercise the reporting path at all.
    big = ["cit_" + "x" * 40 for _ in range(8000)]
    with provider.get_tracer("test").start_as_current_span("node"):
        otel_module.add_span_event("claim_merge", {"citation_ids": big})
    spans = exporter.get_finished_spans()
    names = [e.name for e in spans[0].events]
    assert "claim_merge" in names
    assert "inqtrix.truncation" in names
    trunc = next(e for e in spans[0].events if e.name == "inqtrix.truncation")
    from inqtrix.observability import semconv

    assert trunc.attributes[semconv.TRUNCATION_LIMIT_NAME] == (
        "claim_merge.citation_ids"
    )
    assert trunc.attributes[semconv.TRUNCATION_ORIGINAL_SIZE] > (
        trunc.attributes[semconv.TRUNCATION_CAPPED_SIZE]
    )


def test_forensic_without_recording_sink_warns(monkeypatch, caplog):
    """Forensic depth with tracing off/local or
    a sampled-out rate records NOTHING — that must never be silent."""
    from inqtrix.settings import Settings

    monkeypatch.setenv("OBSERVABILITY_PROFILE", "forensic")
    monkeypatch.setenv("INQTRIX_TRACING", "local")
    with caplog.at_level("WARNING", logger="inqtrix"):
        Settings()
    assert any("KEIN Span" in r.message for r in caplog.records)

    caplog.clear()
    monkeypatch.setenv("INQTRIX_TRACING", "otlp")
    monkeypatch.setenv("INQTRIX_TRACE_SAMPLE_RATE", "0.2")
    with caplog.at_level("WARNING", logger="inqtrix"):
        Settings()
    assert any("Head-Sampling" in r.message for r in caplog.records)


def test_server_span_makes_the_persisted_payload_carry_the_trace(monkeypatch):
    """The trace chain must hold through the REAL ASGI stack.

    Hand-opening a span around ``store.submit`` proves only that the
    helper works; it cannot detect that no span is active during an
    actual request — in which case ``inject_traceparent`` writes nothing
    and the worker starts a disconnected root span.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from inqtrix.observability.propagation import (
        TELEMETRY_FIELD,
        inject_traceparent,
    )
    from inqtrix.server.request_context import RequestContextMiddleware

    provider = TracerProvider()
    monkeypatch.setattr(otel_trace, "get_tracer_provider", lambda: provider)
    captured: dict = {}

    async def _submit(request):
        # Stands in for the run store's submit: it injects whatever
        # trace context the request path made current.
        payload: dict = {"body": {}}
        inject_traceparent(payload)
        captured.update(payload)
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/v1/runs", _submit, methods=["POST"])])
    client = TestClient(RequestContextMiddleware(app))
    client.post("/v1/runs", json={})

    assert TELEMETRY_FIELD in captured, (
        "no traceparent persisted — the request path has no active span, "
        "so the worker would open a disconnected root span"
    )
    assert "traceparent" in captured[TELEMETRY_FIELD]


def test_server_span_continues_an_upstream_trace(monkeypatch):
    """An incoming traceparent must be adopted, not replaced."""
    from opentelemetry.sdk.trace import TracerProvider
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from inqtrix.server.request_context import RequestContextMiddleware

    provider = TracerProvider()
    monkeypatch.setattr(otel_trace, "get_tracer_provider", lambda: provider)
    seen: dict = {}

    async def _echo(request):
        seen["trace_id"] = otel_trace.get_current_span(

        ).get_span_context().trace_id
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/v1/runs", _echo, methods=["POST"])])
    client = TestClient(RequestContextMiddleware(app))
    upstream = "4bf92f3577b34da6a3ce929d0e0e4736"
    client.post(
        "/v1/runs",
        json={},
        headers={
            "traceparent": f"00-{upstream}-00f067aa0ba902b7-01",
        },
    )
    assert format(seen["trace_id"], "032x") == upstream


def test_probe_paths_are_not_traced(monkeypatch):
    """Liveness/readiness/scrape traffic must not flood the waterfall."""
    from opentelemetry.sdk.trace import TracerProvider
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from inqtrix.server.request_context import RequestContextMiddleware

    provider = TracerProvider()
    monkeypatch.setattr(otel_trace, "get_tracer_provider", lambda: provider)
    recorded: dict = {}

    async def _health(request):
        recorded["recording"] = otel_trace.get_current_span().is_recording()
        return JSONResponse({"status": "ok"})

    app = Starlette(
        routes=[
            Route("/health", _health),
            Route("/internal/collaboration/policy-events", _health),
        ]
    )
    client = TestClient(RequestContextMiddleware(app))
    client.get("/health")
    assert recorded["recording"] is False
    # Service-to-service polling is continuous background chatter; one
    # knowledge run's window carried twice as many of these as the run
    # itself, which would bury every user-initiated trace.
    recorded.clear()
    client.get("/internal/collaboration/policy-events")
    assert recorded["recording"] is False
