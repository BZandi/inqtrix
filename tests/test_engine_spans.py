"""C2 engine spans: forensic→span-event bridge, tool spans, event flattening.

The helpers under test resolve the process-global tracer; the fixture
swaps the module's ``_otel_trace`` for a proxy bound to a test-local
provider, so nothing global is installed and suites stay isolated.
"""

from __future__ import annotations

import json
import logging

import pytest
from opentelemetry import trace as real_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from inqtrix.agents.kernel import tools as kernel_tools
from inqtrix.observability import otel as otel_module
from inqtrix.observability.otel import add_span_event, operation_span
from inqtrix.runtime_logging import emit_runtime_event


class _ProxyTrace:
    """`_otel_trace` stand-in routing get_tracer to a test provider."""

    def __init__(self, provider: TracerProvider) -> None:
        self._provider = provider

    def get_tracer(self, name: str, tracer_provider=None):
        return (tracer_provider or self._provider).get_tracer(name)

    def get_current_span(self):
        return real_trace.get_current_span()

    def set_tracer_provider(self, provider) -> None:  # pragma: no cover
        raise AssertionError("tests must never install a global provider")


@pytest.fixture()
def recording(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel_module, "_otel_trace", _ProxyTrace(provider))
    yield exporter
    provider.shutdown()


def test_add_span_event_flattens_nested_values(recording):
    with operation_span("probe"):
        add_span_event(
            "stop_cascade",
            {
                "confidence": 82,
                "final_done": True,
                "reason": "confidence_stop",
                "scores": {"coverage": 3, "evidence": 5},
                "skipped": None,
            },
        )
    (span,) = recording.get_finished_spans()
    (event,) = [e for e in span.events if e.name == "stop_cascade"]
    attrs = dict(event.attributes)
    assert attrs["confidence"] == 82
    assert attrs["final_done"] is True
    assert attrs["reason"] == "confidence_stop"
    assert json.loads(attrs["scores"]) == {"coverage": 3, "evidence": 5}
    assert "skipped" not in attrs


def test_forensic_events_bridge_onto_the_active_span(recording):
    """The bridge fires even when the log level gate is CLOSED."""
    logger = logging.getLogger("inqtrix")
    previous_level = logger.level
    logger.setLevel(logging.CRITICAL)  # log sink off — trace sink stays on
    try:
        with operation_span("evaluate"):
            emit_runtime_event(
                "stop_cascade",
                {
                    "run_id": "run_1",
                    "confidence": 77,
                    "final_done": False,
                    "round": 2,
                },
            )
    finally:
        logger.setLevel(previous_level)
    (span,) = recording.get_finished_spans()
    (event,) = [e for e in span.events if e.name == "stop_cascade"]
    attrs = dict(event.attributes)
    assert attrs["confidence"] == 77
    assert attrs["round"] == 2
    assert attrs["run_id"] == "run_1"


def test_bridge_is_silent_without_active_span(recording):
    logger = logging.getLogger("inqtrix")
    previous_level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        emit_runtime_event(
            "stop_cascade", {"run_id": "run_1"}
        )
    finally:
        logger.setLevel(previous_level)
    assert recording.get_finished_spans() == ()


def test_capability_invocation_gets_a_tool_span(recording, monkeypatch):
    from inqtrix.observability.content import ContentCapturePolicy

    monkeypatch.setattr(
        kernel_tools,
        "_invoke_capability_inner",
        lambda capability_id, payload: {"ok": True},
    )
    monkeypatch.setattr(
        kernel_tools,
        "_tool_content_policy",
        lambda: ContentCapturePolicy(
            capture_content=False, max_attr_bytes=32_768
        ),
    )
    result = kernel_tools._invoke_capability(
        "web.instant", {"query": "hello https://x?api_key=SECRET"}
    )
    assert result == {"ok": True}
    (span,) = recording.get_finished_spans()
    attrs = dict(span.attributes)
    assert span.name == "web.instant"
    assert attrs["gen_ai.operation.name"] == "execute_tool"
    assert attrs["inqtrix.tool.capability"] == "web.instant"
    # Tool args are user content: without content capture the span
    # carries NO argument text in any form.
    assert "inqtrix.tool.args" not in attrs
    assert "inqtrix.tool.args_preview" not in attrs
    assert "inqtrix.tool.failure_code" not in attrs


def test_capability_args_appear_only_with_content_capture(
    recording, monkeypatch
):
    from inqtrix.observability.content import ContentCapturePolicy

    monkeypatch.setattr(
        kernel_tools,
        "_invoke_capability_inner",
        lambda capability_id, payload: {"ok": True},
    )
    monkeypatch.setattr(
        kernel_tools,
        "_tool_content_policy",
        lambda: ContentCapturePolicy(
            capture_content=True, max_attr_bytes=32_768
        ),
    )
    kernel_tools._invoke_capability(
        "web.instant", {"query": "hello https://x?api_key=SECRET"}
    )
    (span,) = recording.get_finished_spans()
    attrs = dict(span.attributes)
    args_text = attrs["inqtrix.tool.args"]
    assert "hello" in args_text
    # The redaction pipeline must have stripped the credential.
    assert "SECRET" not in args_text


def test_capability_failure_text_marks_the_span(recording, monkeypatch):
    failure = kernel_tools._CapabilityFailureText(
        "nicht verfuegbar", code="capability_denied"
    )
    monkeypatch.setattr(
        kernel_tools,
        "_invoke_capability_inner",
        lambda capability_id, payload: failure,
    )
    result = kernel_tools._invoke_capability("knowledge.search", {})
    assert result is failure
    (span,) = recording.get_finished_spans()
    assert (
        dict(span.attributes)["inqtrix.tool.failure_code"]
        == "capability_denied"
    )
