"""C1 root-span helpers and the spool replay CLI."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from inqtrix.observability.otel import (
    current_trace_id_hex,
    enrich_current_span,
    operation_span,
    traced_thread_call,
)
from inqtrix.observability.replay import (
    _headers_from_env,
    _traces_url,
    replay_path,
)


@pytest.fixture()
def exporter_and_provider():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter, provider
    provider.shutdown()


def test_enrich_current_span_sets_attributes(exporter_and_provider):
    exporter, provider = exporter_and_provider
    with operation_span(
        "inqtrix.run", {"inqtrix.mode": "research"}, tracer_provider=provider
    ):
        enrich_current_span(
            {
                "langfuse.user.id": "usr_0123456789abcdef",
                "langfuse.session.id": "",  # empty values are skipped
                "inqtrix.outcome": "completed",
            }
        )
        assert current_trace_id_hex() is not None
    (span,) = exporter.get_finished_spans()
    attrs = dict(span.attributes)
    assert attrs["langfuse.user.id"] == "usr_0123456789abcdef"
    assert attrs["inqtrix.outcome"] == "completed"
    assert "langfuse.session.id" not in attrs


def test_enrich_without_active_span_is_noop():
    enrich_current_span({"inqtrix.outcome": "completed"})
    assert current_trace_id_hex() is None


def test_traced_thread_call_opens_span_inside_thread(exporter_and_provider):
    exporter, provider = exporter_and_provider

    def work() -> str:
        # The root span must be ACTIVE in the executing thread so that
        # nested provider spans parent correctly.
        assert current_trace_id_hex() is not None
        with operation_span("child", tracer_provider=provider):
            pass
        return "done"

    def enrich(span, result) -> None:
        span.set_attribute("inqtrix.result", result)

    runner = traced_thread_call(
        "inqtrix.chat",
        {"inqtrix.mode": "direct_chat"},
        work,
        enrich=enrich,
        tracer_provider=provider,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(runner).result() == "done"

    spans = {span.name: span for span in exporter.get_finished_spans()}
    root = spans["inqtrix.chat"]
    child = spans["child"]
    assert dict(root.attributes)["inqtrix.result"] == "done"
    assert child.parent is not None
    assert child.parent.span_id == root.get_span_context().span_id
    assert (
        child.get_span_context().trace_id
        == root.get_span_context().trace_id
    )


def test_traced_thread_call_records_errors(exporter_and_provider):
    exporter, provider = exporter_and_provider

    def boom() -> None:
        raise RuntimeError("agent failed")

    runner = traced_thread_call(
        "inqtrix.chat", None, boom, tracer_provider=provider
    )
    with pytest.raises(RuntimeError, match="agent failed"):
        runner()
    (span,) = exporter.get_finished_spans()
    assert not span.status.is_ok


# ------------------------------------------------------------------ #
# Replay CLI
# ------------------------------------------------------------------ #


def test_traces_url_appends_v1_traces_once():
    assert (
        _traces_url("https://x/api/public/otel")
        == "https://x/api/public/otel/v1/traces"
    )
    assert (
        _traces_url("https://x/api/public/otel/v1/traces")
        == "https://x/api/public/otel/v1/traces"
    )


def test_headers_from_env(monkeypatch):
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_HEADERS",
        "Authorization=Basic abc,x-langfuse-ingestion-version=4",
    )
    headers = _headers_from_env()
    assert headers["Authorization"] == "Basic abc"
    assert headers["x-langfuse-ingestion-version"] == "4"


def test_terminate_native_run_marks_the_open_run_span(exporter_and_provider):
    """A failed segment must never render as a clean run: the terminal
    write happens while the run root span is still current, and the
    shared failure chokepoint stamps ERROR + outcome=failed onto it."""
    from opentelemetry.trace import StatusCode

    from inqtrix.execution_failures import terminate_native_run

    exporter, provider = exporter_and_provider

    class _Handle:
        def __init__(self):
            self.failed_with: tuple | None = None

        def cancel(self, reason):  # pragma: no cover — not this path
            raise AssertionError("failure must not cancel")

        def fail(self, message, *, error_type):
            self.failed_with = (message, error_type)

    handle = _Handle()
    with operation_span("inqtrix.run", None, tracer_provider=provider):
        error_type = terminate_native_run(handle, RuntimeError("boom"))
    assert handle.failed_with is not None
    assert error_type != "client_requested_cancel"
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert dict(span.attributes)["inqtrix.outcome"] == "failed"


def test_replay_path_posts_valid_lines_and_reports_failures(tmp_path):
    spool = tmp_path / "trace-spool-1-000001.otlp.jsonl"
    spool.write_text(
        json.dumps({"resourceSpans": []})
        + "\n"
        + "not-json\n"
        + json.dumps({"resourceSpans": []})
        + "\n"
    )
    calls: list[tuple[str, str]] = []

    def fake_post(url, *, content, headers):
        calls.append((url, content))
        assert headers["x-langfuse-ingestion-version"] == "4"
        return SimpleNamespace(status_code=207 if len(calls) == 2 else 200)

    sent, failed = replay_path(
        Path(tmp_path),
        endpoint="https://langfuse.local/api/public/otel",
        headers={"x-langfuse-ingestion-version": "4"},
        post=fake_post,
    )
    assert sent == 2
    assert failed == 1  # the junk line
    assert all(
        url == "https://langfuse.local/api/public/otel/v1/traces"
        for url, _ in calls
    )


class _FakeHttpxClient:
    """Captures the headers replay.main() builds; never talks HTTP."""

    last_headers: dict[str, str] | None = None

    def __init__(self, timeout):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, *, content, headers):
        _FakeHttpxClient.last_headers = dict(headers)
        return SimpleNamespace(status_code=200)


@pytest.fixture()
def replay_spool(tmp_path):
    spool = tmp_path / "trace-spool-1-000001.otlp.jsonl"
    spool.write_text(json.dumps({"resourceSpans": []}) + "\n")
    return tmp_path


def _run_replay_main(monkeypatch, replay_spool, argv_extra=()):
    import httpx

    from inqtrix.observability import replay as replay_module

    _FakeHttpxClient.last_headers = None
    monkeypatch.setattr(httpx, "Client", _FakeHttpxClient)
    return replay_module.main(
        [
            str(replay_spool),
            "--endpoint",
            "https://langfuse.local/api/public/otel",
            *argv_extra,
        ]
    )


def test_replay_main_reads_auth_from_env(monkeypatch, replay_spool):
    monkeypatch.setenv("LANGFUSE_REPLAY_AUTH", "pk-lf-x:sk-lf-y")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    assert _run_replay_main(monkeypatch, replay_spool) == 0
    headers = _FakeHttpxClient.last_headers
    assert headers is not None
    import base64

    expected = base64.b64encode(b"pk-lf-x:sk-lf-y").decode("ascii")
    assert headers["Authorization"] == f"Basic {expected}"
    assert headers["x-langfuse-ingestion-version"] == "4"


def test_replay_main_rejects_credentials_on_argv(monkeypatch, replay_spool):
    """Credentials come from the environment ONLY — argv is visible in ps
    and shell history, so the CLI must not accept an auth flag at all."""
    import pytest as _pytest

    monkeypatch.setenv("LANGFUSE_REPLAY_AUTH", "pk-env:sk-env")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    with _pytest.raises(SystemExit):
        _run_replay_main(monkeypatch, replay_spool, ("--auth", "pk:sk"))
