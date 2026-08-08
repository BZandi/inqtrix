"""Provider tracing wrappers (Paket C0): spans, duration_ms, content policy.

Tests run against their OWN TracerProvider (passed into the wrappers) —
no process-global tracing state is touched.
"""

from __future__ import annotations

import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from inqtrix.observability.content import ContentCapturePolicy
from inqtrix.observability.provider_tracing import (
    TracingLLMProvider,
    instrument_embeddings,
    instrument_llm,
    instrument_search,
)
from inqtrix.providers.base import (
    ChatTurn,
    LLMProvider,
    LLMResponse,
    SearchProvider,
    StructuredLLMResponse,
    ToolCallRequest,
)
from inqtrix.providers.embeddings import EmbeddingProvider
from inqtrix.runtime_logging import describe_llm_provider
from inqtrix.search_result import GroundedSearchResult, GroundedSource
from inqtrix.services.health_service import provider_label

SUMMARY = ContentCapturePolicy(capture_content=False, max_attr_bytes=32_768)
FORENSIC = ContentCapturePolicy(capture_content=True, max_attr_bytes=32_768)


class FakeLLM(LLMProvider):
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def complete(self, prompt, **kwargs) -> str:
        # Real providers report a bare completion's usage only by adding it
        # to the state accumulator (`track_tokens`); there is no return
        # value carrying it. Mirror that here, otherwise the meter this
        # path reads stays empty and the test proves nothing.
        state = kwargs.get("state")
        if state is not None:
            state["total_prompt_tokens"] = (
                int(state.get("total_prompt_tokens", 0) or 0) + 11
            )
            state["total_completion_tokens"] = (
                int(state.get("total_completion_tokens", 0) or 0) + 7
            )
        return "text-answer"

    def complete_with_metadata(self, prompt, **kwargs) -> LLMResponse:
        if self.fail:
            raise RuntimeError("backend down")
        return LLMResponse(
            content="answer https://api.example.com/x?api_key=SECRET123",
            prompt_tokens=11,
            completion_tokens=7,
            model="fake-1",
            finish_reason="stop",
            raw={"api_key": "SECRET123", "usage": {"total": 18}},
            request_max_tokens=256,
        )

    def complete_structured(self, prompt, **kwargs) -> StructuredLLMResponse:
        return StructuredLLMResponse(
            parsed={"ok": True},
            content='{"ok": true}',
            prompt_tokens=3,
            completion_tokens=2,
            model="fake-1",
            finish_reason="stop",
            schema_name=kwargs.get("schema_name", ""),
        )

    def supports_tool_calls(self, *, model=None) -> bool:
        return True

    def supports_structured_output(self, *, model=None) -> bool:
        return True

    def chat(self, messages, **kwargs) -> ChatTurn:
        return ChatTurn(
            text="",
            tool_calls=(
                ToolCallRequest(id="c1", name="web_search", arguments={"q": "x"}),
            ),
            finish_reason="tool_calls",
            model="fake-1",
            prompt_tokens=5,
            completion_tokens=4,
        )

    def is_available(self) -> bool:
        return True


class FakeSearch(SearchProvider):
    def search(self, query, **kwargs) -> GroundedSearchResult:
        return GroundedSearchResult(
            answer="the answer",
            sources=[GroundedSource(url="https://example.com", rank=1)],
            prompt_tokens=9,
            completion_tokens=6,
        )

    def is_available(self) -> bool:
        return True

    @property
    def search_model(self) -> str:
        return "fake-engine"


class FakeEmbeddings(EmbeddingProvider):
    @property
    def default_model(self) -> str:
        return "fake-embed"

    def embed_documents(self, texts, *, model=None):
        return [[0.0] for _ in texts]

    def embed_query(self, text, *, model=None):
        return [0.0]


@pytest.fixture()
def exporter_and_provider():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter, provider
    provider.shutdown()


def _wrap_llm(policy, provider, fail: bool = False) -> TracingLLMProvider:
    return TracingLLMProvider(
        FakeLLM(fail=fail),
        provider_name="fake",
        policy=policy,
        tracer_provider=provider,
    )


def test_metadata_span_and_duration(exporter_and_provider):
    exporter, provider = exporter_and_provider
    llm = _wrap_llm(SUMMARY, provider)
    response = llm.complete_with_metadata("why?")
    assert response.duration_ms > 0
    (span,) = exporter.get_finished_spans()
    assert span.name == "text_completion"
    attrs = dict(span.attributes)
    assert attrs["gen_ai.operation.name"] == "text_completion"
    assert attrs["gen_ai.provider.name"] == "fake"
    assert attrs["gen_ai.response.model"] == "fake-1"
    assert attrs["gen_ai.usage.input_tokens"] == 11
    assert attrs["gen_ai.usage.output_tokens"] == 7
    assert attrs["gen_ai.request.max_tokens"] == 256
    assert tuple(attrs["gen_ai.response.finish_reasons"]) == ("stop",)
    # Summary policy: no content attributes at all.
    assert "gen_ai.input.messages" not in attrs
    assert "gen_ai.output.messages" not in attrs
    assert "inqtrix.response.raw" not in attrs


def test_forensic_content_is_redacted(exporter_and_provider):
    exporter, provider = exporter_and_provider
    llm = _wrap_llm(FORENSIC, provider)
    llm.complete_with_metadata(
        "check https://api.example.com/x?api_key=SECRET123"
    )
    (span,) = exporter.get_finished_spans()
    attrs = dict(span.attributes)
    assert "SECRET123" not in attrs["gen_ai.input.messages"]
    assert "SECRET123" not in attrs["gen_ai.output.messages"]
    raw = attrs["inqtrix.response.raw"]
    assert "SECRET123" not in raw
    # Non-sensitive structure survives the sanitizer.
    assert "usage" in raw


def test_truncation_emits_visible_event(exporter_and_provider):
    exporter, provider = exporter_and_provider
    tiny = ContentCapturePolicy(capture_content=True, max_attr_bytes=64)
    llm = TracingLLMProvider(
        FakeLLM(),
        provider_name="fake",
        policy=tiny,
        tracer_provider=provider,
    )
    llm.complete_with_metadata("x" * 500)
    (span,) = exporter.get_finished_spans()
    events = [e for e in span.events if e.name == "inqtrix.truncation"]
    assert events
    attrs = dict(events[0].attributes)
    assert attrs["original_size"] > attrs["capped_size"]
    assert attrs["limit_name"]


def test_error_is_recorded_on_span(exporter_and_provider):
    exporter, provider = exporter_and_provider
    llm = _wrap_llm(SUMMARY, provider, fail=True)
    with pytest.raises(RuntimeError, match="backend down"):
        llm.complete_with_metadata("boom")
    (span,) = exporter.get_finished_spans()
    assert not span.status.is_ok
    assert any(e.name == "exception" for e in span.events)


def test_chat_span_counts_tool_calls_and_sets_duration(
    exporter_and_provider,
):
    exporter, provider = exporter_and_provider
    llm = _wrap_llm(FORENSIC, provider)
    turn = llm.chat([{"role": "user", "content": "hi"}])
    assert turn.duration_ms > 0
    (span,) = exporter.get_finished_spans()
    attrs = dict(span.attributes)
    assert span.name == "chat"
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["inqtrix.response.tool_call_count"] == 1
    output = json.loads(attrs["gen_ai.output.messages"])
    assert output[0]["tool_calls"][0]["name"] == "web_search"


def test_structured_span_carries_schema_name(exporter_and_provider):
    exporter, provider = exporter_and_provider
    llm = _wrap_llm(SUMMARY, provider)
    response = llm.complete_structured(
        "extract", schema={"type": "object"}, schema_name="claims_v1"
    )
    assert response.duration_ms > 0
    (span,) = exporter.get_finished_spans()
    assert dict(span.attributes)["inqtrix.request.schema_name"] == "claims_v1"


def test_search_span_metadata_and_content_policy(exporter_and_provider):
    exporter, provider = exporter_and_provider
    search = instrument_search(
        FakeSearch(), policy=SUMMARY, tracer_provider=provider
    )
    result = search.search("find https://x.example?token=SECRET987 things")
    assert result.answer == "the answer"
    (span,) = exporter.get_finished_spans()
    attrs = dict(span.attributes)
    assert span.name == "web_search"
    assert attrs["inqtrix.search.provider"] == "FakeSearch"
    assert attrs["inqtrix.search.engine"] == "fake-engine"
    assert attrs["inqtrix.search.source_count"] == 1
    assert attrs["inqtrix.search.input_tokens"] == 9
    # Summary policy: query and answer are user content and stay out.
    assert "inqtrix.search.query" not in attrs
    assert "inqtrix.search.answer" not in attrs


def test_search_query_is_content_gated_and_redacted(exporter_and_provider):
    exporter, provider = exporter_and_provider
    search = instrument_search(
        FakeSearch(), policy=FORENSIC, tracer_provider=provider
    )
    search.search("find https://x.example?token=SECRET987 things")
    (span,) = exporter.get_finished_spans()
    attrs = dict(span.attributes)
    assert "inqtrix.search.query" in attrs
    assert "SECRET987" not in attrs["inqtrix.search.query"]
    assert "inqtrix.search.answer" in attrs


def test_embeddings_span(exporter_and_provider):
    exporter, provider = exporter_and_provider
    embeddings = instrument_embeddings(
        FakeEmbeddings(), policy=SUMMARY, tracer_provider=provider
    )
    embeddings.embed_documents(["a", "b", "c"])
    (span,) = exporter.get_finished_spans()
    attrs = dict(span.attributes)
    assert attrs["gen_ai.operation.name"] == "embeddings"
    assert attrs["inqtrix.embeddings.text_count"] == 3
    assert attrs["gen_ai.request.model"] == "fake-embed"


def test_wrapper_is_transparent_for_labels_and_capabilities():
    llm = instrument_llm(FakeLLM(), provider_name="fake", policy=SUMMARY)
    assert provider_label(llm) == "FakeLLM"
    assert describe_llm_provider(llm)["provider"] == "FakeLLM"
    assert llm.supports_tool_calls() is True
    assert llm.supports_structured_output() is True
    assert llm.is_available() is True
    # Idempotent: instrumenting twice never stacks wrappers.
    assert instrument_llm(llm, provider_name="fake", policy=SUMMARY) is llm


def test_bare_complete_exports_the_same_usage_the_ledger_books(
    exporter_and_provider,
):
    """A bare `complete()` span carries usage, like every other call path.

    Without it the span reaches Langfuse with no usage attribute at all and
    Langfuse infers token counts from the message text — producing numbers
    that silently disagree with the ledger and the quota. Observed live:
    two research generations showed 934/175 and 7328/240 in Langfuse while
    the ledger booked 846/142 and 6193/210 for the same calls.
    """
    exporter, provider = exporter_and_provider
    llm = _wrap_llm(SUMMARY, provider)
    state: dict = {}

    assert llm.complete("why?", state=state) == "text-answer"

    (span,) = exporter.get_finished_spans()
    attrs = dict(span.attributes)
    # The exported numbers are the accumulator delta — the same source the
    # ledger and Prometheus read, so the three can never diverge.
    assert attrs["gen_ai.usage.input_tokens"] == 11
    assert attrs["gen_ai.usage.output_tokens"] == 7
    assert "inqtrix.usage_unavailable" not in attrs


def test_bare_complete_without_state_marks_usage_unavailable(
    exporter_and_provider,
):
    """No accumulator means no usage — say so instead of exporting zero.

    Two call sites pass `state=None`. Exporting `0` there would look like a
    free call; staying silent would let Langfuse invent a number again.
    A visible marker keeps the gap honest, matching `inqtrix.raw_unavailable`.
    """
    exporter, provider = exporter_and_provider
    llm = _wrap_llm(SUMMARY, provider)

    assert llm.complete("why?") == "text-answer"

    (span,) = exporter.get_finished_spans()
    attrs = dict(span.attributes)
    assert attrs["inqtrix.usage_unavailable"] is True
    assert "gen_ai.usage.input_tokens" not in attrs


def test_wrappers_work_without_any_tracer_provider():
    """No global tracing installed: pure duration measurement, no error."""
    llm = instrument_llm(FakeLLM(), provider_name="fake", policy=SUMMARY)
    response = llm.complete_with_metadata("plain")
    assert response.duration_ms > 0
    assert llm.complete("plain") == "text-answer"
