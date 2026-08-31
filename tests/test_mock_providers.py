"""Every endpoint of the load-measurement provider stand-in must answer.

A stand-in that returns errors does not stop a load run: the pipeline treats
a failed provider call as non-fatal, produces an evidence-free answer, and
reports the run as completed. The measurement then describes a system that
did almost no work — and looks like a success.

So these tests exercise each route the application actually calls and require
a well-formed reply. They deliberately run without the response corpus, which
is untracked operator data: a fresh checkout must still be able to prove the
routes are wired.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_MOCK = (
    Path(__file__).resolve().parent / "load" / "mock-providers" / "mock_providers.py"
)


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    """Load the stand-in with an empty corpus and return a test client.

    A missing corpus is a supported, reported state — the stand-in serves
    synthesised replies and says so on /healthz. The environment is restored
    afterwards so the module under test cannot leak settings into the rest
    of the session.
    """
    from pytest import MonkeyPatch

    corpus = tmp_path_factory.mktemp("corpus-absent")
    patch = MonkeyPatch()
    patch.setenv("INQTRIX_MOCK_CORPUS", str(corpus))
    patch.setenv("INQTRIX_MOCK_LATENCY_SCALE", "0")
    patch.delenv("INQTRIX_MOCK_SELECT", raising=False)

    spec = importlib.util.spec_from_file_location("mock_providers_under_test", _MOCK)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        yield TestClient(module.app)
    finally:
        patch.undo()
        sys.modules.pop(spec.name, None)


def test_health_reports_the_missing_corpus_rather_than_hiding_it(client) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "degraded"
    assert any("corpus" in note for note in body["degraded"])


@pytest.mark.parametrize(
    "path",
    [
        "/azure/openai/v1/chat/completions",
        "/v1/chat/completions",
    ],
)
def test_chat_completion_routes_answer(client, path) -> None:
    response = client.post(
        path,
        json={"messages": [{"role": "system", "content": "Beliebige Anweisung."}]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert "usage" in body


def test_structured_request_returns_json_matching_the_requested_schema(client) -> None:
    """Chunk contextualisation needs exactly as many entries as it asked for."""
    schema = {
        "type": "object",
        "properties": {
            "contexts": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk_number": {"type": "integer"},
                        "context": {"type": "string", "minLength": 1},
                    },
                },
            }
        },
    }
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "system", "content": "Du situierst Textabschnitte."}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "inqtrix_knowledge_chunk_contexts_v1",
                                "schema": schema},
            },
        },
    )
    assert response.status_code == 200, response.text
    parsed = json.loads(response.json()["choices"][0]["message"]["content"])
    contexts = parsed["contexts"]
    assert len(contexts) == 3
    assert [entry["chunk_number"] for entry in contexts] == [1, 2, 3]
    assert all(entry["context"] for entry in contexts)


def test_embeddings_return_one_vector_per_input_in_order(client) -> None:
    response = client.post(
        "/v1/embeddings", json={"input": ["alpha", "beta", "gamma"]}
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert [entry["index"] for entry in data] == [0, 1, 2]
    widths = {len(entry["embedding"]) for entry in data}
    assert len(widths) == 1


@pytest.mark.parametrize(
    "path",
    ["/perplexity/v1/responses", "/v1/responses"],
)
def test_search_routes_return_positional_results(client, path) -> None:
    """Extracted claims reference results by position, so ids start at one."""
    response = client.post(path, json={"input": [{"role": "user", "content": "q"}]})
    assert response.status_code == 200, response.text
    outputs = response.json()["output"]
    results = next(o for o in outputs if o["type"] == "search_results")["results"]
    assert results
    assert [entry["id"] for entry in results] == list(range(1, len(results) + 1))
    assert all(entry["url"] and entry["snippet"] for entry in results)


def test_foundry_and_anthropic_routes_answer(client) -> None:
    foundry = client.post(
        "/foundry/openai/v1/responses",
        json={"input": [{"role": "user", "content": "q"}]},
    )
    assert foundry.status_code == 200, foundry.text
    assert foundry.json()["output"][0]["content"][0]["type"] == "output_text"

    anthropic = client.post(
        "/anthropic/v1/messages",
        json={"system": "Beliebige Anweisung.", "messages": []},
    )
    assert anthropic.status_code == 200, anthropic.text
    assert anthropic.json()["content"][0]["type"] == "text"


def test_stats_count_every_served_call(client) -> None:
    """The evidence that a measurement reached no real provider."""
    before = client.get("/admin/stats").json()["total_calls"]
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "system", "content": "Noch eine Anweisung."}]},
    )
    after = client.get("/admin/stats").json()
    assert after["total_calls"] > before
    # An unrecognised instruction must be reported, never served quietly.
    assert after["unmatched_instructions"]
