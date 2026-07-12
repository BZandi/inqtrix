"""Tests for the optional Prometheus ``/metrics`` endpoint (measure 2.5).

The endpoint is off by default, mounted only when
``INQTRIX_METRICS_ENABLED`` is set AND the optional ``metrics`` extra is
installed, Bearer-gated when an API key is configured, and carries only
bounded-cardinality series (no run-id/subject/session labels). The
admission counter is driven by the real 429 rejection path, so a broken
wiring (counter in a different registry than the one exposed) shows up
here as a stale zero.
"""

from __future__ import annotations

import sys
import threading

import pytest

pytest.importorskip(
    "prometheus_client", reason="optional 'metrics' extra not installed"
)

from prometheus_client.parser import text_string_to_metric_families  # noqa: E402

import inqtrix.research.web_research as web_research_module
import inqtrix.server.metrics as metrics_module
from inqtrix.settings import ServerSettings

from tests.contract._app import (
    make_contract_client,
    minimal_agent_result,
    wait_for_run_status,
)


def _sample_value(text: str, name: str, **labels: str) -> float | None:
    """Return the value of one sample, or ``None`` if absent."""
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name == name and sample.labels == labels:
                return sample.value
    return None


def test_metrics_absent_by_default():
    """Default settings never mount /metrics (opt-in only)."""
    with make_contract_client() as client:
        assert client.get("/metrics").status_code == 404


def test_metrics_missing_extra_stays_off(monkeypatch):
    """Flag on but the extra missing: /metrics unmounted, no crash."""
    # Force the guarded import to fail even though the extra is installed.
    monkeypatch.setitem(sys.modules, "prometheus_client", None)
    with make_contract_client(
        server_settings=ServerSettings(metrics_enabled=True)
    ) as client:
        assert client.get("/metrics").status_code == 404


def test_metrics_endpoint_exposes_run_gauges():
    """The endpoint renders the run gauges with bounded cardinality."""
    with make_contract_client(
        server_settings=ServerSettings(metrics_enabled=True)
    ) as client:
        client.get("/health")  # one templated HTTP sample
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert _sample_value(body, "inqtrix_run_queue_depth") == 0.0
    assert _sample_value(body, "inqtrix_run_active") == 0.0
    # capacity is emitted for the in-process memory backend.
    assert _sample_value(body, "inqtrix_run_capacity") is not None
    # HTTP histograms are labelled by route TEMPLATE, never a raw id.
    assert "http_request" in body


def test_metrics_bearer_gated_when_api_key_set():
    """With an API key, /metrics requires the same Bearer token."""
    with make_contract_client(
        server_settings=ServerSettings(
            api_key="secret-token-123", metrics_enabled=True
        )
    ) as client:
        assert client.get("/metrics").status_code == 401
        ok = client.get(
            "/metrics",
            headers={"Authorization": "Bearer secret-token-123"},
        )
        assert ok.status_code == 200
        assert "inqtrix_run_queue_depth" in ok.text


def test_queue_full_increments_admission_counter(monkeypatch):
    """A real queue-full 429 bumps the reason=queue_full counter.

    The counter must live in the SAME registry the endpoint exposes, so
    this goes red if the admission wiring regresses to a dead counter.
    """
    release = threading.Event()

    def blocking_run(*args, **kwargs):
        release.wait(timeout=5)
        return minimal_agent_result()

    monkeypatch.setattr(web_research_module, "run_web_graph", blocking_run)

    with make_contract_client(
        server_settings=ServerSettings(
            run_max_concurrent=1,
            run_queue_max_size=0,
            metrics_enabled=True,
        ),
    ) as client:
        first = client.post("/v1/runs", json={"question": "blockiert"})
        assert first.status_code == 202
        run_id = first.json()["run_id"]
        wait_for_run_status(client, run_id, "running")

        overflow = client.post("/v1/runs", json={"question": "zu viel"})
        release.set()
        wait_for_run_status(client, run_id, "completed")

        assert overflow.status_code == 429
        body = client.get("/metrics").text

    assert (
        _sample_value(
            body,
            "inqtrix_run_admission_rejected_total",
            reason="queue_full",
        )
        == 1.0
    )
    # The run summary polling never leaks the concrete id into a label.
    assert run_id not in body


def test_record_admission_rejected_is_noop_when_disabled(monkeypatch):
    """Off by default: the call site helper never touches a counter."""
    monkeypatch.setattr(metrics_module, "_admission_counter", None)
    # Must not raise even though metrics were never set up.
    metrics_module.record_admission_rejected("queue_full")
