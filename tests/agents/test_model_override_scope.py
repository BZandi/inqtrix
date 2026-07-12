"""Per-run model overrides scope to the THINKING nodes (R3).

An explicit request override (model/tier/effort) and a skill pin must
strengthen plan/synthesis/answer — and ONLY those. Assembly-line nodes
(intake, critic, ...) stay on the tier map; paying frontier prices for
classification calls is exactly what the scope prevents. Pinned via the
`model_resolution` provenance events, so the assertion covers what the
run actually reports.
"""

from __future__ import annotations

from typing import Any

from tests.agents.test_workspace_agent import (
    make_agent_client,
    wait_status,
)
from tests.agents.test_workspace_agent_skills import _create_skill


def _resolutions(client, run_id: str) -> dict[str, dict[str, Any]]:
    # format=json returns the replay buffer IMMEDIATELY — the SSE
    # stream would stay open on a PARKED (non-terminal) run and block
    # the test forever.
    events = client.get(
        f"/v1/runs/{run_id}/events", params={"format": "json"}
    ).json()["data"]
    return {
        event["data"]["node"]: event["data"]
        for event in events
        if event["type"] == "inqtrix.node.model_resolution"
    }


def _submit(client, body: dict[str, Any]) -> str:
    response = client.post(
        "/v1/runs",
        json={
            "question": "Erstelle eine Marktanalyse.",
            "mode": "workspace_agent",
            **body,
        },
    )
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def test_explicit_model_and_effort_hit_only_thinking_nodes(monkeypatch):
    client = make_agent_client(monkeypatch)
    with client:
        run_id = _submit(
            client,
            {"agent_overrides": {"model": "opus-x", "effort": "xhigh"}},
        )
        wait_status(client, run_id, {"waiting_for_approval"})
        nodes = _resolutions(client, run_id)
        for node in ("agent_plan", "agent_synthesis", "agent_answer"):
            assert nodes[node]["model"] == "opus-x", node
            assert nodes[node]["model_source"] == "explicit_request"
            assert nodes[node]["effort"] == "xhigh"
        # Assembly-line nodes keep the tier map untouched.
        assert nodes["agent_intake"]["model"] == "fast-model"
        assert nodes["agent_intake"]["model_source"] == "tier:fast"
        assert nodes["agent_intake"]["effort"] == ""
        assert nodes["agent_critic"]["model"] == "fast-model"


def test_tier_override_scopes_the_same_way(monkeypatch):
    client = make_agent_client(monkeypatch)
    with client:
        run_id = _submit(client, {"agent_overrides": {"model_tier": "mid"}})
        wait_status(client, run_id, {"waiting_for_approval"})
        nodes = _resolutions(client, run_id)
        assert nodes["agent_plan"]["model"] == "mid-model"
        assert nodes["agent_plan"]["model_source"] == "tier:mid"
        assert nodes["agent_intake"]["model"] == "fast-model"


def test_skill_pin_scopes_to_thinking_nodes(monkeypatch):
    client = make_agent_client(monkeypatch)
    with client:
        skill_id = _create_skill(
            client,
            clarification_points=[],
            instructions_markdown="Arbeite knapp.",
            model_tier="high",
            effort="xhigh",
            # The default payload says never — that SKIPS the plan gate
            # and the test would wait for an approval that never comes.
            requires_plan="always",
        )
        run_id = _submit(client, {"skill_ids": [skill_id]})
        wait_status(client, run_id, {"waiting_for_approval"})
        nodes = _resolutions(client, run_id)
        assert nodes["agent_plan"]["model"] == "high-model"
        assert nodes["agent_plan"]["effort"] == "xhigh"
        assert nodes["agent_plan"]["effort_source"] == "explicit_request"
        # The pin must not inflate the cheap calls either.
        assert nodes["agent_intake"]["model"] == "fast-model"
        assert nodes["agent_intake"]["effort"] == ""
