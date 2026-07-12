"""Canonical execution-block preservation in compact run snapshots."""

from __future__ import annotations

from inqtrix.state import build_run_snapshot


def test_build_run_snapshot_preserves_canonical_agent_execution() -> None:
    execution = {
        "execution_directive": "knowledge_only",
        "effective_mode": "agent_kernel",
        "response_form": "chat",
        "depth": "normal",
        "model": "model-a",
        "reasoning_effort": "high",
        "source_policy": {"web": "disabled", "knowledge": "available"},
        "consent_reason": "permission_policy",
        "tool_use_counts": {"web": 0, "knowledge": 2},
    }

    snapshot = build_run_snapshot(
        {"done": True, "execution": execution},
        current_node="agent_kernel",
        last_message="completed",
    )

    assert snapshot["execution"] == execution
