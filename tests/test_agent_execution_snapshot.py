"""Canonical execution-block preservation in compact run snapshots."""

from __future__ import annotations

import logging

import pytest

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
        "limits": {},
    }

    snapshot = build_run_snapshot(
        {"done": True, "execution": execution},
        current_node="agent_kernel",
        last_message="completed",
    )

    assert snapshot["execution"] == execution


def test_invalid_execution_snapshot_logs_validation_code_without_payload(caplog) -> None:
    logger = logging.getLogger("inqtrix")
    previous_level = logger.level
    logger.addHandler(caplog.handler)
    logger.setLevel(logging.WARNING)
    private_invalid_value = "PRIVATE_INVALID_EXECUTION_VALUE"

    try:
        with pytest.raises(RuntimeError, match="ungueltigen Agent-Ausfuehrungsblock"):
            build_run_snapshot(
                {
                    "done": False,
                    "execution": {
                        "execution_directive": private_invalid_value,
                    },
                },
                current_node="agent_kernel",
                last_message="failed",
            )
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(previous_level)

    assert "error_code=ValidationError" in caplog.text
    assert "error_count=" in caplog.text
    assert private_invalid_value not in caplog.text
