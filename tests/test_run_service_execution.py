"""Run-service terminal typing independent of the execution backend."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from inqtrix.core.results import AgentResult
from inqtrix.services.run_service import execute_run_request


class _Handle:
    run_id = "run-terminal-type"
    parked = False

    def __init__(self, *, cancelled: bool = False) -> None:
        self.cancel_event = threading.Event()
        if cancelled:
            self.cancel_event.set()
        self.calls: list[tuple[str, str]] = []

    def emit(self, _event_type, _payload) -> None:
        return

    def wait(self, _status) -> None:
        return

    def cancel(self, reason: str) -> None:
        self.calls.append(("cancel", reason))

    def fail(self, message: str, *, error_type: str) -> None:
        self.calls.append((error_type, message))


class _Algorithm:
    def __init__(self, reason: str) -> None:
        self._reason = reason

    def run(self, _request, *, runtime, context) -> AgentResult:
        del runtime, context
        return AgentResult(
            answer="",
            raw={
                "answer": "",
                "usage": {},
                "result_state": {
                    "cancelled": True,
                    "cancel_reason": self._reason,
                },
            },
        )


def _execute(handle: _Handle, *, reason: str) -> None:
    execute_run_request(
        handle,
        algorithm=_Algorithm(reason),
        run_request=SimpleNamespace(),
        resolved=SimpleNamespace(
            providers=SimpleNamespace(),
            strategies=SimpleNamespace(),
            agent_settings=SimpleNamespace(),
        ),
        runtime=SimpleNamespace(
            settings=SimpleNamespace(
                quota=SimpleNamespace(max_tokens_per_run=500)
            )
        ),
        principal=None,
    )


def test_token_budget_cancellation_is_a_typed_failure() -> None:
    handle = _Handle()

    _execute(handle, reason="token_budget_exceeded")

    assert handle.calls == [
        (
            "token_budget_exceeded",
            "Der Lauf hat das serverseitige Tokenbudget erreicht.",
        )
    ]


def test_client_cancel_wins_even_when_a_global_token_cap_is_active() -> None:
    handle = _Handle(cancelled=True)

    _execute(handle, reason="token_budget_exceeded")

    assert handle.calls == [("cancel", "client_requested_cancel")]
