"""deepagents contract tests for the kernel seam (plan M2 step 3).

These tests ARE the deepagents upgrade gate: they freeze the exact
behavior the kernel runtime depends on — built-in tool exclusion via the
``inqtrix:kernel`` harness profile, the HITL interrupt payload and
resume-decision shapes, and park/resume across a REBUILT agent (a parked
run resumes in another worker process with a fresh compiled graph over
the shared checkpointer). A deepagents bump that changes any of these
must fail here before it can reach production.
"""

from __future__ import annotations

import pytest

pytest.importorskip("deepagents")

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from inqtrix.agents.harness import build_kernel_agent
from inqtrix.agents.kernel.chat_bridge import build_tool_chat_model
from inqtrix.agents.kernel.middleware import (
    KernelToolBudgetExceeded,
    KernelToolBudgetMiddleware,
)
from inqtrix.providers.base import ChatTurn, LLMProvider, ToolCallRequest


def test_deepagents_version_is_pinned_to_contracted_range():
    import importlib.metadata

    version = importlib.metadata.version("deepagents")
    parts = tuple(int(p) for p in version.split(".")[:3])
    assert (0, 6, 12) <= parts < (0, 7, 0), (
        f"deepagents {version} is outside the contracted >=0.6.12,<0.7 "
        "range — re-run these contract tests deliberately and update the "
        "pin, never bump silently."
    )


class _ScriptedProvider(LLMProvider):
    """Replays scripted ChatTurns; records every chat() call."""

    def __init__(self, turns: list[ChatTurn]) -> None:
        self._turns = list(turns)
        self.calls: list[dict] = []

    def complete(self, prompt, **kwargs):  # type: ignore[override]
        raise AssertionError("kernel path must use chat(), not complete()")

    def is_available(self) -> bool:
        return True

    def supports_tool_calls(self, *, model=None) -> bool:
        return True

    def chat(self, messages, *, tools=None, model=None, **kwargs):  # type: ignore[override]
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self._turns:
            raise AssertionError("scripted provider ran out of turns")
        return self._turns.pop(0)


def _text_turn(text: str) -> ChatTurn:
    return ChatTurn(
        text=text,
        tool_calls=(),
        finish_reason="stop",
        model="scripted",
        prompt_tokens=1,
        completion_tokens=1,
        raw=None,
    )


def _tool_call_turn(call_id: str, name: str, arguments: dict) -> ChatTurn:
    return ChatTurn(
        text="",
        tool_calls=(
            ToolCallRequest(id=call_id, name=name, arguments=arguments),
        ),
        finish_reason="tool_calls",
        model="scripted",
        prompt_tokens=1,
        completion_tokens=1,
        raw=None,
    )


def _tool_call_batch_turn(*calls: ToolCallRequest) -> ChatTurn:
    return ChatTurn(
        text="",
        tool_calls=tuple(calls),
        finish_reason="tool_calls",
        model="scripted",
        prompt_tokens=1,
        completion_tokens=1,
        raw=None,
    )


@tool
def echo_tool(text: str) -> str:
    """Gibt den Text zurueck."""
    return f"echo: {text}"


def _user_input(question: str) -> dict:
    return {"messages": [{"role": "user", "content": question}]}


def test_kernel_toolset_is_inqtrix_tools_plus_todos_only():
    provider = _ScriptedProvider([_text_turn("Fertig.")])
    agent = build_kernel_agent(
        build_tool_chat_model(provider),
        tools=[echo_tool],
        system_prompt="Test.",
    )

    agent.invoke(_user_input("Hallo"))

    exposed = {t["function"]["name"] for t in provider.calls[0]["tools"]}
    # Filesystem built-ins excluded via the inqtrix:kernel profile, the
    # general-purpose subagent (task tool) disabled, write_todos kept.
    assert exposed == {"echo_tool", "write_todos"}, (
        f"kernel tool surface drifted: {sorted(exposed)} — a deepagents "
        "upgrade added/renamed built-ins or profile resolution broke."
    )


def test_kernel_tool_budget_allows_30_and_rejects_31() -> None:
    middleware = KernelToolBudgetMiddleware(30)
    allowed = AIMessage(
        content="",
        tool_calls=[
            {"id": f"call_{index}", "name": "echo_tool", "args": {}}
            for index in range(30)
        ],
    )
    assert middleware.after_model({"messages": [allowed]}, None) is None

    overflow = AIMessage(
        content="",
        tool_calls=[
            {"id": f"call_{index}", "name": "echo_tool", "args": {}}
            for index in range(31)
        ],
    )
    with pytest.raises(KernelToolBudgetExceeded) as raised:
        middleware.after_model({"messages": [overflow]}, None)
    assert raised.value.attempted == 31
    assert raised.value.batch_size == 31


def test_kernel_tool_budget_rejects_whole_overflowing_batch() -> None:
    executed: list[str] = []

    @tool
    def recording_tool(text: str) -> str:
        """Record execution for the batch-rejection contract."""
        executed.append(text)
        return text

    provider = _ScriptedProvider(
        [
            _tool_call_batch_turn(
                ToolCallRequest(
                    id="call_batch1",
                    name="recording_tool",
                    arguments={"text": "first"},
                ),
                ToolCallRequest(
                    id="call_batch2",
                    name="recording_tool",
                    arguments={"text": "second"},
                ),
            )
        ]
    )
    agent = build_kernel_agent(
        build_tool_chat_model(provider),
        tools=[recording_tool],
        system_prompt="Test.",
        max_tool_calls=1,
    )

    with pytest.raises(KernelToolBudgetExceeded):
        agent.invoke(_user_input("Run batch"))
    assert executed == []


def test_hitl_interrupt_payload_shape_is_frozen():
    provider = _ScriptedProvider(
        [
            _tool_call_turn("call_gate1", "echo_tool", {"text": "hi"}),
            _text_turn("Fertig."),
        ]
    )
    agent = build_kernel_agent(
        build_tool_chat_model(provider),
        tools=[echo_tool],
        system_prompt="Test.",
        interrupt_on={"echo_tool": True},
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "contract-payload"}}

    interrupts = [
        item
        for update in agent.stream(
            _user_input("Frage"), config=config, stream_mode="updates"
        )
        if "__interrupt__" in update
        for item in update["__interrupt__"]
    ]

    assert len(interrupts) == 1
    payload = interrupts[0].value
    assert set(payload.keys()) >= {"action_requests", "review_configs"}
    request = payload["action_requests"][0]
    assert request["name"] == "echo_tool"
    assert request["args"] == {"text": "hi"}
    review = payload["review_configs"][0]
    assert review["action_name"] == "echo_tool"
    assert review["allowed_decisions"] == [
        "approve",
        "edit",
        "reject",
        "respond",
    ]


def test_resume_approve_across_rebuilt_agent_executes_tool():
    checkpointer = MemorySaver()
    config = {"configurable": {"thread_id": "contract-resume"}}
    provider = _ScriptedProvider(
        [_tool_call_turn("call_gate2", "echo_tool", {"text": "weiter"})]
    )
    first = build_kernel_agent(
        build_tool_chat_model(provider),
        tools=[echo_tool],
        system_prompt="Test.",
        interrupt_on={"echo_tool": True},
        checkpointer=checkpointer,
        max_tool_calls=1,
    )
    parked = any(
        "__interrupt__" in update
        for update in first.stream(
            _user_input("Frage"), config=config, stream_mode="updates"
        )
    )
    assert parked

    # A parked run resumes in ANOTHER worker: fresh compiled graph over
    # the shared checkpointer, resume decisions rebuilt from control rows.
    resume_provider = _ScriptedProvider([_text_turn("Erledigt.")])
    second = build_kernel_agent(
        build_tool_chat_model(resume_provider),
        tools=[echo_tool],
        system_prompt="Test.",
        interrupt_on={"echo_tool": True},
        checkpointer=checkpointer,
        max_tool_calls=1,
    )
    updates = list(
        second.stream(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config,
            stream_mode="updates",
        )
    )

    assert not any("__interrupt__" in u for u in updates)
    state = second.get_state(config)
    messages = state.values["messages"]
    tool_results = [
        m for m in messages if getattr(m, "type", "") == "tool"
    ]
    assert any("echo: weiter" in str(m.content) for m in tool_results)
    final = messages[-1]
    assert isinstance(final, AIMessage)
    assert final.content == "Erledigt."


def test_interrupt_id_is_stable_across_reentry_and_rebuild():
    """The kernel derives its tool-approval row ids from interrupt ids.

    A crash between park and decision re-enters the graph and must find
    the SAME interrupt id (idempotent approval create) — as must a
    resume through a rebuilt agent in another worker. An upstream
    change to id derivation breaks park/resume idempotency and must
    fail here.
    """
    checkpointer = MemorySaver()
    config = {"configurable": {"thread_id": "contract-intr-id"}}
    provider = _ScriptedProvider(
        [_tool_call_turn("call_gate9", "echo_tool", {"text": "x"})]
    )
    first = build_kernel_agent(
        build_tool_chat_model(provider),
        tools=[echo_tool],
        system_prompt="Test.",
        interrupt_on={"echo_tool": True},
        checkpointer=checkpointer,
    )
    first_ids = [
        item.id
        for update in first.stream(
            _user_input("Frage"), config=config, stream_mode="updates"
        )
        if "__interrupt__" in update
        for item in update["__interrupt__"]
    ]

    rebuilt = build_kernel_agent(
        build_tool_chat_model(_ScriptedProvider([])),
        tools=[echo_tool],
        system_prompt="Test.",
        interrupt_on={"echo_tool": True},
        checkpointer=checkpointer,
    )
    # Crash re-entry: stream(None) replays to the same pending gate.
    reentry_ids = [
        item.id
        for update in rebuilt.stream(
            None, config=config, stream_mode="updates"
        )
        if "__interrupt__" in update
        for item in update["__interrupt__"]
    ]
    state_ids = [
        item.id
        for task in rebuilt.get_state(config).tasks
        for item in task.interrupts
    ]
    assert first_ids and first_ids == reentry_ids == state_ids


def test_resume_reject_skips_tool_and_continues():
    checkpointer = MemorySaver()
    config = {"configurable": {"thread_id": "contract-reject"}}
    executed: list[str] = []

    @tool
    def gated_tool(text: str) -> str:
        """Aufzeichnendes Tool."""
        executed.append(text)
        return "ran"

    provider = _ScriptedProvider(
        [
            _tool_call_turn("call_gate3", "gated_tool", {"text": "nein"}),
            _text_turn("Verstanden, ohne Tool beantwortet."),
        ]
    )
    agent = build_kernel_agent(
        build_tool_chat_model(provider),
        tools=[gated_tool],
        system_prompt="Test.",
        interrupt_on={"gated_tool": True},
        checkpointer=checkpointer,
    )
    assert any(
        "__interrupt__" in update
        for update in agent.stream(
            _user_input("Frage"), config=config, stream_mode="updates"
        )
    )

    updates = list(
        agent.stream(
            Command(resume={"decisions": [{"type": "reject"}]}),
            config=config,
            stream_mode="updates",
        )
    )

    assert not any("__interrupt__" in u for u in updates)
    assert executed == []
    final = agent.get_state(config).values["messages"][-1]
    assert final.content == "Verstanden, ohne Tool beantwortet."
