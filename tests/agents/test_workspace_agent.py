"""Workspace-agent trajectory catalog (M5 gate, plan §10).

Every scenario runs the REAL wiring — container, routes, run store,
control store, capability registry — with a scripted LLM whose calls
record the trajectory (tool ORDER, not just the end state) and the
existing ``run_web_graph`` seam standing in for child research runs.
"""

from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import inqtrix.research.web_research as web_research_module
from inqtrix.providers.base import StructuredLLMResponse
from inqtrix.search_result import GroundedSearchResult, GroundedSource
from inqtrix.server.routes import create_router, register_routes
from inqtrix.settings import (
    AgentPlatformSettings,
    AgentSettings,
    ModelSettings,
    ServerSettings,
    Settings,
    StorageSettings,
)

# -- scripted providers ------------------------------------------------------ #

PROFILE = {
    "language": "de",
    "intent": "Marktanalyse erstellen",
    "scope_clarity": "clear",
    "needs_web": True,
    "needs_internal": True,
    "needs_files": False,
    "recency_sensitive": True,
    "contested_topic": False,
    "sub_goals": ["Anbieter identifizieren"],
    "clarification_questions": [],
    "response_form": "canvas",
    "success_criteria": ["Alle Anbieter mit Quelle belegt."],
}

DISCOVERY = {
    "known_facts": [
        {"fact": "Interner Bericht existiert", "source": "doc:d1#0", "fresh": True}
    ],
    "gaps": [
        {
            "gap_id": "g1",
            "kind": "outdated",
            "description": "Aktuelle Marktlage fehlt",
            "recommended_capability": "web_research",
            "suggested_queries": ["Marktlage 2026"],
            "blocking": False,
        }
    ],
    "questions_for_user": [],
    "sufficient_to_plan": True,
}

PLAN = {
    "summary_markdown": "Eine Instant-Suche, dann Synthese.",
    "tasks": [
        {
            "id": "t1",
            "title": "Aktuelle Marktlage",
            "tool_kind": "web_instant",
            "gap_ids": ["g1"],
            "queries": [
                "Wie hat sich die relevante Marktlage im Jahr 2026 entwickelt?"
            ],
            "params": {"recency": "month"},
        },
        {
            "id": "s",
            "title": "Synthese",
            "tool_kind": "synthesis",
            "depends_on": ["t1"],
        },
    ],
    "success_criteria": ["Alle Anbieter mit Quelle belegt."],
}


def replan_delta(
    *new_tasks: dict[str, Any], summary: str = "Additiver Replan."
) -> dict[str, Any]:
    """Return the delta-only schema used after the initial plan."""
    return {
        "summary_markdown": summary,
        "new_tasks": list(new_tasks),
        "skip_task_ids": [],
        "assumptions": [],
    }

RESEARCH_PLAN = {
    **PLAN,
    "summary_markdown": "Ein expliziter Recherchelauf, dann Synthese.",
    "tasks": [
        {
            **PLAN["tasks"][0],
            "tool_kind": "web_research",
            "queries": ["Welche belastbaren Quellen beschreiben die Marktlage 2026?"],
            "params": {"profile": "compact", "recency": "month"},
        },
        PLAN["tasks"][1],
    ],
}

SUFFICIENCY = {"coverage": "covered", "missing": []}

OUTLINE = {
    "title": "Marktanalyse",
    "sections": [
        {
            "title": "Ergebnis",
            "focus": "Marktlage zusammenfassen",
            "criterion_ids": ["0"],
            "evidence_labels": ["W1"],
        },
        {
            "title": "Offene Punkte",
            "focus": "Luecken benennen",
            "criterion_ids": [],
            "evidence_labels": [],
        },
    ],
}

CRITIC_PASS = {
    "findings": [],
    "criteria_covered": ["Alle Anbieter mit Quelle belegt."],
    "criteria_uncovered": [],
    "verdict": "pass",
}


def _outline_for_prompt(prompt: str) -> dict[str, Any]:
    """Make the test planner select evidence that the prompt actually exposes."""
    outline = json.loads(json.dumps(OUTLINE))
    match = re.search(r"\[([KW]\d+)\]", prompt)
    outline["sections"][0]["evidence_labels"] = (
        [match.group(1)] if match else []
    )
    return outline


def _section_for_prompt(prompt: str) -> dict[str, str]:
    """Cite the first evidence label available to the scripted section."""
    match = re.search(r"\[([KW]\d+)\]", prompt)
    suffix = f" [{match.group(1)}]" if match else ""
    return {"markdown": f"Die Marktlage ist stabil{suffix}."}


class ScriptedLLM:
    """Deterministic structured/text replies keyed by schema name."""

    def __init__(self, overrides: dict[str, Any] | None = None) -> None:
        self.models = ModelSettings(
            reasoning_model="base-model",
            tier_high_model="high-model",
            tier_mid_model="mid-model",
            tier_fast_model="fast-model",
        )
        self.calls: list[str] = []
        self.call_models: list[tuple[str, str | None]] = []
        self.prompts: list[tuple[str, str]] = []
        self.scripts: dict[str, Any] = {
            "AssignmentProfile": dict(PROFILE),
            "DiscoveryResult": dict(DISCOVERY),
            "ExecutionPlanModel": dict(PLAN),
            "SufficiencyJudgement": dict(SUFFICIENCY),
            "ContradictionReport": {"contradictions": []},
            "ReportOutline": _outline_for_prompt,
            "AgentCriticReport": dict(CRITIC_PASS),
            "FileAnalysisSummary": {
                "summary": "Kernaussagen aus dem Dokument.",
                "key_quotes": [],
            },
            "SectionText": _section_for_prompt,
        }
        self.scripts.update(overrides or {})

    def is_available(self) -> bool:
        return True

    def supports_structured_output(self, *, model: str | None = None) -> bool:
        return True

    def complete_structured(
        self, prompt: str, *, schema_name: str, **kwargs: Any
    ) -> StructuredLLMResponse:
        self.calls.append(f"structured:{schema_name}")
        self.call_models.append((schema_name, kwargs.get("model")))
        self.prompts.append((schema_name, prompt))
        script = self.scripts.get(schema_name)
        parsed = (
            script(prompt) if callable(script) else dict(script or {})
        )
        return StructuredLLMResponse(
            parsed=parsed,
            content=json.dumps(parsed),
            prompt_tokens=7,
            completion_tokens=5,
        )

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        self.calls.append("complete:section")
        return SimpleNamespace(
            text="Die Marktlage ist stabil [W1].",
            usage={"prompt_tokens": 3, "completion_tokens": 4},
        )


class RetryNotifyingLLM(ScriptedLLM):
    """Scripted provider that exposes one transient retry observation."""

    def __init__(self) -> None:
        super().__init__()
        self._retry_callback: Callable[[dict[str, Any]], None] | None = None
        self._retry_emitted = False

    @contextmanager
    def observe_retries(
        self,
        callback: Callable[[dict[str, Any]], None],
    ) -> Any:
        previous = self._retry_callback
        self._retry_callback = callback
        try:
            yield self
        finally:
            self._retry_callback = previous

    def complete_structured(
        self, prompt: str, *, schema_name: str, **kwargs: Any
    ) -> StructuredLLMResponse:
        if (
            schema_name == "AssignmentProfile"
            and not self._retry_emitted
            and self._retry_callback is not None
        ):
            self._retry_emitted = True
            self._retry_callback({
                "attempt": 1,
                "max_attempts": 3,
                "delay_seconds": 0.25,
                "error_code": "provider_timeout",
                "operation": "complete_structured",
            })
        return super().complete_structured(
            prompt,
            schema_name=schema_name,
            **kwargs,
        )

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        if not self._retry_emitted and self._retry_callback is not None:
            self._retry_emitted = True
            self._retry_callback({
                "attempt": 1,
                "max_attempts": 3,
                "delay_seconds": 0.25,
                "error_code": "provider_timeout",
                "operation": "complete",
            })
        return super().complete(prompt, **kwargs)


class FakeSearch:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def is_available(self) -> bool:
        return True

    def search(self, query: str, **kwargs: Any) -> GroundedSearchResult:
        self.queries.append(query)
        return GroundedSearchResult(
            answer="Web-Vorschau",
            sources=[
                GroundedSource(
                    url="https://example.com/markt",
                    title="Marktbericht",
                    snippet="Der Markt waechst.",
                )
            ],
        )


class RetryNotifyingSearch(FakeSearch):
    """Search stub that exposes one provider retry to its active observer."""

    def __init__(self) -> None:
        super().__init__()
        self._retry_callback: Callable[[dict[str, Any]], None] | None = None
        self._retry_emitted = False

    @contextmanager
    def observe_retries(
        self,
        callback: Callable[[dict[str, Any]], None],
    ) -> Any:
        previous = self._retry_callback
        self._retry_callback = callback
        try:
            yield self
        finally:
            self._retry_callback = previous

    def search(self, query: str, **kwargs: Any) -> GroundedSearchResult:
        if not self._retry_emitted and self._retry_callback is not None:
            self._retry_emitted = True
            self._retry_callback({
                "attempt": 1,
                "max_attempts": 3,
                "delay_seconds": 0.2,
                "error_code": "provider_timeout",
                "operation": "search",
            })
        return super().search(query, **kwargs)


class NoEvidenceSearch:
    """Available provider whose answer contains no usable evidence."""

    def is_available(self) -> bool:
        return True

    def search(self, query: str, **kwargs: Any) -> GroundedSearchResult:
        return GroundedSearchResult(answer="", sources=[])


def fake_child_graph(question: str, **kwargs: Any) -> dict[str, Any]:
    """Stands in for a child research run's graph execution."""
    sink = kwargs.get("run_event_sink")
    if sink is not None:
        sink(
            "inqtrix.node.started",
            {"node": "search", "snapshot": {"current_node": "search"}},
        )
    return {
        "answer": f"Kindbericht zu: {question}",
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        "result_state": {
            "answer": f"Kindbericht zu: {question}",
            "round": 1,
            "report_references": [
                {
                    "label": "E1",
                    "url": "https://example.com/quelle",
                    "tier": "mainstream",
                }
            ],
            "consolidated_claims": [],
        },
    }


def make_agent_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    llm: ScriptedLLM | None = None,
    platform: AgentPlatformSettings | None = None,
    strategies: SimpleNamespace | None = None,
    search: Any = None,
    max_tokens_per_run: int = 0,
) -> TestClient:
    monkeypatch.setattr(
        web_research_module, "run_web_graph", fake_child_graph
    )
    app = FastAPI()
    router = create_router()
    settings = Settings(
        models=ModelSettings(),
        agent=AgentSettings(),
        server=ServerSettings(),
        storage=StorageSettings(backend="memory", database_url=""),
    )
    settings.agent_platform = platform or AgentPlatformSettings(
        INQTRIX_AGENT_ALLOW_VOLATILE=True
    )
    settings.quota.max_tokens_per_run = max_tokens_per_run
    scripted = llm or ScriptedLLM()
    container = register_routes(
        router,
        providers=SimpleNamespace(llm=scripted, search=search or FakeSearch()),
        strategies=strategies or SimpleNamespace(),
        settings=settings,
        semaphore_factory=lambda: __import__("asyncio").Semaphore(2),
    )
    app.include_router(router)
    client = TestClient(app)
    client.llm = scripted  # type: ignore[attr-defined]
    client.container = container  # type: ignore[attr-defined]
    return client


def wait_status(
    client: TestClient, run_id: str, statuses: set[str], *, timeout: float = 10.0
) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        summary = client.get(f"/v1/runs/{run_id}").json()
        if summary["status"] in statuses:
            return summary
        time.sleep(0.02)
    pytest.fail(f"run {run_id} never reached {statuses}")


def submit_agent(
    client: TestClient,
    *,
    autonomy: str | None = None,
    session_id: str | None = None,
    source_policy: dict[str, str] | None = None,
    tool_directives: list[str] | None = None,
    agent_overrides: dict[str, Any] | None = None,
) -> str:
    body: dict[str, Any] = {
        "question": "Erstelle eine Marktanalyse.",
        "mode": "workspace_agent",
    }
    if autonomy:
        body["autonomy"] = autonomy
    if session_id:
        body["session_id"] = session_id
    if source_policy is not None:
        body["source_policy"] = source_policy
    if tool_directives is not None:
        body["tool_directives"] = tool_directives
    if agent_overrides is not None:
        body["agent_overrides"] = agent_overrides
    response = client.post("/v1/runs", json=body)
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def approve_pending(client: TestClient, run_id: str) -> dict[str, Any]:
    approvals = client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
    pending = [a for a in approvals if a["status"] == "pending"]
    assert pending, approvals
    decided = client.post(
        f"/v1/runs/{run_id}/approvals/{pending[0]['approval_id']}",
        json={"decision": "approve"},
    )
    assert decided.status_code == 200, decided.text
    return decided.json()


def event_types(client: TestClient, run_id: str) -> list[str]:
    with client.stream("GET", f"/v1/runs/{run_id}/events") as stream:
        body = stream.read().decode("utf-8")
    return [
        json.loads(line[6:])["type"]
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def run_events(client: TestClient, run_id: str) -> list[dict[str, Any]]:
    with client.stream("GET", f"/v1/runs/{run_id}/events") as stream:
        body = stream.read().decode("utf-8")
    return [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def human_wait_statuses(client: TestClient, run_id: str) -> list[str]:
    """Waiting statuses that stem from a HUMAN interrupt.

    The slot-free ``waiting_for_children`` park is store-internal (the
    last child's terminal write resumes the run) and is filtered out —
    tests about approval/clarification behaviour must not break merely
    because a plan spawned child runs.
    """
    with client.stream("GET", f"/v1/runs/{run_id}/events") as stream:
        body = stream.read().decode("utf-8")
    return [
        status
        for line in body.splitlines()
        if line.startswith("data: ")
        and (event := json.loads(line[6:]))["type"] == "inqtrix.run.waiting"
        and (status := event["data"]["status"]) != "waiting_for_children"
    ]


# -- scenarios ---------------------------------------------------------------- #


def test_structured_model_retries_use_agent_activity_channel(monkeypatch):
    llm = RetryNotifyingLLM()
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        run_id = submit_agent(client)
        wait_status(client, run_id, {"waiting_for_approval"})
        approve_pending(client, run_id)
        wait_status(client, run_id, {"completed"})
        retries = [
            event["data"]
            for event in run_events(client, run_id)
            if event["type"] == "inqtrix.agent.activity"
            and event["data"].get("retry")
        ]

    assert len(retries) == 1
    assert retries[0]["scope"] == "run"
    assert retries[0]["phase"] == "intake"
    assert retries[0]["purpose"] == "Auftrag wird eingeordnet"
    assert retries[0]["retry"] == {
        "failed_attempt": 1,
        "next_attempt": 2,
        "max_attempts": 3,
        "delay_seconds": 0.25,
        "error_code": "provider_timeout",
    }


def test_file_analysis_model_bridge_consumes_bound_retry_sink():
    from langchain_core.messages import HumanMessage

    from inqtrix.agents.model_bridge import build_chat_model
    from inqtrix.agents.patterns._structured import observe_structured_retries

    llm = RetryNotifyingLLM()
    observed: list[tuple[str, dict[str, Any]]] = []
    with observe_structured_retries(
        lambda node, notice: observed.append((node, notice))
    ):
        model = build_chat_model(llm, node="agent_file_analysis")
        model.invoke([HumanMessage(content="Analysiere das Dokument.")])

    assert len(observed) == 1
    assert observed[0][0] == "agent_file_analysis"
    assert observed[0][1]["operation"] == "complete"


def test_balanced_full_arc_with_child_and_memo(monkeypatch):
    """Happy path: discovery -> plan approval -> child wave -> memo."""
    llm = ScriptedLLM(overrides={"ExecutionPlanModel": RESEARCH_PLAN})
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        client.headers["x-inqtrix-workspace-id"] = "ws_mission_contract"
        run_id = submit_agent(
            client,
            session_id="sess-demo",
            tool_directives=["web_research"],
        )
        summary = wait_status(client, run_id, {"waiting_for_approval"})
        assert summary["kind"] == "agent"
        assert summary["workspace_id"] == "ws_mission_contract"
        # The stored row carries the effective autonomy mode (the
        # default resolves to balanced) so summaries can show it.
        assert summary["agent_overrides"]["autonomy"] == "balanced"

        plan = client.get(f"/v1/runs/{run_id}/plan").json()
        assert plan["status"] == "proposed"
        assert [task["task_id"] for task in plan["tasks"]] == ["t1", "s"]
        assert plan["tasks"][0]["params"]["profile"] == "compact"
        approvals = client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
        assert approvals[0]["subject_id"] == plan["plan_id"]

        approve_pending(client, run_id)
        wait_status(client, run_id, {"completed"})

        # Exactly one child research run with the task's profile.
        children = client.get(f"/v1/runs/{run_id}/children").json()["data"]
        assert len(children) == 1
        child = children[0]
        assert child["kind"] == "agent_child"
        assert child["workspace_id"] == "ws_mission_contract"
        assert child["agent_overrides"]["report_profile"] == "compact"
        assert child["status"] == "completed"
        completed_plan = client.get(f"/v1/runs/{run_id}/plan").json()
        assert completed_plan["status"] == "approved"
        completed_task = completed_plan["tasks"][0]
        assert completed_task["status"] == "completed"
        assert completed_task["child_run_id"] == child["run_id"]
        assert completed_task["result_summary"]
        assert completed_plan["tasks"][1]["status"] == "completed"

        # The memo artifact is ready and anchored to the session.
        artifacts = client.get(f"/v1/runs/{run_id}/artifacts").json()["data"]
        memos = [a for a in artifacts if a["kind"] == "memo"]
        assert len(memos) == 1
        assert memos[0]["status"] == "ready"
        assert memos[0]["session_id"] == "sess-demo"
        detail = client.get(
            f"/v1/runs/{run_id}/artifacts/{memos[0]['artifact_id']}"
        ).json()
        assert "[W1]" in detail["content_markdown"]

        # Result payload carries the memo answer.
        result = client.get(f"/v1/runs/{run_id}/result").json()
        assert "Marktanalyse" in result["answer"]

        # Trajectory: intake -> discovery analyst -> plan -> (park) ->
        # sufficiency -> outline -> sections -> critic. Order is pinned.
        calls = client.llm.calls
        assert calls[0] == "structured:AssignmentProfile"
        assert calls[1] == "structured:DiscoveryResult"
        assert calls[2] == "structured:ExecutionPlanModel"
        assert "structured:SufficiencyJudgement" in calls
        assert calls.index("structured:ReportOutline") > calls.index(
            "structured:SufficiencyJudgement"
        )
        assert calls[-1] == "structured:AgentCriticReport"

        # Event catalog: phases, plan.proposed, approval flow, task
        # lifecycle, artifact stream, terminal.
        types = event_types(client, run_id)
        assert "inqtrix.agent.phase.changed" in types
        assert "inqtrix.agent.plan.proposed" in types
        assert "inqtrix.agent.approval.requested" in types
        assert types.index("inqtrix.agent.approval.requested") < types.index(
            "inqtrix.run.waiting"
        )
        assert "inqtrix.agent.task.started" in types
        assert "inqtrix.agent.task.finished" in types
        assert "inqtrix.agent.child.progress" in types
        assert "inqtrix.agent.artifact.created" in types
        assert types[-1] == "inqtrix.run.completed"


def test_research_child_receives_structured_recency_without_raw_params(
    monkeypatch,
):
    """A mission passes freshness structurally, not as prompt-shaped data."""
    captured: list[tuple[str, dict[str, Any]]] = []
    plan = json.loads(json.dumps(RESEARCH_PLAN))
    research_task = plan["tasks"][0]
    research_task["objective"] = "Belastbare aktuelle Quellen einordnen."
    research_task["expected_output"] = "Eine belegte Kurzbewertung."
    research_task["params"]["recency"] = "year"

    def capture_child(question: str, **kwargs: Any) -> dict[str, Any]:
        captured.append((question, kwargs))
        return fake_child_graph(question, **kwargs)

    client = make_agent_client(
        monkeypatch,
        llm=ScriptedLLM(overrides={"ExecutionPlanModel": plan}),
    )
    monkeypatch.setattr(
        web_research_module, "run_web_graph", capture_child
    )

    with client:
        run_id = submit_agent(
            client,
            autonomy="autonomous",
            tool_directives=["web_research"],
        )
        wait_status(client, run_id, {"completed"})

    assert len(captured) == 1
    question, kwargs = captured[0]
    assert kwargs["web_recency"] == "year"
    assert "Task: Aktuelle Marktlage" in question
    assert "Objective: Belastbare aktuelle Quellen einordnen." in question
    assert "Welche belastbaren Quellen" in question
    assert "Expected output: Eine belegte Kurzbewertung." in question
    assert "Tool parameters:" not in question


def test_mixed_wave_overlaps_child_and_instant_with_one_shared_limit(
    monkeypatch,
):
    """Independent child/local work starts together instead of serializing."""
    import threading

    child_started = threading.Event()
    local_started = threading.Event()

    mixed_plan = {
        "summary_markdown": "Recherche und Instant-Frage parallel.",
        "tasks": [
            {
                "id": "r",
                "title": "Quellen vertiefen",
                "tool_kind": "web_research",
                "gap_ids": ["g1"],
                "queries": ["Welche Quellen belegen die Marktlage 2026?"],
                "params": {"profile": "compact"},
            },
            {
                "id": "i",
                "title": "Aktuellen Fakt prüfen",
                "tool_kind": "web_instant",
                "gap_ids": ["g1"],
                "queries": ["Wie groß ist der Markt im Jahr 2026?"],
            },
            {
                "id": "s",
                "title": "Synthese",
                "tool_kind": "synthesis",
                "depends_on": ["r", "i"],
            },
        ],
    }

    class OverlapSearch(FakeSearch):
        def search(self, query: str, **kwargs: Any) -> GroundedSearchResult:
            local_started.set()
            assert child_started.wait(5), "child did not overlap instant work"
            return super().search(query, **kwargs)

    def overlapping_child(question: str, **kwargs: Any) -> dict[str, Any]:
        child_started.set()
        assert local_started.wait(5), "instant work did not overlap child"
        return fake_child_graph(question, **kwargs)

    llm = ScriptedLLM(
        overrides={
            "AssignmentProfile": {**PROFILE, "needs_web": False},
            "ExecutionPlanModel": mixed_plan,
        }
    )
    platform = AgentPlatformSettings(
        INQTRIX_AGENT_ALLOW_VOLATILE=True,
        max_parallel_children=2,
    )
    client = make_agent_client(
        monkeypatch,
        llm=llm,
        search=OverlapSearch(),
        platform=platform,
    )
    monkeypatch.setattr(
        web_research_module, "run_web_graph", overlapping_child
    )
    with client:
        run_id = submit_agent(
            client,
            autonomy="autonomous",
            tool_directives=["web_research"],
        )
        wait_status(client, run_id, {"completed"})

    assert child_started.is_set()
    assert local_started.is_set()


def test_mixed_wave_two_children_two_locals_respects_shared_limit(monkeypatch):
    """A larger mixed wave never marks or executes beyond shared capacity."""
    import threading

    lock = threading.Lock()
    active = 0
    max_active = 0
    child_calls = 0
    local_calls = 0

    def tracked_call(callback):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.05)
            return callback()
        finally:
            with lock:
                active -= 1

    class CappedSearch(FakeSearch):
        def search(self, query: str, **kwargs: Any) -> GroundedSearchResult:
            nonlocal local_calls
            if not query.startswith("Instant-Cap-"):
                return super().search(query, **kwargs)
            local_calls += 1
            return tracked_call(
                lambda: super(CappedSearch, self).search(query, **kwargs)
            )

    def capped_child(question: str, **kwargs: Any) -> dict[str, Any]:
        nonlocal child_calls
        child_calls += 1
        return tracked_call(lambda: fake_child_graph(question, **kwargs))

    tasks = [
        {
            "id": task_id,
            "title": f"Recherche {task_id}",
            "tool_kind": "web_research",
            "gap_ids": ["g1"],
            "queries": [f"Welche Belege liefert Research-Cap-{task_id}?"],
            "params": {"profile": "compact"},
        }
        for task_id in ("r1", "r2")
    ] + [
        {
            "id": task_id,
            "title": f"Instant {task_id}",
            "tool_kind": "web_instant",
            "gap_ids": ["g1"],
            "queries": [f"Instant-Cap-{task_id}"],
        }
        for task_id in ("i1", "i2")
    ]
    tasks.append(
        {
            "id": "s",
            "title": "Synthese",
            "tool_kind": "synthesis",
            "depends_on": ["r1", "r2", "i1", "i2"],
        }
    )
    platform = AgentPlatformSettings(
        INQTRIX_AGENT_ALLOW_VOLATILE=True,
        max_parallel_children=2,
    )
    client = make_agent_client(
        monkeypatch,
        llm=ScriptedLLM(
            overrides={
                "AssignmentProfile": {**PROFILE, "needs_web": False},
                "ExecutionPlanModel": {
                    "summary_markdown": "Vier unabhaengige Aufgaben.",
                    "tasks": tasks,
                },
            }
        ),
        search=CappedSearch(),
        platform=platform,
    )
    monkeypatch.setattr(web_research_module, "run_web_graph", capped_child)
    with client:
        run_id = submit_agent(
            client,
            autonomy="autonomous",
            tool_directives=["web_research"],
        )
        wait_status(client, run_id, {"completed"})
        stored = client.get(f"/v1/runs/{run_id}/plan").json()

    assert child_calls == 2
    assert local_calls == 2
    assert max_active == 2
    assert all(task["status"] != "running" for task in stored["tasks"])


def test_synthesis_failure_closes_its_plan_task(monkeypatch):
    """A provider failure cannot strand the synthesis row in running."""

    class FailingSynthesisLLM(ScriptedLLM):
        def complete_structured(
            self, prompt: str, *, schema_name: str, **kwargs: Any
        ) -> StructuredLLMResponse:
            if schema_name == "SectionText":
                raise RuntimeError("synthesis unavailable")
            return super().complete_structured(
                prompt, schema_name=schema_name, **kwargs
            )

    client = make_agent_client(monkeypatch, llm=FailingSynthesisLLM())
    with client:
        run_id = submit_agent(client, autonomy="autonomous")
        wait_status(client, run_id, {"failed"})
        plan = client.get(f"/v1/runs/{run_id}/plan").json()

    synthesis_task = next(
        task for task in plan["tasks"] if task["tool_kind"] == "synthesis"
    )
    assert synthesis_task["status"] == "failed"
    assert synthesis_task["result_summary"] == "Synthese fehlgeschlagen."
    assert "structured:AgentCriticReport" not in client.llm.calls


def test_mission_source_policy_is_enforced_and_inherited_by_child(monkeypatch):
    """A mission and its delegated research child share one source policy."""
    llm = ScriptedLLM(overrides={"ExecutionPlanModel": RESEARCH_PLAN})
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        run_id = submit_agent(
            client,
            source_policy={"web": "available", "knowledge": "disabled"},
            tool_directives=["web_research"],
        )
        waiting = wait_status(client, run_id, {"waiting_for_approval"})
        assert waiting["agent_overrides"]["source_policy"] == {
            "web": "available",
            "knowledge": "disabled",
        }
        approve_pending(client, run_id)
        completed = wait_status(client, run_id, {"completed"})

        children = client.get(
            f"/v1/runs/{run_id}/children"
        ).json()["data"]
        assert len(children) == 1
        assert children[0]["agent_overrides"]["source_policy"] == {
            "web": "available",
            "knowledge": "disabled",
        }
        execution = completed["snapshot"]["execution"]
        assert execution["source_policy"] == {
            "web": "available",
            "knowledge": "disabled",
        }
        assert execution["tool_use_counts"] == {
            "web": 1,
            "knowledge": 0,
        }


def test_mission_rejects_planned_task_for_disabled_source(monkeypatch):
    """A disabled source fails at plan validation before dispatch."""
    client = make_agent_client(monkeypatch)
    with client:
        run_id = submit_agent(
            client,
            source_policy={"web": "disabled", "knowledge": "available"},
        )
        summary = wait_status(client, run_id, {"failed"})
        assert "plan_invalid" in summary["error"]["message"]
        assert client.get(f"/v1/runs/{run_id}/children").json()["data"] == []
        assert client.container.providers.search.queries == []


def test_followup_turn_continues_the_session_memo_lineage(monkeypatch):
    """E15: a second turn in the session READS the prior memo and CONTINUES
    the SAME artifact — it does not start a fresh one nor ignore prior text."""
    outline_prompts: list[str] = []

    def capture_outline(prompt: str) -> dict[str, Any]:
        outline_prompts.append(prompt)
        return dict(OUTLINE)

    llm = ScriptedLLM(overrides={"ReportOutline": capture_outline})
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        run1 = submit_agent(client, session_id="sess-lineage")
        wait_status(client, run1, {"waiting_for_approval"})
        approve_pending(client, run1)
        wait_status(client, run1, {"completed"})
        memo1 = [
            a
            for a in client.get(f"/v1/runs/{run1}/artifacts").json()["data"]
            if a["kind"] == "memo"
        ][0]
        # Turn 1's outline had no prior memo to continue.
        assert "Bisheriges Memo" not in outline_prompts[-1]

        run2 = submit_agent(client, session_id="sess-lineage")
        wait_status(client, run2, {"waiting_for_approval"})
        approve_pending(client, run2)
        wait_status(client, run2, {"completed"})
        memo2 = [
            a
            for a in client.get(f"/v1/runs/{run2}/artifacts").json()["data"]
            if a["kind"] == "memo"
        ][0]

    # Same memo canvas (one per session), revision ADVANCED — continued,
    # not recreated.
    assert memo2["artifact_id"] == memo1["artifact_id"]
    assert memo2["revision"] > memo1["revision"]
    # Turn 2's outline prompt carried the turn-1 memo (the lineage read
    # reached synthesis) so the agent continues instead of restarting.
    assert "Bisheriges Memo" in outline_prompts[-1]
    assert "Die Marktlage ist stabil" in outline_prompts[-1]


def test_autonomous_mode_skips_plan_interrupt(monkeypatch):
    client = make_agent_client(monkeypatch)
    with client:
        run_id = submit_agent(client, autonomy="autonomous")
        wait_status(client, run_id, {"completed"})
        assert "inqtrix.agent.plan.proposed" in event_types(client, run_id)
        # Autonomy skips HUMAN decision waits; the slot-free
        # waiting_for_children park is not a human interrupt and may
        # legitimately appear whenever the plan spawns child runs.
        assert human_wait_statuses(client, run_id) == []
        plan = client.get(f"/v1/runs/{run_id}/plan").json()
        assert plan["version"] == 1
        assert plan["status"] == "approved"


def test_strict_mode_approves_discovery_before_probes(monkeypatch):
    client = make_agent_client(monkeypatch)
    with client:
        run_id = submit_agent(client, autonomy="strict")
        wait_status(client, run_id, {"waiting_for_approval"})
        approvals = client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
        assert approvals[0]["kind"] == "discovery"
        assert approvals[0]["payload"]["probes"]
        # No discovery-analyst call happened before the approval.
        assert "structured:DiscoveryResult" not in client.llm.calls

        approve_pending(client, run_id)
        wait_status(client, run_id, {"waiting_for_approval"})
        approvals = client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
        kinds = {a["kind"]: a["status"] for a in approvals}
        assert kinds["discovery"] == "approved"
        approve_pending(client, run_id)
        wait_status(client, run_id, {"completed"})


def test_rejected_plan_completes_with_receipt(monkeypatch):
    """Rejecting the plan is a clean user decision, not a run failure:
    the run COMPLETES with a deterministic receipt as its chat answer and
    ``plan_decision=rejected`` in the result payload (the durable
    ``rejected`` status lives on the approval/plan rows)."""
    client = make_agent_client(monkeypatch)
    with client:
        run_id = submit_agent(client)
        wait_status(client, run_id, {"waiting_for_approval"})
        approvals = client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
        rejected = client.post(
            f"/v1/runs/{run_id}/approvals/{approvals[0]['approval_id']}",
            json={"decision": "reject", "note": "Bitte enger fassen."},
        )
        assert rejected.status_code == 200
        wait_status(client, run_id, {"completed"})
        approvals = client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
        assert approvals[0]["status"] == "rejected"
        result = client.get(f"/v1/runs/{run_id}/result").json()
        assert "Plan abgelehnt" in result["answer"]
        assert "Bitte enger fassen." in result["answer"]
        # The receipt is durable as the run-local answer artifact, so a
        # reload rehydrates it like any other chat answer.
        artifacts = client.get(f"/v1/runs/{run_id}/artifacts").json()["data"]
        assert any(artifact["kind"] == "answer" for artifact in artifacts)


def test_gruendlich_tier_honors_in_ceiling_task_profile(monkeypatch):
    """Published == enforced at EXECUTION: a validated per-task profile
    inside the tier ceiling must reach the child run, never be silently
    normalized back to the tier default."""
    llm = ScriptedLLM(overrides={"ExecutionPlanModel": dict(RESEARCH_PLAN)})
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        run_id = submit_agent(
            client, agent_overrides={"agent_tier": "gruendlich"}
        )
        wait_status(client, run_id, {"waiting_for_approval"})
        approve_pending(client, run_id)
        wait_status(client, run_id, {"completed"})
        children = client.get(f"/v1/runs/{run_id}/children").json()["data"]
        assert len(children) == 1
        # The planner asked for compact (== gruendlich ceiling); the
        # tier DEFAULT is schnell — the explicit in-ceiling choice wins.
        assert children[0]["agent_overrides"]["report_profile"] == "compact"


def test_tier_never_influences_model_routing(monkeypatch):
    """P5 orthogonality contract: the Stufe changes budgets, never the
    model resolution — same nodes resolve to the same models."""
    def node_models(client: TestClient) -> dict[str, str | None]:
        return dict(client.llm.call_models)

    baseline = make_agent_client(monkeypatch)
    with baseline:
        run_id = submit_agent(baseline)
        wait_status(baseline, run_id, {"waiting_for_approval"})
        approve_pending(baseline, run_id)
        wait_status(baseline, run_id, {"completed"})
    tiered = make_agent_client(monkeypatch)
    with tiered:
        run_id = submit_agent(
            tiered, agent_overrides={"agent_tier": "tief"}
        )
        wait_status(tiered, run_id, {"waiting_for_approval"})
        approve_pending(tiered, run_id)
        wait_status(tiered, run_id, {"completed"})
    base_models = node_models(baseline)
    tier_models = node_models(tiered)
    shared = set(base_models) & set(tier_models)
    assert shared, "no shared nodes to compare"
    for node in shared:
        assert tier_models[node] == base_models[node], node


def test_report_guidance_on_non_plan_gate_is_a_loud_400(monkeypatch):
    """Guidance a gate cannot honor must never be silently dropped."""
    client = make_agent_client(monkeypatch)
    with client:
        run_id = submit_agent(client, autonomy="strict")
        wait_status(client, run_id, {"waiting_for_approval"})
        approvals = client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
        assert approvals[0]["kind"] == "discovery"
        response = client.post(
            f"/v1/runs/{run_id}/approvals/{approvals[0]['approval_id']}",
            json={
                "decision": "approve",
                "report_guidance": "Bitte als Sprechzettel.",
            },
        )
        assert response.status_code == 400
        assert "report_guidance" in response.text


def test_tief_tier_escalates_unverified_web_quotes_once(monkeypatch):
    """P7: on the tief tier an unverified web-cited quote flips a passing
    critic verdict to revise EXACTLY once (existing revision budget)."""
    quote = "Die Compliance-Kosten steigen 2026 um vierzig Prozent"

    def child_with_excerpt(question: str, **kwargs: Any) -> dict[str, Any]:
        result = fake_child_graph(question, **kwargs)
        result["result_state"]["report_references"][0]["excerpt"] = (
            "Der Auszug behandelt die Marktlage, enthaelt dieses Zitat "
            "aber nicht."
        )
        return result

    def section_with_quote(prompt: str) -> dict[str, str]:
        match = re.search(r"\[([KW]\d+)\]", prompt)
        label = f" [{match.group(1)}]" if match else ""
        if "Kritikpunkte" in prompt:
            return {"markdown": f"Ueberarbeitet ohne Zitat{label}."}
        return {"markdown": f'Analyse: "{quote}"{label}.'}

    llm = ScriptedLLM(overrides={"SectionText": section_with_quote})
    client = make_agent_client(monkeypatch, llm=llm)
    monkeypatch.setattr(
        web_research_module, "run_web_graph", child_with_excerpt
    )
    with client:
        run_id = submit_agent(
            client, agent_overrides={"agent_tier": "tief"}
        )
        wait_status(client, run_id, {"waiting_for_approval"})
        approve_pending(client, run_id)
        wait_status(client, run_id, {"completed"})
        critic_calls = client.llm.calls.count("structured:AgentCriticReport")
        assert critic_calls >= 2
        revision_prompts = [
            prompt
            for name, prompt in client.llm.prompts
            if name == "SectionText" and "Quellenauszuege" in prompt
        ]
        assert revision_prompts, "escalation never reached the revision call"


def test_gruendlich_tier_keeps_unverified_quotes_advisory(monkeypatch):
    """Default tier: the same unverified web quote stays advisory —
    exactly one critic call, no forced revision."""
    quote = "Die Compliance-Kosten steigen 2026 um vierzig Prozent"

    def child_with_excerpt(question: str, **kwargs: Any) -> dict[str, Any]:
        result = fake_child_graph(question, **kwargs)
        result["result_state"]["report_references"][0]["excerpt"] = (
            "Der Auszug behandelt die Marktlage, enthaelt dieses Zitat "
            "aber nicht."
        )
        return result

    def section_with_quote(prompt: str) -> dict[str, str]:
        match = re.search(r"\[([KW]\d+)\]", prompt)
        label = f" [{match.group(1)}]" if match else ""
        return {"markdown": f'Analyse: "{quote}"{label}.'}

    llm = ScriptedLLM(overrides={"SectionText": section_with_quote})
    client = make_agent_client(monkeypatch, llm=llm)
    monkeypatch.setattr(
        web_research_module, "run_web_graph", child_with_excerpt
    )
    with client:
        run_id = submit_agent(client)
        wait_status(client, run_id, {"waiting_for_approval"})
        approve_pending(client, run_id)
        wait_status(client, run_id, {"completed"})
        assert client.llm.calls.count("structured:AgentCriticReport") == 1


def test_report_guidance_reaches_the_synthesis_prompts(monkeypatch):
    """P6: guidance attached to the plan approval must shape the report —
    it rides the decision payload and renders into the synthesis
    prompts (outline/sections/answer)."""
    client = make_agent_client(monkeypatch)
    with client:
        run_id = submit_agent(client)
        wait_status(client, run_id, {"waiting_for_approval"})
        pending = [
            a
            for a in client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
            if a["status"] == "pending"
        ]
        decided = client.post(
            f"/v1/runs/{run_id}/approvals/{pending[0]['approval_id']}",
            json={
                "decision": "approve",
                "report_guidance": "Gliedere als Sprechzettel fuer die GF.",
            },
        )
        assert decided.status_code == 200
        wait_status(client, run_id, {"completed"})
        synthesis_prompts = [
            prompt
            for name, prompt in client.llm.prompts
            if name in ("ReportOutline", "SectionText")
        ]
        assert synthesis_prompts, "synthesis never ran"
        assert any(
            "Nutzer-Vorgaben zum Bericht" in prompt
            and "Sprechzettel" in prompt
            for prompt in synthesis_prompts
        )


def test_report_guidance_rejects_oversized_input(monkeypatch):
    client = make_agent_client(monkeypatch)
    with client:
        run_id = submit_agent(client)
        wait_status(client, run_id, {"waiting_for_approval"})
        pending = client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
        response = client.post(
            f"/v1/runs/{run_id}/approvals/{pending[0]['approval_id']}",
            json={"decision": "approve", "report_guidance": "x" * 2001},
        )
        assert response.status_code == 400


def test_schnell_tier_answers_in_chat_without_gates(monkeypatch):
    """Stufe Schnell (tier table): intake -> plan -> execute -> chat
    answer. No discovery analyst, no clarification interrupt, no plan
    gate under balanced autonomy, an instant search instead of research
    children, and the critic pass skipped."""
    llm = ScriptedLLM(
        overrides={
            "AssignmentProfile": {
                **PROFILE,
                # Even an ambiguous assignment must not park a schnell
                # run — open questions become plan assumptions.
                "scope_clarity": "ambiguous",
                "clarification_questions": [
                    {
                        "prompt": "Welcher Markt?",
                        "options": [
                            {"label": "Europa", "description": ""},
                            {"label": "USA", "description": ""},
                        ],
                        "multi_select": False,
                    }
                ],
            }
        }
    )
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        run_id = submit_agent(
            client, agent_overrides={"agent_tier": "schnell"}
        )
        summary = wait_status(client, run_id, {"completed"})
        assert summary["agent_overrides"]["agent_tier"] == "schnell"
        assert summary["agent_overrides"]["depth"] == "normal"
        # Neither the discovery analyst nor the critic ever ran.
        assert "structured:DiscoveryResult" not in client.llm.calls
        assert "structured:AgentCriticReport" not in client.llm.calls
        # The deliverable is a CHAT answer, not a canvas memo.
        artifacts = client.get(f"/v1/runs/{run_id}/artifacts").json()["data"]
        kinds = {artifact["kind"] for artifact in artifacts}
        assert "answer" in kinds
        assert "memo" not in kinds
        # No round ever parked the run.
        clarifications = client.get(
            f"/v1/runs/{run_id}/clarifications"
        ).json()["data"]
        assert clarifications == []
        approvals = client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
        assert [a for a in approvals if a["status"] == "pending"] == []


def test_schnell_tier_still_gates_under_strict_autonomy(monkeypatch):
    """Permission beats speed: strict autonomy keeps its gates even on
    the fastest tier."""
    client = make_agent_client(monkeypatch)
    with client:
        run_id = submit_agent(
            client,
            autonomy="strict",
            agent_overrides={"agent_tier": "schnell"},
        )
        wait_status(client, run_id, {"waiting_for_approval"})
        approve_pending(client, run_id)
        wait_status(client, run_id, {"completed"})


def test_answered_round_is_never_reasked_after_discovery(monkeypatch):
    """P2 root fix (observed live bug): the discovery analyst sees the
    answered intake round in its prompt, and a rephrased duplicate in
    ``questions_for_user`` is deterministically skipped — the run
    reaches the plan gate without a second input interrupt, with a
    visible skip narration."""
    llm = ScriptedLLM(
        overrides={
            "AssignmentProfile": {
                **PROFILE,
                "scope_clarity": "ambiguous",
                "clarification_questions": [
                    {
                        "prompt": (
                            "Auf welchen KI-Markt soll sich die Analyse "
                            "primaer beziehen?"
                        ),
                        "options": [
                            {"label": "Generative KI", "description": ""},
                            {"label": "Gesamtmarkt", "description": ""},
                        ],
                        "multi_select": False,
                    }
                ],
            },
            "DiscoveryResult": {
                **DISCOVERY,
                "gaps": [
                    {
                        "gap_id": "g1",
                        "kind": "unknown_scope",
                        "description": "Themenabgrenzung unklar",
                        "recommended_capability": "web_research",
                        "suggested_queries": ["KI-Markt 2026"],
                        "blocking": True,
                    }
                ],
                "questions_for_user": [
                    {
                        "prompt": (
                            "Welchen KI-Markt sollen wir thematisch "
                            "analysieren?"
                        ),
                        "options": [
                            {"label": "Generative KI", "description": ""},
                            {"label": "KI-Software", "description": ""},
                        ],
                        "multi_select": False,
                    }
                ],
                "sufficient_to_plan": False,
            },
        }
    )
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        run_id = submit_agent(client)
        wait_status(client, run_id, {"waiting_for_input"})
        clarifications = client.get(
            f"/v1/runs/{run_id}/clarifications"
        ).json()["data"]
        answered = client.post(
            f"/v1/runs/{run_id}/clarifications/"
            f"{clarifications[0]['clarification_id']}",
            json={"answers": {"q1": {"option_ids": ["q1_o1"], "text": ""}}},
        )
        assert answered.status_code == 200
        # The rephrased question is deduplicated: the plan gate is the
        # NEXT interrupt, never a second input round.
        wait_status(client, run_id, {"waiting_for_approval"})
        rounds = client.get(f"/v1/runs/{run_id}/clarifications").json()[
            "data"
        ]
        assert len(rounds) == 1
        # The analyst prompt carried the answered round verbatim.
        analyst_prompt = next(
            prompt
            for name, prompt in llm.prompts
            if name == "DiscoveryResult"
        )
        assert "Bereits geklaert" in analyst_prompt
        assert "Generative KI" in analyst_prompt
        # The planner saw the answers too (same root: the plan must
        # honor what the user pinned down).
        planner_prompt = next(
            prompt
            for name, prompt in llm.prompts
            if name == "ExecutionPlanModel"
        )
        assert "Bereits geklaert" in planner_prompt
        # The skip is visible, not silent.
        events = client.get(
            f"/v1/runs/{run_id}/events?format=json"
        ).json()["data"]
        narration_texts = [
            str((event.get("data") or {}).get("text", ""))
            for event in events
            if "narration" in str(event.get("type", ""))
        ]
        assert any("uebersprungen" in text for text in narration_texts)


def test_clarified_answers_reach_the_discovery_probes(monkeypatch):
    """Graph-level pin for ``clarified_context`` (P2/P8): the answered
    round must survive the LangGraph state schema into the probe
    queries — undeclared state keys are dropped silently between nodes,
    which is exactly how this feature was once a no-op."""
    llm = ScriptedLLM(
        overrides={
            "AssignmentProfile": {
                **PROFILE,
                "scope_clarity": "ambiguous",
                "sub_goals": [],
                "clarification_questions": [
                    {
                        "prompt": "Auf welchen Markt beziehen?",
                        "options": [
                            {"label": "Generative KI", "description": ""},
                            {"label": "Gesamtmarkt", "description": ""},
                        ],
                        "multi_select": False,
                    }
                ],
            },
        }
    )
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        run_id = submit_agent(client)
        wait_status(client, run_id, {"waiting_for_input"})
        clarifications = client.get(
            f"/v1/runs/{run_id}/clarifications"
        ).json()["data"]
        answered = client.post(
            f"/v1/runs/{run_id}/clarifications/"
            f"{clarifications[0]['clarification_id']}",
            json={"answers": {"q1": {"option_ids": ["q1_o1"], "text": ""}}},
        )
        assert answered.status_code == 200
        wait_status(client, run_id, {"waiting_for_approval"})
        events = client.get(
            f"/v1/runs/{run_id}/events?format=json"
        ).json()["data"]
        probe_queries = [
            str((event.get("data") or {}).get("query", ""))
            for event in events
            if event.get("type") == "inqtrix.agent.activity"
            and (event.get("data") or {}).get("phase") == "discovery"
        ]
        assert any(
            "Generative KI" in query for query in probe_queries
        ), probe_queries


def test_rejected_discovery_completes_with_receipt(monkeypatch):
    """The discovery gate (strict mode) follows the same rejection
    contract as the plan gate: clean completion with a receipt, no
    failure."""
    client = make_agent_client(monkeypatch)
    with client:
        run_id = submit_agent(client, autonomy="strict")
        wait_status(client, run_id, {"waiting_for_approval"})
        approvals = client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
        assert approvals[0]["kind"] == "discovery"
        rejected = client.post(
            f"/v1/runs/{run_id}/approvals/{approvals[0]['approval_id']}",
            json={"decision": "reject"},
        )
        assert rejected.status_code == 200
        wait_status(client, run_id, {"completed"})
        result = client.get(f"/v1/runs/{run_id}/result").json()
        assert "Erkundung abgelehnt" in result["answer"]


def test_ambiguous_assignment_asks_before_any_probe(monkeypatch):
    """Scenario 4: ask_user_first — no probe before the clarification."""
    llm = ScriptedLLM(
        overrides={
            "AssignmentProfile": {
                **PROFILE,
                "scope_clarity": "ambiguous",
                "clarification_questions": [
                    {
                        "prompt": "Welcher Markt?",
                        "options": [
                            {"label": "Europa", "description": ""},
                            {
                                "label": "USA",
                                "description": "Nordamerikanischer Markt",
                            },
                        ],
                        "multi_select": False,
                    }
                ],
            }
        }
    )
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        run_id = submit_agent(client)
        wait_status(client, run_id, {"waiting_for_input"})
        # Only the intake call ran — no discovery, no plan.
        assert client.llm.calls == ["structured:AssignmentProfile"]
        clarifications = client.get(
            f"/v1/runs/{run_id}/clarifications"
        ).json()["data"]
        assert clarifications[0]["status"] == "pending"
        # The structured round carries sanitized questions with
        # deterministic ids; the single question mirrors into the
        # legacy options column.
        questions = clarifications[0]["questions"]
        assert [q["id"] for q in questions] == ["q1"]
        assert [o["id"] for o in questions[0]["options"]] == [
            "q1_o1",
            "q1_o2",
        ]
        assert clarifications[0]["options"] == questions[0]["options"]
        answered = client.post(
            f"/v1/runs/{run_id}/clarifications/"
            f"{clarifications[0]['clarification_id']}",
            json={"answer": "Der europaeische Markt."},
        )
        assert answered.status_code == 200
        wait_status(client, run_id, {"waiting_for_approval"})
        approve_pending(client, run_id)
        wait_status(client, run_id, {"completed"})


def test_structured_round_answer_maps_options(monkeypatch):
    """A structured answers map resolves the round and lands in history."""
    llm = ScriptedLLM(
        overrides={
            "AssignmentProfile": {
                **PROFILE,
                "scope_clarity": "ambiguous",
                "clarification_questions": [
                    {
                        "prompt": "Welcher Markt?",
                        "options": [
                            {"label": "Europa", "description": ""},
                            {"label": "USA", "description": ""},
                        ],
                        "multi_select": False,
                    },
                    {
                        "prompt": "Welche Aspekte?",
                        "options": [
                            {"label": "Preise", "description": ""},
                            {"label": "Anbieter", "description": ""},
                            {"label": "Regulierung", "description": ""},
                        ],
                        "multi_select": True,
                    },
                ],
            }
        }
    )
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        run_id = submit_agent(client)
        wait_status(client, run_id, {"waiting_for_input"})
        clarifications = client.get(
            f"/v1/runs/{run_id}/clarifications"
        ).json()["data"]
        clarification_id = clarifications[0]["clarification_id"]
        # Validation matrix: partial answers are rejected loudly.
        partial = client.post(
            f"/v1/runs/{run_id}/clarifications/{clarification_id}",
            json={"answers": {"q1": {"option_ids": ["q1_o1"]}}},
        )
        assert partial.status_code == 400
        # Single-select rejects multiple picks.
        multi_on_single = client.post(
            f"/v1/runs/{run_id}/clarifications/{clarification_id}",
            json={
                "answers": {
                    "q1": {"option_ids": ["q1_o1", "q1_o2"]},
                    "q2": {"option_ids": ["q2_o1"]},
                }
            },
        )
        assert multi_on_single.status_code == 400
        answered = client.post(
            f"/v1/runs/{run_id}/clarifications/{clarification_id}",
            json={
                "answers": {
                    "q1": {"option_ids": ["q1_o1"], "text": ""},
                    "q2": {
                        "option_ids": ["q2_o1", "q2_o3"],
                        "text": "Fokus auf B2B",
                    },
                }
            },
        )
        assert answered.status_code == 200
        assert answered.json()["answers"]["q2"]["option_ids"] == [
            "q2_o1",
            "q2_o3",
        ]
        wait_status(client, run_id, {"waiting_for_approval"})
        approve_pending(client, run_id)
        wait_status(client, run_id, {"completed"})


def test_discovery_web_preview_gated_to_autonomous(monkeypatch):
    """E16 amendment: in Standard/balanced the discovery phase makes NO
    web contact even with the preview setting on — all web goes through
    the approved plan. Autonomous keeps the preview."""
    balanced = make_agent_client(monkeypatch)
    with balanced:
        run_id = submit_agent(balanced)  # default balanced
        wait_status(balanced, run_id, {"waiting_for_approval"})
        # The plan gate parks AFTER discovery ran — no web probe fired.
        assert balanced.container.providers.search.queries == []
        approve_pending(balanced, run_id)
        wait_status(balanced, run_id, {"completed"})

    autonomous = make_agent_client(
        monkeypatch,
        search=RetryNotifyingSearch(),
    )
    with autonomous:
        run_id = submit_agent(autonomous, autonomy="autonomous")
        wait_status(autonomous, run_id, {"completed"})
        # The preview ran during discovery (plus any plan web_instant).
        assert autonomous.container.providers.search.queries
        retry = next(
            event["data"]
            for event in run_events(autonomous, run_id)
            if event["type"] == "inqtrix.agent.activity"
            and event["data"].get("retry")
        )
        assert retry["scope"] == "discovery"
        assert retry["purpose"] == "Webvorschau in der Erkundung"


def test_discovery_activity_progress_is_scoped_per_operation(monkeypatch):
    goals = [f"Teilfrage {index}" for index in range(1, 5)]
    llm = ScriptedLLM(
        overrides={
            "AssignmentProfile": {
                **PROFILE,
                "sub_goals": goals,
                "needs_web": False,
            }
        }
    )
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        run_id = submit_agent(client)
        wait_status(client, run_id, {"waiting_for_approval"})
        approve_pending(client, run_id)
        wait_status(client, run_id, {"completed"})
        activities = [
            event["data"]
            for event in run_events(client, run_id)
            if event["type"] == "inqtrix.agent.activity"
            and event["data"].get("scope") == "discovery"
            and event["data"].get("status") == "started"
        ]

    searches = [
        activity
        for activity in activities
        if activity["operation"] == "knowledge.search"
    ]
    assert [activity["current"] for activity in searches] == [1, 2, 3, 4]
    assert {activity["total"] for activity in searches} == {4}
    assert [activity["detail"] for activity in searches] == goals
    collections = [
        activity
        for activity in activities
        if activity["operation"] == "knowledge.collections.list"
    ]
    assert [(item["current"], item["total"]) for item in collections] == [
        (1, 1)
    ]


def test_chat_deliverable_writes_answer_artifact_not_memo(monkeypatch):
    """S3: response_form=chat -> run-local answer artifact, NO memo."""
    llm = ScriptedLLM(
        overrides={
            "AssignmentProfile": {**PROFILE, "response_form": "chat"},
            "SectionText": {
                "markdown": "Direkte Antwort: Der Markt waechst [W1]."
            },
        }
    )
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        run_id = submit_agent(client, session_id="sess-chat")
        wait_status(client, run_id, {"waiting_for_approval"})
        approve_pending(client, run_id)
        summary = wait_status(client, run_id, {"completed"})
        artifacts = client.get(f"/v1/runs/{run_id}/artifacts").json()["data"]
        kinds = {a["kind"] for a in artifacts}
        assert "answer" in kinds
        assert "memo" not in kinds
        answer_meta = [a for a in artifacts if a["kind"] == "answer"][0]
        # Run-local: the answer never joins the session-memo lineage.
        assert answer_meta["session_id"] is None
        assert answer_meta["status"] == "ready"
        detail = client.get(
            f"/v1/runs/{run_id}/artifacts/{answer_meta['artifact_id']}"
        ).json()
        assert (
            detail["content_markdown"]
            == "Direkte Antwort: Der Markt waechst [W1]."
        )
        # AgentResult.answer == artifact body (one truth).
        result = client.get(f"/v1/runs/{run_id}/result").json()
        assert result["answer"] == detail["content_markdown"]
        assert summary["status"] == "completed"


def test_response_form_request_override_forces_chat(monkeypatch):
    """S3: the composer override beats the intake profile."""
    client = make_agent_client(monkeypatch)  # profile says canvas
    with client:
        response = client.post(
            "/v1/runs",
            json={
                "question": "Erstelle eine Marktanalyse.",
                "mode": "workspace_agent",
                "session_id": "sess-override",
                "response_form": "chat",
            },
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]
        # The stored overrides mirror the form for summaries/audit (K3).
        assert (
            response.json()["agent_overrides"]["response_form"] == "chat"
        )
        wait_status(client, run_id, {"waiting_for_approval"})
        approve_pending(client, run_id)
        wait_status(client, run_id, {"completed"})
        kinds = {
            a["kind"]
            for a in client.get(
                f"/v1/runs/{run_id}/artifacts"
            ).json()["data"]
        }
        assert "answer" in kinds
        assert "memo" not in kinds


def test_response_form_invalid_is_rejected(monkeypatch):
    client = make_agent_client(monkeypatch)
    with client:
        response = client.post(
            "/v1/runs",
            json={
                "question": "Frage.",
                "mode": "workspace_agent",
                "response_form": "pdf",
            },
        )
        assert response.status_code == 400
        assert "response_form" in response.text


def test_answer_node_routes_from_approved_plan_tasks():
    """R1 uses authorized work, including a web task with zero results."""
    from inqtrix.agents.algorithm import answer_node_for

    assert answer_node_for([]) == "agent_answer_light"
    assert (
        answer_node_for([{"tool_kind": "rag_query"}, {"tool_kind": "synthesis"}])
        == "agent_answer_light"
    )
    assert (
        answer_node_for([{"tool_kind": "web_research"}, {"tool_kind": "synthesis"}])
        == "agent_answer"
    )


def test_follow_up_turn_receives_session_history(monkeypatch):
    """K1: a follow-up turn's intake sees the prior turn's Q + answer."""
    captured: list[str] = []

    def capturing_profile(prompt: str) -> dict[str, Any]:
        captured.append(prompt)
        return dict(PROFILE)

    llm = ScriptedLLM()
    llm.scripts["AssignmentProfile"] = capturing_profile
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        run1 = submit_agent(client, session_id="sess-k1")
        wait_status(client, run1, {"waiting_for_approval"})
        approve_pending(client, run1)
        wait_status(client, run1, {"completed"})
        run2 = submit_agent(client, session_id="sess-k1")
        wait_status(client, run2, {"waiting_for_approval"})
        approve_pending(client, run2)
        wait_status(client, run2, {"completed"})
    assert len(captured) >= 2
    follow_up = captured[1]
    assert "Bisheriger Verlauf" in follow_up
    assert "Nutzer: Erstelle eine Marktanalyse." in follow_up
    # The prior turn's memo body reaches the follow-up as the agent
    # answer (rows are truth — built server-side, not sent by the FE).
    assert "Agent: " in follow_up
    assert "Die Marktlage ist stabil" in follow_up
    # The first turn ran WITHOUT history (nothing prior in the session).
    assert "Bisheriger Verlauf" not in captured[0]


def test_plan_repair_retry_then_loud_failure(monkeypatch):
    """Scenario: two invalid plans -> run fails plan_invalid (no memo)."""
    bad_plan = {"summary_markdown": "x", "tasks": [
        {"id": "t1", "title": "Nur Suche", "tool_kind": "rag_query"}
    ]}
    llm = ScriptedLLM(overrides={"ExecutionPlanModel": bad_plan})
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        run_id = submit_agent(client)
        summary = wait_status(client, run_id, {"failed"})
        assert "plan_invalid" in summary["error"]["message"]
        # Exactly two planner attempts (one repair round).
        assert client.llm.calls.count("structured:ExecutionPlanModel") == 2


def test_web_instant_task_spawns_no_child(monkeypatch):
    """Scenario 13a: a punctual gap uses web_instant, zero children."""
    instant_plan = {
        "summary_markdown": "Ein Instant-Task.",
        "tasks": [
            {
                "id": "t1",
                "title": "Fakt nachschlagen",
                "tool_kind": "web_instant",
                "objective": "Aktuelles Marktvolumen kritisch pruefen.",
                "queries": [
                    "Welches belastbare Marktvolumen wird fuer 2026 berichtet?"
                ],
                "expected_output": "Eine belegte Kurzantwort mit Risiken.",
                "is_falsification": True,
            },
            {
                "id": "s",
                "title": "Synthese",
                "tool_kind": "synthesis",
                "depends_on": ["t1"],
            },
        ],
    }
    llm = ScriptedLLM(overrides={"ExecutionPlanModel": instant_plan})
    search = FakeSearch()
    client = make_agent_client(monkeypatch, llm=llm, search=search)
    with client:
        run_id = submit_agent(client, autonomy="autonomous")
        wait_status(client, run_id, {"completed"})
        approved_query = (
            "Welches belastbare Marktvolumen wird fuer 2026 berichtet?"
        )
        assert search.queries[-1] == approved_query
        assert search.queries.count(approved_query) == 1
        children = client.get(f"/v1/runs/{run_id}/children").json()["data"]
        assert children == []
        # The instant source became a W-reference in the memo refs.
        artifacts = client.get(f"/v1/runs/{run_id}/artifacts").json()["data"]
        memo = [a for a in artifacts if a["kind"] == "memo"][0]
        detail = client.get(
            f"/v1/runs/{run_id}/artifacts/{memo['artifact_id']}"
        ).json()
        assert any(
            ref.get("url") == "https://example.com/markt"
            for ref in detail["refs"]
        )


def test_url_only_instant_evidence_reaches_synthesis_and_artifact(monkeypatch):
    """Azure-shaped URL citations retain grounded, non-quote support."""
    synthesis_prompts: list[str] = []

    def section_text(prompt: str) -> dict[str, str]:
        synthesis_prompts.append(prompt)
        return {
            "markdown": "Der Markt erreicht US-$1.5T [W1]; Formel $x$.",
        }

    class UrlOnlySearch(FakeSearch):
        def search(self, query: str, **kwargs: Any) -> GroundedSearchResult:
            self.queries.append(query)
            return GroundedSearchResult(
                answer=(
                    "Gartner beziffert die weltweiten KI-Ausgaben 2025 "
                    "auf US-$1.5T "
                    "([Gartner](https://example.com/markt))."
                ),
                sources=[
                    GroundedSource(
                        url="https://example.com/markt",
                        title="Gartner",
                        snippet="",
                        origin="url_citation",
                    ),
                    GroundedSource(
                        url="https://example.com/zweite-quelle",
                        title="Second source",
                        snippet="Second evidence that belongs to W2 only.",
                    ),
                ],
            )

    llm = ScriptedLLM(overrides={"SectionText": section_text})
    client = make_agent_client(monkeypatch, llm=llm, search=UrlOnlySearch())
    with client:
        run_id = submit_agent(client, autonomy="autonomous")
        wait_status(client, run_id, {"completed"})
        artifacts = client.get(f"/v1/runs/{run_id}/artifacts").json()["data"]
        memo = [item for item in artifacts if item["kind"] == "memo"][0]
        detail = client.get(
            f"/v1/runs/{run_id}/artifacts/{memo['artifact_id']}"
        ).json()
        result = client.get(f"/v1/runs/{run_id}/result").json()

    assert synthesis_prompts
    assert any(
        "weltweiten KI-Ausgaben 2025" in prompt
        for prompt in synthesis_prompts
    )
    assert all("Second evidence" not in prompt for prompt in synthesis_prompts)
    ref = next(
        item
        for item in detail["refs"]
        if item.get("url") == "https://example.com/markt"
    )
    assert ref.get("excerpt") in (None, "")
    assert "weltweiten KI-Ausgaben 2025" in ref["grounded_support"]
    assert "\\$" not in ref["grounded_support"]
    exported_ref = next(
        item
        for item in result["references"]
        if item.get("url") == "https://example.com/markt"
    )
    assert exported_ref["label"] == "W1"
    assert "weltweiten KI-Ausgaben 2025" in exported_ref[
        "grounded_support"
    ]
    assert r"US-\$1.5T" in detail["content_markdown"]
    assert "$x$" in detail["content_markdown"]


def test_independent_instant_tasks_overlap_and_call_each_query_once(
    monkeypatch,
):
    """One plan wave runs independent Instant calls concurrently."""
    import threading

    questions = (
        "Welche belastbaren Quellen beziffern das Marktvolumen 2026?",
        "Welche Gegenbelege relativieren das Marktwachstum 2026?",
    )
    parallel_plan = {
        "summary_markdown": "Zwei unabhaengige Instant-Fragen.",
        "tasks": [
            {
                "id": "t1",
                "title": "Marktvolumen",
                "tool_kind": "web_instant",
                "queries": [questions[0]],
            },
            {
                "id": "t2",
                "title": "Gegenbelege",
                "tool_kind": "web_instant",
                "queries": [questions[1]],
                "is_falsification": True,
            },
            {
                "id": "s",
                "title": "Synthese",
                "tool_kind": "synthesis",
                "depends_on": ["t1", "t2"],
            },
        ],
    }
    barrier = threading.Barrier(2)

    class ParallelSearch(FakeSearch):
        def __init__(self) -> None:
            super().__init__()
            self._lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def search(self, query: str, **kwargs: Any) -> GroundedSearchResult:
            with self._lock:
                self.queries.append(query)
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            barrier.wait(timeout=5)
            try:
                return GroundedSearchResult(
                    answer=f"Antwort zu {query}",
                    sources=[
                        GroundedSource(
                            url=f"https://example.com/{len(self.queries)}",
                            title="Quelle",
                            snippet="Beleg",
                        )
                    ],
                )
            finally:
                with self._lock:
                    self.active -= 1

    llm = ScriptedLLM(
        overrides={
            "AssignmentProfile": {**PROFILE, "needs_web": False},
            "ExecutionPlanModel": parallel_plan,
        }
    )
    search = ParallelSearch()
    client = make_agent_client(monkeypatch, llm=llm, search=search)
    with client:
        run_id = submit_agent(client, autonomy="autonomous")
        wait_status(client, run_id, {"completed"})

    assert sorted(search.queries) == sorted(questions)
    assert all(search.queries.count(question) == 1 for question in questions)
    assert search.max_active == 2


def test_fast_instant_task_persists_before_its_slow_sibling(monkeypatch):
    """A completed work unit must not look active until its whole wave ends."""
    import threading

    fast_query = "Welche Quelle beantwortet die schnelle Frage?"
    slow_query = "Welche Quelle beantwortet die langsame Frage?"
    parallel_plan = {
        "summary_markdown": "Zwei zeitlich unterschiedliche Aufgaben.",
        "tasks": [
            {
                "id": "fast",
                "title": "Schnelle Frage",
                "tool_kind": "web_instant",
                "queries": [fast_query],
            },
            {
                "id": "slow",
                "title": "Langsame Frage",
                "tool_kind": "web_instant",
                "queries": [slow_query],
            },
            {
                "id": "s",
                "title": "Synthese",
                "tool_kind": "synthesis",
                "depends_on": ["fast", "slow"],
            },
        ],
    }
    fast_finished = threading.Event()
    slow_started = threading.Event()
    release_slow = threading.Event()

    class StaggeredSearch(FakeSearch):
        def search(self, query: str, **kwargs: Any) -> GroundedSearchResult:
            if query == slow_query:
                slow_started.set()
                assert release_slow.wait(5)
            elif query == fast_query:
                fast_finished.set()
            else:  # pragma: no cover - planner contract makes this impossible
                raise AssertionError(query)
            return GroundedSearchResult(
                answer=f"Antwort zu {query}",
                sources=[
                    GroundedSource(
                        url=f"https://example.com/{'fast' if query == fast_query else 'slow'}",
                        title="Quelle",
                        snippet="Beleg",
                    )
                ],
            )

    llm = ScriptedLLM(
        overrides={
            "AssignmentProfile": {**PROFILE, "needs_web": False},
            "ExecutionPlanModel": parallel_plan,
        }
    )
    client = make_agent_client(
        monkeypatch, llm=llm, search=StaggeredSearch()
    )
    try:
        with client:
            run_id = submit_agent(client, autonomy="autonomous")
            assert fast_finished.wait(5)
            assert slow_started.wait(5)

            deadline = time.monotonic() + 5
            task_statuses: dict[str, str] = {}
            while time.monotonic() < deadline:
                plan = client.get(f"/v1/runs/{run_id}/plan").json()
                task_statuses = {
                    task["task_id"]: task["status"]
                    for task in plan["tasks"]
                }
                if task_statuses.get("fast") == "completed":
                    break
                time.sleep(0.02)

            assert task_statuses["fast"] == "completed"
            assert task_statuses["slow"] == "running"
            release_slow.set()
            wait_status(client, run_id, {"completed"})
    finally:
        release_slow.set()


def test_knowledge_hits_flow_through_capability_layer(monkeypatch):
    """Scenario 1b: knowledge.search wire shape (``hits``) reaches both
    consumers — the discovery probe digest and file_analysis evidence.

    Uses the REAL catalog output models through the real registry, so a
    key rename in the capability layer breaks this test instead of
    silently reporting "Keine Treffer" against a filled index.
    """
    from inqtrix.capabilities.catalog.knowledge import (
        CollectionsListInput,
        CollectionsListOutput,
        CollectionSummary,
        KnowledgeHit,
        KnowledgeSearchInput,
        KnowledgeSearchOutput,
    )
    from inqtrix.capabilities.contracts import CapabilityDefinition, Effect

    async def _list(payload, context):
        return CollectionsListOutput(
            collections=[
                CollectionSummary(
                    id="c1",
                    name="Marktberichte",
                    embedding_model="emb-1",
                    document_count=1,
                )
            ]
        )

    capability_queries: list[str] = []

    async def _search(payload, context):
        capability_queries.append(payload.query)
        return KnowledgeSearchOutput(
            query=payload.query,
            hits=[
                KnowledgeHit(
                    document_id="doc-1",
                    collection_id="c1",
                    document_title="Interner Marktbericht",
                    chunk_index=0,
                    chunk_id="ch-1",
                    rank=1,
                    text="Interner Marktbericht: Anbieter A fuehrt.",
                    source_text="Interner Marktbericht: Anbieter A fuehrt.",
                    page_number=None,
                    score=0.9,
                )
            ],
        )

    analyst_prompts: list[str] = []

    def discovery_script(prompt: str) -> dict[str, Any]:
        analyst_prompts.append(prompt)
        return dict(DISCOVERY)

    file_plan = {
        "summary_markdown": "Ein Datei-Analyse-Task, dann Synthese.",
        "tasks": [
            {
                "id": "t1",
                "title": "Bericht auswerten",
                "tool_kind": "file_analysis",
                "gap_ids": ["g1"],
                "objective": "Interne Anbieterbefunde pruefen.",
                "queries": ["Anbieteruebersicht"],
                "expected_output": "Relevante interne Befunde mit Belegstellen.",
                "is_falsification": True,
            },
            {
                "id": "s",
                "title": "Synthese",
                "tool_kind": "synthesis",
                "depends_on": ["t1"],
            },
        ],
    }
    llm = ScriptedLLM(
        overrides={
            "ExecutionPlanModel": file_plan,
            "DiscoveryResult": discovery_script,
        }
    )
    client = make_agent_client(monkeypatch, llm=llm)
    client.container.capability_registry.register_all(
        [
            CapabilityDefinition(
                id="knowledge.collections.list",
                summary="test stub",
                input_model=CollectionsListInput,
                output_model=CollectionsListOutput,
                effect=Effect.READ,
                idempotent=True,
                handler=_list,
            ),
            CapabilityDefinition(
                id="knowledge.search",
                summary="test stub",
                input_model=KnowledgeSearchInput,
                output_model=KnowledgeSearchOutput,
                effect=Effect.READ,
                idempotent=True,
                handler=_search,
            ),
        ]
    )
    with client:
        run_id = submit_agent(client, autonomy="autonomous")
        wait_status(client, run_id, {"completed"})
        # Probe hits reached the analyst digest instead of "Keine Treffer".
        assert analyst_prompts, llm.calls
        assert "[Intern doc:doc-1#0]" in analyst_prompts[0]
        assert "Anbieter A fuehrt" in analyst_prompts[0]
        assert "Keine Treffer" not in analyst_prompts[0]
        # file_analysis consumed the hits (past the insufficient branch)
        # and its evidence became a K-reference on the memo.
        assert "structured:FileAnalysisSummary" in llm.calls
        assert any(
            "Expected output: Relevante interne Befunde mit Belegstellen."
            in query
            and "actively seek counter-evidence" in query
            for query in capability_queries
        )
        children = client.get(f"/v1/runs/{run_id}/children").json()["data"]
        assert children == []
        artifacts = client.get(f"/v1/runs/{run_id}/artifacts").json()["data"]
        memo = [a for a in artifacts if a["kind"] == "memo"][0]
        detail = client.get(
            f"/v1/runs/{run_id}/artifacts/{memo['artifact_id']}"
        ).json()
        assert any(
            ref.get("document_id") == "doc-1" for ref in detail["refs"]
        )


def test_rag_query_task_completes_with_memo_k_references(monkeypatch):
    """Scenario: a rag_query task drives the registered ``knowledge`` mode
    to a COMPLETED run and its ``report_references`` become K-references on
    the memo.

    The default agent harness registers no ``knowledge`` algorithm, so
    _run_rag_query's success branch (reference aggregation) is only reached
    when one is present — the bad-plan and no-web scenarios exercise only
    its failure branches.
    """
    from inqtrix.core.results import AgentResult

    class _FakeKnowledgeAlgorithm:
        id = "knowledge"
        display_name = "Knowledge (test stub)"

        def capabilities(self) -> dict:
            return {}

        def run(self, request, *, runtime, context):
            answer = "Interne Befunde: Anbieter A fuehrt."
            return AgentResult(
                answer=answer,
                raw={
                    "answer": answer,
                    "usage": {"prompt_tokens": 4, "completion_tokens": 3},
                    "result_state": {
                        "answer": answer,
                        "report_references": [
                            {
                                "url": None,
                                "title": "Interner Marktbericht",
                                "document_id": "doc-9",
                                "chunk_index": 0,
                                "source_text": "Anbieter A fuehrt.",
                            }
                        ],
                    },
                },
            )

    rag_plan = {
        "summary_markdown": "Ein RAG-Task, dann Synthese.",
        "tasks": [
            {
                "id": "t1",
                "title": "Interne Quellen befragen",
                "tool_kind": "rag_query",
                "gap_ids": ["g1"],
                "objective": "Interne Anbieterbefunde pruefen.",
                "queries": ["Anbieteruebersicht"],
                "expected_output": "Belegte interne Befunde.",
                "params": {"profile": "gruendlich"},
            },
            {
                "id": "s",
                "title": "Synthese",
                "tool_kind": "synthesis",
                "depends_on": ["t1"],
            },
        ],
    }
    llm = ScriptedLLM(overrides={"ExecutionPlanModel": rag_plan})
    client = make_agent_client(monkeypatch, llm=llm)
    client.container.registry.register(_FakeKnowledgeAlgorithm())
    with client:
        run_id = submit_agent(client, autonomy="autonomous")
        wait_status(client, run_id, {"completed"})
        # A rag_query task never spawns child research runs.
        children = client.get(f"/v1/runs/{run_id}/children").json()["data"]
        assert children == []
        artifacts = client.get(f"/v1/runs/{run_id}/artifacts").json()["data"]
        memo = [a for a in artifacts if a["kind"] == "memo"][0]
        detail = client.get(
            f"/v1/runs/{run_id}/artifacts/{memo['artifact_id']}"
        ).json()
        assert any(
            ref.get("document_id") == "doc-9" for ref in detail["refs"]
        )


def test_rag_answer_without_references_is_insufficient_evidence(monkeypatch):
    """An answer string alone is not a completed evidence task."""
    from inqtrix.core.results import AgentResult

    class _UnreferencedKnowledgeAlgorithm:
        id = "knowledge"
        display_name = "Knowledge without evidence (test stub)"

        def capabilities(self) -> dict:
            return {}

        def run(self, request, *, runtime, context):
            answer = "Eine unreferenzierte interne Antwort."
            return AgentResult(
                answer=answer,
                raw={
                    "answer": answer,
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                    "result_state": {
                        "answer": answer,
                        "report_references": [],
                    },
                },
            )

    plan = {
        "summary_markdown": "Interne Evidenz pruefen.",
        "tasks": [
            {
                "id": "t1",
                "title": "Interne Quellen befragen",
                "tool_kind": "rag_query",
                "gap_ids": ["g1"],
                "queries": ["Welche Belege liegen intern vor?"],
            },
            {
                "id": "s",
                "title": "Synthese",
                "tool_kind": "synthesis",
                "depends_on": ["t1"],
            },
        ],
    }
    client = make_agent_client(
        monkeypatch,
        llm=ScriptedLLM(overrides={"ExecutionPlanModel": plan}),
    )
    client.container.registry.register(_UnreferencedKnowledgeAlgorithm())
    with client:
        run_id = submit_agent(client, autonomy="autonomous")
        wait_status(client, run_id, {"completed"})
        stored = client.get(f"/v1/runs/{run_id}/plan").json()

    task = next(item for item in stored["tasks"] if item["task_id"] == "t1")
    assert task["status"] == "insufficient_evidence"


def _put_target_document(client: TestClient, document_id: str) -> None:
    saved = client.put(
        f"/v1/editor/documents/{document_id}",
        json={
            "title": "Compliance-Roadmap",
            "content_markdown": (
                "# Compliance-Roadmap\n\n"
                "Alte Kernaussage zur Regulierung.\n\n"
                "Weitere Details folgen."
            ),
            "created_at": 1_700_000_000,
            "updated_at": 1_700_000_000,
            "folder_id": None,
            "revision": 1,
            "source": "blank",
        },
    )
    assert saved.status_code == 200, saved.text


PATCH_INSTRUCT = {
    "assistant_message": "Kernaussage aktualisiert.",
    "edits": [
        {
            "find": "Alte Kernaussage zur Regulierung.",
            "quote_before": "",
            "quote_after": "",
            "position": "replace",
            "text": "Neue Kernaussage auf Basis der Recherche [W1].",
            "note": "Aktualisiert.",
        }
    ],
    "warnings": [],
}


def test_document_target_patch_is_always_gated_and_never_self_applied(
    monkeypatch,
):
    """M7 scenario: a document-targeted assignment proposes an editor patch
    and parks for the patch approval EVEN in autonomous mode (the E16
    write invariant). Approving resumes the run; the patch row stays
    PENDING — apply happens only through the user-driven route."""
    llm = ScriptedLLM(
        overrides={"inqtrix_editor_instruction_v1": dict(PATCH_INSTRUCT)}
    )
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        _put_target_document(client, "doc-m7")
        response = client.post(
            "/v1/runs",
            json={
                "question": "Arbeite die Rechercheergebnisse ein.",
                "mode": "workspace_agent",
                "autonomy": "autonomous",
                "document_id": "doc-m7",
            },
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]

        # Autonomous skips the PLAN gate — the ONLY interrupt is the patch.
        wait_status(client, run_id, {"waiting_for_approval"})
        approvals = client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
        pending = [a for a in approvals if a["status"] == "pending"]
        assert len(pending) == 1
        assert pending[0]["kind"] == "patch"
        assert pending[0]["subject_type"] == "editor_patch"
        patch_id = pending[0]["subject_id"]
        assert patch_id

        # The proposal row exists, source=agent, still pending.
        patches = client.get("/v1/editor/documents/doc-m7/patches").json()
        assert [p["patch_id"] for p in patches["data"]] == [patch_id]
        assert patches["data"][0]["source"] == "agent"
        assert patches["data"][0]["status"] == "pending"

        decided = client.post(
            f"/v1/runs/{run_id}/approvals/{pending[0]['approval_id']}",
            json={"decision": "approve"},
        )
        assert decided.status_code == 200, decided.text
        wait_status(client, run_id, {"completed"})

        # Rows are the truth: the approval row records the decision.
        approvals = client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
        assert approvals[0]["kind"] == "patch"
        assert approvals[0]["status"] == "approved"
        # E14: approval resumes the run, it never applies — the row is
        # still pending until the user hits the ONE apply route.
        detail = client.get(f"/v1/editor/patches/{patch_id}").json()
        assert detail["status"] == "pending"

        # The artifact row references the patch id only (rule R3).
        artifacts = client.get(f"/v1/runs/{run_id}/artifacts").json()["data"]
        kinds = {artifact["kind"] for artifact in artifacts}
        assert "editor_patch" in kinds

        # The user applies through route #16; the document text changes.
        applied = client.post(
            f"/v1/editor/patches/{patch_id}:apply",
            json={"expected_revision": detail["document_revision"]},
        )
        assert applied.status_code == 200, applied.text
        assert applied.json()["applied_edit_ids"] == ["ed_1"]
        document = client.get("/v1/editor/documents/doc-m7").json()
        assert "Neue Kernaussage auf Basis der Recherche" in (
            document["content_markdown"]
        )


def test_patch_rejection_ends_the_run_normally(monkeypatch):
    """Rejecting the patch is a normal outcome: the memo stays the
    deliverable, the run COMPLETES with patch_decision=rejected."""
    llm = ScriptedLLM(
        overrides={"inqtrix_editor_instruction_v1": dict(PATCH_INSTRUCT)}
    )
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        _put_target_document(client, "doc-m7b")
        response = client.post(
            "/v1/runs",
            json={
                "question": "Arbeite die Rechercheergebnisse ein.",
                "mode": "workspace_agent",
                "autonomy": "autonomous",
                "document_id": "doc-m7b",
            },
        )
        run_id = response.json()["run_id"]
        wait_status(client, run_id, {"waiting_for_approval"})
        pending = [
            a
            for a in client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
            if a["status"] == "pending"
        ]
        client.post(
            f"/v1/runs/{run_id}/approvals/{pending[0]['approval_id']}",
            json={"decision": "reject"},
        )
        summary = wait_status(client, run_id, {"completed"})
        assert summary["status"] == "completed"
        approvals = client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
        assert approvals[0]["status"] == "rejected"
        # The reject decision cascades onto the patch ROW too — rejected
        # edits must never stay pending/appliable (review finding).
        patch_id = approvals[0]["subject_id"]
        detail = client.get(f"/v1/editor/patches/{patch_id}").json()
        assert detail["status"] == "rejected"
        applied = client.post(
            f"/v1/editor/patches/{patch_id}:apply",
            json={"expected_revision": detail["document_revision"]},
        )
        assert applied.status_code == 409
        # The memo artifact survives the rejection.
        artifacts = client.get(f"/v1/runs/{run_id}/artifacts").json()["data"]
        assert any(artifact["kind"] == "memo" for artifact in artifacts)


def test_schnell_tier_document_target_still_proposes_the_patch(monkeypatch):
    """Skipping the critic (labels tier) must never skip the deliverable:
    a document-targeted schnell run still reaches the patch phase."""
    llm = ScriptedLLM(
        overrides={"inqtrix_editor_instruction_v1": dict(PATCH_INSTRUCT)}
    )
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        _put_target_document(client, "doc-schnell")
        response = client.post(
            "/v1/runs",
            json={
                "question": "Arbeite die Rechercheergebnisse ein.",
                "mode": "workspace_agent",
                "autonomy": "autonomous",
                "document_id": "doc-schnell",
                "agent_overrides": {"agent_tier": "schnell"},
            },
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]
        wait_status(client, run_id, {"waiting_for_approval"})
        approvals = client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
        pending = [a for a in approvals if a["status"] == "pending"]
        assert len(pending) == 1
        assert pending[0]["kind"] == "patch"


def test_invisible_target_document_fails_loudly(monkeypatch):
    """A document id the owner cannot see is a hard, loud failure."""
    client = make_agent_client(monkeypatch)
    with client:
        response = client.post(
            "/v1/runs",
            json={
                "question": "Arbeite die Rechercheergebnisse ein.",
                "mode": "workspace_agent",
                "autonomy": "autonomous",
                "document_id": "doc-does-not-exist",
            },
        )
        run_id = response.json()["run_id"]
        summary = wait_status(client, run_id, {"failed"})
        assert summary["error"]["message"] == "patch_document_not_found"


def test_cancel_while_waiting_cancels_cleanly(monkeypatch):
    client = make_agent_client(monkeypatch)
    with client:
        run_id = submit_agent(client)
        wait_status(client, run_id, {"waiting_for_approval"})
        cancelled = client.post(f"/v1/runs/{run_id}/cancel")
        assert cancelled.json()["status"] == "cancelled"
        plan = client.get(f"/v1/runs/{run_id}/plan").json()
        assert {task["status"] for task in plan["tasks"]} == {"skipped"}
        types = event_types(client, run_id)
        assert types[-1] == "inqtrix.run.cancelled"


def test_cancel_during_execution_ends_cancelled_not_failed(monkeypatch):
    """A client cancel DURING the task wave ends the run CANCELLED —
    the all-hard-failed check must not reinterpret tasks that failed
    BECAUSE of the cancel as ``all_tasks_failed``."""
    import threading

    started = threading.Event()
    release = threading.Event()

    class BlockingSearch(FakeSearch):
        def search(self, query: str, **kwargs: Any) -> GroundedSearchResult:
            started.set()
            release.wait(10)
            raise RuntimeError("Suche waehrend Cancel abgebrochen")

    instant_plan = {
        "summary_markdown": "Ein Instant-Task.",
        "tasks": [
            {
                "id": "t1",
                "title": "Fakt nachschlagen",
                "tool_kind": "web_instant",
                "gap_ids": ["g1"],
                "queries": ["Marktvolumen 2026"],
            },
            {
                "id": "s",
                "title": "Synthese",
                "tool_kind": "synthesis",
                "depends_on": ["t1"],
            },
        ],
    }
    # No discovery web preview: the block must hit during EXECUTION.
    profile = dict(PROFILE, needs_web=False)
    llm = ScriptedLLM(
        overrides={
            "AssignmentProfile": profile,
            "ExecutionPlanModel": instant_plan,
        }
    )
    client = make_agent_client(monkeypatch, llm=llm, search=BlockingSearch())
    with client:
        run_id = submit_agent(client, autonomy="autonomous")
        assert started.wait(10), "web_instant task never started"
        cancel = client.post(f"/v1/runs/{run_id}/cancel")
        assert cancel.status_code == 200, cancel.text
        release.set()
        final = wait_status(client, run_id, {"cancelled", "failed"})
        assert final["status"] == "cancelled", final
        plan = client.get(f"/v1/runs/{run_id}/plan").json()
        assert all(task["status"] != "running" for task in plan["tasks"])
        assert plan["tasks"][0]["status"] == "failed"
        assert plan["tasks"][1]["status"] == "skipped"
        types = event_types(client, run_id)
        assert types[-1] == "inqtrix.run.cancelled"


def test_task_agent_cancel_marks_parent_and_remaining_tasks(monkeypatch):
    """A task-level AgentCancelled is folded before the parent cancels."""
    from inqtrix.exceptions import AgentCancelled

    class CancellingSearch(FakeSearch):
        def search(self, query: str, **kwargs: Any) -> GroundedSearchResult:
            raise AgentCancelled("provider observed cancellation")

    llm = ScriptedLLM(
        overrides={"AssignmentProfile": {**PROFILE, "needs_web": False}}
    )
    client = make_agent_client(
        monkeypatch,
        llm=llm,
        search=CancellingSearch(),
    )
    with client:
        run_id = submit_agent(client, autonomy="autonomous")
        final = wait_status(client, run_id, {"cancelled", "failed"})
        assert final["status"] == "cancelled"
        plan = client.get(f"/v1/runs/{run_id}/plan").json()

    assert plan["tasks"][0]["status"] == "failed"
    assert plan["tasks"][1]["status"] == "skipped"


def test_operator_token_budget_fails_distinct_from_client_cancel(monkeypatch):
    client = make_agent_client(monkeypatch, max_tokens_per_run=10)
    with client:
        run_id = submit_agent(client, autonomy="autonomous")
        summary = wait_status(client, run_id, {"failed", "cancelled"})

    assert summary["status"] == "failed"
    assert summary["error"]["type"] == "token_budget_exceeded"
    assert "Tokenbudget" in summary["error"]["message"]


def test_budget_stop_persists_every_finished_parallel_task(monkeypatch):
    import inqtrix.agents.algorithm as agent_algorithm
    from inqtrix.agents.scheduler import TaskOutcome

    parallel_plan = {
        "summary_markdown": "Zwei parallele Aufgaben.",
        "tasks": [
            {
                "id": "t1",
                "title": "Erste Frage",
                "tool_kind": "web_instant",
                "queries": ["Welche Quelle beantwortet die erste Frage?"],
            },
            {
                "id": "t2",
                "title": "Zweite Frage",
                "tool_kind": "web_instant",
                "queries": ["Welche Quelle beantwortet die zweite Frage?"],
            },
            {
                "id": "s",
                "title": "Synthese",
                "tool_kind": "synthesis",
                "depends_on": ["t1", "t2"],
            },
        ],
    }
    llm = ScriptedLLM(
        overrides={
            "AssignmentProfile": {**PROFILE, "needs_web": False},
            "ExecutionPlanModel": parallel_plan,
        }
    )

    def completed_task(*_args: Any, **_kwargs: Any) -> TaskOutcome:
        return TaskOutcome(
            status="completed",
            summary="Belegt.",
            usage={"prompt_tokens": 6, "completion_tokens": 0},
        )

    monkeypatch.setattr(agent_algorithm, "_run_web_instant", completed_task)
    client = make_agent_client(
        monkeypatch,
        llm=llm,
        max_tokens_per_run=40,
    )
    with client:
        run_id = submit_agent(client, autonomy="autonomous")
        summary = wait_status(client, run_id, {"failed"})
        plan = client.get(f"/v1/runs/{run_id}/plan").json()

    assert summary["error"]["type"] == "token_budget_exceeded"
    assert [task["status"] for task in plan["tasks"][:2]] == [
        "completed",
        "completed",
    ]


def test_replan_appends_additive_version_balanced_auto(monkeypatch):
    """Scenario 9 (E16 amendment): insufficient evidence -> ONE replan,
    auto-approved in balanced mode because the delta is INTERNAL
    read-only; plan v2 created by agent."""
    sufficiency_calls = {"n": 0}

    def sufficiency_script(prompt: str) -> dict[str, Any]:
        sufficiency_calls["n"] += 1
        if sufficiency_calls["n"] == 1:
            return {"coverage": "uncovered", "missing": ["Quellen fehlen"]}
        return {"coverage": "covered", "missing": []}

    delta = replan_delta(
        {
            "id": "t2",
            "title": "Bestand nachpruefen",
            "tool_kind": "rag_query",
            "queries": ["Marktlage Details intern"],
        }
    )
    llm = ScriptedLLM(
        overrides={
            "SufficiencyJudgement": sufficiency_script,
            "ReplanDeltaModel": delta,
        }
    )
    client = make_agent_client(
        monkeypatch, llm=llm, search=NoEvidenceSearch()
    )
    with client:
        run_id = submit_agent(client)
        wait_status(client, run_id, {"waiting_for_approval"})
        approve_pending(client, run_id)
        wait_status(client, run_id, {"completed"})

        plan = client.get(f"/v1/runs/{run_id}/plan").json()
        assert plan["version"] == 2
        assert [v["version"] for v in plan["versions"]] == [2, 1]
        tasks_by_id = {task["task_id"]: task for task in plan["tasks"]}
        assert tasks_by_id["t1"]["status"] == "insufficient_evidence"
        assert tasks_by_id["t2"]["status"] != "pending"
        types = event_types(client, run_id)
        assert "inqtrix.agent.plan.revised" in types
        # Balanced auto-approval: exactly ONE HUMAN waiting interval
        # (children parks are store-internal and don't count).
        assert human_wait_statuses(client, run_id) == [
            "waiting_for_approval"
        ]


def test_replan_repairs_changed_completed_task_id_instead_of_skipping(
    monkeypatch,
):
    """Changed work under a completed id enters the planner repair loop."""
    sufficiency_calls = {"n": 0}

    def sufficiency_script(prompt: str) -> dict[str, Any]:
        sufficiency_calls["n"] += 1
        if sufficiency_calls["n"] == 1:
            return {"coverage": "uncovered", "missing": ["Detail fehlt"]}
        return {"coverage": "covered", "missing": []}

    changed_id = replan_delta(
        {
            **PLAN["tasks"][0],
            "queries": ["Eine inhaltlich geaenderte Evidenzfrage?"],
        },
        summary="Ungueltiger Replan mit wiederverwendeter ID.",
    )
    repaired = replan_delta(
        {
            "id": "t2",
            "title": "Zusaetzliche Evidenzfrage",
            "tool_kind": "web_instant",
            "gap_ids": ["g1"],
            "queries": ["Welche Quelle schliesst die verbleibende Luecke?"],
        },
        summary="Additiver Replan mit neuer Task-ID.",
    )
    deltas = iter([changed_id, repaired])
    client = make_agent_client(
        monkeypatch,
        llm=ScriptedLLM(
            overrides={
                "ReplanDeltaModel": lambda prompt: next(deltas),
                "SufficiencyJudgement": sufficiency_script,
            }
        ),
        search=NoEvidenceSearch(),
    )
    with client:
        run_id = submit_agent(client, autonomy="autonomous")
        wait_status(client, run_id, {"completed"})
        plan = client.get(f"/v1/runs/{run_id}/plan").json()

    tasks = {task["task_id"]: task for task in plan["tasks"]}
    assert plan["version"] == 2
    assert set(tasks) == {"t1", "t2", "s"}
    assert tasks["t1"]["status"] == "insufficient_evidence"
    assert tasks["t2"]["status"] == "insufficient_evidence"
    assert tasks["t1"]["queries"] == PLAN["tasks"][0]["queries"]
    assert client.llm.calls.count("structured:ExecutionPlanModel") == 1
    assert client.llm.calls.count("structured:ReplanDeltaModel") == 2


def test_empty_replan_delta_proceeds_directly_to_synthesis(monkeypatch):
    """A model-confirmed no-op cannot trigger repair or another execute loop."""
    llm = ScriptedLLM(
        overrides={
            "SufficiencyJudgement": {
                "coverage": "uncovered",
                "missing": ["No additional source is justified"],
            },
            "ReplanDeltaModel": replan_delta(
                summary="Proceed with the visible evidence gap."
            ),
        }
    )
    client = make_agent_client(
        monkeypatch,
        llm=llm,
        search=NoEvidenceSearch(),
    )

    with client:
        run_id = submit_agent(client, autonomy="autonomous")
        wait_status(client, run_id, {"completed"})
        plan = client.get(f"/v1/runs/{run_id}/plan").json()

    assert client.llm.calls.count("structured:ReplanDeltaModel") == 1
    assert plan["version"] == 2


def test_replan_with_web_delta_regates_in_balanced(monkeypatch):
    """E16 amendment (plan M1 S7): a replan that ADDS web queries parks
    for approval again in Standard/balanced — the approved plan is the
    web-search consent, unseen queries never run."""
    sufficiency_calls = {"n": 0}

    def sufficiency_script(prompt: str) -> dict[str, Any]:
        sufficiency_calls["n"] += 1
        if sufficiency_calls["n"] == 1:
            return {"coverage": "uncovered", "missing": ["Quellen fehlen"]}
        return {"coverage": "covered", "missing": []}

    delta = replan_delta(
        {
            "id": "t2",
            "title": "Nachrecherche",
            "tool_kind": "web_instant",
            "queries": ["Welche Quellen ergaenzen die Marktlage im Detail?"],
        }
    )
    llm = ScriptedLLM(
        overrides={
            "SufficiencyJudgement": sufficiency_script,
            "ReplanDeltaModel": delta,
        }
    )
    client = make_agent_client(
        monkeypatch, llm=llm, search=NoEvidenceSearch()
    )
    with client:
        run_id = submit_agent(client)
        wait_status(client, run_id, {"waiting_for_approval"})
        approve_pending(client, run_id)
        # The web replan parks AGAIN (no auto-approve for new queries).
        wait_status(client, run_id, {"waiting_for_approval"})
        approve_pending(client, run_id)
        wait_status(client, run_id, {"completed"})
        assert human_wait_statuses(client, run_id) == [
            "waiting_for_approval",
            "waiting_for_approval",
        ]


def test_critic_research_routes_to_additive_replan(monkeypatch):
    critic_calls = {"n": 0}

    def critic_script(prompt: str) -> dict[str, Any]:
        critic_calls["n"] += 1
        if critic_calls["n"] == 1:
            return {
                "findings": [
                    {
                        "kind": "criterion_unmet",
                        "detail": "Aktuelle Gegenposition fehlt.",
                        "suggested_fix": "Weitere Quelle recherchieren.",
                    }
                ],
                "criteria_covered": [],
                "criteria_uncovered": ["Gegenposition belegt."],
                "verdict": "research",
            }
        return dict(CRITIC_PASS)

    delta = replan_delta(
        {
            "id": "t2",
            "title": "Gegenposition pruefen",
            "tool_kind": "web_instant",
            "queries": [
                "Welche belastbaren Gegenbelege widersprechen der Marktlage 2026?"
            ],
            "is_falsification": True,
        }
    )
    llm = ScriptedLLM(
        overrides={
            "ReplanDeltaModel": delta,
            "AgentCriticReport": critic_script,
        }
    )
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        run_id = submit_agent(client, autonomy="autonomous")
        wait_status(client, run_id, {"completed"})

        plan = client.get(f"/v1/runs/{run_id}/plan").json()
        assert plan["version"] == 2
        assert "t2" in [task["task_id"] for task in plan["tasks"]]
        assert client.llm.calls.count("structured:ExecutionPlanModel") == 1
        assert client.llm.calls.count("structured:ReplanDeltaModel") == 1
        assert critic_calls["n"] == 2
        children = client.get(f"/v1/runs/{run_id}/children").json()["data"]
        assert children == []
        types = event_types(client, run_id)
        assert "inqtrix.agent.plan.revised" in types
        assert "inqtrix.agent.activity" in types


def test_replan_round_budget_is_shared_by_gate_and_critic(monkeypatch):
    """The deterministic gate (insufficient evidence) and the critic
    ('research') draw from ONE shared ``replan_rounds`` budget: a gate
    replan plus a critic-research replan consume both of
    ``max_replan_rounds=2``, so a persistently research-demanding critic is
    exhausted after just a single research replan (no third plan version).

    Pins the current accounting — were the two increment sites to use
    separate budgets, the critic would win a second research replan and the
    plan would reach version 4 with four planner calls.
    """
    sufficiency = iter([{"coverage": "uncovered", "missing": ["Marktlage"]}])

    def sufficiency_script(prompt: str) -> dict[str, Any]:
        return next(sufficiency, dict(SUFFICIENCY))

    critic_calls = {"n": 0}

    def critic_script(prompt: str) -> dict[str, Any]:
        critic_calls["n"] += 1
        return {
            "findings": [],
            "criteria_covered": [],
            "criteria_uncovered": ["Gegenposition belegt."],
            "verdict": "research",
        }

    def additive(*extra_ids: str) -> dict[str, Any]:
        return replan_delta(
            *[
            {
                "id": extra_id,
                "title": f"Zusatz {extra_id}",
                "tool_kind": "web_instant",
                "queries": [f"Zusatzfrage {extra_id}"],
            }
            for extra_id in extra_ids
            ]
        )

    deltas = iter([additive("t2"), additive("t3")])
    llm = ScriptedLLM(
        overrides={
            "ReplanDeltaModel": lambda prompt: next(deltas),
            "SufficiencyJudgement": sufficiency_script,
            "AgentCriticReport": critic_script,
        }
    )
    client = make_agent_client(
        monkeypatch, llm=llm, search=NoEvidenceSearch()
    )
    with client:
        run_id = submit_agent(client, autonomy="autonomous")
        wait_status(client, run_id, {"completed"})
        plan = client.get(f"/v1/runs/{run_id}/plan").json()
        assert plan["version"] == 3
        assert client.llm.calls.count("structured:ExecutionPlanModel") == 1
        assert client.llm.calls.count("structured:ReplanDeltaModel") == 2
        assert critic_calls["n"] == 2


def test_no_web_plan_when_search_unavailable(monkeypatch):
    """Scenario 8: without web the planner is FORCED off web tasks."""
    rag_plan = {
        "summary_markdown": "Nur intern.",
        "tasks": [
            {
                "id": "t1",
                "title": "Interner Bestand",
                "tool_kind": "rag_query",
                "queries": ["Marktanalyse Bestand"],
            },
            {
                "id": "s",
                "title": "Synthese",
                "tool_kind": "synthesis",
                "depends_on": ["t1"],
            },
        ],
    }

    plans = iter([dict(PLAN), rag_plan])

    def plan_script(prompt: str) -> dict[str, Any]:
        return next(plans)

    llm = ScriptedLLM(overrides={"ExecutionPlanModel": plan_script})
    monkeypatch.setattr(
        web_research_module, "run_web_graph", fake_child_graph
    )
    app = FastAPI()
    router = create_router()
    settings = Settings(
        models=ModelSettings(),
        agent=AgentSettings(),
        server=ServerSettings(),
        storage=StorageSettings(backend="memory", database_url=""),
    )
    settings.agent_platform = AgentPlatformSettings(
        INQTRIX_AGENT_ALLOW_VOLATILE=True
    )

    class NoSearch:
        def is_available(self) -> bool:
            return False

        def search(self, *a: Any, **k: Any) -> GroundedSearchResult:
            raise AssertionError("web must not be called")

    container = register_routes(
        router,
        providers=SimpleNamespace(llm=llm, search=None),
        strategies=SimpleNamespace(),
        settings=settings,
        semaphore_factory=lambda: __import__("asyncio").Semaphore(2),
    )
    app.include_router(router)
    client = TestClient(app)
    client.llm = llm  # type: ignore[attr-defined]
    with client:
        run_id = submit_agent(client, autonomy="autonomous")
        # This harness has neither web nor a knowledge stack: the plan
        # is forced off web tasks (repair round), and with every
        # remaining task hard-failing the run ends LOUDLY instead of
        # synthesizing an evidence-free memo.
        summary = wait_status(client, run_id, {"failed"})
        assert "all_tasks_failed" in summary["error"]["message"]
        plan = client.get(f"/v1/runs/{run_id}/plan").json()
        kinds = [task["tool_kind"] for task in plan["tasks"]]
        assert "web_research" not in kinds and "web_instant" not in kinds
        synthesis_task = next(
            task for task in plan["tasks"] if task["tool_kind"] == "synthesis"
        )
        assert synthesis_task["status"] == "skipped"
        # The web plan was rejected via the repair round (2 attempts).
        assert llm.calls.count("structured:ExecutionPlanModel") == 2
        children = client.get(f"/v1/runs/{run_id}/children").json()["data"]
        assert children == []


def test_contradictions_flow_into_critic_facts(monkeypatch):
    """Scenario 3: overlapping claims -> contradiction analysis -> the
    critic receives the contradiction count as a precomputed fact."""
    def child_graph(question: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "answer": f"Kindbericht: {question}",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "result_state": {
                "answer": "x",
                "round": 1,
                "report_references": [
                    {"label": "E1", "url": "https://example.com/a", "tier": "unknown"}
                ],
                "consolidated_claims": [
                    {
                        "claim_text": "Der Markt waechst stark",
                        "status": "verified",
                        "claim_type": "fact",
                        "needs_primary": False,
                        "status_reason": "",
                        "support_count": 2,
                        "contradict_count": 0,
                        "source_tier_counts": {},
                        "sources": ["https://example.com/a"],
                    },
                    {
                        "claim_text": "Der Markt waechst stark",
                        "status": "contested",
                        "claim_type": "fact",
                        "needs_primary": False,
                        "status_reason": "",
                        "support_count": 1,
                        "contradict_count": 1,
                        "source_tier_counts": {},
                        "sources": ["https://example.com/b"],
                    },
                ],
            },
        }

    contradiction = {
        "contradictions": [
            {
                "internal_position": "Der Markt waechst stark",
                "external_position": "Der Markt schrumpft",
                "severity": "hard",
                "likely_cause": "Unterschiedliche Zeitraeume",
            }
        ]
    }
    llm = ScriptedLLM(
        overrides={
            "ContradictionReport": contradiction,
            "ExecutionPlanModel": RESEARCH_PLAN,
        }
    )

    class LexicalConsolidator:
        """Only the signature surface the contradiction prefilter uses."""

        def claim_signature(self, text: str) -> str:
            return " ".join(text.lower().split())

    client = make_agent_client(
        monkeypatch,
        llm=llm,
        strategies=SimpleNamespace(claim_consolidation=LexicalConsolidator()),
    )
    monkeypatch.setattr(web_research_module, "run_web_graph", child_graph)
    with client:
        run_id = submit_agent(
            client,
            autonomy="autonomous",
            tool_directives=["web_research"],
        )
        wait_status(client, run_id, {"completed"})
        result = client.get(f"/v1/runs/{run_id}/result").json()
        assert result  # completed with a payload
        # Contradiction analysis ran and its result reached result_state.
        assert "structured:ContradictionReport" in client.llm.calls
        assert client.llm.calls.index(
            "structured:ContradictionReport"
        ) < client.llm.calls.index("structured:SufficiencyJudgement")


def test_critic_revise_buys_exactly_one_revision(monkeypatch):
    critic_calls = {"n": 0}

    def critic_script(prompt: str) -> dict[str, Any]:
        critic_calls["n"] += 1
        return {
            "findings": [
                {
                    "kind": "uncited_claim",
                    "detail": "Absatz 2 ohne Beleg",
                    "suggested_fix": "Label ergaenzen",
                }
            ],
            "criteria_covered": [],
            "criteria_uncovered": [],
            "verdict": "revise",
        }

    llm = ScriptedLLM(overrides={"AgentCriticReport": critic_script})
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        run_id = submit_agent(client, autonomy="autonomous")
        wait_status(client, run_id, {"completed"})
        # verdict stays revise, but only ONE revision attempt was made,
        # followed by the required second critic check.
        # section calls = 2 outline sections + 1 revision.
        assert client.llm.calls.count("structured:SectionText") == 3
        assert critic_calls["n"] == 2


def test_edit_decision_reloads_user_plan_before_execution(monkeypatch):
    """The edited plan (user version) is what executes (rule R5)."""
    client = make_agent_client(monkeypatch)
    with client:
        run_id = submit_agent(client)
        wait_status(client, run_id, {"waiting_for_approval"})
        approvals = client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
        edited_plan = {
            "summary_markdown": "Vom Nutzer verschlankt.",
            "tasks": [
                {
                    "id": "u1",
                    "title": "Nur Instant-Suche",
                    "tool_kind": "web_instant",
                    "queries": ["Marktvolumen"],
                },
                {
                    "id": "s",
                    "title": "Synthese",
                    "tool_kind": "synthesis",
                    "depends_on": ["u1"],
                },
            ],
        }
        decided = client.post(
            f"/v1/runs/{run_id}/approvals/{approvals[0]['approval_id']}",
            json={"decision": "edit", "plan": edited_plan},
        )
        assert decided.status_code == 200, decided.text
        wait_status(client, run_id, {"completed"})
        plan = client.get(f"/v1/runs/{run_id}/plan").json()
        assert plan["created_by"] == "user"
        assert [task["task_id"] for task in plan["tasks"]] == ["u1", "s"]
        # The USER plan executed: instant task only, no children.
        children = client.get(f"/v1/runs/{run_id}/children").json()["data"]
        assert children == []


def test_user_research_edit_consent_survives_critic_replan(monkeypatch):
    """A later agent-authored version cannot erase explicit user consent."""
    critic_calls = {"count": 0}

    def critic_script(_prompt: str) -> dict[str, Any]:
        critic_calls["count"] += 1
        if critic_calls["count"] == 1:
            return {
                "findings": [
                    {
                        "kind": "criterion_unmet",
                        "detail": "Eine Gegenposition fehlt.",
                        "suggested_fix": "Weitere Quellen vergleichen.",
                    }
                ],
                "criteria_covered": [],
                "criteria_uncovered": ["Gegenposition belegt."],
                "verdict": "research",
            }
        return dict(CRITIC_PASS)

    delta = replan_delta(
        {
            "id": "t2",
            "title": "Gegenposition vertiefen",
            "tool_kind": "web_research",
            "queries": [
                "Welche belastbaren Gegenbelege widersprechen der "
                "Markteinschaetzung 2026?"
            ],
            "params": {"profile": "compact", "recency": "year"},
        },
        summary="Zweiter kompakter Rechercheauftrag.",
    )
    llm = ScriptedLLM(
        overrides={
            "ReplanDeltaModel": delta,
            "AgentCriticReport": critic_script,
        }
    )
    client = make_agent_client(monkeypatch, llm=llm)

    with client:
        run_id = submit_agent(client)
        wait_status(client, run_id, {"waiting_for_approval"})
        approval = client.get(
            f"/v1/runs/{run_id}/approvals"
        ).json()["data"][0]
        edited = client.post(
            f"/v1/runs/{run_id}/approvals/{approval['approval_id']}",
            json={"decision": "edit", "plan": RESEARCH_PLAN},
        )
        assert edited.status_code == 200, edited.text

        # The critic's agent-authored research replan still re-gates its new
        # web question in balanced mode, but policy validation must retain the
        # user's earlier compact-research consent.
        wait_status(client, run_id, {"waiting_for_approval"})
        approve_pending(client, run_id)
        wait_status(client, run_id, {"completed"})

        plan = client.get(f"/v1/runs/{run_id}/plan").json()
        assert plan["version"] == 3
        assert plan["created_by"] == "agent"
        children = client.get(f"/v1/runs/{run_id}/children").json()["data"]
        assert len(children) == 2
        assert {
            child["agent_overrides"]["report_profile"]
            for child in children
        } == {"compact"}
        assert client.llm.calls.count("structured:ExecutionPlanModel") == 1
        assert client.llm.calls.count("structured:ReplanDeltaModel") == 1
