"""Skill runtime in the phase machine (plan M3 `3.4`/`3.5`, M3a).

Trajectories over the full platform path: the point check answers from
context or queues a structured question through the M1 gate, answers
substitute into the ``{{name}}`` slots of the planner/answer prompts,
``requires_plan`` overrides the plan gate (strictest wins), and the
runs router rejects invisible skills and unknown directives loudly.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tests.agents.test_workspace_agent import (
    ScriptedLLM,
    make_agent_client,
    wait_status,
)

SKILL_PAYLOAD = {
    "label": "sprechzettel",
    "title": "Sprechzettel",
    "description": "Kompakter Sprechzettel.",
    "when_to_use": "Fuer Termine.",
    "instructions_markdown": (
        "Erstelle einen Sprechzettel fuer {{anlass}} mit Blick auf "
        "{{publikum}}."
    ),
    "clarification_points": [
        {
            "name": "anlass",
            "question": "Fuer welchen Anlass?",
            "options": [
                {"label": "Vorstandssitzung"},
                {"label": "Kundentermin"},
            ],
            "required": True,
            "default_assumption": "Interner Termin",
        },
        {
            "name": "publikum",
            "question": "Wer ist das Publikum?",
            "options": [],
            "required": False,
            "default_assumption": "Fachpublikum",
        },
    ],
    "deliverable": "",
    "requires_plan": "never",
    "invocation": "model_allowed",
}


def _create_skill(client, **overrides) -> str:
    payload = {**SKILL_PAYLOAD, **overrides}
    service = client.container.skill_service
    record = asyncio.run(
        service.create(payload, tenant_id="default", owner_sub=None)
    )
    return record.id


def _point_check(points: list[dict[str, Any]]):
    return {"points": points}


def _submit(client, *, skill_ids=None, tool_directives=None, autonomy=None):
    body: dict[str, Any] = {
        "question": "Bereite den Termin naechste Woche vor.",
        "mode": "workspace_agent",
    }
    if skill_ids is not None:
        body["skill_ids"] = skill_ids
    if tool_directives is not None:
        body["tool_directives"] = tool_directives
    if autonomy:
        body["autonomy"] = autonomy
    return client.post("/v1/runs", json=body)


def test_missing_required_point_asks_then_substitutes(monkeypatch):
    """The full M3a loop: check -> structured ask -> substitution."""
    planner_prompts: list[str] = []

    def capture_plan(prompt: str) -> dict[str, Any]:
        planner_prompts.append(prompt)
        return {
            "summary_markdown": "Plan",
            "tasks": [
                {
                    "id": "t1",
                    "title": "Recherche",
                    "tool_kind": "web_research",
                    "queries": ["Termin-Unterlagen"],
                    "params": {"profile": "compact"},
                },
                {
                    "id": "s",
                    "title": "Synthese",
                    "tool_kind": "synthesis",
                    "depends_on": ["t1"],
                },
            ],
            "success_criteria": ["Sprechzettel steht."],
        }

    llm = ScriptedLLM(
        overrides={
            "SkillPointCheck": _point_check(
                [
                    {"id": "p1", "answered": False, "answer_from_context": ""},
                    {"id": "p2", "answered": False, "answer_from_context": ""},
                ]
            ),
            "ExecutionPlanModel": capture_plan,
        }
    )
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        skill_id = _create_skill(client)
        run_id = _submit(
            client, skill_ids=[skill_id], tool_directives=["web_research"]
        ).json()["run_id"]

        # The missing REQUIRED point parks the run on a structured round.
        wait_status(client, run_id, {"waiting_for_input"})
        rows = client.get(f"/v1/runs/{run_id}/clarifications").json()["data"]
        questions = rows[0]["questions"]
        assert [q["prompt"] for q in questions] == ["Fuer welchen Anlass?"]
        assert [o["label"] for o in questions[0]["options"]] == [
            "Vorstandssitzung",
            "Kundentermin",
        ]
        answered = client.post(
            f"/v1/runs/{run_id}/clarifications/{rows[0]['clarification_id']}",
            json={"answers": {"q1": {"option_ids": ["q1_o1"]}}},
        )
        assert answered.status_code == 200, answered.text

        # requires_plan=never: no plan gate even in balanced — straight
        # through to completion.
        wait_status(client, run_id, {"completed"})
        prompt = planner_prompts[0]
        # The answered point substituted into the {{anlass}} slot ...
        assert "Sprechzettel fuer Vorstandssitzung" in prompt
        # ... the optional one rides as a VISIBLE assumption ...
        assert "Annahme: Fachpublikum" in prompt
        # ... inside the delimited user-content block, with the
        # explicit tool directive named.
        assert "[Skill 'sprechzettel'" in prompt
        assert "AUSDRUECKLICH verlangt" in prompt


def test_context_answered_point_skips_the_gate(monkeypatch):
    llm = ScriptedLLM(
        overrides={
            "SkillPointCheck": _point_check(
                [
                    {
                        "id": "p1",
                        "answered": True,
                        "answer_from_context": "Kundentermin",
                    },
                    {"id": "p2", "answered": False, "answer_from_context": ""},
                ]
            ),
        }
    )
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        skill_id = _create_skill(client)
        run_id = _submit(client, skill_ids=[skill_id]).json()["run_id"]
        # No waiting_for_input: the required point came from context,
        # the optional one falls back to its visible assumption.
        summary = wait_status(client, run_id, {"completed"})
        assert summary["status"] == "completed"
        assert (
            client.get(f"/v1/runs/{run_id}/clarifications").json()["data"]
            == []
        )


def test_requires_plan_always_gates_even_autonomous(monkeypatch):
    llm = ScriptedLLM(
        overrides={"SkillPointCheck": _point_check([])}
    )
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        skill_id = _create_skill(
            client,
            label="pflichtplan",
            instructions_markdown="Arbeite streng nach Plan.",
            clarification_points=[],
            requires_plan="always",
        )
        run_id = _submit(
            client, skill_ids=[skill_id], autonomy="autonomous"
        ).json()["run_id"]
        summary = wait_status(
            client, run_id, {"waiting_for_approval", "completed"}
        )
        # The one way a skill reins in Auto (plan 3.5): the gate parks.
        assert summary["status"] == "waiting_for_approval"


def test_router_rejects_invisible_skill_and_unknown_directive(monkeypatch):
    client = make_agent_client(monkeypatch)
    with client:
        missing = _submit(client, skill_ids=["sk_missing"])
        assert missing.status_code == 404
        assert "sk_missing" in missing.json()["error"]["message"]

        unknown = _submit(client, tool_directives=["rm_rf"])
        assert unknown.status_code == 400

        too_many = _submit(
            client,
            skill_ids=[
                _create_skill(client, label=f"skill-{i}")
                for i in range(4)
            ],
        )
        assert too_many.status_code == 400
        assert "Hoechstens 3" in too_many.json()["error"]["message"]


def _record(**overrides):
    from inqtrix.content.skills import SkillRecord

    base: dict[str, Any] = {
        "id": "sk_unit",
        "tenant_id": "default",
        "owner_sub": None,
        "label": "unit",
        "title": "Unit",
        "description": "",
        "when_to_use": "",
        "instructions_markdown": "Nutze {{ ort }} und {{ort}}.",
        "clarification_points": [
            {
                "id": "p1",
                "name": "ort",
                "question": "Wo?",
                "options": [],
                "required": True,
                "default_assumption": "",
            },
            {
                "id": "p2",
                "name": "",
                "question": "Gibt es Sperrfristen?",
                "options": [],
                "required": True,
                "default_assumption": "",
            },
        ],
        "deliverable": "",
        "allowed_tools": [],
        "requires_plan": "auto",
        "invocation": "model_allowed",
        "argument_hint": "",
        "model_tier": "",
        "effort": "",
        "include_in_autocomplete": True,
        "created_at": 0.0,
        "updated_at": 0.0,
    }
    base.update(overrides)
    return SkillRecord(**base)


def test_substitution_covers_every_validated_slot_shape():
    """A marker that passed the coupling rule never survives unfilled
    (the extraction regex tolerates inner whitespace — substitution
    must match it exactly, review M3-M1)."""
    from inqtrix.agents.skills_runtime import substitute_placeholders

    filled = substitute_placeholders(
        "Nutze {{ ort }}, {{ort}} und {{  ort  }}.", {"ort": "Bonn"}
    )
    assert filled == "Nutze Bonn, Bonn und Bonn."


def test_free_point_answer_reaches_the_prompt_lines():
    """Answers to NAMELESS (context-only) points key on the point id —
    writer and reader share skill_point_key (review M3-K3: a divergent
    key silently dropped the user's answer)."""
    from inqtrix.agents.skills_runtime import (
        skill_input_lines,
        skill_point_key,
        unanswered_required_points,
    )

    skill = _record()
    free_point = skill.clarification_points[1]
    key = skill_point_key(free_point)
    assert key == "p2"
    lines = skill_input_lines(skill, {key: "Sperrfrist 1.8."})
    assert any("Sperrfrist 1.8." in line for line in lines)
    # An answered required point leaves the ask-the-user set.
    missing = unanswered_required_points(skill, {key: "Sperrfrist 1.8."})
    assert [point["id"] for point in missing] == ["p1"]


def test_skill_model_pins_strongest_wins():
    from inqtrix.agents.skills_runtime import skill_model_pins

    pins = skill_model_pins(
        [
            _record(model_tier="mid", effort="xhigh"),
            _record(id="sk_unit2", model_tier="high", effort="low"),
        ]
    )
    assert pins == ("high", "xhigh")
    assert skill_model_pins([_record()]) == ("", "")


def test_deep_mission_children_run_the_deep_wire_profile(monkeypatch):
    """depth=deep forces every phase-machine child research run onto the
    DEEP report profile — as the literal wire value 'deep' (review F1:
    'gruendlich' would 400 the child resolve and fail every task)."""
    from tests.agents.test_workspace_agent import (
        RESEARCH_PLAN,
        ScriptedLLM,
        approve_pending,
        make_agent_client,
        wait_status,
    )

    deep_plan = {
        **RESEARCH_PLAN,
        "tasks": [
            {
                **RESEARCH_PLAN["tasks"][0],
                "params": {"profile": "deep", "recency": "month"},
            },
            RESEARCH_PLAN["tasks"][1],
        ],
    }
    llm = ScriptedLLM(overrides={"ExecutionPlanModel": deep_plan})
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        response = client.post(
            "/v1/runs",
            json={
                "question": "Erstelle eine Marktanalyse.",
                "mode": "workspace_agent",
                "session_id": "sess-deep-mission",
                "agent_overrides": {"depth": "deep"},
            },
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]
        wait_status(client, run_id, {"waiting_for_approval"})
        approve_pending(client, run_id)
        wait_status(client, run_id, {"completed"})
        children = client.get(f"/v1/runs/{run_id}/children").json()["data"]
        assert len(children) == 1
        assert children[0]["agent_overrides"]["report_profile"] == "deep"
        assert children[0]["status"] == "completed"
