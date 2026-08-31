"""Skill runtime in the phase machine.

Trajectories over the full platform path: the point check answers from
context or queues a structured question through the clarification gate, answers
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
        service.create(payload, tenant_id="default", owner_user_id=None)
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
        "owner_user_id": None,
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


def test_skill_instructions_reach_the_critic(monkeypatch):
    """S1: a skill states the FORM the report must take, and the writing
    prompts already call it binding ("Form und Ton folgen ihnen"). The
    critic has to see that same instruction — otherwise a memo can
    ignore an attached skill and still pass, which would make a skill a
    suggestion while the plan-gate guidance field is a contract.
    """
    llm = ScriptedLLM(overrides={"SkillPointCheck": _point_check([])})
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        skill_id = _create_skill(
            client,
            label="kritikform",
            instructions_markdown="Gliedere als Sprechzettel fuer die GF.",
            clarification_points=[],
        )
        run_id = _submit(client, skill_ids=[skill_id]).json()["run_id"]
        wait_status(client, run_id, {"completed"})
        critic_prompts = [
            prompt
            for name, prompt in client.llm.prompts
            if name == "AgentCriticReport"
        ]
        assert critic_prompts, "critic never ran"
        assert any(
            "Ergebnisvorgabe" in prompt
            and "[Skill 'kritikform'" in prompt
            and "Sprechzettel fuer die GF" in prompt
            for prompt in critic_prompts
        )


def _create_rule(client, *, label: str, content: str) -> str:
    """One prompt-library rule, created the way the library route does."""
    service = client.container.prompt_template_service
    record = asyncio.run(
        service.create(
            {
                "title": label.title(),
                "label": label,
                "category": "instruction",
                "content_markdown": content,
                "visibility": {"agent": True, "chat": True, "editor": True},
            },
            tenant_id="default",
            owner_user_id=None,
        )
    )
    return record.id


def test_attached_library_rule_shapes_the_report(monkeypatch):
    """S5: the reusable half of the requirement.

    A rule the user attaches at the plan gate must reach the writing
    prompts with an origin marker — the whole point is not having to
    retype the same structure for every run, and the critic can only
    name a broken requirement it can tell apart from the others.
    """
    llm = ScriptedLLM(overrides={"SkillPointCheck": _point_check([])})
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        rule_id = _create_rule(
            client,
            label="sprechzettel",
            content="Gliedere in genau fuenf Punkte.",
        )
        run_id = _submit(client).json()["run_id"]
        wait_status(client, run_id, {"waiting_for_approval"})
        pending = [
            item
            for item in client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
            if item["status"] == "pending"
        ]
        decided = client.post(
            f"/v1/runs/{run_id}/approvals/{pending[0]['approval_id']}",
            json={
                "decision": "approve",
                "report_guidance": "Zielgruppe: Laien.",
                "report_rule_ids": [rule_id],
            },
        )
        assert decided.status_code == 200, decided.text
        wait_status(client, run_id, {"completed"})
        writing = [
            prompt
            for name, prompt in client.llm.prompts
            if name in ("ReportOutline", "SectionText", "AgentCriticReport")
        ]
        assert writing, "synthesis never ran"
        assert any(
            "[Regel: sprechzettel]" in prompt
            and "Gliedere in genau fuenf Punkte." in prompt
            and "[Freie Vorgabe]" in prompt
            and "Zielgruppe: Laien." in prompt
            for prompt in writing
        )
        # The decision keeps the parts, so the surface can show what the
        # user chose instead of the composed prompt text.
        approval = client.get(f"/v1/runs/{run_id}/approvals").json()["data"][0]
        requirement = approval["decision_payload"]["report_requirement"]
        assert requirement["free_text"] == "Zielgruppe: Laien."
        assert requirement["rules"][0]["label"] == "sprechzettel"
        assert requirement["rules"][0]["template_id"] == rule_id


def test_unknown_rule_id_is_refused_not_dropped(monkeypatch):
    """Approving under requirements that silently vanished would be the
    worst outcome: the user believes a rule is in force and it is not."""
    llm = ScriptedLLM(overrides={"SkillPointCheck": _point_check([])})
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        run_id = _submit(client).json()["run_id"]
        wait_status(client, run_id, {"waiting_for_approval"})
        pending = [
            item
            for item in client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
            if item["status"] == "pending"
        ]
        response = client.post(
            f"/v1/runs/{run_id}/approvals/{pending[0]['approval_id']}",
            json={"decision": "approve", "report_rule_ids": ["gibt-es-nicht"]},
        )
        assert response.status_code == 400
        assert "gibt-es-nicht" in response.text


def test_a_pinned_email_format_reaches_the_prompt():
    """S7: a skill author picks one of four deliverable values. Two of
    them named a surface and were routed; the other two named a FORM and
    changed nothing at all — folded to canvas without a word."""
    from inqtrix.agents.skills_runtime import build_skills_block

    def skill(deliverable: str):
        return _record(
            instructions_markdown="Fasse den Stand zusammen.",
            clarification_points=[],
            deliverable=deliverable,
        )

    email = build_skills_block([skill("email")])
    assert "Zielformat: E-Mail" in email
    points = build_skills_block([skill("talking_points")])
    assert "Zielformat: Sprechzettel" in points
    # A surface pin is enforced by routing; repeating it in the prompt
    # would be noise, so it stays out.
    assert "Zielformat" not in build_skills_block([skill("canvas")])
    assert "Zielformat" not in build_skills_block([skill("chat")])
    assert "Zielformat" not in build_skills_block([skill("")])


def test_a_section_writer_is_not_told_to_write_a_whole_email():
    """Review finding: the mission writes a memo section by section and
    handed EVERY section prompt the whole-deliverable format line. A
    four-section memo would have become four complete emails — four
    subject lines, four salutations, four closings — assembled under the
    memo's own headings."""
    from inqtrix.agents.skills_runtime import build_skills_block

    skill = _record(
        instructions_markdown="Fasse den Stand zusammen.",
        clarification_points=[],
        deliverable="email",
    )
    whole = build_skills_block([skill])
    part = build_skills_block([skill], scope="section")
    # The whole deliverable carries the envelope...
    assert "Betreffzeile" in whole
    assert "Gruss" in whole
    # ...one section carries the register and REFUSES the envelope.
    assert "KEINE eigene Betreffzeile" in part
    assert "E-Mail-Ton" in part


def test_every_pinnable_deliverable_has_a_defined_effect():
    """Published == enforced: a value the UI offers must change
    something. This pins the pair — a fifth value added to the picker
    without an effect fails here instead of quietly doing nothing."""
    from inqtrix.agents.skills_runtime import (
        SKILL_DELIVERABLE_FORMAT_LINES,
        SKILL_DELIVERABLE_SECTION_LINES,
    )
    from inqtrix.content.skills import SKILL_DELIVERABLES

    routed = {"", "chat", "canvas"}
    for value in SKILL_DELIVERABLES:
        assert value in routed or value in SKILL_DELIVERABLE_FORMAT_LINES, (
            f"deliverable {value!r} is selectable but has no effect"
        )
        # A form without a section-scoped twin would reappear once per
        # section in the assembled memo.
        assert value in routed or value in SKILL_DELIVERABLE_SECTION_LINES, (
            f"deliverable {value!r} has no section-scoped wording"
        )


def test_the_section_loop_really_uses_the_part_scoped_wording(monkeypatch):
    """The wiring, not just the wording: a pinned email must reach the
    per-section writer WITHOUT its envelope, while the outline (which
    decides the whole deliverable) still gets the full form."""
    llm = ScriptedLLM(overrides={"SkillPointCheck": _point_check([])})
    client = make_agent_client(monkeypatch, llm=llm)
    with client:
        skill_id = _create_skill(
            client,
            label="mailentwurf",
            instructions_markdown="Fasse den Stand zusammen.",
            clarification_points=[],
            deliverable="email",
            requires_plan="never",
        )
        run_id = _submit(
            client, skill_ids=[skill_id], autonomy="autonomous"
        ).json()["run_id"]
        wait_status(client, run_id, {"completed"})
        by_name: dict[str, list[str]] = {}
        for name, prompt in client.llm.prompts:
            by_name.setdefault(name, []).append(prompt)
        sections = by_name.get("SectionText") or []
        outlines = by_name.get("ReportOutline") or []
        assert sections, "no section was written — this proves nothing"
        assert outlines, "no outline was written — this proves nothing"
        for prompt in sections:
            assert "KEINE eigene Betreffzeile" in prompt
            assert "schreibe Betreffzeile, Anrede" not in prompt
        for prompt in outlines:
            assert "schreibe Betreffzeile, Anrede" in prompt
