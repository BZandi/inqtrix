"""Contracts of the Stufen ladder (tier_policy + its consumers).

What these tests pin: the policy table drives web budgets and plan
validation deterministically (published == enforced), the legacy
``tier=None`` path stays byte-identical, and the override seam bridges
tier -> depth without ever touching model selection.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from inqtrix.agents.plan_models import (
    ExecutionPlanModel,
    WEB_RESEARCH_PROFILES,
)
from inqtrix.agents.plan_validation import validate_plan
from inqtrix.agents.tier_policy import (
    AGENT_TIERS,
    TIER_POLICIES,
    resolve_tier_policy,
    tier_capabilities_payload,
)
from inqtrix.agents.web_execution_policy import derive_web_research_policy
from inqtrix.services.overrides import (
    apply_overrides,
    parse_overrides_payload,
)
from inqtrix.settings import AgentSettings


def _plan(profile: str | None) -> ExecutionPlanModel:
    return ExecutionPlanModel.model_validate(
        {
            "summary_markdown": "Plan",
            "tasks": [
                {
                    "id": "t1",
                    "title": "Recherche",
                    "tool_kind": "web_research",
                    "queries": ["Frage eins"],
                    **(
                        {"params": {"profile": profile}}
                        if profile is not None
                        else {}
                    ),
                },
                {
                    "id": "s",
                    "title": "Synthese",
                    "tool_kind": "synthesis",
                    "depends_on": ["t1"],
                },
            ],
        }
    )


def test_web_policy_per_tier_matches_the_table():
    schnell = derive_web_research_policy(depth="normal", tier="schnell")
    assert schnell.allowed is False
    assert schnell.profile is None

    gruendlich = derive_web_research_policy(depth="normal", tier="gruendlich")
    assert gruendlich.allowed is True
    assert gruendlich.profile == "schnell"
    assert gruendlich.max_profile == "compact"

    tief = derive_web_research_policy(depth="deep", tier="tief")
    assert tief.allowed is True
    assert tief.profile == "compact"
    assert tief.max_profile == "deep"


def test_web_policy_without_tier_is_byte_identical_to_legacy():
    assert derive_web_research_policy(depth="deep") == (
        derive_web_research_policy(depth="deep", tier=None)
    )
    legacy = derive_web_research_policy(depth="normal", admitted_directive=True)
    assert legacy.allowed is True
    assert legacy.profile == "compact"
    assert legacy.max_profile is None
    denied = derive_web_research_policy(depth="normal")
    assert denied.allowed is False


def test_validate_plan_enforces_the_tier_ceiling():
    over = validate_plan(
        _plan("deep"),
        web_research_allowed=True,
        web_research_profile_ceiling="compact",
    )
    assert any("uebersteigt die erlaubte Suchtiefe" in e for e in over)
    within = validate_plan(
        _plan("schnell"),
        web_research_allowed=True,
        web_research_profile_ceiling="compact",
    )
    assert within == []
    # Missing profile is fine — the server default applies at execution.
    defaulted = validate_plan(
        _plan(None),
        web_research_allowed=True,
        web_research_profile_ceiling="compact",
    )
    assert defaulted == []


def test_validate_plan_rejects_research_when_the_tier_forbids_it():
    errors = validate_plan(_plan("schnell"), web_research_allowed=False)
    assert any("web_instant" in e for e in errors)


def test_overrides_bridge_tier_into_depth_and_reject_contradictions():
    base = AgentSettings()
    tief = apply_overrides(
        base, parse_overrides_payload({"agent_tier": "tief"})
    )
    assert tief.agent_tier == "tief"
    assert tief.depth == "deep"
    schnell = apply_overrides(
        base, parse_overrides_payload({"agent_tier": "schnell"})
    )
    assert schnell.depth == "normal"
    # The CONSISTENT pair replays fine (worker resume body).
    replay = parse_overrides_payload(
        {"agent_tier": "tief", "depth": "deep"}
    )
    assert replay is not None and replay.agent_tier == "tief"
    with pytest.raises(HTTPException) as denied:
        parse_overrides_payload({"agent_tier": "tief", "depth": "normal"})
    assert denied.value.status_code == 400
    with pytest.raises(HTTPException):
        parse_overrides_payload({"agent_tier": "extrem"})


def test_capabilities_payload_mirrors_the_policy_table():
    payload = tier_capabilities_payload()
    assert [entry["id"] for entry in payload] == list(AGENT_TIERS)
    for entry in payload:
        policy = TIER_POLICIES[entry["id"]]  # type: ignore[index]
        assert entry["latency_hint"] == policy.latency_hint
        assert entry["web_child_ceiling"] == policy.web_child_ceiling
        if entry["web_child_profile"] is not None:
            assert entry["web_child_profile"] in WEB_RESEARCH_PROFILES


def test_unknown_tier_fails_loudly():
    with pytest.raises(ValueError):
        resolve_tier_policy("extrem")


def test_schnell_instant_budget_is_validator_enforced():
    """The published 'exactly one web_instant' budget is never
    prompt-only: a second instant task is a repairable violation."""
    from inqtrix.agents.plan_models import ExecutionPlanModel
    from inqtrix.agents.plan_validation import validate_plan
    from inqtrix.agents.web_execution_policy import derive_web_research_policy

    policy = derive_web_research_policy(depth="normal", tier="schnell")
    assert policy.max_instant_tasks == 1
    plan = ExecutionPlanModel.model_validate({
        "summary_markdown": "s",
        "assumptions": [],
        "success_criteria": [],
        "tasks": [
            {
                "id": "t1",
                "title": "Frage eins beantworten",
                "tool_kind": "web_instant",
                "objective": "",
                "queries": ["Frage eins"],
                "gap_ids": [],
                "depends_on": [],
                "budget": {},
                "params": {},
                "expected_output": "",
                "is_falsification": False,
            },
            {
                "id": "t2",
                "title": "Frage zwei beantworten",
                "tool_kind": "web_instant",
                "objective": "",
                "queries": ["Frage zwei"],
                "gap_ids": [],
                "depends_on": [],
                "budget": {},
                "params": {},
                "expected_output": "",
                "is_falsification": False,
            },
            {
                "id": "s",
                "title": "Memo",
                "tool_kind": "synthesis",
                "objective": "",
                "queries": [],
                "gap_ids": [],
                "depends_on": ["t1", "t2"],
                "budget": {},
                "params": {},
                "expected_output": "",
                "is_falsification": False,
            },
        ],
    })
    errors = validate_plan(
        plan,
        web_research_allowed=False,
        max_web_instant_tasks=policy.max_instant_tasks,
    )
    assert any("hoechstens 1 web_instant" in error for error in errors)
    assert any("t1, t2" in error for error in errors)
