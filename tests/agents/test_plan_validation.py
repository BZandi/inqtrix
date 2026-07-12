"""Deterministic plan-validation rules (M4; reused by the M5 planner)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inqtrix.agents.plan_models import ExecutionPlanModel
from inqtrix.agents.plan_validation import validate_plan


def _plan(tasks: list[dict]) -> ExecutionPlanModel:
    return ExecutionPlanModel.model_validate(
        {"summary_markdown": "Testplan", "tasks": tasks}
    )


def _task(task_id: str, **overrides) -> dict:
    base = {
        "id": task_id,
        "title": f"Aufgabe {task_id}",
        "tool_kind": "rag_query",
        # Retrieval kinds require concrete queries since the plan
        # transparency change; tests override to [] to probe the rule.
        "queries": [f"Frage zu {task_id}"],
    }
    base.update(overrides)
    return base


def _valid_tasks() -> list[dict]:
    return [
        _task("t1", tool_kind="web_research"),
        _task("t2", tool_kind="rag_query"),
        _task("s", tool_kind="synthesis", depends_on=["t1", "t2"]),
    ]


def test_valid_plan_has_no_errors() -> None:
    assert validate_plan(_plan(_valid_tasks())) == []


def test_exactly_one_synthesis_required() -> None:
    errors = validate_plan(
        _plan([_task("t1"), _task("t2", tool_kind="web_research")])
    )
    assert any("genau einen synthesis-Task" in error for error in errors)


def test_synthesis_must_depend_on_every_other_task() -> None:
    tasks = _valid_tasks()
    tasks[2]["depends_on"] = ["t1"]
    errors = validate_plan(_plan(tasks))
    assert any("fehlend: t2" in error for error in errors)


def test_duplicate_task_ids_reported() -> None:
    tasks = [
        _task("t1"),
        _task("t1", tool_kind="web_research"),
        _task("s", tool_kind="synthesis", depends_on=["t1"]),
    ]
    errors = validate_plan(_plan(tasks))
    assert any("Doppelte Task-IDs: t1" in error for error in errors)


def test_unknown_dependency_and_self_dependency() -> None:
    tasks = [
        _task("t1", depends_on=["ghost"]),
        _task("t2", depends_on=["t2"]),
        _task("s", tool_kind="synthesis", depends_on=["t1", "t2"]),
    ]
    errors = validate_plan(_plan(tasks))
    assert any("unbekannte Task-ID 'ghost'" in error for error in errors)
    assert any("haengt von sich selbst ab" in error for error in errors)


def test_dependency_cycle_detected() -> None:
    tasks = [
        _task("a", depends_on=["b"]),
        _task("b", depends_on=["a"]),
        _task("s", tool_kind="synthesis", depends_on=["a", "b"]),
    ]
    errors = validate_plan(_plan(tasks))
    assert any("Zyklische Abhaengigkeiten" in error for error in errors)
    assert any("a, b" in error for error in errors)


def test_profile_domain_is_tool_specific() -> None:
    tasks = [
        _task("t1", tool_kind="web_research", params={"profile": "tief"}),
        _task("t2", tool_kind="rag_query", params={"profile": "deep"}),
        _task("t3", tool_kind="file_analysis", params={"profile": "schnell"}),
        _task("s", tool_kind="synthesis", depends_on=["t1", "t2", "t3"]),
    ]
    errors = validate_plan(_plan(tasks))
    assert any("t1: unbekanntes Profil 'tief'" in error for error in errors)
    assert any("t2: unbekanntes Profil 'deep'" in error for error in errors)
    assert any("t3: file_analysis kennt kein Profil" in error for error in errors)


def test_task_budgets_are_server_managed() -> None:
    tasks = [
        _task("t1", budget={"max_tokens": 6000}),
        _task("t2", budget={"max_tokens": 6000}),
        _task("s", tool_kind="synthesis", depends_on=["t1", "t2"]),
    ]
    errors = validate_plan(_plan(tasks))
    assert sum("serverseitig verwaltet" in error for error in errors) == 2


def test_web_instant_requires_exactly_one_query() -> None:
    tasks = [
        _task(
            "t1",
            tool_kind="web_instant",
            queries=["Frage eins?", "Frage zwei?"],
        ),
        _task("s", tool_kind="synthesis", depends_on=["t1"]),
    ]
    errors = validate_plan(_plan(tasks))
    assert any("web_instant braucht genau eine" in error for error in errors)


def test_normal_plan_blocks_research_and_explicit_profile_is_fixed() -> None:
    plan = _plan(_valid_tasks())
    normal_errors = validate_plan(plan, web_research_allowed=False)
    assert any(
        "in dieser Stufe" in error and "web_instant" in error
        for error in normal_errors
    )
    explicit_errors = validate_plan(
        plan,
        web_research_allowed=True,
        web_research_profile="compact",
    )
    assert any("profile=compact" in error for error in explicit_errors)


def test_recency_uses_the_capability_vocabulary() -> None:
    plan = _plan([_task("t1", params={"recency": ""})])
    assert plan.tasks[0].params.recency is None
    with pytest.raises(ValidationError):
        _plan([_task("t1", params={"recency": "365d"})])


def test_gap_ids_checked_only_when_universe_given() -> None:
    tasks = [
        _task("t1", gap_ids=["g1", "ghost"]),
        _task("s", tool_kind="synthesis", depends_on=["t1"]),
    ]
    assert validate_plan(_plan(tasks)) == []
    errors = validate_plan(_plan(tasks), known_gap_ids={"g1"})
    assert any("unbekannte Gap-ID 'ghost'" in error for error in errors)


def test_max_tasks_ceiling() -> None:
    tasks = [_task(f"t{i}") for i in range(8)] + [
        _task("s", tool_kind="synthesis", depends_on=[f"t{i}" for i in range(8)])
    ]
    errors = validate_plan(_plan(tasks), max_tasks=8)
    assert any("Zu viele Tasks (9" in error for error in errors)


def test_retrieval_tasks_require_concrete_queries() -> None:
    """web/rag tasks without literal query strings are invalid; the
    exempt kinds (synthesis by design, file_analysis via its
    objective/title executor fallback) stay valid without queries."""
    tasks = [
        _task("t1", tool_kind="web_research", queries=[]),
        _task("t2", tool_kind="web_instant", queries=["   "]),
        _task("t3", tool_kind="rag_query", queries=[]),
        _task("t4", tool_kind="file_analysis", queries=[]),
        _task(
            "s",
            tool_kind="synthesis",
            queries=[],
            depends_on=["t1", "t2", "t3", "t4"],
        ),
    ]
    errors = validate_plan(_plan(tasks))
    assert any(
        "t1: web_research braucht mindestens" in error for error in errors
    )
    assert any(
        "t2: web_instant braucht mindestens" in error for error in errors
    )
    assert any(
        "t3: rag_query braucht mindestens" in error for error in errors
    )
    assert not any("t4" in error for error in errors)
    assert not any("s:" in error for error in errors)


def test_allowed_collection_ids_gate() -> None:
    """Explicit collection references outside the caller-visible set are
    violations; ``None`` skips the check (no knowledge service), an
    empty set rejects every explicit reference."""
    tasks = [
        _task("t1", params={"collection_ids": ["kc_1", "kc_ghost"]}),
        _task("s", tool_kind="synthesis", depends_on=["t1"]),
    ]
    assert validate_plan(_plan(tasks)) == []
    errors = validate_plan(
        _plan(tasks), allowed_collection_ids={"kc_1"}
    )
    assert errors == [
        "Task t1: Sammlung 'kc_ghost' ist nicht sichtbar oder unbekannt."
    ]
    errors = validate_plan(_plan(tasks), allowed_collection_ids=set())
    assert any("'kc_1'" in error for error in errors)
    assert any("'kc_ghost'" in error for error in errors)


def test_schema_rejects_unknown_fields_and_empty_plans() -> None:
    with pytest.raises(ValidationError):
        ExecutionPlanModel.model_validate({"tasks": []})
    with pytest.raises(ValidationError):
        ExecutionPlanModel.model_validate(
            {"tasks": [_task("t1")], "surprise": True}
        )
    with pytest.raises(ValidationError):
        ExecutionPlanModel.model_validate(
            {"tasks": [_task("t1", params={"typo_knob": 1})]}
        )


def test_planner_schema_omits_legacy_task_budget() -> None:
    schema = ExecutionPlanModel.model_json_schema()
    task_properties = schema["$defs"]["PlanTaskModel"]["properties"]
    assert "budget" not in task_properties
