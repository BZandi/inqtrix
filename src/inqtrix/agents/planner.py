"""Phase 3 — planning (§4): one high-tier structured call + repair.

The planner produces THE plan shape (:mod:`inqtrix.agents.plan_models`)
and is validated by THE validator (:mod:`inqtrix.agents.plan_validation`)
— the same pair the user-edit endpoint runs, so agent plans and user
edits can never drift apart. One repair retry with the full error list;
a second invalid plan fails the run loudly (``plan_invalid``).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from inqtrix.agents.control_ports import (
    PlanRecord,
    PlanTaskRecord,
    additive_replan_errors,
    carry_forward_terminal_task_results,
)
from inqtrix.agents.patterns._structured import structured_call
from inqtrix.agents.phase_models import AssignmentProfile, DiscoveryResult
from inqtrix.agents.plan_collections import (
    CollectionCatalogEntry,
    resolve_plan_collections,
)
from inqtrix.agents.plan_models import (
    ExecutionPlanModel,
    PlanTaskModel,
    ReplanDeltaModel,
)
from inqtrix.agents.plan_validation import validate_plan
from inqtrix.agents.web_execution_policy import derive_web_research_policy
from inqtrix.agents.prompts import (
    agent_planner_system_prompt,
    build_agent_planner_prompt,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from inqtrix.providers.base import LLMProvider


class PlanningFailed(RuntimeError):
    """Planner produced no valid plan after the repair round (loud).

    Attributes:
        errors: The final validation error list (German).
    """

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors) or "plan_invalid")
        self.errors = errors


def run_planner(
    llm: "LLMProvider",
    *,
    question: str,
    discovery: DiscoveryResult | None,
    profile: AssignmentProfile | None,
    max_tasks: int,
    web_allowed: bool,
    knowledge_allowed: bool,
    model: str | None,
    reasoning_effort: str | None,
    timeout: float,
    replan_context: str = "",
    memory_briefing: str = "",
    collection_catalog: "Sequence[CollectionCatalogEntry] | None" = None,
    skills_block: str = "",
    allowed_task_kinds: "set[str] | None" = None,
    depth: str = "normal",
    explicit_web_research: bool = False,
    previous_plan: "PlanRecord | None" = None,
    previous_tasks: "Sequence[PlanTaskRecord] | None" = None,
    history: str = "",
    tier: str | None = None,
) -> tuple[ExecutionPlanModel, dict[str, int]]:
    """Plan, validate, repair once; raises :class:`PlanningFailed`.

    ``collection_catalog`` is the caller-visible knowledge catalog: it is
    shown to the planner (name AND canonical id), name references in the
    produced plan are canonicalized to ids, and unresolvable references
    join the repair errors — a saved plan can then never fail retrieval
    with a raw unknown-collection error. ``None`` (no knowledge service)
    skips catalog handling; the runtime E5 gate still guards retrieval.

    ``previous_plan`` activates the delta-only replan contract. The model can
    then emit only new work (and optional pending-task skips); the server
    reconstructs the complete plan with immutable terminal results and one
    synthesis task before running the normal validator. Returns ``(plan,
    usage)`` with the accumulated token usage of the one or two calls.
    """
    criteria = list(profile.success_criteria) if profile else []
    research_policy = derive_web_research_policy(
        depth=depth,
        admitted_directive=explicit_web_research,
        tier=tier,
    )
    known_gap_ids = (
        {gap.gap_id for gap in discovery.gaps} if discovery else None
    )
    digest = summarize_discovery(discovery)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    errors: list[str] = []
    is_replan = previous_plan is not None
    if is_replan and previous_tasks is None:
        raise ValueError("previous_tasks are required for a replan")
    for attempt in range(2):
        outcome = structured_call(
            llm,
            prompt=build_agent_planner_prompt(
                question,
                digest,
                criteria,
                max_tasks=max_tasks,
                web_allowed=web_allowed,
                knowledge_allowed=knowledge_allowed,
                replan_context=replan_context,
                memory_briefing=memory_briefing,
                repair_errors=errors or None,
                collection_catalog=collection_catalog,
                skills_block=skills_block,
                research_allowed=research_policy.allowed,
                research_profile=research_policy.profile,
                research_profile_ceiling=research_policy.max_profile,
                max_web_instant_tasks=research_policy.max_instant_tasks,
                replan_mode=is_replan,
                history=history,
            ),
            model_cls=ReplanDeltaModel if is_replan else ExecutionPlanModel,
            node="agent_plan",
            system=agent_planner_system_prompt(),
            model=model,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
        usage["prompt_tokens"] += outcome.usage.get("prompt_tokens", 0)
        usage["completion_tokens"] += outcome.usage.get(
            "completion_tokens", 0
        )
        value = outcome.value
        if value is None:
            errors = ["Die Antwort war kein gueltiges Plan-JSON."]
            continue
        if is_replan:
            assert isinstance(value, ReplanDeltaModel)
            plan, errors = merge_replan_delta(
                value,
                previous_plan=previous_plan,
                previous_tasks=list(previous_tasks or ()),
            )
            if plan is None:
                continue
        else:
            assert isinstance(value, ExecutionPlanModel)
            plan = value
            errors = []
        # Canonicalize BEFORE validation so the allowed-ids check only
        # ever sees references the resolver could not map (one combined
        # repair list, no double-reporting).
        errors += (
            resolve_plan_collections(plan, collection_catalog)
            if collection_catalog is not None
            else []
        )
        errors += validate_plan(
            plan,
            max_tasks=max_tasks,
            known_gap_ids=known_gap_ids,
            allowed_collection_ids=(
                {entry.collection_id for entry in collection_catalog}
                if collection_catalog is not None
                else None
            ),
            web_research_allowed=research_policy.allowed,
            # A tier publishes a CEILING (per-task choice up to it); the
            # legacy path keeps the exact server pin.
            web_research_profile=(
                None
                if research_policy.max_profile is not None
                else research_policy.profile
            ),
            web_research_profile_ceiling=research_policy.max_profile,
            max_web_instant_tasks=research_policy.max_instant_tasks,
        )
        if not web_allowed:
            web_tasks = [
                task.id
                for task in plan.tasks
                if task.tool_kind in ("web_research", "web_instant")
            ]
            if web_tasks:
                errors.append(
                    "Web-Recherche ist nicht erlaubt; entferne die Tasks "
                    f"{', '.join(web_tasks)}."
                )
        if allowed_task_kinds is not None:
            # Activated skill allowlists restrict
            # the tool families; a task outside the union is a repair
            # error, so a forbidden task never even gets persisted.
            blocked = [
                f"{task.id} ({task.tool_kind})"
                for task in plan.tasks
                if task.tool_kind not in allowed_task_kinds
            ]
            if blocked:
                allowed = ", ".join(sorted(allowed_task_kinds))
                errors.append(
                    "Die aktivierten Skills erlauben nur diese "
                    f"Werkzeuge: {allowed}. Entferne oder ersetze: "
                    f"{', '.join(blocked)}."
                )
        if previous_tasks:
            _candidate_record, candidate_tasks = plan_to_records(
                plan,
                run_id="replan-validation",
                created_by="agent",
            )
            errors += additive_replan_errors(
                list(previous_tasks), candidate_tasks
            )
        if not errors:
            return plan, usage
    raise PlanningFailed(errors)


def merge_replan_delta(
    delta: ReplanDeltaModel,
    *,
    previous_plan: PlanRecord | None,
    previous_tasks: list[PlanTaskRecord],
) -> tuple[ExecutionPlanModel | None, list[str]]:
    """Merge a model-authored delta with server-owned prior task truth."""
    if previous_plan is None:
        return None, ["Ein Replan braucht einen bestehenden Plan."]
    prior_by_id = {task.task_id: task for task in previous_tasks}
    synthesis = next(
        (task for task in previous_tasks if task.tool_kind == "synthesis"),
        None,
    )
    errors: list[str] = []
    skip_ids = set(delta.skip_task_ids)
    unknown_skips = sorted(skip_ids - set(prior_by_id))
    if unknown_skips:
        errors.append(
            "skip_task_ids enthaelt unbekannte Tasks: "
            + ", ".join(unknown_skips)
            + "."
        )
    invalid_skips = sorted(
        task_id
        for task_id in skip_ids & set(prior_by_id)
        if prior_by_id[task_id].tool_kind == "synthesis"
        or prior_by_id[task_id].status != "pending"
    )
    if invalid_skips:
        errors.append(
            "Nur noch nicht gestartete Quell-Tasks duerfen uebersprungen "
            "werden: " + ", ".join(invalid_skips) + "."
        )
    new_ids = [task.id for task in delta.new_tasks]
    collisions = sorted(set(new_ids) & set(prior_by_id))
    if collisions:
        errors.append(
            "Neue Replan-Tasks brauchen neue IDs; bereits vorhanden: "
            + ", ".join(collisions)
            + "."
        )
    synthesis_new = sorted(
        task.id for task in delta.new_tasks if task.tool_kind == "synthesis"
    )
    if synthesis_new:
        errors.append(
            "Das Replan-Delta darf keinen synthesis-Task enthalten: "
            + ", ".join(synthesis_new)
            + "."
        )
    running = sorted(
        task.task_id
        for task in previous_tasks
        if task.tool_kind != "synthesis" and task.status == "running"
    )
    if running:
        errors.append(
            "Ein Replan darf nicht waehrend laufender Tasks entstehen: "
            + ", ".join(running)
            + "."
        )
    if errors:
        return None, errors

    source_tasks = [
        _task_record_to_model(task)
        for task in previous_tasks
        if task.tool_kind != "synthesis" and task.task_id not in skip_ids
    ]
    source_tasks.extend(delta.new_tasks)
    source_ids = [task.id for task in source_tasks]
    synthesis_id = synthesis.task_id if synthesis is not None else "s"
    if synthesis_id in set(new_ids):
        return None, [
            f"Die neue Task-ID {synthesis_id!r} ist fuer die serverseitige "
            "Synthese reserviert."
        ]
    synthesis_task = PlanTaskModel(
        id=synthesis_id,
        title=synthesis.title if synthesis is not None else "Synthese",
        tool_kind="synthesis",
        objective=synthesis.objective if synthesis is not None else "",
        depends_on=source_ids,
        expected_output=(
            synthesis.expected_output if synthesis is not None else ""
        ),
    )
    assumptions = list(previous_plan.assumptions)
    for assumption in delta.assumptions:
        if assumption not in assumptions:
            assumptions.append(assumption)
    return (
        ExecutionPlanModel(
            summary_markdown=(
                delta.summary_markdown or previous_plan.summary_markdown
            ),
            tasks=[*source_tasks, synthesis_task],
            assumptions=assumptions,
            success_criteria=list(previous_plan.success_criteria),
        ),
        [],
    )


def _task_record_to_model(task: PlanTaskRecord) -> PlanTaskModel:
    """Project one server-owned task definition back into plan validation."""
    return PlanTaskModel.model_validate(
        {
            "id": task.task_id,
            "title": task.title,
            "tool_kind": task.tool_kind,
            "objective": task.objective,
            "queries": list(task.queries),
            "gap_ids": list(task.gap_ids),
            "depends_on": list(task.depends_on),
            "params": dict(task.params),
            "expected_output": task.expected_output,
            "is_falsification": task.is_falsification,
        }
    )


def summarize_discovery(discovery: DiscoveryResult | None) -> str:
    """Compact digest of facts + gaps for the planner prompt."""
    if discovery is None:
        return "(keine Erkundung durchgefuehrt)"
    lines = [
        f"Bekannt: {finding.fact} [{finding.source}]"
        + ("" if finding.fresh else " (moeglicherweise veraltet)")
        for finding in discovery.known_facts
    ]
    lines += [
        f"Gap {gap.gap_id} ({gap.kind}"
        + (", blockierend" if gap.blocking else "")
        + f"): {gap.description} -> {gap.recommended_capability}"
        for gap in discovery.gaps
    ]
    return "\n".join(lines) or "(keine Befunde)"


def plan_to_records(
    plan: ExecutionPlanModel,
    *,
    run_id: str,
    created_by: str,
    reason: str = "",
    previous_tasks: "Sequence[PlanTaskRecord] | None" = None,
) -> tuple[PlanRecord, list[PlanTaskRecord]]:
    """Convert the wire plan into control-store records (version assigned
    by the store as latest+1)."""
    plan_id = f"plan_{uuid.uuid4().hex}"
    record = PlanRecord(
        plan_id=plan_id,
        run_id=run_id,
        version=0,
        status="proposed",
        created_by=created_by,
        summary_markdown=plan.summary_markdown,
        assumptions=tuple(plan.assumptions),
        success_criteria=tuple(plan.success_criteria),
        reason=reason,
    )
    tasks = [
        PlanTaskRecord(
            task_id=task.id,
            plan_id=plan_id,
            run_id=run_id,
            ordinal=index,
            title=task.title,
            tool_kind=task.tool_kind,
            objective=task.objective,
            queries=tuple(task.queries),
            gap_ids=tuple(task.gap_ids),
            depends_on=tuple(task.depends_on),
            budget={},
            params=task.params.model_dump(exclude_none=True),
            expected_output=task.expected_output,
            is_falsification=task.is_falsification,
        )
        for index, task in enumerate(plan.tasks)
    ]
    if previous_tasks:
        tasks = carry_forward_terminal_task_results(
            list(previous_tasks), tasks
        )
    return record, tasks
