"""Phase 1 — read-only discovery (§4).

Probes run DETERMINISTICALLY (code decides which, the LLM never steers
them) against the capability registry under a hard, server-side call
budget. Raw probe texts stay in this module (context quarantine): the
analyst call sees a compressed digest, the caller only the structured
:class:`~inqtrix.agents.phase_models.DiscoveryResult` plus the probe
stats for the discovery event.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from inqtrix.agents.patterns._structured import StructuredOutcome, structured_call
from inqtrix.agents.phase_models import AssignmentProfile, DiscoveryResult
from inqtrix.agents.prompts import (
    agent_analyst_system_prompt,
    build_agent_analyst_prompt,
)

if TYPE_CHECKING:
    from inqtrix.capabilities import CapabilityContext, CapabilityRegistry
    from inqtrix.providers.base import LLMProvider

log = logging.getLogger("inqtrix")

_PROBE_SNIPPET_CHARS = 400
"""Per-hit compression cap for the analyst digest."""


@dataclass
class ProbePlan:
    """The deterministic probe list (shown to the user in strict mode)."""

    probes: list[dict[str, Any]] = field(default_factory=list)

    def as_payload(self) -> list[dict[str, Any]]:
        """Wire shape for the discovery approval / discovery event."""
        return [dict(probe) for probe in self.probes]


def build_probe_plan(
    profile: AssignmentProfile | None,
    *,
    question: str,
    collection_ids: list[str],
    max_calls: int,
    web_preview_allowed: bool,
    knowledge_allowed: bool = True,
    clarified_context: str = "",
) -> ProbePlan:
    """Assemble the deterministic probe list, budget-capped.

    Scope decision (deliberate, not an oversight): of the plan's §4
    Phase-1 probe set (P1-P6) this builds P1 (``knowledge.collections.
    list``), P2 (``knowledge.search`` per sub-goal, top_k=5) and P6
    (``web.search.instant`` preview). P3-P5 are intentionally NOT probed
    here because each is either input-less or already covered:

    * P3 (prior-run reuse) would need a runs-read capability plus a
      question-similarity heuristic — a NEW catalog for a single consumer
      (Prinzip 7); within a session the prior context is already carried
      by the memo lineage (E15 intake read). It is a named follow-up, not
      a gap-fill.
    * P4 (referenced files) has no input: an agent run carries collection
      ids and (M7) one patch-target ``document_id``, never a list of
      @-mentioned individual files — a probe for them would have nothing
      to read.
    * P5 (canvas/editor context) is redundant: the canvas/memo context is
      read at intake (the session memo, E15) and the editor patch-target
      document is read in the patch phase — probing it here would
      duplicate existing reads (Prinzip 4).
    """
    # Answered clarification rounds sharpen the free-text probes: the
    # user just pinned down market/region/timeframe — a probe ignoring
    # that would re-explore what is already settled (P2). Bounded so a
    # long free-text answer cannot blow up the query.
    context = clarified_context.strip()[:200]
    scoped_question = f"{question} — {context}" if context else question
    probes: list[dict[str, Any]] = []
    if knowledge_allowed:
        probes.append({"kind": "knowledge.collections.list", "query": ""})
    goals = list(profile.sub_goals) if profile else []
    if knowledge_allowed:
        for goal in (goals or [scoped_question])[:4]:
            probes.append(
                {
                    "kind": "knowledge.search",
                    "query": goal,
                    "collection_ids": list(collection_ids),
                }
            )
    if web_preview_allowed and (profile is None or profile.needs_web):
        probes.append(
            {"kind": "web.search.instant", "query": scoped_question}
        )
    if len(probes) > max_calls:
        log.warning(
            "Discovery-Probenplan auf das Budget gekuerzt (%d -> %d).",
            len(probes),
            max_calls,
        )
        probes = probes[:max_calls]
    return ProbePlan(probes=probes)


def execute_probes(
    plan: ProbePlan,
    *,
    registry: "CapabilityRegistry",
    capability_context: "CapabilityContext",
    invoke: Any = None,
) -> tuple[str, dict[str, Any]]:
    """Run the probes; returns ``(digest_for_analyst, probe_stats)``.

    *invoke* overrides the capability invocation (test seam); defaults
    to ``registry.invoke``. Probe failures are collected VISIBLY into
    the digest and stats — a discovery with dead probes still plans,
    but the analyst and the user see the holes.
    """
    import asyncio

    call = invoke or (
        lambda capability_id, payload: asyncio.run(
            registry.invoke(capability_id, payload, capability_context)
        )
    )
    lines: list[str] = []
    executed = 0
    failed = 0
    source_tool_counts = {"web": 0, "knowledge": 0}
    for probe in plan.probes:
        kind = probe["kind"]
        try:
            if kind == "knowledge.collections.list":
                output = call("knowledge.collections.list", {})
                described = [
                    _describe_collection(item) for item in _rows(output)
                ]
                lines.append(
                    "[Bestand] Sammlungen: "
                    f"{'; '.join(described) or '(keine)'}"
                )
            elif kind == "knowledge.search":
                payload: dict[str, Any] = {
                    "query": probe["query"],
                    "top_k": 5,
                }
                if probe.get("collection_ids"):
                    payload["collection_ids"] = probe["collection_ids"]
                output = call("knowledge.search", payload)
                for hit in _rows(output)[:5]:
                    text = str(hit.get("text", ""))[:_PROBE_SNIPPET_CHARS]
                    lines.append(
                        "[Intern doc:"
                        f"{hit.get('document_id', '?')}#"
                        f"{hit.get('chunk_index', '?')}] {text}"
                    )
                if not _rows(output):
                    lines.append(
                        f"[Intern] Keine Treffer fuer: {probe['query']}"
                    )
            elif kind == "web.search.instant":
                output = call(
                    "web.search.instant",
                    {"query": probe["query"], "max_sources": 5},
                )
                for source in _rows(output, key="sources")[:5]:
                    lines.append(
                        f"[Web {source.get('url', '?')}] "
                        f"{str(source.get('snippet', ''))[:_PROBE_SNIPPET_CHARS]}"
                    )
            else:  # pragma: no cover - probe kinds are code-defined
                raise ValueError(f"unknown probe kind {kind!r}")
            executed += 1
            source = "web" if kind.startswith("web.") else "knowledge"
            source_tool_counts[source] += 1
        except Exception as exc:  # noqa: BLE001 — probe failures stay visible
            failed += 1
            log.warning(
                "Discovery-Probe %s fehlgeschlagen: %s", kind, exc
            )
            lines.append(f"[FEHLGESCHLAGEN {kind}] {exc}")
    digest = "\n".join(lines) or "(keine Sondierungsergebnisse)"
    stats = {
        "planned": len(plan.probes),
        "executed": executed,
        "failed": failed,
        "source_tool_counts": source_tool_counts,
    }
    return digest, stats


def _describe_collection(item: dict[str, Any]) -> str:
    """Digest line for one collection: display name PLUS canonical id.

    The planner writes ``params.collection_ids`` from what it reads
    here. Showing only the display name (the pre-fix behavior) forced it
    to guess an identifier, and the guess — the name — later failed
    retrieval as an unknown id. Name and id travel together from the
    first planner-facing surface on; model and document count let the
    planner judge scope without another probe.
    """
    name = str(item.get("name") or item.get("id") or "?")
    details = [f"id: {item.get('id') or '?'}"]
    model = str(item.get("embedding_model") or "")
    if model:
        details.append(model)
    count = item.get("document_count")
    if isinstance(count, int):
        details.append(f"{count} Dokumente")
    return f"{name} ({', '.join(details)})"


def _rows(output: Any, *, key: str = "results") -> list[dict[str, Any]]:
    """Rows of a capability output model or plain dict."""
    if hasattr(output, "model_dump"):
        output = output.model_dump()
    if isinstance(output, dict):
        value = output.get(key)
        if value is None and key == "results":
            # The knowledge.search capability names its rows ``hits``,
            # collections listing ``collections`` — cover the catalog's
            # actual output keys, never silently pretend emptiness.
            value = output.get("hits") or output.get("collections")
        return list(value or [])
    return []


def run_discovery_analysis(
    llm: "LLMProvider",
    *,
    question: str,
    probe_digest: str,
    profile: AssignmentProfile | None,
    model: str | None,
    reasoning_effort: str | None,
    timeout: float,
    history: str = "",
) -> StructuredOutcome:
    """The mid-tier analyst call; value is a DiscoveryResult.

    Args:
        history: Conversation plus ANSWERED clarification rounds. The
            analyst proposes ``questions_for_user`` — without the
            already-given answers it re-asks the intake round in new
            words (the observed duplicate-clarification bug).
    """
    return structured_call(
        llm,
        prompt=build_agent_analyst_prompt(
            question,
            probe_digest,
            list(profile.sub_goals) if profile else [],
            history=history,
        ),
        model_cls=DiscoveryResult,
        node="agent_discovery_analyst",
        system=agent_analyst_system_prompt(),
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )
