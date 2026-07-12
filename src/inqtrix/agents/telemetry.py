"""Shared semantic telemetry projections for both Agent engines."""

from __future__ import annotations

from typing import Any


_MODEL_RETRY_CONTEXT: dict[str, tuple[str, str]] = {
    "agent_intake": ("Auftrag wird eingeordnet", "intake"),
    "agent_discovery_analyst": ("Erkundung wird ausgewertet", "discovery"),
    "agent_skill_point_check": ("Skill-Eingaben werden geprueft", "intake"),
    "agent_plan": ("Ausfuehrungsplan wird erstellt", "planning"),
    "agent_contradiction": ("Gegenbelege werden geprueft", "evidence"),
    "agent_sufficiency": ("Beleglage wird bewertet", "evidence"),
    "agent_synthesis": ("Ergebnisse werden zusammengefuehrt", "synthesis"),
    "agent_citation_repair": (
        "Quellenverweise werden geprueft",
        "synthesis",
    ),
    "agent_answer": ("Antwort wird ausgearbeitet", "synthesis"),
    "agent_answer_light": ("Antwort wird ausgearbeitet", "synthesis"),
    "agent_critic": ("Ergebnis wird geprueft", "critic"),
    "agent_memory_reflection": ("Sitzungskontext wird aktualisiert", "critic"),
    "agent_file_analysis": ("Dateiinhalte werden ausgewertet", "discovery"),
    "agent_patch": ("Dokumentaenderung wird vorbereitet", "patch"),
    "agent_kernel": ("Agent-Antwort wird fortgesetzt", "execution"),
    "agent_quick_web_query": ("Webfrage wird vorbereitet", "execution"),
    "agent_quick_web_answer": (
        "Direkte Webantwort wird formuliert",
        "execution",
    ),
    "agent_deep_review": ("Ergebnis wird tiefengeprueft", "critic"),
    "agent_model": ("Modellschritt wird fortgesetzt", "execution"),
}


def provider_retry_activity(
    notice: dict[str, object],
    *,
    task_id: str = "",
    purpose: str = "",
    scope: str = "task",
    phase: str = "execution",
    kind: str = "searching",
) -> dict[str, Any]:
    """Project a provider retry notice onto the existing activity event.

    The provider owns retry policy and exact timeout diagnostics; this helper
    only turns the notice into the one Agent activity vocabulary consumed by
    the timeline and task detail views.
    """
    failed_attempt = max(1, int(notice.get("attempt", 1) or 1))
    max_attempts = max(
        failed_attempt,
        int(notice.get("max_attempts", failed_attempt) or failed_attempt),
    )
    next_attempt = min(max_attempts, failed_attempt + 1)
    delay_seconds = max(0.0, float(notice.get("delay_seconds", 0.0) or 0.0))
    query = str(notice.get("query") or "").strip()
    error_code = str(
        notice.get("error_code")
        or notice.get("status_code")
        or "transient_error"
    )
    retry_text = (
        f"Provider-Versuch {failed_attempt}/{max_attempts} fehlgeschlagen "
        f"({error_code}); Versuch {next_attempt}/{max_attempts} startet nach "
        f"{delay_seconds:.1f} s."
    )
    return {
        "kind": kind,
        "scope": scope,
        "phase": phase,
        "operation": str(notice.get("operation") or "web.search.instant"),
        # The bare query keeps the retry notice on the SAME transcript
        # row as its invocation (the FE keys rows by query/detail) —
        # without it every retry opened a ghost row that never settled.
        "query": query or None,
        "detail": " · ".join(
            part for part in (query or purpose, retry_text) if part
        ),
        "status": "started",
        "attempt": next_attempt,
        "task_id": task_id or None,
        "purpose": purpose or None,
        "retry": {
            "failed_attempt": failed_attempt,
            "next_attempt": next_attempt,
            "max_attempts": max_attempts,
            "delay_seconds": delay_seconds,
            "error_code": error_code,
        },
        "configured_timeout_seconds": notice.get(
            "configured_timeout_seconds"
        ),
        "effective_timeout_seconds": notice.get(
            "effective_timeout_seconds"
        ),
        "transport_timeout_seconds": notice.get(
            "transport_timeout_seconds"
        ),
    }


def model_retry_activity(
    notice: dict[str, object],
    *,
    node: str,
) -> dict[str, Any]:
    """Project one LLM retry with a user-facing Agent phase description."""
    purpose, phase = _MODEL_RETRY_CONTEXT.get(
        node,
        _MODEL_RETRY_CONTEXT["agent_model"],
    )
    return provider_retry_activity(
        notice,
        purpose=purpose,
        scope="run",
        phase=phase,
        kind="working",
    )
