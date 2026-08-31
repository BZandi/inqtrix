"""Durable user decisions for recoverable Agent Desk run limits.

Recoverable limits reuse the existing clarification row plus the native
run's ``waiting_for_input -> queued`` compare-and-swap.  The clarification is
the durable decision record; run events are only signals.  This deliberately
does not add another run lifecycle, queue, or model-authored budget channel.

Only checkpoint-safe limits belong here.  A caller may park after reaching a
limit when replay resumes from an authoritative checkpoint without repeating
already charged work.  Provider/token ceilings whose latest work is not yet
checkpointed remain non-recoverable typed stops.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

from inqtrix.agents.control_ports import (
    AgentControlStore,
    ClarificationNotFound,
    ClarificationRecord,
)

LIMIT_REACHED_EVENT = "inqtrix.agent.limit.reached"
"""Audit/event signal emitted when a run parks at a recoverable limit."""
LIMIT_DECIDED_EVENT = "inqtrix.agent.limit.decided"
"""Audit signal emitted when a terminal partial/cancel choice is applied.

Extensions are already durably visible through the canonical
``clarification.answered`` event and row. Emitting a second apply event on
every checkpoint redelivery would make an exactly-once claim the event stream
cannot guarantee.
"""

LIMIT_CHOICE_EXTEND = "extend"
LIMIT_CHOICE_PARTIAL = "partial"
LIMIT_CHOICE_CANCEL = "cancel"

QUICK_WEB_SEARCH_LIMIT = 1
"""The direct quick-web lane performs exactly one provider search."""

LimitKind = Literal["tool_calls", "steps"]
LimitChoice = Literal["extend", "partial", "cancel"]

_LIMIT_ID = re.compile(
    r"^clr_[A-Za-z0-9]+_limit_"
    r"(?P<kind>tool_calls|steps)_"
    r"(?P<used>[0-9]+)_"
    r"(?P<current>[0-9]+)_"
    r"(?P<proposed>[0-9]+)_"
    r"(?P<ceiling>[0-9]+)$"
)


@dataclass(frozen=True)
class AgentLimitGate:
    """One server-authored recoverable limit decision."""

    kind: LimitKind
    current: int
    proposed: int
    ceiling: int
    used: int

    @property
    def extendable(self) -> bool:
        return self.proposed > self.current and self.proposed <= self.ceiling

    def clarification_id(self, run_id: str) -> str:
        return (
            f"clr_{run_id[-12:]}_limit_{self.kind}_"
            f"{self.used}_{self.current}_{self.proposed}_{self.ceiling}"
        )

    def payload(self, *, clarification_id: str) -> dict[str, Any]:
        return {
            "clarification_id": clarification_id,
            "kind": self.kind,
            "used": self.used,
            "limit": self.current,
            "next_limit": self.proposed if self.extendable else None,
            "ceiling": self.ceiling,
            "extendable": self.extendable,
            "choices": [
                *([LIMIT_CHOICE_EXTEND] if self.extendable else []),
                LIMIT_CHOICE_PARTIAL,
                LIMIT_CHOICE_CANCEL,
            ],
        }


def next_extended_limit(*, current: int, ceiling: int, required: int = 0) -> int:
    """Return the next monotonic limit inside the operator ceiling.

    The step doubles the current allowance, while an overflowing model batch
    can require a larger minimum.  ``current`` is returned when the ceiling is
    already binding; the UI then offers partial completion or cancellation,
    never a fake extension.
    """

    current = max(1, int(current))
    ceiling = max(current, int(ceiling))
    required = max(current + 1, int(required or 0))
    return min(ceiling, max(current * 2, required))


def create_or_get_limit_gate(
    control: AgentControlStore,
    *,
    run_id: str,
    gate: AgentLimitGate,
    run_async: Callable[[Any], Any],
) -> tuple[ClarificationRecord, bool]:
    """Create the deterministic clarification once; return ``(row, created)``."""

    clarification_id = gate.clarification_id(run_id)
    try:
        return run_async(
            control.get_clarification(run_id, clarification_id)
        ), False
    except ClarificationNotFound:
        pass

    noun = "Werkzeugaufrufe" if gate.kind == "tool_calls" else "Ausführungsschritte"
    options: list[dict[str, str]] = []
    if gate.extendable:
        options.append(
            {
                "id": LIMIT_CHOICE_EXTEND,
                "label": f"Auf {gate.proposed} erweitern",
                "description": (
                    "Der Lauf setzt am gespeicherten Stand fort; bereits "
                    "verbrauchte Aufrufe und Tokens bleiben angerechnet."
                ),
            }
        )
    options.extend(
        [
            {
                "id": LIMIT_CHOICE_PARTIAL,
                "label": "Teilstand übernehmen",
                "description": (
                    "Der Lauf schließt mit den bisher gesicherten Belegen "
                    "und einer klar gekennzeichneten Teilantwort ab."
                ),
            },
            {
                "id": LIMIT_CHOICE_CANCEL,
                "label": "Lauf abbrechen",
                "description": "Der gespeicherte Lauf wird beendet.",
            },
        ]
    )
    ceiling_text = (
        f" Eine Erweiterung bis {gate.proposed} ist innerhalb der "
        f"Betreibergrenze {gate.ceiling} möglich."
        if gate.extendable
        else f" Die Betreibergrenze {gate.ceiling} lässt keine Erweiterung zu."
    )
    reached_text = (
        f"Der Agent hat bisher {gate.used} {noun} verwendet. Der nächste "
        f"atomare Werkzeug-Batch würde das aktuelle Limit {gate.current} "
        "überschreiten. Kein Aufruf dieses Batches wurde ausgeführt."
        if gate.kind == "tool_calls"
        else (
            f"Der Agent hat {gate.used} {noun} verwendet und damit das "
            f"aktuelle Limit {gate.current} erreicht."
        )
    )
    record = ClarificationRecord(
        clarification_id=clarification_id,
        run_id=run_id,
        question=(
            f"{reached_text}{ceiling_text} "
            "Wie soll der Lauf fortgesetzt werden?"
        ),
        options=tuple(options),
        default_assumption=(
            "Ohne Entscheidung bleibt der Lauf pausiert; es gibt keinen "
            "automatischen Teilabschluss."
        ),
    )
    return run_async(control.create_clarification(record)), True


def parse_limit_gate(record: ClarificationRecord) -> AgentLimitGate | None:
    """Parse server-authored coordinates from a clarification id."""

    match = _LIMIT_ID.fullmatch(record.clarification_id)
    if match is None:
        return None
    kind = match.group("kind")
    used = int(match.group("used"))
    current = int(match.group("current"))
    proposed = int(match.group("proposed"))
    ceiling = int(match.group("ceiling"))
    if used < 0 or current < 1 or proposed < current or ceiling < proposed:
        return None
    return AgentLimitGate(
        kind=kind,  # type: ignore[arg-type]
        current=current,
        proposed=proposed,
        ceiling=ceiling,
        used=used,
    )


def recorded_limit_choice(record: ClarificationRecord) -> LimitChoice | None:
    """Return the validated choice of one answered limit clarification."""

    if record.status != "answered" or parse_limit_gate(record) is None:
        return None
    raw = (record.option_id or record.answer).strip().casefold()
    aliases = {
        LIMIT_CHOICE_EXTEND: LIMIT_CHOICE_EXTEND,
        "erweitern": LIMIT_CHOICE_EXTEND,
        LIMIT_CHOICE_PARTIAL: LIMIT_CHOICE_PARTIAL,
        "teilstand übernehmen": LIMIT_CHOICE_PARTIAL,
        "teilstand uebernehmen": LIMIT_CHOICE_PARTIAL,
        LIMIT_CHOICE_CANCEL: LIMIT_CHOICE_CANCEL,
        "lauf abbrechen": LIMIT_CHOICE_CANCEL,
        "abbrechen": LIMIT_CHOICE_CANCEL,
    }
    return aliases.get(raw)  # type: ignore[return-value]


def effective_extended_limit(
    control: AgentControlStore,
    *,
    run_id: str,
    kind: LimitKind,
    base: int,
    ceiling: int,
    run_async: Callable[[Any], Any],
) -> int:
    """Fold answered extension rows into one bounded effective limit."""

    effective = max(1, int(base))
    hard_ceiling = max(effective, int(ceiling))
    records = run_async(control.list_clarifications(run_id))
    for record in reversed(records):
        gate = parse_limit_gate(record)
        if gate is None or gate.kind != kind:
            continue
        if recorded_limit_choice(record) != LIMIT_CHOICE_EXTEND:
            continue
        effective = max(
            effective,
            min(hard_ceiling, gate.ceiling, gate.proposed),
        )
    return effective


def latest_terminal_limit_choice(
    control: AgentControlStore,
    *,
    run_id: str,
    run_async: Callable[[Any], Any],
) -> tuple[AgentLimitGate, LimitChoice, ClarificationRecord] | None:
    """Return the newest explicit partial/cancel choice, if any."""

    for record in run_async(control.list_clarifications(run_id)):
        gate = parse_limit_gate(record)
        choice = recorded_limit_choice(record)
        if gate is not None and choice in {
            LIMIT_CHOICE_PARTIAL,
            LIMIT_CHOICE_CANCEL,
        }:
            return gate, choice, record
    return None


def effective_tool_grants(
    control: AgentControlStore,
    *,
    run_id: str,
    run_async: Callable[[Any], Any],
) -> frozenset[str]:
    """Fold approved run-wide tool grants into one per-segment set (P6B).

    Mirrors :func:`effective_extended_limit`: the decided approval row is
    the durable decision record, folded fresh at every segment start. A
    grant is exactly an ``approve`` on a ``kind="tool"`` gate whose
    ``decision_payload`` carries ``approval_scope == "run"``; the granted
    names are the action tools of that gate. ``ALWAYS_GATED_TOOLS`` are
    excluded by construction — the service refuses to store such a grant,
    and this fold refuses to honor one that slipped in anyway.
    """
    from inqtrix.agents.kernel.policy import ALWAYS_GATED_TOOLS

    granted: set[str] = set()
    records = run_async(control.list_approvals(run_id))
    for record in records:
        if record.kind != "tool" or record.decision != "approve":
            continue
        if dict(record.decision_payload).get("approval_scope") != "run":
            continue
        for action in record.payload.get("actions") or []:
            name = str(action.get("tool") or "")
            if name and name not in ALWAYS_GATED_TOOLS:
                granted.add(name)
    return frozenset(granted)
