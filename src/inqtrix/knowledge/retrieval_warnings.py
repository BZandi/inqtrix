"""Shared public warnings for candidates excluded by Knowledge retrieval.

Stores report source-integrity decisions as typed, text-free
``RetrievalExclusion`` values.  This module is the single projection from those
internal reasons to the stable warning codes consumed by the synchronous search
surface, Agent capability and native Knowledge runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from inqtrix.knowledge.stores.ports import RetrievalExclusion


@dataclass(frozen=True)
class KnowledgeRetrievalWarning:
    """Bounded, evidence-free warning safe for API, run and UI projection."""

    code: str
    message: str
    reason: str
    stage: str
    count: int
    recommended_action: str | None = None

    def as_dict(self, *, include_message: bool = True) -> dict[str, object]:
        return {
            "code": self.code,
            **({"message": self.message} if include_message else {}),
            "reason": self.reason,
            "stage": self.stage,
            "count": self.count,
            "recommended_action": self.recommended_action,
        }


_WARNING_CONTRACT: dict[str, tuple[str, str]] = {
    "source_unverified": (
        "chunks_require_reindex",
        "Treffer ohne verifizierbaren Originaltext wurden ausgeschlossen und "
        "müssen neu indiziert werden.",
    ),
    "canonical_chunk_unavailable": (
        "chunks_pending_reconciliation",
        "Vektortreffer ohne kanonischen Datenbanksatz wurden ausgeschlossen; "
        "der Indexabgleich muss abgeschlossen werden.",
    ),
    "duplicate_document": (
        "duplicate_documents_collapsed",
        "Inhaltsgleiche Dokumente wurden auf eines zusammengefasst, damit "
        "nicht mehrere Belegplätze dieselben Passagen tragen.",
    ),
}

_UNKNOWN_WARNING = (
    "retrieval_candidates_excluded",
    "Treffer wurden durch eine Integritätsprüfung der Wissensquellen "
    "ausgeschlossen; die Ergebnisabdeckung kann unvollständig sein.",
)


def project_retrieval_exclusion_warnings(
    exclusions: Iterable[RetrievalExclusion],
) -> tuple[KnowledgeRetrievalWarning, ...]:
    """Aggregate exclusions losslessly and project stable warning codes.

    Repeated retrievals in a decomposed or gate-rewrite run can report the same
    exclusion kind. Counts are therefore accumulated by the full typed identity
    (reason, stage, recommended action), while first-seen order remains stable.
    The resulting count is explicitly a number of exclusion *observations*, not
    unique chunks: chunk ids deliberately do not cross this aggregate boundary,
    so the same unsafe point observed by two queries cannot be deduplicated
    safely. Unknown future reasons are deliberately surfaced through a generic
    warning instead of being silently discarded.
    """

    order: list[tuple[str, str, str | None]] = []
    counts: dict[tuple[str, str, str | None], int] = {}
    for exclusion in exclusions:
        key = (
            exclusion.reason,
            exclusion.stage,
            exclusion.recommended_action,
        )
        if key not in counts:
            order.append(key)
            counts[key] = 0
        counts[key] += exclusion.count

    warnings: list[KnowledgeRetrievalWarning] = []
    for reason, stage, recommended_action in order:
        code, message = _WARNING_CONTRACT.get(reason, _UNKNOWN_WARNING)
        warnings.append(
            KnowledgeRetrievalWarning(
                code=code,
                message=message,
                reason=reason,
                stage=stage,
                count=counts[(reason, stage, recommended_action)],
                recommended_action=recommended_action,
            )
        )
    return tuple(warnings)
