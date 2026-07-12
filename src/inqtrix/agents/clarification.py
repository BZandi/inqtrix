"""Phase 2 — clarification records (§4).

Small by design: which questions get asked is decided by intake routing
(``ask_user_first``) or blocking discovery gaps; this module only shapes
the control-store records. Non-blocking uncertainty becomes plan
ASSUMPTIONS — the approval is the correction chance, not a question.

Structured option questions (decision #8 refinement): the LLM proposes
questions with LIKELY answer options; :func:`sanitize_questions` turns
them into the deterministic wire shape the record stores. Ids are
assigned positionally here — never trusted from the model — so a
re-executed clarify node (interrupt semantics) regenerates the exact
same payload from the checkpointed phase state.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Sequence

from inqtrix.agents.control_ports import ClarificationRecord
from inqtrix.agents.phase_models import (
    AssignmentProfile,
    ClarificationQuestionModel,
    DiscoveryResult,
)

log = logging.getLogger("inqtrix")

MAX_QUESTIONS_PER_ROUND = 3
"""Hard cap of questions per gate round (one record, one interrupt)."""

MAX_OPTIONS_PER_QUESTION = 4
"""Hard cap of options per question (AskUserQuestion-informed)."""

_MAX_LABEL_CHARS = 60
_MAX_DESCRIPTION_CHARS = 160


def sanitize_questions(
    models: Sequence[ClarificationQuestionModel],
) -> list[dict[str, Any]]:
    """The deterministic wire shape of LLM-proposed questions.

    Guards (all LOUD, nothing silently dropped without a log line):

    - at most :data:`MAX_QUESTIONS_PER_ROUND` questions survive;
    - a question without a usable prompt is dropped with a warning (an
      empty question cannot be asked);
    - a question with fewer than 2 usable options keeps the question but
      loses the chips (free text always works, decision #8);
    - options beyond :data:`MAX_OPTIONS_PER_QUESTION` are cut, labels and
      descriptions are length-capped, duplicate labels collapse.

    Ids are positional (``q1``, ``q1_o2`` ...) so the SAME input always
    yields the SAME payload — required by the interrupt re-execution
    contract (pre-interrupt writes must be idempotent).
    """
    sanitized: list[dict[str, Any]] = []
    for model in models:
        if len(sanitized) >= MAX_QUESTIONS_PER_ROUND:
            log.warning(
                "Rueckfragen-Runde auf %d Fragen gekappt (%d vorgeschlagen).",
                MAX_QUESTIONS_PER_ROUND,
                len(models),
            )
            break
        prompt = model.prompt.strip()
        if not prompt:
            log.warning(
                "Rueckfrage ohne Fragetext verworfen (Optionen: %d).",
                len(model.options),
            )
            continue
        number = len(sanitized) + 1
        options: list[dict[str, Any]] = []
        seen_labels: set[str] = set()
        for option in model.options:
            label = option.label.strip()[:_MAX_LABEL_CHARS]
            if not label or label.casefold() in seen_labels:
                continue
            if len(options) >= MAX_OPTIONS_PER_QUESTION:
                log.warning(
                    "Rueckfrage %d: Optionen auf %d gekappt.",
                    number,
                    MAX_OPTIONS_PER_QUESTION,
                )
                break
            seen_labels.add(label.casefold())
            options.append(
                {
                    "id": f"q{number}_o{len(options) + 1}",
                    "label": label,
                    "description": option.description.strip()[
                        :_MAX_DESCRIPTION_CHARS
                    ],
                }
            )
        if len(options) < 2:
            if options:
                log.warning(
                    "Rueckfrage %d hat nur %d nutzbare Option(en) — "
                    "wird als Freitext-Frage gestellt.",
                    number,
                    len(options),
                )
            options = []
        sanitized.append(
            {
                "id": f"q{number}",
                "prompt": prompt,
                "options": options,
                "multi_select": bool(model.multi_select) and bool(options),
            }
        )
    return sanitized


def build_clarification(
    run_id: str,
    *,
    questions: Sequence[dict[str, Any]],
    default_assumption: str = "",
    clarification_id: str = "",
) -> ClarificationRecord:
    """One pending clarification row for a whole gate round.

    ``question`` (the legacy single-text column) carries the joined
    prompts so older readers and the composer free-text path keep one
    human-readable string; the structured ``questions`` payload is the
    chips source. Legacy single-option rounds keep working: with exactly
    one question its options mirror into the legacy ``options`` column.

    ``clarification_id`` overrides the random id with a DETERMINISTIC
    one — required by every interrupt caller (phase-machine clarify
    node, kernel ``ask_user``): the node re-executes on resume, so the
    get-or-create must land on the same row.
    """
    question_text = " ".join(q["prompt"] for q in questions)
    legacy_options: tuple[dict[str, Any], ...] = ()
    if len(questions) == 1:
        legacy_options = tuple(questions[0]["options"])
    return ClarificationRecord(
        clarification_id=clarification_id or f"clr_{uuid.uuid4().hex[:12]}",
        run_id=run_id,
        question=question_text,
        options=legacy_options,
        questions=tuple(questions),
        default_assumption=default_assumption,
    )


def round_qa_lines(
    *,
    questions: Sequence[dict[str, Any]],
    question: str,
    options: Sequence[dict[str, Any]],
    answers: dict[str, Any],
    answer: str,
    option_id: str,
) -> list[tuple[str, str]]:
    """Human-readable (prompt, answer) pairs of one ANSWERED round.

    Structured answers compose per question (picked labels, then free
    text); a whole-round free-text or legacy option answer yields the
    single legacy pair. Empty while unanswered. Shared by the clarify
    node's history composition AND the session-context builder (K1) so
    transcript and follow-up context can never phrase a round
    differently.
    """
    if answers and questions:
        lines: list[tuple[str, str]] = []
        for item in questions:
            entry = answers.get(item.get("id", "")) or {}
            labels = {
                option["id"]: option["label"]
                for option in item.get("options", [])
            }
            picked = [
                labels[oid]
                for oid in entry.get("option_ids", [])
                if oid in labels
            ]
            free_text = str(entry.get("text", "")).strip()
            parts = [part for part in ("; ".join(picked), free_text) if part]
            if parts:
                lines.append((str(item.get("prompt", "")), " — ".join(parts)))
        return lines
    legacy = answer or next(
        (
            str(option.get("label", ""))
            for option in options
            if str(option.get("id", "")) == option_id and option_id
        ),
        "",
    ) or option_id
    return [(question, legacy)] if legacy else []


_QUESTION_STOPWORDS = frozenset(
    {
        "an", "am", "auch", "auf", "aus", "bei", "beziehen", "bitte",
        "das", "dem", "den", "der", "des", "die", "du", "ein", "eine",
        "einem", "einen", "einer", "es", "fuer", "für", "im", "in",
        "ist", "mit", "nach", "primaer", "primär", "sich", "sie",
        "soll", "sollen", "sollte", "um", "und", "uns", "unsere",
        "von", "was", "welche", "welchem", "welchen", "welcher",
        "welches", "wie", "wir", "zu", "zum", "zur",
    }
)
"""German question scaffolding that carries no topical signal."""

_TOKEN_PREFIX_CHARS = 6
"""Poor-man's stemming: 'Analyse' and 'analysieren' meet at 'analys'.
A full German stemmer would be a new dependency for one comparison."""

_DUPLICATE_JACCARD = 0.6
_DUPLICATE_CONTAINMENT = 0.8


def _question_tokens(prompt: str) -> frozenset[str]:
    """Topical token set of a question prompt (normalized, prefix-cut)."""
    words = re.findall(r"[0-9a-zA-ZÀ-ſ]+", prompt.casefold())
    return frozenset(
        word[:_TOKEN_PREFIX_CHARS]
        for word in words
        if len(word) >= 2 and word not in _QUESTION_STOPWORDS
    )


def filter_repeated_questions(
    questions: Sequence[dict[str, Any]],
    asked_prompts: Sequence[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Drop questions that near-duplicate an ALREADY ASKED round.

    Deterministic backstop behind the prompt-side never-re-ask rule: the
    discovery analyst is an independent LLM call and can rephrase an
    intake question ("Auf welchen KI-Markt ...?" -> "Welchen KI-Markt
    sollen wir ...?"). Token-set overlap (Jaccard >=
    :data:`_DUPLICATE_JACCARD` or containment >=
    :data:`_DUPLICATE_CONTAINMENT` of the smaller set) marks the
    rephrase as the same question.

    Returns:
        ``(kept, dropped_prompts)`` — the caller narrates every dropped
        prompt (No Silent Fallbacks) and proceeds without an interrupt
        when nothing is left to ask.
    """
    asked = [
        tokens
        for prompt in asked_prompts
        if (tokens := _question_tokens(prompt))
    ]
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for question in questions:
        prompt = str(question.get("prompt", ""))
        tokens = _question_tokens(prompt)
        duplicate = False
        for prior in asked:
            if not tokens or not prior:
                continue
            overlap = len(tokens & prior)
            jaccard = overlap / len(tokens | prior)
            containment = overlap / min(len(tokens), len(prior))
            if (
                jaccard >= _DUPLICATE_JACCARD
                or containment >= _DUPLICATE_CONTAINMENT
            ):
                duplicate = True
                break
        if duplicate:
            dropped.append(prompt)
        else:
            kept.append(question)
    return kept, dropped


def intake_questions(
    profile: AssignmentProfile | None,
) -> list[dict[str, Any]]:
    """The ask-user-first questions from the intake profile (sanitized)."""
    if profile is None:
        return []
    return sanitize_questions(profile.clarification_questions)


def blocking_questions(
    discovery: DiscoveryResult | None,
) -> list[dict[str, Any]]:
    """Post-discovery questions — ONLY blocking gaps ask (§4 Phase 2)."""
    if discovery is None:
        return []
    return sanitize_questions(discovery.questions_for_user)
