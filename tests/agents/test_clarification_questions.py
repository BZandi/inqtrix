"""Sanitizer contract of structured clarification rounds (decision #8).

These tests secure the deterministic wire shape the clarify node stores:
positional ids (interrupt re-execution regenerates the SAME payload),
hard caps, and the loud degradation rules (a bad question never vanishes
silently — it either logs and degrades to free text or logs and drops).
"""

from __future__ import annotations

import logging

from inqtrix.agents.clarification import (
    MAX_OPTIONS_PER_QUESTION,
    MAX_QUESTIONS_PER_ROUND,
    build_clarification,
    sanitize_questions,
)
from inqtrix.agents.phase_models import (
    ClarificationOptionModel,
    ClarificationQuestionModel,
)


def _question(
    prompt: str = "Welcher Markt?",
    labels: tuple[str, ...] = ("Europa", "USA"),
    multi_select: bool = False,
) -> ClarificationQuestionModel:
    return ClarificationQuestionModel(
        prompt=prompt,
        options=[
            ClarificationOptionModel(label=label, description="")
            for label in labels
        ],
        multi_select=multi_select,
    )


def test_sanitize_assigns_positional_ids_deterministically():
    models = [_question(), _question("Welche Aspekte?", ("A", "B", "C"))]
    first = sanitize_questions(models)
    second = sanitize_questions(models)
    assert first == second
    assert [q["id"] for q in first] == ["q1", "q2"]
    assert [o["id"] for o in first[1]["options"]] == [
        "q2_o1",
        "q2_o2",
        "q2_o3",
    ]


def test_sanitize_caps_questions_and_options(caplog):
    handler_target = logging.getLogger("inqtrix")
    handler_target.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="inqtrix"):
            questions = sanitize_questions(
                [
                    _question(labels=("A", "B", "C", "D", "E", "F")),
                    _question("F2?"),
                    _question("F3?"),
                    _question("F4?"),
                ]
            )
    finally:
        handler_target.removeHandler(caplog.handler)
    assert len(questions) == MAX_QUESTIONS_PER_ROUND
    assert len(questions[0]["options"]) == MAX_OPTIONS_PER_QUESTION
    assert any("gekappt" in record.message for record in caplog.records)


def test_sanitize_degrades_single_option_to_free_text(caplog):
    handler_target = logging.getLogger("inqtrix")
    handler_target.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="inqtrix"):
            questions = sanitize_questions([_question(labels=("Europa",))])
    finally:
        handler_target.removeHandler(caplog.handler)
    # The question survives WITHOUT chips (free text always works).
    assert len(questions) == 1
    assert questions[0]["options"] == []
    assert questions[0]["multi_select"] is False
    assert any("Freitext" in record.message for record in caplog.records)


def test_sanitize_drops_empty_prompt_loudly(caplog):
    handler_target = logging.getLogger("inqtrix")
    handler_target.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="inqtrix"):
            questions = sanitize_questions([_question(prompt="   ")])
    finally:
        handler_target.removeHandler(caplog.handler)
    assert questions == []
    assert any("verworfen" in record.message for record in caplog.records)


def test_sanitize_collapses_duplicate_labels_and_caps_length():
    long_label = "X" * 200
    questions = sanitize_questions(
        [_question(labels=("Europa", "europa", long_label))]
    )
    options = questions[0]["options"]
    assert [option["label"] for option in options][0] == "Europa"
    assert len(options) == 2
    assert len(options[1]["label"]) == 60


def test_build_clarification_mirrors_single_question_into_legacy_options():
    questions = sanitize_questions([_question()])
    record = build_clarification("run_1", questions=questions)
    assert record.question == "Welcher Markt?"
    assert record.questions == tuple(questions)
    assert list(record.options) == questions[0]["options"]


def test_build_clarification_multi_question_keeps_legacy_options_empty():
    questions = sanitize_questions([_question(), _question("F2?")])
    record = build_clarification("run_1", questions=questions)
    assert record.question == "Welcher Markt? F2?"
    assert record.options == ()
    assert len(record.questions) == 2


def test_filter_repeated_questions_drops_rephrased_duplicates():
    """The observed live bug: the discovery analyst rephrased the intake
    round ('Auf welchen KI-Markt ...?' -> 'Welchen KI-Markt sollen wir
    ...?'). The token-overlap backstop must catch the rephrase while a
    genuinely new question survives."""
    from inqtrix.agents.clarification import filter_repeated_questions

    asked = [
        "Auf welchen KI-Markt soll sich die Analyse primaer beziehen?",
        "Auf welche Region soll sich die Marktanalyse beziehen?",
        "Welchen Zeithorizont soll die Analyse abdecken?",
    ]
    candidates = sanitize_questions(
        [
            _question("Welchen KI-Markt sollen wir thematisch analysieren?"),
            _question(
                "Auf welche Region soll sich die Marktanalyse primaer "
                "beziehen?"
            ),
            _question("Welches Budget steht fuer die Studie zur Verfuegung?"),
        ]
    )
    kept, dropped = filter_repeated_questions(candidates, asked)
    assert [q["prompt"] for q in kept] == [
        "Welches Budget steht fuer die Studie zur Verfuegung?"
    ]
    assert len(dropped) == 2


def test_filter_repeated_questions_keeps_everything_without_history():
    from inqtrix.agents.clarification import filter_repeated_questions

    candidates = sanitize_questions([_question("Welcher Markt?")])
    kept, dropped = filter_repeated_questions(candidates, [])
    assert kept == candidates
    assert dropped == []
