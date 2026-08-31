"""The result requirement composed at a plan gate.

Two origins, one value: library rules the user attached and the text
they typed. Both carry a marker, because the critic's
``instruction_violation`` finding is only useful when it can name which
requirement was broken.
"""

from __future__ import annotations

import pytest

from inqtrix.agents.report_requirement import (
    MAX_COMPOSED_REQUIREMENT_CHARS,
    compose_report_requirement,
    composed_requirement_is_oversized,
)


def test_rules_come_first_and_every_part_is_marked():
    composed = compose_report_requirement(
        free_text="  Zielgruppe: Laien.  ",
        rules=[("sprechzettel", "Gliedere in fuenf Punkte."), ("ton", "Sachlich.")],
    )
    assert composed == (
        "[Regel: sprechzettel]\n"
        "Gliedere in fuenf Punkte.\n"
        "[Ende Regel: sprechzettel]\n\n"
        "[Regel: ton]\n"
        "Sachlich.\n"
        "[Ende Regel: ton]\n\n"
        "[Freie Vorgabe]\n"
        "Zielgruppe: Laien."
    )


@pytest.mark.parametrize(
    ("free_text", "rules", "expected"),
    [
        ("", (), ""),
        ("   ", (("leer", "   "),), ""),
        ("Nur Text.", (), "[Freie Vorgabe]\nNur Text."),
        (
            "",
            (("nur_regel", "Nur Regel."),),
            "[Regel: nur_regel]\nNur Regel.\n[Ende Regel: nur_regel]",
        ),
    ],
)
def test_only_real_contributions_appear(free_text, rules, expected):
    assert compose_report_requirement(free_text=free_text, rules=rules) == expected


def test_the_ceiling_is_a_refusal_not_a_truncation():
    """Attached rules are whole documents and the requirement is repeated
    into every writing prompt of the run. Shortening one behind the
    user's back would change what they approved without saying so."""
    huge = "x" * (MAX_COMPOSED_REQUIREMENT_CHARS + 1)
    composed = compose_report_requirement(rules=[("gross", huge)])
    assert composed_requirement_is_oversized(composed)
    assert huge in composed, "nothing may be cut on the way"
