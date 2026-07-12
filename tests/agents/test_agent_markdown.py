"""Markdown hygiene at agent-generation boundaries."""

from __future__ import annotations

from inqtrix.agents.markdown import normalize_agent_markdown
from inqtrix.agents.scheduler import task_result_summary


def test_currency_dollars_are_escaped_without_creating_cross_prose_math() -> None:
    markdown = (
        "Gartner meldet $1.5T, waehrend Statista US-$244 Mrd. ausweist."
    )

    assert normalize_agent_markdown(markdown) == (
        r"Gartner meldet \$1.5T, waehrend Statista US-\$244 Mrd. ausweist."
    )
    assert normalize_agent_markdown(
        "Szenario $1.5T + Forecast, Vergleich US-$244B."
    ) == r"Szenario \$1.5T + Forecast, Vergleich US-\$244B."


def test_genuine_inline_and_block_math_are_preserved() -> None:
    markdown = (
        "Die Variable $x$, $O(n)$, $x, y$, $1.5$ und die Formel "
        "$x + y = z$ bleiben.\n\n"
        "$$\nE = mc^2\n$$\n"
    )

    assert normalize_agent_markdown(markdown) == markdown


def test_code_spans_and_fences_are_preserved_while_prose_is_normalized() -> None:
    markdown = (
        "`price = '$1.5T'` und ausserhalb $2T.\n\n"
        "```python\nprice = '$3T'\n```\n"
    )
    assert normalize_agent_markdown("Offen ` und Preis $4T") == (
        r"Offen ` und Preis \$4T"
    )

    assert normalize_agent_markdown(markdown) == (
        "`price = '$1.5T'` und ausserhalb \\$2T.\n\n"
        "```python\nprice = '$3T'\n```\n"
    )


def test_normalization_is_idempotent_and_keeps_existing_escapes() -> None:
    markdown = r"Umsatz: \$1.5T; Formel: $x-y$."
    normalized = normalize_agent_markdown(markdown)

    assert normalized == markdown
    assert normalize_agent_markdown(normalized) == normalized


def test_url_query_dollars_are_byte_preserved() -> None:
    markdown = (
        "Quelle: https://example.com/report?q=$20&formula=$x$ "
        "und Marktwert $20B."
    )

    assert normalize_agent_markdown(markdown) == (
        "Quelle: https://example.com/report?q=$20&formula=$x$ "
        r"und Marktwert \$20B."
    )


def test_task_result_summary_uses_the_same_currency_boundary() -> None:
    assert task_result_summary("Marktwert US-$1.5T; Formel $x$.") == (
        r"Marktwert US-\$1.5T; Formel $x$."
    )
