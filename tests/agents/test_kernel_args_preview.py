"""The transcript line that says what a tool was asked to do.

A preview is what a user READS while a tool runs. It used to be built by
serialising the arguments and clipping the result, so every tool except
``web_instant`` showed a JSON dump cut mid-token — rubble for the reader
and unparseable for the surface, which then rendered it verbatim.
"""

from inqtrix.agents.kernel.algorithm import (
    _ARGS_PREVIEW_LIMIT,
    _PROSE_PREVIEW_LIMIT,
    _args_preview,
)


def test_delegation_assignment_previews_as_prose() -> None:
    """The regression: a delegation is described, not dumped."""
    preview = _args_preview(
        {
            "assignment": "Erstelle ein belegtes Memo zum AI Act.",
            "depth": "deep",
        }
    )
    assert preview == "Erstelle ein belegtes Memo zum AI Act."
    assert not preview.startswith("{")


def test_a_long_assignment_is_cut_as_a_sentence_not_as_json() -> None:
    """Over the cap the line stays prose AND stays visibly cut."""
    preview = _args_preview({"assignment": "A" * 900, "depth": "deep"})
    assert preview.startswith("A")
    assert preview.endswith("…")
    assert len(preview) == _PROSE_PREVIEW_LIMIT + 1


def test_a_bare_query_still_previews_as_prose() -> None:
    """The one case that always worked must keep working."""
    assert _args_preview({"query": "EU AI Act Artikel 50"}) == (
        "EU AI Act Artikel 50"
    )


def test_query_wins_over_a_second_argument() -> None:
    """A scoped knowledge search is still described by its question."""
    preview = _args_preview(
        {"query": "Pflichten der Betreiber", "collection_ids": ["c1", "c2"]}
    )
    assert preview == "Pflichten der Betreiber"


def test_an_id_only_call_previews_as_json_but_stays_whole() -> None:
    """No intent argument: JSON is honest, and short enough to survive."""
    preview = _args_preview({"artifact_id": "art_123"})
    assert preview == '{"artifact_id": "art_123"}'
    assert not preview.endswith("…")


def test_a_document_body_never_reaches_the_prose_lane() -> None:
    """``write_canvas`` carries a whole document; it must stay clipped
    tight, so the event log never becomes a second copy of an artifact."""
    preview = _args_preview(
        {"title": "Memo", "content_markdown": "x" * 40000}
    )
    assert preview.endswith("…")
    assert len(preview) == _ARGS_PREVIEW_LIMIT + 1
