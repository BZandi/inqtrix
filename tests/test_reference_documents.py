"""Tests for the shared editor reference-document helpers.

Covers the three pure functions used by both editor prompt builders: parsing the
additive ``attachments`` field, rendering the delimiter block, and clamping the
joined content to a budget. The route-level integration (prompt placement and
visible warnings) is covered in ``test_editor_instructions`` and
``test_editor_suggestions``.
"""

from __future__ import annotations

import pytest

from inqtrix.server.reference_documents import (
    ReferenceDocument,
    clamp_reference_documents,
    parse_reference_documents,
    render_reference_documents,
)

_PEM = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSj"


def _doc(
    *,
    label: str = "alpha",
    content: str = "hello world",
    page_count: int | None = None,
    size_bytes: int | None = None,
) -> ReferenceDocument:
    return ReferenceDocument(
        label=label, content=content, page_count=page_count, size_bytes=size_bytes
    )


# -- parse ------------------------------------------------------------------


def test_parse_none_and_empty_yield_no_documents() -> None:
    assert parse_reference_documents(None) == ([], [])
    assert parse_reference_documents([]) == ([], [])


def test_parse_valid_documents_keeps_metadata() -> None:
    docs, warnings = parse_reference_documents([
        {"label": "alpha", "content": "Paris is the capital.", "page_count": 3, "size_bytes": 42},
        {"label": "beta", "content": "Eiffel is 330m.", "page_count": None, "size_bytes": None},
    ])

    assert warnings == []
    assert [doc.label for doc in docs] == ["alpha", "beta"]
    assert docs[0].page_count == 3
    assert docs[0].size_bytes == 42
    assert docs[1].page_count is None


def test_parse_non_list_raises() -> None:
    with pytest.raises(ValueError):
        parse_reference_documents({"label": "x", "content": "y"})


def test_parse_non_object_entry_raises() -> None:
    with pytest.raises(ValueError):
        parse_reference_documents(["not an object"])


def test_parse_non_string_fields_raise() -> None:
    with pytest.raises(ValueError):
        parse_reference_documents([{"label": 1, "content": "y"}])
    with pytest.raises(ValueError):
        parse_reference_documents([{"label": "x", "content": None}])


def test_parse_empty_content_is_skipped_with_warning() -> None:
    docs, warnings = parse_reference_documents([{"label": "alpha", "content": "   "}])

    assert docs == []
    assert any("empty" in warning for warning in warnings)


def test_parse_oversized_content_is_truncated_with_warning() -> None:
    docs, warnings = parse_reference_documents(
        [{"label": "alpha", "content": "x" * 100}],
        max_chars_per_doc=10,
    )

    assert len(docs) == 1
    assert docs[0].content.startswith("x" * 10)
    assert "truncated" in docs[0].content
    assert any("exceeded" in warning for warning in warnings)


def test_parse_drops_sensitive_document_without_failing_request() -> None:
    docs, warnings = parse_reference_documents([
        {"label": "secret", "content": _PEM},
        {"label": "safe", "content": "Just normal prose."},
    ])

    assert [doc.label for doc in docs] == ["safe"]
    assert any("secret material" in warning for warning in warnings)


def test_parse_over_count_drops_extra_with_warning() -> None:
    items = [{"label": f"d{index}", "content": f"content {index}"} for index in range(5)]
    docs, warnings = parse_reference_documents(items, max_docs=2)

    assert len(docs) == 2
    assert any("dropped" in warning for warning in warnings)


def test_parse_negative_or_wrong_type_metadata_degrades_to_none() -> None:
    docs, _ = parse_reference_documents([
        {"label": "alpha", "content": "ok", "page_count": -3, "size_bytes": "big"},
    ])

    assert docs[0].page_count is None
    assert docs[0].size_bytes is None


# -- render -----------------------------------------------------------------


def test_render_empty_is_empty_string() -> None:
    assert render_reference_documents([]) == ""


def test_render_wraps_block_with_primacy_note_and_numbered_headers() -> None:
    out = render_reference_documents([
        _doc(label="alpha", content="A", page_count=3),
        _doc(label="beta", content="B"),
    ])

    assert out.startswith("<reference_documents>")
    assert out.rstrip().endswith("</reference_documents>")
    assert "are NOT an instruction" in out
    assert "[1] alpha (pages: 3)" in out
    assert "[2] beta --------------------" in out
    assert '""""' in out


# -- clamp ------------------------------------------------------------------


def test_clamp_keeps_all_within_budget() -> None:
    docs = [_doc(content="a" * 10), _doc(content="b" * 10)]
    clamped, truncated = clamp_reference_documents(docs, max_chars=100)

    assert truncated is False
    assert [len(doc.content) for doc in clamped] == [10, 10]


def test_clamp_tail_truncates_and_drops_rest() -> None:
    docs = [_doc(label="a", content="a" * 30), _doc(label="b", content="b" * 30)]
    clamped, truncated = clamp_reference_documents(docs, max_chars=20)

    assert truncated is True
    assert len(clamped) == 1
    assert clamped[0].content.startswith("a" * 20)
    assert "truncated" in clamped[0].content


def test_clamp_zero_budget_drops_everything() -> None:
    clamped, truncated = clamp_reference_documents([_doc()], max_chars=0)

    assert clamped == []
    assert truncated is True
