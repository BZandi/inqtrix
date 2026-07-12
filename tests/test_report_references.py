"""Report-reference de-duplication for the knowledge citation feature.

The default knowledge deployment cites chunks via
``inqtrix://documents/{id}#chunk-{n}``; ``normalize_url`` strips the
fragment, so de-duping by normalized URL would silently collapse every
cited chunk of one document into a single reference (dead ``[K#]`` links,
missing excerpts). These tests pin the (document_id, chunk_index) identity
for knowledge references while keeping URL-dedup for web references.
"""

from __future__ import annotations

from inqtrix.result import _report_references_from_state


class _Tiering:
    def tier_for_url(self, url: str) -> str:  # noqa: D401 - test stub
        return "unknown"


def test_same_document_chunks_survive_url_dedup() -> None:
    references = [
        {
            "label": "K1",
            "url": "inqtrix://documents/doc1#chunk-0",
            "tier": "primary",
            "document_id": "doc1",
            "chunk_index": 0,
            "excerpt": "Chunk null.",
        },
        {
            "label": "K2",
            "url": "inqtrix://documents/doc1#chunk-5",
            "tier": "primary",
            "document_id": "doc1",
            "chunk_index": 5,
            "excerpt": "Chunk fuenf.",
        },
    ]

    out = _report_references_from_state(
        references, tier_by_url={}, tiering=_Tiering()
    )

    assert [reference.label for reference in out] == ["K1", "K2"]
    assert out[0].excerpt == "Chunk null."
    assert out[1].chunk_index == 5


def test_same_document_same_chunk_is_still_deduped() -> None:
    references = [
        {"label": "K1", "url": "inqtrix://documents/doc1#chunk-0",
         "document_id": "doc1", "chunk_index": 0},
        {"label": "K2", "url": "inqtrix://documents/doc1#chunk-0",
         "document_id": "doc1", "chunk_index": 0},
    ]

    out = _report_references_from_state(
        references, tier_by_url={}, tiering=_Tiering()
    )

    assert [reference.label for reference in out] == ["K1"]


def test_web_references_still_dedup_by_url() -> None:
    references = [
        {"label": "E1", "url": "https://example.com/a", "tier": "unknown"},
        {"label": "E2", "url": "https://example.com/a", "tier": "unknown"},
    ]

    out = _report_references_from_state(
        references, tier_by_url={}, tiering=_Tiering()
    )

    assert [reference.label for reference in out] == ["E1"]
    assert out[0].document_id is None


def test_internal_reference_without_url_keeps_identity_and_support_fields() -> None:
    out = _report_references_from_state(
        [
            {
                "label": "K1",
                "document_id": "doc1",
                "chunk_index": 4,
                "source_text": "Exact internal passage",
            }
        ],
        tier_by_url={},
        tiering=_Tiering(),
    )

    assert len(out) == 1
    assert out[0].url == "inqtrix://documents/doc1#chunk-4"
    assert out[0].document_id == "doc1"
    assert out[0].chunk_index == 4
    assert out[0].source_text == "Exact internal passage"
