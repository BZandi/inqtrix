"""P7 report-quality building blocks: web-excerpt grounding, the
deterministic evidence rank, and the tief-tier escalation trigger.

The moved citation helpers keep their behavior contract in
``test_synthesis_citations.py`` (imports via the synthesis re-exports —
the refactor guard); this module covers what P7 ADDED.
"""

from __future__ import annotations

from inqtrix.agents.report_quality import (
    rank_evidence,
    unverified_web_quotes,
    verify_quotes,
)


def _web_ref(label: str, url: str, excerpt: str = "", tasks: int = 1) -> dict:
    return {
        "label": label,
        "url": url,
        "excerpt": excerpt,
        "tasks": [f"t{i}" for i in range(tasks)],
    }


def _doc_ref(label: str, excerpt: str = "") -> dict:
    return {
        "label": label,
        "document_id": f"doc-{label}",
        "excerpt": excerpt,
        "tasks": ["t1"],
    }


class TestVerifyQuotesIncludesWebExcerpts:
    def test_quote_matching_a_web_excerpt_verifies(self):
        quote = "Die Compliance-Kosten steigen fuer kleine Anbieter deutlich"
        refs = [
            _web_ref("W1", "https://example.org/a", excerpt=f"Studie: {quote}."),
        ]
        checks = verify_quotes(f'Analyse: "{quote}" [W1].', refs)
        assert checks == [{"quote": quote, "verified": True}]

    def test_quote_without_any_stored_excerpt_stays_unverified(self):
        quote = "Diese Behauptung steht in keinem gespeicherten Auszug drin"
        refs = [
            _web_ref("W1", "https://example.org/a", excerpt="Anderer Inhalt."),
            _doc_ref("K1", excerpt="Interner Text ohne das Zitat."),
        ]
        checks = verify_quotes(f'Er sagte: "{quote}" [W1].', refs)
        assert checks == [{"quote": quote, "verified": False}]


class TestUnverifiedWebQuotes:
    def test_flags_only_unverified_quotes_in_web_cited_paragraphs(self):
        markdown = (
            'Erster Absatz: "Zitat aus dem Web ohne Auszugstreffer xxxx" [W1].'
            "\n\n"
            'Zweiter Absatz: "Internes Zitat ohne jeden Auszugstreffer" [K1].'
        )
        checks = [
            {"quote": "Zitat aus dem Web ohne Auszugstreffer xxxx", "verified": False},
            {"quote": "Internes Zitat ohne jeden Auszugstreffer", "verified": False},
        ]
        assert unverified_web_quotes(markdown, checks) == [
            "Zitat aus dem Web ohne Auszugstreffer xxxx"
        ]

    def test_verified_quotes_never_escalate(self):
        markdown = 'Absatz: "Sauber belegtes Zitat mit Auszugstreffer" [W1].'
        checks = [
            {"quote": "Sauber belegtes Zitat mit Auszugstreffer", "verified": True},
        ]
        assert unverified_web_quotes(markdown, checks) == []


class TestRankEvidence:
    def test_passthrough_at_or_below_budget(self):
        refs = [_web_ref("W1", "https://a.example"), _doc_ref("K1")]
        assert rank_evidence(refs, budget=2) is refs
        assert rank_evidence(refs, budget=0) is refs

    def test_caps_by_tier_corroboration_and_excerpt(self):
        refs = [
            _web_ref("W1", "https://low.example"),
            _doc_ref("K1", excerpt="Interner Auszug."),
            _web_ref(
                "W2",
                "https://corroborated.example",
                excerpt="Belegter Auszug.",
                tasks=3,
            ),
            _web_ref("W3", "https://plain.example"),
        ]
        tiers = {
            "https://low.example": "low",
            "https://corroborated.example": "mainstream",
            "https://plain.example": "unknown",
        }
        kept = rank_evidence(
            refs, budget=2, tier_for_url=lambda url: tiers[url]
        )
        # Internal doc (primary weight) and the corroborated excerpt win;
        # ledger order is preserved among the kept rows.
        assert [ref["label"] for ref in kept] == ["K1", "W2"]

    def test_never_mutates_or_reorders_the_ledger(self):
        refs = [
            _web_ref("W1", "https://a.example", excerpt="x", tasks=2),
            _web_ref("W2", "https://b.example"),
            _web_ref("W3", "https://c.example", excerpt="y"),
        ]
        before = [dict(ref) for ref in refs]
        kept = rank_evidence(refs, budget=2, tier_for_url=lambda url: "unknown")
        assert refs == before
        assert [ref["label"] for ref in kept] == ["W1", "W3"]
