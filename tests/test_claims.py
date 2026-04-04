""" Tests for claim consolidation, quality metrics, and relevance filtering."""

import pytest


class TestClaimLedger:

    def test_consolidation_and_metrics(self, consolidator):
        ledger = [
            {
                "signature": "no_plan",
                "claim_text": "Die Bundesregierung plant keine Privatisierung von Zahnleistungen.",
                "claim_type": "fact",
                "polarity": "affirmed",
                "needs_primary": False,
                "source_urls": [
                    "https://www.bundesgesundheitsministerium.de/presse/pm",
                    "https://www.aerzteblatt.de/news/x",
                ],
            },
            {
                "signature": "no_plan",
                "claim_text": "Es gibt keine Regierungsplaene zur Privatisierung.",
                "claim_type": "fact",
                "polarity": "affirmed",
                "needs_primary": False,
                "source_urls": ["https://www.bundesregierung.de/breg-de/aktuelles/x"],
            },
            {
                "signature": "savings_18",
                "claim_text": "Eine Auslagerung spare 18 Milliarden Euro pro Jahr.",
                "claim_type": "actor_claim",
                "polarity": "affirmed",
                "needs_primary": True,
                "source_urls": ["https://www.kzbv.de/presse/x"],
            },
            {
                "signature": "majority",
                "claim_text": "Der Vorschlag hat eine parlamentarische Mehrheit.",
                "claim_type": "forecast",
                "polarity": "affirmed",
                "needs_primary": False,
                "source_urls": ["https://www.handelsblatt.com/politik/x"],
            },
            {
                "signature": "majority",
                "claim_text": "Der Vorschlag hat keine parlamentarische Mehrheit.",
                "claim_type": "forecast",
                "polarity": "negated",
                "needs_primary": False,
                "source_urls": ["https://www.tagesspiegel.de/politik/y"],
            },
        ]

        consolidated = consolidator.consolidate(ledger)
        by_sig = {c["signature"]: c for c in consolidated}

        assert by_sig["no_plan"]["status"] == "verified"
        assert by_sig["savings_18"]["status"] == "unverified"
        assert by_sig["majority"]["status"] == "contested"

        counts, score, np_total, np_verified = consolidator.quality_metrics(consolidated)
        assert counts == {"verified": 1, "contested": 1, "unverified": 1}
        assert score == pytest.approx(0.5, abs=1e-3)
        assert np_total == 0
        assert np_verified == 0

    def test_claim_prompt_view_contains_status(self, consolidator):
        consolidated = [
            {
                "signature": "s1",
                "claim_text": "Testclaim",
                "claim_type": "fact",
                "needs_primary": False,
                "support_count": 2,
                "contradict_count": 0,
                "status": "verified",
                "source_urls": ["https://example.com/a"],
            }
        ]
        view = consolidator.claims_prompt_view(consolidated, max_items=1)
        assert "status=verified" in view
        assert "Testclaim" in view

    def test_needs_primary_counts_only_facts(self, consolidator):
        ledger = [
            {
                "signature": "np_fact",
                "claim_text": "Die GKV-Ausgaben betragen 18 Milliarden Euro.",
                "claim_type": "fact",
                "polarity": "affirmed",
                "needs_primary": True,
                "source_urls": ["https://www.bundestag.de/dokumente/x"],
            },
            {
                "signature": "np_actor",
                "claim_text": "Ein Verband fordert Einsparungen von 18 Milliarden Euro.",
                "claim_type": "actor_claim",
                "polarity": "affirmed",
                "needs_primary": True,
                "source_urls": ["https://www.kzbv.de/presse/x"],
            },
        ]
        consolidated = consolidator.consolidate(ledger)
        counts, score, np_total, np_verified = consolidator.quality_metrics(consolidated)
        assert counts["verified"] == 1
        assert score > 0.0
        assert np_total == 1
        assert np_verified == 1


class TestClaimRelevance:

    def test_focus_stems_detect_core_topics(self, consolidator):
        q = "Sollen zahnärztliche Leistungen zukünftig privatisiert werden? In welche Richtung gehen die Diskussionen?"
        stems = consolidator.focus_stems_from_question(q)
        assert any(s.startswith("zahn") for s in stems)
        assert any(s.startswith("priv") for s in stems)

    def test_claim_matches_focus_stems(self, consolidator):
        q = "Sollen zahnärztliche Leistungen zukünftig privatisiert werden?"
        stems = consolidator.focus_stems_from_question(q)

        assert consolidator.claim_matches_focus_stems(
            "Die Bundesregierung plant keine Privatisierung von Zahnleistungen.",
            stems,
        )
        assert not consolidator.claim_matches_focus_stems(
            "Nina Warken ist Mitglied der CDU.",
            stems,
        )
