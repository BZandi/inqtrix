""" Tests for source tiering and quality scoring strategies."""

import pytest


class TestSourceTiering:

    def test_source_tier_primary(self, tiering):
        assert tiering.tier_for_url("https://www.bundestag.de/dokumente") == "primary"

    def test_source_tier_stakeholder(self, tiering):
        assert tiering.tier_for_url("https://www.kzbv.de/presse") == "stakeholder"

    def test_source_tier_low(self, tiering):
        assert tiering.tier_for_url("https://www.tiktok.com/@foo/video/123") == "low"

    def test_source_quality_score_weighted(self, tiering):
        counts, score = tiering.quality_from_urls([
            "https://www.bundestag.de/x",
            "https://www.aerzteblatt.de/y",
            "https://www.kzbv.de/z",
        ])
        assert counts["primary"] == 1
        assert counts["mainstream"] == 1
        assert counts["stakeholder"] == 1
        assert 0.0 < score <= 1.0


class TestRiskScoring:

    def test_risk_score_high_for_policy_numeric_question(self, risk_scorer):
        q = "Sollen Leistungen privatisiert werden, welche Kosten in Mrd Euro und welche Gesetzeslage gilt aktuell?"
        assert risk_scorer.score(q) >= 4

    def test_required_aspects_for_policy_question(self, risk_scorer):
        q = "Sollen zahnärztliche Leistungen privatisiert werden? In welche Richtung gehen die Diskussionen?"
        aspects = risk_scorer.derive_required_aspects(q, "news")
        joined = " ".join(aspects).lower()
        assert "status quo" in joined
        assert "richtung" in joined
        assert "mehrheitslage" in joined

    def test_aspect_coverage(self, risk_scorer):
        aspects = ["Status quo mit konkretem Datum", "Positionen zentraler Akteure"]
        context = [
            "Stand 12. Februar 2026: Die Bundesregierung lehnt den Vorschlag ab.",
            "Positionen: SPD, CDU, Gruene und Verbaende widersprechen der Privatisierung.",
        ]
        uncovered, coverage = risk_scorer.estimate_aspect_coverage(aspects, context)
        assert uncovered == []
        assert coverage == 1.0
