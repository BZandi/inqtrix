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

    # ------------------------------------------------------------------
    # Phase 13: domain-list expansion regressions
    #
    # The lists below were extended with domains that previously fell into
    # the "unknown" tier even though they are clearly government, mainstream
    # press, or stakeholder organisations. These tests defend the new
    # mappings against accidental removal.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("url", [
        "https://dip.bundestag.de/parlamentsmaterial/12345",
        "https://www.das-parlament.de/themen/2026",
        "https://www.bundesrechnungshof.de/de/veroeffentlichungen",
        "https://www.abgeordnetenwatch.de/profile/some-mp",
    ])
    def test_phase13_new_primary_sources(self, tiering, url):
        assert tiering.tier_for_url(url) == "primary"

    @pytest.mark.parametrize("url", [
        "https://www.focus.de/finanzen/krankenkassen-reform",
        "https://taz.de/Gesundheitspolitik/!12345/",
        "https://www.aerztezeitung.de/Politik/article",
        "https://www.deutsche-apotheker-zeitung.de/news/abc",
        "https://www.dasinvestment.com/krankenkassen",
        "https://www.finanztip.de/krankenversicherung/",
        "https://de.statista.com/statistik/daten/studie/gkv",
        "https://www.morningstar.com/stocks/best-ai-stocks",
        "https://www.zacks.com/featured-articles/200/",
    ])
    def test_phase13_new_mainstream_sources(self, tiering, url):
        assert tiering.tier_for_url(url) == "mainstream"

    @pytest.mark.parametrize("url", [
        "https://www.kbv.de/html/positionen.php",
        "https://www.pkv.de/themen/news",
        "https://www.deutscher-pflegerat.de/aktuelles",
        "https://www.physio-deutschland.de/news",
        "https://www.marburger-bund.de/positionen",
        "https://www.dkgev.de/",
        "https://www.arbeitgeber.de/positionen/krankenversicherung",
        "https://www.csu.de/aktuelles/",
    ])
    def test_phase13_new_stakeholder_sources(self, tiering, url):
        assert tiering.tier_for_url(url) == "stakeholder"

    def test_phase13_unknown_domain_still_unknown(self, tiering):
        """Domains we did NOT add must remain 'unknown' (regression guard)."""
        # A random hobby blog or generic content farm should not slip into
        # any tier just because of the expansion.
        assert tiering.tier_for_url(
            "https://random-hobbyblog-not-in-any-list.example/article",
        ) == "unknown"


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

    def test_quality_terms_keep_gkv_scope_health_specific(self, risk_scorer):
        terms = risk_scorer.quality_terms_for_question(
            "Soll ein Tempolimit eingefuehrt werden?",
            "news",
        )

        assert "gkv" not in terms

    def test_quality_site_queries_do_not_pollute_unrelated_policy_topics(self, risk_scorer):
        queries = risk_scorer.inject_quality_site_queries(
            ["Tempolimit Deutschland aktuelle Debatte"],
            search_lang="de",
            question="Soll ein Tempolimit eingefuehrt werden?",
            query_type="news",
            need_primary=True,
            need_mainstream=True,
            max_items=5,
        )

        site_queries = [query for query in queries if query.startswith("site:")]
        assert site_queries
        assert all("gkv" not in query.lower() for query in site_queries)
