""" Tests for source tiering and quality scoring strategies."""

import pytest


class TestSourceTiering:

    def test_source_tier_primary(self, tiering):
        assert tiering.tier_for_url("https://www.bundestag.de/dokumente") == "primary"

    def test_source_tier_stakeholder(self, tiering):
        assert tiering.tier_for_url("https://www.kzbv.de/presse") == "stakeholder"

    def test_source_tier_low(self, tiering):
        assert tiering.tier_for_url("https://www.tiktok.com/@foo/video/123") == "low"

    @pytest.mark.parametrize("url", [
        "https://aitoolsone.com/ai-events",
        "https://aipressroom.com/events-calendar",
        "https://www.tldl.io/blog/ai-news-updates-2026",
        "https://toolscompare.ai/news/may-2026",
    ])
    def test_ai_aggregator_domains_are_low_quality(self, tiering, url):
        assert tiering.tier_for_url(url) == "low"

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
        "https://www.sec.gov/Archives/edgar/data/1318605/000095017026000001/tsla-20251231.htm",
        "https://www.un.org/global-dialogue-ai-governance/report.pdf",
        "https://www.destatis.de/DE/Themen/Wirtschaft/Konjunkturindikatoren/_inhalt.html",
        "https://hai.stanford.edu/ai-index/2026-ai-index-report",
        "https://openai.com/index/gpt-5-5-instant/",
        "https://www.anthropic.com/news/anthropic-raises-30-billion-series-g-funding-380-billion-post-money-valuation",
        "https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/",
        "https://newsroom.ibm.com/2026-05-05-think-2026-ibm-delivers-the-blueprint-for-the-ai-operating-model-as-the-ai-divide-widens",
        "https://investor.atmeta.com/investor-news/press-release-details/2026/Meta-Reports-First-Quarter-2026-Results/default.aspx",
        "https://ir.tesla.com/#quarterly-disclosure",
    ])
    def test_phase13_new_primary_sources(self, tiering, url):
        assert tiering.tier_for_url(url) == "primary"

    @pytest.mark.parametrize(("url", "reason"), [
        (
            "https://www.sec.gov/ixviewer/doc/action",
            "matched_regulator_domain",
        ),
        (
            "https://openai.com/index/gpt-5-5-instant/",
            "matched_official_company_domain",
        ),
        (
            "https://hai.stanford.edu/ai-index/2026-ai-index-report",
            "matched_academic_institution_domain",
        ),
    ])
    def test_source_tier_explains_primary_subcategories(self, tiering, url, reason):
        explanation = tiering.explain_url(url)

        assert explanation["tier"] == "primary"
        assert explanation["tier_reason"] == reason

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
        "https://www.technologyreview.com/2026/01/23/1131559/americas-coming-war-over-ai-regulation",
        "https://spectrum.ieee.org/state-of-ai-index-2026",
        "https://www.theverge.com/ai-artificial-intelligence",
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

    def test_risk_score_accumulates_topic_neutral_signals(self, risk_scorer):
        # No German-policy bonus anymore: the score reflects only the
        # topic-neutral signals -- recency ("aktuell"), numeric ("Mrd Euro"),
        # and normative phrasing ("Sollen") -- which sum to 3.
        q = "Sollen Leistungen privatisiert werden, welche Kosten in Mrd Euro und welche Gesetzeslage gilt aktuell?"
        assert risk_scorer.score(q) == 3

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

    def test_claim_check_question_uses_focused_aspects(self, risk_scorer):
        aspects = risk_scorer.derive_required_aspects(
            "Stimmt die Aussage, dass Meta 2026 weniger in KI investiert?",
            "news",
        )
        joined = " ".join(aspects).lower()

        assert risk_scorer.infer_answer_contract(
            "Stimmt die Aussage, dass Meta 2026 weniger in KI investiert?"
        ) == "claim_check"
        assert "gegenbelege" in joined
        assert "positionen zentraler akteure" not in joined

    def test_data_extraction_question_uses_financial_aspects(self, risk_scorer):
        aspects = risk_scorer.derive_required_aspects(
            "Welche Umsatz- und CapEx-Wachstumsraten meldete Tesla 2026?",
            "news",
        )
        joined = " ".join(aspects).lower()

        assert risk_scorer.infer_answer_contract(
            "Welche Umsatz- und CapEx-Wachstumsraten meldete Tesla 2026?"
        ) == "data_extraction"
        assert "kennzahl" in joined
        assert "primaerquelle" in joined
        assert "richtung der laufenden diskussion" not in joined
