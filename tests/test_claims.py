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

    def test_select_answer_citations_uses_full_claim_source_list(self, consolidator):
        """The full source_urls list of a claim reaches the selection path as
        long as the URLs sit on distinct domains (no Phase 2 cap conflict)."""
        citations = [
            f"https://example-{letter}.com/report/{idx}"
            for idx, letter in enumerate("abcdef")
        ]
        consolidated = [
            {
                "signature": "s1",
                "claim_text": "Testclaim",
                "claim_type": "fact",
                "needs_primary": False,
                "support_count": 2,
                "contradict_count": 0,
                "status": "verified",
                "source_urls": citations,
            }
        ]

        selected = consolidator.select_answer_citations(
            consolidated,
            citations,
            max_items=6,
        )

        assert set(selected) == set(citations)
        assert len(selected) == 6

    def test_select_answer_citations_tier_bias(self, consolidator, tiering):
        """verified+mainstream ranks above unverified+primary; primary still
        wins among verified-tier ties."""
        # mainstream (ard.de) beats a unverified primary claim on bundestag.de
        primary_url = "https://www.bundestag.de/dokumente/primaer"
        mainstream_url = "https://www.tagesschau.de/inland/mainstream"
        primary_url_verified = "https://www.bundesregierung.de/breg-de/primary-v"

        consolidated = [
            {
                "signature": "unv_primary",
                "claim_text": "Unverifizierter primaerer Claim",
                "claim_type": "fact",
                "needs_primary": True,
                "support_count": 1,
                "contradict_count": 0,
                "status": "unverified",
                "source_urls": [primary_url],
            },
            {
                "signature": "ver_mainstream",
                "claim_text": "Verifizierter Mainstream Claim",
                "claim_type": "fact",
                "needs_primary": False,
                "support_count": 2,
                "contradict_count": 0,
                "status": "verified",
                "source_urls": [mainstream_url],
            },
            {
                "signature": "ver_primary",
                "claim_text": "Verifizierter primaerer Claim",
                "claim_type": "fact",
                "needs_primary": True,
                "support_count": 2,
                "contradict_count": 0,
                "status": "verified",
                "source_urls": [primary_url_verified],
            },
        ]

        selected = consolidator.select_answer_citations(
            consolidated,
            [primary_url, mainstream_url, primary_url_verified],
            max_items=3,
            source_tiering=tiering,
        )

        # verified+primary comes first, verified+mainstream second, then the
        # unverified fallback pass picks up the primary unverified URL.
        assert selected[0] == primary_url_verified
        assert selected[1] == mainstream_url
        assert primary_url in selected

    def test_select_answer_citations_domain_diversity(self, consolidator, tiering):
        """Phase 1 picks one URL per domain before adding a second URL from a
        domain already represented. Phase 2 fills remaining slots with
        additional URLs from seen domains in claim-score order."""
        domain_a_urls = [
            "https://www.bundesregierung.de/breg-de/a1",
            "https://www.bundesregierung.de/breg-de/a2",
        ]
        domain_a_urls_2 = ["https://www.bundesregierung.de/breg-de/a3"]
        domain_b_url = "https://www.tagesschau.de/inland/b1"
        domain_c_url = "https://www.bundestag.de/dokumente/c1"

        consolidated = [
            {
                "signature": "s1",
                "claim_text": "Claim A1 (verified, primary)",
                "claim_type": "fact",
                "needs_primary": False,
                "support_count": 2,
                "contradict_count": 0,
                "status": "verified",
                "source_urls": list(domain_a_urls),
            },
            {
                "signature": "s2",
                "claim_text": "Claim A2 (verified, primary, same domain)",
                "claim_type": "fact",
                "needs_primary": False,
                "support_count": 2,
                "contradict_count": 0,
                "status": "verified",
                "source_urls": list(domain_a_urls_2),
            },
            {
                "signature": "s3",
                "claim_text": "Claim B (verified, mainstream)",
                "claim_type": "fact",
                "needs_primary": False,
                "support_count": 2,
                "contradict_count": 0,
                "status": "verified",
                "source_urls": [domain_b_url],
            },
            {
                "signature": "s4",
                "claim_text": "Claim C (unverified, primary)",
                "claim_type": "fact",
                "needs_primary": False,
                "support_count": 1,
                "contradict_count": 0,
                "status": "unverified",
                "source_urls": [domain_c_url],
            },
        ]

        all_urls = (
            domain_a_urls + domain_a_urls_2 + [domain_b_url, domain_c_url]
        )

        selected = consolidator.select_answer_citations(
            consolidated,
            all_urls,
            max_items=10,
            source_tiering=tiering,
        )

        # Phase 1 — one URL per domain (3 unique domains available).
        assert selected[0] == domain_a_urls[0]
        assert selected[1] == domain_b_url
        assert selected[2] == domain_c_url

        # Phase 2 — up to _PHASE2_MAX_PER_DOMAIN additional URLs per seen
        # domain in claim-score order: claim s1 adds a2, claim s2 adds a3;
        # domain A now at the 3-URL cap.
        assert selected[3] == domain_a_urls[1]
        assert selected[4] == domain_a_urls_2[0]
        assert len(selected) == 5

    def test_select_answer_citations_phase2_domain_cap(
        self, consolidator, tiering
    ):
        """Phase 2 honours the per-domain cap (_PHASE2_MAX_PER_DOMAIN=3).
        A domain that monopolises the claim pool cannot fill more than three
        slots even when max_items still has room."""
        domain_a_urls = [
            f"https://www.bundesregierung.de/breg-de/a{i}" for i in range(1, 7)
        ]
        domain_b_url = "https://www.tagesschau.de/inland/b1"

        consolidated = [
            {
                "signature": f"s{idx}",
                "claim_text": f"Claim A{idx} (verified, primary)",
                "claim_type": "fact",
                "needs_primary": False,
                "support_count": 2,
                "contradict_count": 0,
                "status": "verified",
                "source_urls": [url],
            }
            for idx, url in enumerate(domain_a_urls, start=1)
        ] + [
            {
                "signature": "sB",
                "claim_text": "Claim B (verified, mainstream)",
                "claim_type": "fact",
                "needs_primary": False,
                "support_count": 2,
                "contradict_count": 0,
                "status": "verified",
                "source_urls": [domain_b_url],
            }
        ]

        selected = consolidator.select_answer_citations(
            consolidated,
            domain_a_urls + [domain_b_url],
            max_items=10,
            source_tiering=tiering,
        )

        # Phase 1 places one URL per domain (A, B) — 2 slots.
        # Phase 2 tops up from domain A until _PHASE2_MAX_PER_DOMAIN = 3.
        # Remaining A URLs must be skipped even though max_items=10 has room.
        from inqtrix.strategies._claim_consolidation import (
            _PHASE2_MAX_PER_DOMAIN,
        )

        a_count = sum(1 for u in selected if "bundesregierung.de" in u)
        assert a_count == _PHASE2_MAX_PER_DOMAIN
        assert domain_b_url in selected
        assert len(selected) == _PHASE2_MAX_PER_DOMAIN + 1

        # Claim-C's URL must not be duplicated; list has no repeats.
        assert len(selected) == len(set(selected))

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

    def test_negated_only_evidence_does_not_verify_claim(self, consolidator):
        ledger = [
            {
                "signature": "privatisierung",
                "claim_text": "Die Bundesregierung plant keine Privatisierung von Zahnleistungen.",
                "claim_type": "fact",
                "polarity": "negated",
                "needs_primary": False,
                "source_urls": ["https://www.bundesregierung.de/breg-de/aktuelles/x"],
            },
        ]

        consolidated = consolidator.consolidate(ledger)

        assert len(consolidated) == 1
        assert consolidated[0]["support_count"] == 0
        assert consolidated[0]["contradict_count"] == 1
        assert consolidated[0]["status"] == "unverified"


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
