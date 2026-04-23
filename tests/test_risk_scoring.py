"""Regression tests for DEEP aspect derivation and coverage heuristics."""

from __future__ import annotations

from inqtrix.report_profiles import ReportProfile


def test_deep_profile_adds_review_perspectives(risk_scorer):
    aspects = risk_scorer.derive_required_aspects(
        "Soll die GKV reformiert werden?",
        "general",
        report_profile=ReportProfile.DEEP,
    )

    assert "Stakeholder- und Betroffenenperspektiven" in aspects
    assert "Gegenargumente, Risiken und Limitationen" in aspects
    assert "Vergleich mit Alternativen oder Gegenmodellen" in aspects


def test_estimate_aspect_coverage_recognizes_deep_perspective_terms(risk_scorer):
    uncovered, coverage = risk_scorer.estimate_aspect_coverage(
        [
            "Stakeholder- und Betroffenenperspektiven",
            "Gegenargumente, Risiken und Limitationen",
        ],
        [
            "Patientenverbaende, Krankenkassen und Unternehmen vertreten unterschiedliche Positionen.",
            "Kritiker verweisen auf Risiken, Probleme und Unsicherheiten des Vorschlags.",
        ],
    )

    assert uncovered == []
    assert coverage == 1.0
