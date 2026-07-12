"""Agent instant-evidence enrichment and synthesis digest contracts."""

from __future__ import annotations

from inqtrix.agents.evidence import (
    enrich_instant_evidence,
    evidence_digest,
    merge_evidence,
)
from inqtrix.agents.scheduler import TaskOutcome


def test_grounded_answer_associates_each_inline_url_with_its_statement() -> None:
    answer = (
        "Gartner beziffert die weltweiten KI-Ausgaben 2025 auf US-$1.5T "
        "([Gartner](https://example.com/gartner)).\n\n"
        "Statista grenzt den GenAI-Umsatz auf $244 Mrd. ein "
        "([Statista](https://example.com/statista))."
    )
    refs = enrich_instant_evidence(
        answer,
        [
            {
                "url": "https://example.com/gartner",
                "title": "Gartner",
                "excerpt": "",
            },
            {
                "url": "https://example.com/statista",
                "title": "Statista",
                "excerpt": "",
            },
        ],
    )

    assert "KI-Ausgaben 2025" in refs[0]["grounded_support"]
    assert "GenAI-Umsatz" not in refs[0]["grounded_support"]
    assert "GenAI-Umsatz" in refs[1]["grounded_support"]
    assert "KI-Ausgaben 2025" not in refs[1]["grounded_support"]
    assert refs[0]["grounded_support"].count("https://") == 0
    assert "US-$1.5T" in refs[0]["grounded_support"]


def test_empty_provider_snippet_still_yields_a_factual_digest() -> None:
    answer = (
        "Der Umsatz stieg 2025 um 29 Prozent "
        "([Marktbericht](https://example.com/report))."
    )
    enriched = enrich_instant_evidence(
        answer,
        [
            {
                "label": "W1",
                "url": "https://example.com/report",
                "title": "Marktbericht",
                "excerpt": "",
            }
        ],
    )

    digest = evidence_digest(enriched)
    assert digest.startswith("[W1]")
    assert "Umsatz stieg 2025 um 29 Prozent" in digest
    assert digest != "[W1] Marktbericht"


def test_support_is_bounded_and_never_invented_for_an_uncited_url() -> None:
    cited_url = "https://example.com/cited"
    refs = enrich_instant_evidence(
        "A" * 500 + f" ([Quelle]({cited_url})).",
        [
            {"url": cited_url, "title": "Cited"},
            {"url": "https://example.com/uncited", "title": "Uncited"},
        ],
        support_chars=80,
    )

    assert 0 < len(refs[0]["grounded_support"]) <= 80
    assert "grounded_support" not in refs[1]


def test_first_paragraph_keeps_its_first_character_and_urls_match_exactly() -> None:
    answer = (
        "Gartner nennt einen Wert "
        "([Lang](https://example.com/report-extended))."
    )
    refs = enrich_instant_evidence(
        answer,
        [
            {"url": "https://example.com/report", "title": "Short"},
            {
                "url": "https://example.com/report-extended",
                "title": "Long",
            },
        ],
    )

    assert "grounded_support" not in refs[0]
    assert refs[1]["grounded_support"].startswith("Gartner")


def test_web_identity_uses_the_platform_url_normalizer() -> None:
    references, _claims = merge_evidence(
        {
            "a": TaskOutcome(
                status="completed",
                evidence=[
                    {"url": "https://example.com/report?utm_source=one"}
                ],
            ),
            "b": TaskOutcome(
                status="completed",
                evidence=[
                    {"url": "https://example.com/report?utm_source=two"}
                ],
            ),
        }
    )

    assert len(references) == 1
    assert references[0]["tasks"] == ["a", "b"]


def test_digest_prefers_source_content_then_grounded_support_then_title() -> None:
    digest = evidence_digest(
        [
            {
                "label": "W1",
                "excerpt": "provider excerpt",
                "source_text": "canonical source text",
                "grounded_support": "provider answer support",
                "title": "title one",
            },
            {
                "label": "W2",
                "excerpt": "",
                "source_text": "canonical source text",
                "grounded_support": "provider answer support",
                "title": "title two",
            },
            {
                "label": "W3",
                "excerpt": "",
                "source_text": "",
                "grounded_support": "provider answer support",
                "title": "title three",
            },
            {"label": "W4", "title": "title four"},
        ]
    )

    assert digest.splitlines() == [
        "[W1] provider excerpt",
        "[W2] canonical source text",
        (
            "[W3] Geerdeter Antwortkontext (kein Quellenauszug): "
            "provider answer support"
        ),
        "[W4] title four",
    ]

    scoped = evidence_digest(
        [
            {"label": "W1", "excerpt": "one"},
            {"label": "W2", "excerpt": "two"},
        ],
        labels=["W2"],
    )
    assert scoped == "[W2] two"


def test_duplicate_url_fills_missing_stronger_evidence_without_overwrite() -> None:
    references, _claims = merge_evidence(
        {
            "a": TaskOutcome(
                status="completed",
                evidence=[
                    {
                        "url": "https://example.com/report",
                        "title": "First title",
                        "grounded_support": "Provider answer support",
                    }
                ],
            ),
            "b": TaskOutcome(
                status="completed",
                evidence=[
                    {
                        "url": "https://example.com/report/",
                        "title": "Later title",
                        "excerpt": "Verbatim provider excerpt",
                    }
                ],
            ),
            "c": TaskOutcome(
                status="completed",
                evidence=[
                    {
                        "url": "https://example.com/report",
                        "excerpt": "Must not replace the first excerpt",
                    }
                ],
            ),
        }
    )

    assert len(references) == 1
    assert references[0]["label"] == "W1"
    assert references[0]["tasks"] == ["a", "b", "c"]
    assert references[0]["title"] == "First title"
    assert references[0]["grounded_support"] == "Provider answer support"
    assert references[0]["excerpt"] == "Verbatim provider excerpt"


def test_kernel_ledger_assigns_stable_labels_across_resume() -> None:
    from inqtrix.agents.control_memory import MemoryAgentControlStore
    from inqtrix.agents.kernel.deps import KernelDeps

    store = MemoryAgentControlStore()
    first = KernelDeps(
        run_id="run_kernel_evidence_resume",
        control=store,
        platform=None,  # type: ignore[arg-type]
        llm=None,  # type: ignore[arg-type]
        model=None,
        reasoning_effort=None,
        timeout=1.0,
    )

    registered = first.register_references(
        [
            {
                "url": "https://example.com/report?utm_source=first",
                "title": "Web report",
            },
            {
                "document_id": "doc-1",
                "chunk_index": 2,
                "title": "Internal report",
            },
        ]
    )

    assert [item["label"] for item in registered] == ["W1", "K1"]

    resumed = KernelDeps(
        run_id=first.run_id,
        control=store,
        platform=None,  # type: ignore[arg-type]
        llm=None,  # type: ignore[arg-type]
        model=None,
        reasoning_effort=None,
        timeout=1.0,
    )
    resumed.hydrate_evidence()
    after_resume = resumed.register_references(
        [
            {
                "url": "https://example.com/report?utm_source=second",
                "excerpt": "Later source excerpt",
            },
            {"url": "https://example.com/second", "title": "Second web"},
        ]
    )

    assert [item["label"] for item in after_resume] == ["W1", "W2"]
    assert after_resume[0]["excerpt"] == "Later source excerpt"
    assert sorted(
        item["label"] for item in resumed.evidence_refs.values()
    ) == ["K1", "W1", "W2"]
