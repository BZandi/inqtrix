"""Tests for EvidenceLedger assembly and the record-driven evidence overview."""

from __future__ import annotations

from inqtrix.evidence import (
    assemble_evidence_records,
    audit_answer_evidence_bindings,
    build_web_search_ledger,
    derive_claim_ledger_from_evidence,
    evidence_id_for_citation,
    project_claim_verification_to_evidence,
    render_evidence_ledger_overview,
    select_section_evidence_records,
)
from inqtrix.runtime_logging import make_record_id


def test_web_search_ledger_uses_provider_annotation_span_without_page_read():
    answer = "Global input costs 5 USD. Data Zone costs more."
    ledger = build_web_search_ledger(
        run_id="run_1",
        query_records=[
            {
                "query_id": "query_1",
                "query": "model prices",
                "provider": "AzureFoundryWebSearch",
                "status": "completed",
            }
        ],
        query_synthesis={
            "query_1": {
                "query": "model prices",
                "provider_answer": answer,
            }
        },
        citation_records=[
            {
                "annotation_end": 25,
                "annotation_start": 0,
                "canonical_url": "https://example.com/prices",
                "citation_id": "citation_1",
                "query_id": "query_1",
                "source_id": "source_1",
                "title": "Prices",
            }
        ],
    )

    search = ledger["searches"]["query_1"]
    assert search["provider_answer"] == answer
    assert search["citations"][0]["grounded_support"] == (
        "Global input costs 5 USD."
    )
    assert search["citations"][0]["mapping_status"] == (
        "provider_answer_context"
    )


def test_web_search_ledger_labels_link_only_annotation_as_marker_context():
    answer = (
        "Global input costs 5 USD. "
        "([azure.microsoft.com](https://azure.microsoft.com/prices))"
    )
    marker_start = answer.index("([")
    ledger = build_web_search_ledger(
        run_id="run_1",
        query_records=[
            {
                "query_id": "query_1",
                "query": "model prices",
                "provider": "AzureFoundryWebSearch",
                "status": "completed",
            }
        ],
        query_synthesis={
            "query_1": {
                "query": "model prices",
                "provider_answer": answer,
            }
        },
        citation_records=[
            {
                "annotation_end": len(answer),
                "annotation_start": marker_start,
                "canonical_url": "https://azure.microsoft.com/prices",
                "citation_id": "citation_1",
                "query_id": "query_1",
                "source_id": "source_1",
                "title": "Prices",
            }
        ],
    )

    citation = ledger["searches"]["query_1"]["citations"][0]
    assert citation["mapping_status"] == "provider_citation_marker"
    assert citation["grounded_support"] == "Global input costs 5 USD."


def _source(source_id: str, url: str, tier: str = "primary") -> dict:
    return {
        "source_id": source_id,
        "url": url,
        "canonical_url": url,
        "domain": "example.com",
        "provider": "StubSearch",
        "first_seen_query_id": "qry_1",
        "first_seen_rank": 1,
        "origin": "search_results",
        "tier": tier,
        "tier_reason": "matched_primary_domain",
        "access_status": "answer",
    }


def _citation(query_id: str, source_id: str, url: str, rank: int = 1) -> dict:
    return {
        "citation_id": make_record_id("cit", query_id, rank, url, "search_results"),
        "query_id": query_id,
        "source_id": source_id,
        "url": url,
        "canonical_url": url,
        "rank": rank,
        "origin": "search_results",
        "provider": "StubSearch",
        "title": "Example report",
        "snippet": "Meta expects capital expenditures of 115-135 USD billion.",
        "source_date": "2026-05-01",
        "last_updated": "2026-05-02",
    }


def _synthesis(query_id: str, query: str, summary: str, provider_answer: str = "") -> dict:
    """Build a one-query ``query_synthesis`` map entry for the renderer."""
    return {
        query_id: {
            "query": query,
            "round": 0,
            "provider_answer": provider_answer or summary,
            "related_questions": [],
        }
    }


def test_assemble_evidence_records_preserves_claims_and_passages_without_data_points():
    query_id = "qry_1"
    url = "https://example.com/report"
    source = _source("src_1", url)
    citation = _citation(query_id, "src_1", url)
    evidence_id = evidence_id_for_citation(query_id, citation)
    claim = {
        "raw_claim_id": "raw_1",
        "claim_text": "Meta expects 115-135 USD billion of capital expenditures.",
        "claim_type": "fact",
        "polarity": "affirmed",
        "needs_primary": True,
        "source_urls": [url],
        "source_ids": ["src_1"],
        "citation_ids": [citation["citation_id"]],
        "evidence_ids": [evidence_id],
        "published_date": "2026-05-01",
        "signature": "meta capex 115 135 usd billion",
        "evidence_snippet": "The guidance range is 115-135 USD billion.",
    }

    records = assemble_evidence_records(
        query_id=query_id,
        query="meta capex",
        provider="StubSearch",
        source_records=[source],
        citation_records=[citation],
        claim_entries=[claim],
    )

    assert len(records) == 1
    record = records[0]
    assert record["evidence_id"] == evidence_id
    assert "provider_answer_excerpt" not in record
    assert record["claims"][0]["raw_claim_id"] == "raw_1"
    assert record["source_passages"]
    assert "data_points" not in record


def test_multi_source_claim_attaches_to_each_source_record():
    """A claim citing two sources yields one per-source record each, both citable."""
    query_id = "qry_multi_src"
    urls = ["https://example.com/a", "https://example.com/b"]
    citations = [
        _citation(query_id, f"src_{index}", url, rank=index)
        for index, url in enumerate(urls, start=1)
    ]
    evidence_ids = [evidence_id_for_citation(query_id, citation) for citation in citations]
    claim = {
        "raw_claim_id": "raw_multi",
        "claim_text": "Example reported 12 percent growth.",
        "claim_type": "fact",
        "polarity": "affirmed",
        "needs_primary": False,
        "source_urls": urls,
        "source_ids": ["src_1", "src_2"],
        "citation_ids": [citation["citation_id"] for citation in citations],
        "evidence_ids": evidence_ids,
        "published_date": "2026-05-01",
        "signature": "example growth 12 percent",
        "evidence_snippet": "Two sources cite 12 percent growth.",
    }

    records = assemble_evidence_records(
        query_id=query_id,
        query="multi source",
        provider="StubSearch",
        source_records=[_source("src_1", urls[0]), _source("src_2", urls[1])],
        citation_records=citations,
        claim_entries=[claim],
    )

    assert len(records) == 2
    assert all(record["record_type"] == "source" for record in records)
    assert all(record["report_eligible"] for record in records)
    assert all(
        record["claims"] and record["claims"][0]["raw_claim_id"] == "raw_multi"
        for record in records
    )


def test_url_only_source_is_citeable_without_snippet():
    """Azure-style URL-only sources (no per-source body) stay citable anchors."""
    query_id = "qry_urlonly"
    url = "https://www.bloomberg.com/latest/the-ai-race"
    citation = _citation(query_id, "src_1", url)
    citation["snippet"] = ""
    evidence_id = evidence_id_for_citation(query_id, citation)
    claim = {
        "raw_claim_id": "raw_1",
        "claim_text": "Cloudflare announced 1,100 layoffs.",
        "claim_type": "fact",
        "polarity": "affirmed",
        "needs_primary": True,
        "source_urls": [url],
        "source_ids": ["src_1"],
        "citation_ids": [citation["citation_id"]],
        "evidence_ids": [evidence_id],
        "signature": "cloudflare layoffs 1100",
    }

    records = assemble_evidence_records(
        query_id=query_id,
        query="cloudflare layoffs",
        provider="AzureFoundryWebSearch",
        source_records=[_source("src_1", url, tier="mainstream")],
        citation_records=[citation],
        claim_entries=[claim],
    )

    assert len(records) == 1
    assert records[0]["report_eligible"] is True
    assert records[0]["source_snippet"] == ""
    assert records[0]["claims"][0]["citation_set"][0]["url"] == url


def _records_for_query(query_id: str, query: str, count: int) -> list[dict]:
    records: list[dict] = []
    for index in range(count):
        url = f"https://example.com/{query_id}-{index}"
        citation = _citation(query_id, f"src_{query_id}_{index}", url, rank=index + 1)
        records += assemble_evidence_records(
            query_id=query_id,
            query=query,
            provider="StubSearch",
            source_records=[_source(f"src_{query_id}_{index}", url)],
            citation_records=[citation],
            claim_entries=[],
        )
    return records


def test_evidence_ledger_overview_renders_every_eligible_record():
    records = _records_for_query("qrymulti", "multi source query", 3)
    overview = render_evidence_ledger_overview(
        records,
        max_total_chars=8000,
        max_record_chars=1500,
        query_synthesis=_synthesis("qrymulti", "multi source query", "Summary: three sources."),
    )
    assert overview.rendered_record_count == 3
    assert overview.omitted_record_count == 0
    for index in range(1, 4):
        assert f"[E{index}]" in overview.markdown
    assert set(overview.label_urls) == {"E1", "E2", "E3"}
    assert len(overview.allowed_urls) == 3
    # URLs no longer leak into the prompt body (token saving); they live in the
    # label map and are re-attached after synthesis.
    assert overview.label_urls["E1"] not in overview.markdown
    assert "  URL:" not in overview.markdown


def test_evidence_ledger_overview_shows_provider_answer_once_per_query():
    records = _records_for_query("qryone", "query number one", 2)
    records += _records_for_query("qrytwo", "query number two", 2)
    query_synthesis = {
        **_synthesis("qryone", "query number one", "SUMMARY-Q1 digest text."),
        **_synthesis("qrytwo", "query number two", "SUMMARY-Q2 digest text."),
    }
    overview = render_evidence_ledger_overview(
        records, max_total_chars=20000, max_record_chars=2000, query_synthesis=query_synthesis
    )
    assert overview.rendered_record_count == 4
    assert overview.markdown.count("SUMMARY-Q1 digest text.") == 1
    assert overview.markdown.count("SUMMARY-Q2 digest text.") == 1


def test_evidence_ledger_overview_renders_claimless_record_as_context():
    records = _records_for_query("qrycl", "claimless query", 1)
    overview = render_evidence_ledger_overview(
        records,
        max_total_chars=8000,
        max_record_chars=2000,
        query_synthesis=_synthesis(
            "qrycl", "claimless query", "Summary: claimless source still useful."
        ),
    )
    assert overview.rendered_record_count == 1
    assert "Beleglage: source-context" in overview.markdown
    assert "Meta expects capital expenditures" in overview.markdown


def test_evidence_ledger_overview_dedups_repeated_text():
    query_id = "qrydup"
    url = "https://example.com/dup"
    repeated = "The launch event is scheduled for the third quarter."
    citation = _citation(query_id, "src_dup", url)
    citation["snippet"] = repeated
    evidence_id = evidence_id_for_citation(query_id, citation)
    claim = {
        "raw_claim_id": "raw_dup",
        "claim_text": repeated,
        "claim_type": "fact",
        "polarity": "affirmed",
        "needs_primary": False,
        "source_urls": [url],
        "source_ids": ["src_dup"],
        "citation_ids": [citation["citation_id"]],
        "evidence_ids": [evidence_id],
        "published_date": "2026-05-01",
        "signature": "launch event third quarter",
        "evidence_snippet": repeated,
    }
    records = assemble_evidence_records(
        query_id=query_id,
        query="dedup query",
        provider="StubSearch",
        source_records=[_source("src_dup", url)],
        citation_records=[citation],
        claim_entries=[claim],
    )
    overview = render_evidence_ledger_overview(
        records,
        max_total_chars=8000,
        max_record_chars=2000,
        query_synthesis=_synthesis(query_id, "dedup query", "Summary: dedup check."),
    )
    assert overview.markdown.count(repeated) == 1


def test_evidence_ledger_overview_omits_over_budget_records_visibly():
    records = _records_for_query("qrybudget", "budget query", 6)
    overview = render_evidence_ledger_overview(
        records,
        max_total_chars=900,
        max_record_chars=400,
        query_synthesis=_synthesis(
            "qrybudget", "budget query", "Summary: many sources, tiny budget."
        ),
    )
    assert 0 < overview.rendered_record_count < 6
    assert overview.omitted_record_count > 0
    assert overview.rendered_record_count + overview.omitted_record_count == 6
    assert "HINWEIS" in overview.markdown
    assert len(overview.allowed_urls) == overview.rendered_record_count


def test_evidence_ledger_overview_with_zero_rendered_records_is_empty():
    records = _records_for_query("qrynone", "zero budget query", 2)
    overview = render_evidence_ledger_overview(
        records,
        max_total_chars=1,
        max_record_chars=400,
        query_synthesis=_synthesis(
            "qrynone", "zero budget query", "Summary too large for the budget."
        ),
    )

    assert overview.rendered_record_count == 0
    assert overview.omitted_record_count == 2
    assert overview.markdown == ""
    assert overview.allowed_urls == []


def test_render_allowed_urls_only_include_rendered_source_blocks():
    records = _records_for_query("qryvisible", "visible budget query", 5)
    overview = render_evidence_ledger_overview(
        records,
        max_total_chars=900,
        max_record_chars=400,
        query_synthesis=_synthesis("qryvisible", "visible budget query", "Summary."),
    )

    url_by_id = {record["evidence_id"]: record["canonical_url"] for record in records}
    rendered_urls = [
        url_by_id[evidence_id]
        for evidence_id in overview.rendered_evidence_ids
    ]

    assert 0 < overview.rendered_record_count < len(records)
    assert overview.allowed_urls == rendered_urls
    assert set(overview.label_urls.values()) == set(rendered_urls)


def test_render_assigns_one_label_per_canonical_url_across_queries():
    url = "https://example.com/shared-source"
    records: list[dict] = []
    for index, query_id in enumerate(("qry_shared_a", "qry_shared_b"), start=1):
        citation = _citation(query_id, f"src_shared_{index}", url, rank=index)
        records += assemble_evidence_records(
            query_id=query_id,
            query=f"shared query {index}",
            provider="StubSearch",
            source_records=[_source(f"src_shared_{index}", url)],
            citation_records=[citation],
            claim_entries=[],
        )

    overview = render_evidence_ledger_overview(
        records,
        max_total_chars=8000,
        max_record_chars=1200,
        query_synthesis={
            **_synthesis("qry_shared_a", "shared query 1", "Summary one."),
            **_synthesis("qry_shared_b", "shared query 2", "Summary two."),
        },
    )
    labels = {
        overview.label_by_evidence_id[record["evidence_id"]]
        for record in records
    }

    assert labels == {"E1"}
    assert overview.label_urls == {"E1": url}
    assert overview.allowed_urls == [url]


def test_render_remaps_provider_answer_citations_to_labels():
    """Provider inline citations are rewritten to [E#] labels; no raw URLs leak."""
    query_id = "qry_remap"
    url1 = "https://primary.example/a"
    url2 = "https://media.example/b"
    records: list[dict] = []
    for sid, url, rank in (("src_1", url1, 1), ("src_2", url2, 2)):
        citation = _citation(query_id, sid, url, rank=rank)
        records += assemble_evidence_records(
            query_id=query_id,
            query="remap query",
            provider="StubSearch",
            source_records=[_source(sid, url)],
            citation_records=[citation],
            claim_entries=[],
        )
    query_synthesis = {
        query_id: {
            "query": "remap query",
            "round": 0,
            # Perplexity numeric [1] + Azure Markdown-link citation styles:
            "provider_answer": (
                f"Fakt eins [1]. Fakt zwei ([media.example]({url2})). "
                "Beide zusammen [1][2]."
            ),
            "related_questions": [],
            "citation_urls_by_rank": {"1": url1, "2": url2},
        }
    }
    overview = render_evidence_ledger_overview(
        records, max_total_chars=8000, max_record_chars=2000, query_synthesis=query_synthesis
    )
    md = overview.markdown
    label1 = next(label for label, url in overview.label_urls.items() if "primary.example" in url)
    label2 = next(label for label, url in overview.label_urls.items() if "media.example" in url)
    assert f"Fakt eins [{label1}]." in md
    assert f"Fakt zwei [{label2}]." in md
    assert f"Beide zusammen [{label1}] [{label2}]." in md
    assert "[1]" not in md
    assert url1 not in md and url2 not in md
    assert "Provider-Synthese (Kontext; nicht eigenstaendig verifiziert):" in md


def test_render_keeps_unmapped_provider_citations_visible():
    query_id = "qry_unmapped"
    url = "https://primary.example/a"
    citation = _citation(query_id, "src_1", url, rank=1)
    records = assemble_evidence_records(
        query_id=query_id,
        query="unmapped query",
        provider="StubSearch",
        source_records=[_source("src_1", url)],
        citation_records=[citation],
        claim_entries=[],
    )
    query_synthesis = {
        query_id: {
            "query": "unmapped query",
            "round": 0,
            "provider_answer": "Fakt eins [99].",
            "related_questions": [],
            "citation_urls_by_rank": {"1": url},
        }
    }

    overview = render_evidence_ledger_overview(
        records, max_total_chars=8000, max_record_chars=2000, query_synthesis=query_synthesis
    )

    assert "[unmapped:99]" in overview.markdown


def test_provider_synthesis_marks_citations_without_visible_source_block():
    query_id = "qry_not_rendered"
    records: list[dict] = []
    for rank in range(1, 4):
        url = f"https://example.com/not-rendered-{rank}"
        citation = _citation(query_id, f"src_{rank}", url, rank=rank)
        citation["snippet"] = f"Long source {rank}. " * 50
        records += assemble_evidence_records(
            query_id=query_id,
            query="budget remap query",
            provider="StubSearch",
            source_records=[_source(f"src_{rank}", url)],
            citation_records=[citation],
            claim_entries=[],
        )
    overview = render_evidence_ledger_overview(
        records,
        max_total_chars=550,
        max_record_chars=500,
        query_synthesis={
            query_id: {
                "query": "budget remap query",
                "round": 0,
                "provider_answer": "Combined statement [1][2][3].",
                "related_questions": [],
                "citation_urls_by_rank": {
                    str(rank): f"https://example.com/not-rendered-{rank}"
                    for rank in range(1, 4)
                },
            }
        },
    )

    assert 0 < overview.rendered_record_count < 3
    assert "[nicht-gerendert:" in overview.markdown
    assert "[E2] [nicht-gerendert:3]" in overview.markdown
    for label in overview.label_urls:
        assert f"[{label}] " in overview.markdown


def test_derive_claim_ledger_merges_one_claim_across_evidence_records():
    query_id = "qry_1"
    urls = ["https://primary.example/report", "https://media.example/story"]
    citations = [
        _citation(query_id, f"src_{idx}", url, rank=idx)
        for idx, url in enumerate(urls, start=1)
    ]
    records = []
    for idx, (url, citation) in enumerate(zip(urls, citations, strict=True), start=1):
        evidence_id = evidence_id_for_citation(query_id, citation)
        records.extend(
            assemble_evidence_records(
                query_id=query_id,
                query="meta capex",
                provider="StubSearch",
                source_records=[_source(f"src_{idx}", url)],
                citation_records=[citation],
                claim_entries=[
                    {
                        "raw_claim_id": "raw_shared",
                        "claim_text": "Meta expects 115-135 USD billion of capital expenditures.",
                        "claim_type": "fact",
                        "polarity": "affirmed",
                        "needs_primary": True,
                        "source_urls": [url],
                        "source_ids": [f"src_{idx}"],
                        "citation_ids": [citation["citation_id"]],
                        "evidence_ids": [evidence_id],
                        "published_date": "2026-05-01",
                        "signature": "meta capex 115 135 usd billion",
                    }
                ],
            )
        )

    ledger = derive_claim_ledger_from_evidence(records)

    assert len(ledger) == 1
    assert len(ledger[0]["evidence_ids"]) == 2
    assert len(ledger[0]["source_urls"]) == 2


def _claim_record(query_id: str, url: str, claim_text: str, *, status: str) -> dict:
    """Build one evidence record carrying a single projected-verification claim."""
    citation = _citation(query_id, "src_x", url)
    evidence_id = evidence_id_for_citation(query_id, citation)
    records = assemble_evidence_records(
        query_id=query_id,
        query="claim record query",
        provider="StubSearch",
        source_records=[_source("src_x", url)],
        citation_records=[citation],
        claim_entries=[
            {
                "raw_claim_id": f"raw_{query_id}",
                "claim_text": claim_text,
                "claim_type": "fact",
                "polarity": "affirmed",
                "needs_primary": False,
                "source_urls": [url],
                "source_ids": ["src_x"],
                "citation_ids": [citation["citation_id"]],
                "evidence_ids": [evidence_id],
                "published_date": "2026-05-01",
                "signature": claim_text.lower(),
                "evidence_snippet": claim_text,
            }
        ],
    )
    record = records[0]
    for claim in record["claims"]:
        claim["verification_status"] = status
        claim["verification_basis"] = (
            "verified_primary" if status == "verified" else status
        )
    return record


def test_project_claim_verification_makes_ledger_self_describing():
    query_id = "qry_proj"
    url = "https://example.com/proj"
    citation = _citation(query_id, "src_x", url)
    evidence_id = evidence_id_for_citation(query_id, citation)
    records = assemble_evidence_records(
        query_id=query_id,
        query="projection query",
        provider="StubSearch",
        source_records=[_source("src_x", url)],
        citation_records=[citation],
        claim_entries=[
            {
                "raw_claim_id": "raw_proj",
                "claim_text": "Meta expects 115-135 USD billion of capital expenditures.",
                "claim_type": "fact",
                "polarity": "affirmed",
                "needs_primary": True,
                "source_urls": [url],
                "source_ids": ["src_x"],
                "citation_ids": [citation["citation_id"]],
                "evidence_ids": [evidence_id],
                "published_date": "2026-05-01",
                "signature": "meta capex",
                "evidence_snippet": "Guidance range is 115-135 USD billion.",
            }
        ],
    )
    consolidated = [
        {
            "claim_id": "claim_proj",
            "status": "verified",
            "verification_basis": "verified_primary",
            "member_claim_ids": ["raw_proj"],
            "supporting_evidence_ids": [evidence_id],
            "supporting_domain_count": 1,
        }
    ]
    projected = project_claim_verification_to_evidence(records, consolidated)
    overview = render_evidence_ledger_overview(
        projected,
        max_total_chars=8000,
        max_record_chars=2000,
        query_synthesis=_synthesis(query_id, "projection query", "Summary."),
    )
    assert projected[0]["claims"][0]["verification_status"] == "verified"
    assert "Beleglage: primary-source" in overview.markdown


def test_audit_answer_evidence_bindings_classifies_cited_sources():
    verified = _claim_record(
        "qry_v", "https://example.com/verified", "Verified fact stated.", status="verified"
    )
    context = _claim_record(
        "qry_c", "https://example.com/context", "Unverified note stated.", status="unverified"
    )
    answer = (
        "Erstens [E1](https://example.com/verified) belegt den Fakt. "
        "Zweitens [E2](https://example.com/context) liefert Kontext. "
        "Drittens [E9](https://example.com/not-in-ledger) ist unbekannt."
    )
    bindings = audit_answer_evidence_bindings(answer, [verified, context])
    by_url = {b["citation_url"]: b for b in bindings}
    assert by_url["https://example.com/verified"]["binding_status"] == "matched"
    assert by_url["https://example.com/context"]["binding_status"] == "source_context"
    assert by_url["https://example.com/not-in-ledger"]["binding_status"] == "unknown_citation"


def test_select_section_evidence_records_prefers_uncertainty_for_risks_section():
    verified = _claim_record(
        "qry_v", "https://example.com/verified", "Verified fact.", status="verified"
    )
    contested = _claim_record(
        "qry_x", "https://example.com/contested", "Contested claim.", status="contested"
    )
    label_by_id = {
        str(verified["evidence_id"]): "E1",
        str(contested["evidence_id"]): "E2",
    }
    selected = select_section_evidence_records(
        [verified, contested],
        heading="Risiken / Unsicherheiten",
        required_aspects=[],
        used_labels=set(),
        label_by_evidence_id=label_by_id,
        max_records=1,
    )
    assert selected and selected[0]["evidence_id"] == contested["evidence_id"]
