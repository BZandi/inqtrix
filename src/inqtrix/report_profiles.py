"""Report profile types and runtime tuning presets.

Use :class:`ReportProfile` as the public switch for answer style and research
depth. The associated tuning bundle keeps profile-specific defaults in one
place so runtime code can consume them without duplicating literals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

class ReportProfile(StrEnum):
    """Supported report styles for the research agent.

    ``schnell`` runs exactly ONE research round (the broad first-round
    STORM queries) and composes a short two-section answer — the
    latency-first profile for quick lookups and for agent child runs
    that need a fast external sweep rather than a full report.

    ``compact`` preserves the current concise answer style with lower latency.

    ``deep`` keeps more evidence in the pipeline and targets a denser,
    review-style report with broader citation coverage and explicit
    uncertainty handling.
    """

    SCHNELL = "schnell"
    COMPACT = "compact"
    DEEP = "deep"


@dataclass(frozen=True, slots=True)
class AnswerSectionSpec:
    """One named section of the final synthesised answer.

    Each entry drives a single LLM call inside the answer composer:
    ``heading`` becomes the markdown ``##``-header, ``prompt_instruction``
    + ``length_guidance`` are merged into the per-section prompt. Provider
    output budgets stay in provider constructors, not in report profiles.

    Frozen + slotted so instances are safe to share across runs and
    cheap to compare in tests.

    Attributes:
        heading: Markdown ``##`` header text written verbatim into the
            final answer (e.g. ``"Kurzfazit"``).
        prompt_instruction: Section-specific instruction appended to the
            shared answer-prompt scaffold. Should describe what the
            section must contain in business terms.
        length_guidance: Soft qualitative length hint phrased for the
            model. It should steer density and completeness without
            enforcing fixed token or word counts.
        required: When ``False``, the answer composer may skip this
            section if upstream evidence is insufficient. Default
            ``True`` — used today only by extension hooks; stock
            sections are all required.
        write_last: When ``True``, the composer renders this section
            AFTER all other sections in the same plan, even though it
            still appears at its declared position in the final
            answer. Used for summary-style sections (Executive Summary,
            Kurzfazit) that should synthesize the rest of the body and
            therefore benefit from seeing the completed body in their
            ``report_so_far_summary`` and ``used_evidence_labels``.
    """

    heading: str
    prompt_instruction: str
    length_guidance: str
    required: bool = True
    write_last: bool = False


@dataclass(frozen=True, slots=True)
class ReportProfileTuning:
    """Bundled runtime tuning values associated with one report profile.

    A profile (``compact`` / ``deep``) selects an instance of this
    dataclass via :func:`tuning_for_report_profile`. Every numeric
    field is consumed by exactly one component (claim extraction,
    consolidation, answer composer, or evidence renderer), so adjusting
    one value affects only that component.

    Attributes:
        settings_overrides: ``AgentSettings`` fields auto-overridden
            when the profile is selected (see :class:`AgentSettings`
            ``with_report_profile_defaults``). Include profile-owned
            baseline values for every field that another profile may
            raise, so request-time profile switches can move in both
            directions while preserving explicit user settings.
        answer_sections: Ordered tuple of :class:`AnswerSectionSpec`
            describing the per-profile section layout.
        answer_claim_prompt_items: Maximum number of consolidated
            claims passed into the answer prompt's ``Claims`` block.
        claim_input_char_limit: Maximum text length (chars) per source
            forwarded to the claim-extraction call.
        claim_citation_cap: Maximum number of citations included with
            each extracted claim batch.
        claim_max_items: Maximum number of claims extracted per source
            in a single call.
        claim_source_url_cap: Maximum number of source URLs attached
            to a single consolidated claim.
        claim_ledger_cap: Maximum number of entries kept in the
            consolidated claim ledger across the whole run.
        materialize_max_total: Maximum total number of materialised
            (verified + contested + unverified) claims in the result
            view.
        materialize_max_unverified: Sub-cap for ``unverified`` claims
            within ``materialize_max_total``. Profiles may set this equal
            to ``materialize_max_total`` when hidden unverified-evidence
            truncation would be more harmful than a broader uncertainty view.
        prompt_evidence_record_char_limit: Maximum characters rendered
            per source block in the EvidenceLedger overview before its
            evidence lines are compacted (label, URL, and metadata are
            kept).
        prompt_evidence_total_char_budget: Maximum total characters for
            the single EvidenceLedger overview passed to the answer
            composer. This is the only character budget on the answer-
            facing evidence view; records that do not fit are counted as
            omitted rather than dropped silently.
        min_report_eligible_evidence: Minimum report-eligible
            EvidenceRecords expected before early stopping is allowed,
            unless the run already hit ``max_rounds``.
    """

    settings_overrides: dict[str, int]
    answer_sections: tuple[AnswerSectionSpec, ...]
    answer_claim_prompt_items: int
    claim_input_char_limit: int
    claim_citation_cap: int
    claim_max_items: int
    claim_source_url_cap: int
    claim_ledger_cap: int
    materialize_max_total: int
    materialize_max_unverified: int
    prompt_evidence_record_char_limit: int
    prompt_evidence_total_char_budget: int
    min_report_eligible_evidence: int


_COMPACT_ANSWER_SECTIONS = (
    AnswerSectionSpec(
        heading="Kurzfazit",
        prompt_instruction=(
            "Beantworte die Frage direkt und nenne die wichtigsten Erkenntnisse "
            "mit sauberer Einordnung. Stuetze dich auf die bereits geschriebenen "
            "Abschnitte und verwende deren Evidence-Labels, wenn du Aussagen daraus "
            "verdichtest."
        ),
        length_guidance="knapp, aber vollstaendig; keine Bulletpoints",
        write_last=True,
    ),
    AnswerSectionSpec(
        heading="Kernaussagen",
        prompt_instruction=(
            "Formuliere substanzielle Bulletpoints mit den wichtigsten Fakten, "
            "Zahlen und Implikationen. Jeder Punkt soll eigenstaendig "
            "verstaendlich sein."
        ),
        length_guidance="so viele Bulletpoints wie die Evidenz traegt; keine Fuellpunkte",
    ),
    AnswerSectionSpec(
        heading="Detailanalyse",
        prompt_instruction=(
            "Vertiefe die Kernaspekte mit sinnvollen `###`-Unterabschnitten. Erklaere "
            "Zusammenhaenge, Ursachen und Auswirkungen belastbar und ohne Wiederholungen."
        ),
        length_guidance="ausfuehrlich genug fuer die Frage; keine kuenstliche Kuerzung",
    ),
    AnswerSectionSpec(
        heading="Einordnung / Ausblick",
        prompt_instruction=(
            "Ordne die Befunde knapp ein und benenne die wichtigsten naechsten "
            "Entwicklungen oder offenen Fragen."
        ),
        length_guidance="kurz und abgeschlossen",
    ),
)

_DEEP_ANSWER_SECTIONS = (
    AnswerSectionSpec(
        heading="Executive Summary",
        prompt_instruction=(
            "Beantworte die Frage direkt und verdichte die wichtigsten Erkenntnisse "
            "mit Stand-Einordnung. Stuetze dich auf die bereits geschriebenen "
            "Abschnitte und verwende deren Evidence-Labels, wenn du Aussagen daraus "
            "verdichtest. Es ist legitim, wenige bis keine eigenen Zitate hinzuzufuegen, "
            "wenn die ausfuehrlichen Sektionen die Belege bereits tragen."
        ),
        length_guidance="knapp, dicht und vollstaendig; keine Bulletpoints",
        write_last=True,
    ),
    AnswerSectionSpec(
        heading="Hintergrund / Kontext",
        prompt_instruction=(
            "Erklaere Ausgangspunkt, zeitlichen Rahmen und relevanten Kontext."
        ),
        length_guidance="kompakt, aber mit allen noetigen Kontextankern",
    ),
    AnswerSectionSpec(
        heading="Analyse",
        prompt_instruction=(
            "Schreibe mehrere sinnvolle `###`-Unterabschnitte. Integriere relevante Zahlen, "
            "Vergleichswerte, Mechanismen, Umsetzungsfragen und sachliche Kausalzusammenhaenge."
        ),
        length_guidance="die ausfuehrlichste Sektion; Tiefe nach Evidenzlage statt Wortlimit",
    ),
    AnswerSectionSpec(
        heading="Perspektiven / Positionen",
        prompt_instruction=(
            "Stelle die wesentlichen Perspektiven getrennt und neutral dar, inklusive "
            "treibender Akteure, Gegenpositionen und Betroffenen- oder Umsetzungssicht."
        ),
        length_guidance="klar getrennte Absaetze; Umfang nach Zahl belastbarer Perspektiven",
    ),
    AnswerSectionSpec(
        heading="Risiken / Unsicherheiten",
        prompt_instruction=(
            "Liste praezise Bulletpoints zu Evidenzgrenzen, offenen Punkten, "
            "methodischen Einschraenkungen oder Gegenargumenten."
        ),
        length_guidance="nur echte Risiken und Unsicherheiten, keine Pflichtfuellung",
    ),
    AnswerSectionSpec(
        heading="Fazit / Ausblick",
        prompt_instruction=(
            "Ziehe eine belastbare Gesamteinordnung und benenne den plausibelsten "
            "naechsten Entwicklungspfad."
        ),
        length_guidance="kurz, klar und abgeschlossen",
    ),
)


_SCHNELL_ANSWER_SECTIONS = (
    AnswerSectionSpec(
        heading="Kurzfazit",
        prompt_instruction=(
            "Beantworte die Frage direkt in wenigen Saetzen und nenne die "
            "tragenden Erkenntnisse mit Evidence-Labels. Stuetze dich auf die "
            "bereits geschriebenen Kernaussagen."
        ),
        length_guidance="sehr knapp; keine Bulletpoints",
        write_last=True,
    ),
    AnswerSectionSpec(
        heading="Kernaussagen",
        prompt_instruction=(
            "Formuliere die wichtigsten Fakten und Zahlen als eigenstaendig "
            "verstaendliche Bulletpoints. Keine Vertiefung, keine Wiederholungen."
        ),
        length_guidance="nur die tragfaehigsten Punkte; keine Fuellpunkte",
    ),
)


_SCHNELL_TUNING = ReportProfileTuning(
    # One round, first-iteration STORM breadth only: the profile exists
    # for latency (quick lookups, agent child runs), so every budget is
    # the smallest value that still yields a citable two-section answer.
    # Same field set as the other profiles (bidirectional request-time
    # profile switches, see ``settings_overrides`` docstring).
    settings_overrides={
        "max_rounds": 1,
        "min_rounds": 1,
        "confidence_stop": 6,
        "first_round_queries": 6,
        "answer_prompt_citations_max": 40,
        "reasoning_timeout": 600,
        "editor_assistant_timeout": 600,
        "claim_extract_timeout": 600,
        "search_timeout": 600,
        "max_total_seconds": 3600,
    },
    answer_sections=_SCHNELL_ANSWER_SECTIONS,
    answer_claim_prompt_items=12,
    claim_input_char_limit=16000,
    claim_citation_cap=6,
    claim_max_items=6,
    claim_source_url_cap=3,
    claim_ledger_cap=200,
    materialize_max_total=16,
    materialize_max_unverified=6,
    prompt_evidence_record_char_limit=1800,
    prompt_evidence_total_char_budget=20000,
    min_report_eligible_evidence=2,
)


_COMPACT_TUNING = ReportProfileTuning(
    settings_overrides={
        "max_rounds": 2,
        "min_rounds": 1,
        "confidence_stop": 7,
        "first_round_queries": 6,
        "answer_prompt_citations_max": 60,
        "reasoning_timeout": 600,
        "editor_assistant_timeout": 600,
        "claim_extract_timeout": 600,
        "search_timeout": 600,
        "max_total_seconds": 3600,
    },
    answer_sections=_COMPACT_ANSWER_SECTIONS,
    answer_claim_prompt_items=20,
    claim_input_char_limit=24000,
    claim_citation_cap=8,
    claim_max_items=8,
    claim_source_url_cap=4,
    claim_ledger_cap=400,
    materialize_max_total=24,
    materialize_max_unverified=8,
    prompt_evidence_record_char_limit=2200,
    prompt_evidence_total_char_budget=30000,
    min_report_eligible_evidence=3,
)

_DEEP_TUNING = ReportProfileTuning(
    # Eight broad questions retain the established DEEP source breadth while
    # bounding the costly first wave before iterative gap filling begins.
    settings_overrides={
        "max_rounds": 4,
        "min_rounds": 2,
        "confidence_stop": 8,
        "first_round_queries": 8,
        "answer_prompt_citations_max": 500,
        "reasoning_timeout": 600,
        "editor_assistant_timeout": 600,
        "claim_extract_timeout": 600,
        "search_timeout": 600,
        "max_total_seconds": 3600,
    },
    answer_sections=_DEEP_ANSWER_SECTIONS,
    answer_claim_prompt_items=40,
    claim_input_char_limit=48000,
    claim_citation_cap=20,
    claim_max_items=24,
    claim_source_url_cap=6,
    claim_ledger_cap=800,
    materialize_max_total=48,
    materialize_max_unverified=48,
    prompt_evidence_record_char_limit=2600,
    # Raised so the record-driven evidence overview holds substantially more
    # than the 60-80 records a 110k-char budget rendered; the answer prompt
    # consumes this single view and the LLM context window (Claude Sonnet 4.x
    # = 200k tokens) handles a ~180k-char system prompt comfortably.
    prompt_evidence_total_char_budget=180000,
    min_report_eligible_evidence=8,
)


def tuning_for_report_profile(profile: ReportProfile | str) -> ReportProfileTuning:
    """Return the runtime tuning bundle for ``profile``.

    Args:
        profile: Either a :class:`ReportProfile` enum value or its
            string representation (``"compact"`` / ``"deep"``). Any
            value that does not parse to a known profile silently
            falls back to ``ReportProfile.COMPACT`` to keep callers
            robust against malformed config.

    Returns:
        The frozen :class:`ReportProfileTuning` instance for the
        profile. The same instance is returned for repeat calls (the
        underlying tuning objects are module-level constants), so
        identity-based caching by callers is safe.
    """
    try:
        normalized = ReportProfile(profile)
    except ValueError:
        normalized = ReportProfile.COMPACT
    if normalized is ReportProfile.DEEP:
        return _DEEP_TUNING
    if normalized is ReportProfile.SCHNELL:
        return _SCHNELL_TUNING
    return _COMPACT_TUNING


def settings_overrides_for_report_profile(profile: ReportProfile | str) -> dict[str, int]:
    """Return the ``AgentSettings`` overrides implied by ``profile``.

    Args:
        profile: Either a :class:`ReportProfile` enum value or its
            string representation. Unknown values fall back to
            ``ReportProfile.COMPACT``.

    Returns:
        A new ``dict`` mapping ``AgentSettings`` field names to the
        profile-specific override values. Caller may mutate the returned
        dict freely — it is a fresh copy of the bundle's
        ``settings_overrides``.
    """
    return dict(tuning_for_report_profile(profile).settings_overrides)
