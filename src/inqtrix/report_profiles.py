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

    ``compact`` preserves the current concise answer style with lower latency.

    ``deep`` keeps more evidence in the pipeline and targets a denser,
    review-style report with broader citation coverage and explicit
    uncertainty handling.
    """

    COMPACT = "compact"
    DEEP = "deep"


@dataclass(frozen=True, slots=True)
class AnswerSectionSpec:
    """One named section of the final synthesised answer.

    Each entry drives a single LLM call inside the answer composer:
    ``heading`` becomes the markdown ``##``-header, ``prompt_instruction``
    + ``length_guidance`` are merged into the per-section prompt, and
    ``max_output_tokens`` caps that call's completion budget.

    Frozen + slotted so instances are safe to share across runs and
    cheap to compare in tests.

    Attributes:
        heading: Markdown ``##`` header text written verbatim into the
            final answer (e.g. ``"Kurzfazit"``).
        prompt_instruction: Section-specific instruction appended to the
            shared answer-prompt scaffold. Should describe what the
            section must contain in business terms.
        length_guidance: Soft length hint phrased for the model
            (e.g. ``"ca. 120-180 Woerter, keine Bulletpoints"``).
            Combined with ``max_output_tokens`` for hard enforcement.
        max_output_tokens: Hard upper bound on the section's completion
            tokens. Sized to leave ~20-25 % headroom on top of the
            upper word-count guidance; over-tight values cause
            mid-sentence truncation.
        required: When ``False``, the answer composer may skip this
            section if upstream evidence is insufficient. Default
            ``True`` — used today only by extension hooks; stock
            sections are all required.
    """

    heading: str
    prompt_instruction: str
    length_guidance: str
    max_output_tokens: int
    required: bool = True


@dataclass(frozen=True, slots=True)
class ReportProfileTuning:
    """Bundled runtime tuning values associated with one report profile.

    A profile (``compact`` / ``deep``) selects an instance of this
    dataclass via :func:`tuning_for_report_profile`. Every numeric
    field is consumed by exactly one component (summarise loop,
    consolidation, answer composer, session store), so adjusting one
    value affects only that component.

    Attributes:
        settings_overrides: ``AgentSettings`` fields auto-overridden
            when the profile is selected (see :class:`AgentSettings`
            ``with_report_profile_defaults``). Empty dict for profiles
            that intentionally inherit user-supplied settings.
        summarize_input_char_limit: Maximum input-text length (chars)
            forwarded to the summarize call per search hit. Larger
            values preserve more raw content but raise prompt cost.
        summarize_fallback_char_limit: Reduced char limit used by the
            summarize-fallback path triggered on rate-limit / oversize
            errors.
        summarize_max_output_tokens: Output-token cap for summarize
            calls. ``None`` lets the provider default apply (used by
            COMPACT, where the model is small enough that the default
            is conservative).
        answer_body_citation_cap: Maximum number of citation markers
            embedded in the body of the final answer (separate from
            the references list).
        answer_sections: Ordered tuple of :class:`AnswerSectionSpec`
            describing the per-profile section layout.
        answer_claim_prompt_items: Maximum number of consolidated
            claims passed into the answer prompt's ``Claims`` block.
        result_citation_cap: Maximum number of citations preserved in
            the public :class:`~inqtrix.result.ResearchResult.top_sources`
            list.
        claim_input_char_limit: Maximum text length (chars) per source
            forwarded to the claim-extraction call.
        claim_citation_cap: Maximum number of citations included with
            each extracted claim batch.
        claim_max_items: Maximum number of claims extracted per source
            in a single call.
        claim_source_url_cap: Maximum number of source URLs attached
            to a single consolidated claim.
        answer_citation_block_char_budget: Total character budget for
            the references block at the end of the answer. ``None``
            falls back to the answer-prompt's default scaffold size
            (used by COMPACT).
        claim_ledger_cap: Maximum number of entries kept in the
            consolidated claim ledger across the whole run.
        materialize_max_total: Maximum total number of materialised
            (verified + contested + unverified) claims in the result
            view.
        materialize_max_unverified: Sub-cap for ``unverified`` claims
            within ``materialize_max_total``; prevents the result from
            being dominated by weak evidence.
        session_max_context_blocks: Per-session cap for follow-up turns;
            mirrors but may differ from
            :attr:`~inqtrix.settings.ServerSettings.session_max_context_blocks`.
        session_max_claim_ledger: Per-session cap for the consolidated
            claim ledger preserved across follow-ups.
        session_max_answer_chars: Hard cap on the answer text length
            (chars) preserved in session state.
    """

    settings_overrides: dict[str, int]
    summarize_input_char_limit: int
    summarize_fallback_char_limit: int
    summarize_max_output_tokens: int | None
    answer_body_citation_cap: int
    answer_sections: tuple[AnswerSectionSpec, ...]
    answer_claim_prompt_items: int
    result_citation_cap: int
    claim_input_char_limit: int
    claim_citation_cap: int
    claim_max_items: int
    claim_source_url_cap: int
    answer_citation_block_char_budget: int | None
    claim_ledger_cap: int
    materialize_max_total: int
    materialize_max_unverified: int
    session_max_context_blocks: int
    session_max_claim_ledger: int
    session_max_answer_chars: int


# NOTE on max_output_tokens sizing
# --------------------------------
# Limits leave ~20-25 % headroom on top of the upper word-count guidance.
# Live-traffic measurements (no-thinking COMPACT, 18.04.2026) showed
# "Kernaussagen" at 95 % utilization with the previous 2000-token cap —
# borderline truncation risk. Raised to 2500. Similar safety margin added
# to "Einordnung / Ausblick".
# When extended thinking is enabled the provider auto-raises max_tokens
# to >= 16384 anyway, so these limits only bite in the no-thinking path.
_COMPACT_ANSWER_SECTIONS = (
    AnswerSectionSpec(
        heading="Kurzfazit",
        prompt_instruction=(
            "Beantworte die Frage direkt in 2-4 Saetzen und nenne die wichtigsten "
            "1-2 Erkenntnisse mit sauberer Einordnung."
        ),
        length_guidance="ca. 120-180 Woerter, keine Bulletpoints",
        max_output_tokens=1500,
    ),
    AnswerSectionSpec(
        heading="Kernaussagen",
        prompt_instruction=(
            "Formuliere 5-8 substanzielle Bulletpoints mit den wichtigsten Fakten, "
            "Zahlen und Implikationen. Jeder Punkt soll 1-2 Saetze lang sein."
        ),
        length_guidance="ca. 250-450 Woerter als Bulletpoints",
        max_output_tokens=2500,
    ),
    AnswerSectionSpec(
        heading="Detailanalyse",
        prompt_instruction=(
            "Vertiefe 2-4 Kernaspekte mit 2-4 `###`-Unterabschnitten. Erklaere "
            "Zusammenhaenge, Ursachen und Auswirkungen belastbar und ohne Wiederholungen."
        ),
        length_guidance="ca. 350-600 Woerter mit `###`-Unterabschnitten",
        max_output_tokens=4000,
    ),
    AnswerSectionSpec(
        heading="Einordnung / Ausblick",
        prompt_instruction=(
            "Ordne die Befunde knapp ein und benenne die wichtigsten naechsten "
            "Entwicklungen oder offenen Fragen."
        ),
        length_guidance="ca. 120-180 Woerter, 2-4 Saetze oder ein kurzer Absatz",
        max_output_tokens=1800,
    ),
)

# DEEP sizing follows the same headroom rule as COMPACT. Limits adjusted
# upward where the previous values were too tight against the upper
# word-count guidance: Executive Summary, Risiken, Fazit. Analyse and
# Hintergrund were already comfortable.
_DEEP_ANSWER_SECTIONS = (
    AnswerSectionSpec(
        heading="Executive Summary",
        prompt_instruction=(
            "Beantworte die Frage direkt in 3-5 Saetzen und verdichte die 1-2 "
            "wichtigsten Erkenntnisse mit Stand-Einordnung."
        ),
        length_guidance="ca. 140-220 Woerter, keine Bulletpoints",
        max_output_tokens=1800,
    ),
    AnswerSectionSpec(
        heading="Hintergrund / Kontext",
        prompt_instruction=(
            "Erklaere Ausgangspunkt, zeitlichen Rahmen und relevanten Kontext in 3-5 "
            "Saetzen."
        ),
        length_guidance="ca. 180-260 Woerter als zusammenhaengender Abschnitt",
        max_output_tokens=2600,
    ),
    AnswerSectionSpec(
        heading="Analyse",
        prompt_instruction=(
            "Schreibe 3-5 `###`-Unterabschnitte. Integriere relevante Zahlen, "
            "Vergleichswerte, Mechanismen, Umsetzungsfragen und sachliche Kausalzusammenhaenge."
        ),
        length_guidance="ca. 700-1000 Woerter mit 3-5 `###`-Unterabschnitten",
        max_output_tokens=6000,
    ),
    AnswerSectionSpec(
        heading="Perspektiven / Positionen",
        prompt_instruction=(
            "Stelle die wesentlichen Perspektiven getrennt und neutral dar, inklusive "
            "treibender Akteure, Gegenpositionen und Betroffenen- oder Umsetzungssicht."
        ),
        length_guidance="ca. 300-500 Woerter in klar getrennten Absaetzen",
        max_output_tokens=4000,
    ),
    AnswerSectionSpec(
        heading="Risiken / Unsicherheiten",
        prompt_instruction=(
            "Liste 3-5 praezise Bulletpoints zu Evidenzgrenzen, offenen Punkten, "
            "methodischen Einschraenkungen oder Gegenargumenten."
        ),
        length_guidance="ca. 120-220 Woerter als Bulletpoints",
        max_output_tokens=2000,
    ),
    AnswerSectionSpec(
        heading="Fazit / Ausblick",
        prompt_instruction=(
            "Ziehe eine belastbare Gesamteinordnung und benenne den plausibelsten "
            "naechsten Entwicklungspfad in 2-4 Saetzen."
        ),
        length_guidance="ca. 100-160 Woerter, 2-4 Saetze",
        max_output_tokens=1600,
    ),
)


_COMPACT_TUNING = ReportProfileTuning(
    settings_overrides={},
    summarize_input_char_limit=6000,
    summarize_fallback_char_limit=800,
    summarize_max_output_tokens=None,
    answer_body_citation_cap=14,
    answer_sections=_COMPACT_ANSWER_SECTIONS,
    answer_claim_prompt_items=20,
    result_citation_cap=8,
    claim_input_char_limit=7000,
    claim_citation_cap=8,
    claim_max_items=8,
    claim_source_url_cap=4,
    answer_citation_block_char_budget=None,
    claim_ledger_cap=400,
    materialize_max_total=24,
    materialize_max_unverified=8,
    session_max_context_blocks=8,
    session_max_claim_ledger=50,
    session_max_answer_chars=2000,
)

_DEEP_TUNING = ReportProfileTuning(
    # ``first_round_queries`` raised from 8 to 10 in Phase 13: when an
    # over-confident evaluator (e.g. GPT-4.1) stops the loop after Round 0,
    # the first round is the ONLY chance to broaden the source pool —
    # 8 queries left the answer with only ~11 unique domains in the
    # synthesis, 10 lifts that to ~14-15 without exceeding sensible
    # parallelization (still ``min(len, first_round_queries)`` workers).
    settings_overrides={
        "max_rounds": 5,
        "confidence_stop": 9,
        "max_context": 24,
        "first_round_queries": 10,
        "answer_prompt_citations_max": 500,
        "max_total_seconds": 540,
    },
    summarize_input_char_limit=16000,
    summarize_fallback_char_limit=2500,
    summarize_max_output_tokens=1400,
    answer_body_citation_cap=30,
    answer_sections=_DEEP_ANSWER_SECTIONS,
    answer_claim_prompt_items=40,
    result_citation_cap=20,
    claim_input_char_limit=10000,
    claim_citation_cap=20,
    claim_max_items=12,
    claim_source_url_cap=6,
    answer_citation_block_char_budget=9000,
    claim_ledger_cap=800,
    materialize_max_total=48,
    materialize_max_unverified=16,
    session_max_context_blocks=16,
    session_max_claim_ledger=180,
    session_max_answer_chars=6000,
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
    return _COMPACT_TUNING


def settings_overrides_for_report_profile(profile: ReportProfile | str) -> dict[str, int]:
    """Return the ``AgentSettings`` overrides implied by ``profile``.

    Args:
        profile: Either a :class:`ReportProfile` enum value or its
            string representation. Unknown values fall back to
            ``ReportProfile.COMPACT`` (which has no overrides).

    Returns:
        A new ``dict`` mapping ``AgentSettings`` field names to the
        profile-specific override values. Returns an empty dict for
        the COMPACT profile (no overrides). Caller may mutate the
        returned dict freely — it is a fresh copy of the bundle's
        ``settings_overrides``.
    """
    return dict(tuning_for_report_profile(profile).settings_overrides)
