"""Stop criteria strategy — evaluate whether the research loop should stop."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from inqtrix.domains import is_de_policy_question
from inqtrix.settings import AgentSettings
from inqtrix.text import is_none_value

log = logging.getLogger("inqtrix")


class StopCriteriaStrategy(ABC):
    """Contract for the multi-hook stop cascade in the evaluate node.

    The evaluate node calls ten ordered hooks per round. Each hook
    either adjusts the running ``conf`` value (capped or unchanged),
    mutates the agent state ``s`` to record a signal, or returns a
    boolean directly. The final :meth:`should_stop` rolls all signals
    into the loop-control decision.

    State-mutation contract:

    - ``s`` is the live agent state dict — mutate sparingly and only
      via documented keys (``s["context"]``, ``s["competing_events_*"]``,
      ``s["falsification_active"]``, etc.). Every mutation should be
      mirrored in :data:`~inqtrix.state.AgentState`.
    - Progress updates flow through :func:`_emit_progress`; never
      block the queue.

    Implementations may replace the cascade entirely (build their own
    :meth:`should_stop`) or override individual hooks while inheriting
    from :class:`MultiSignalStopCriteria`.
    """

    @abstractmethod
    def check_contradictions(self, s: dict, eval_text: str, conf: int) -> int:
        """Cap confidence when the evaluator reported source contradictions.

        Args:
            s: Agent state dict. May emit progress messages via
                :func:`_emit_progress` to surface contradiction hits.
                No persistent fields are mutated by the default
                implementation.
            eval_text: Raw evaluator-model output. Implementations
                parse a ``CONTRADICTIONS:`` block from it.
            conf: Current confidence value (1-10) entering this hook.

        Returns:
            Possibly-capped confidence value in ``[0, conf]``. Hard
            contradictions cap to ``confidence_stop - 2``; light
            contradictions cap to ``confidence_stop - 1``.
        """
        ...

    @abstractmethod
    def filter_irrelevant_blocks(self, s: dict, eval_text: str) -> None:
        """Drop context blocks that the evaluator marked as irrelevant.

        Args:
            s: Agent state dict. Mutates ``s["context"]`` by
                removing entries at indices the evaluator listed in
                its ``IRRELEVANT:`` block (1-based indices).
            eval_text: Raw evaluator-model output to parse.

        Returns:
            None. Side-effect-only hook.
        """
        ...

    @abstractmethod
    def extract_competing_events(self, s: dict, eval_text: str, conf: int) -> int:
        """Persist competing-event signals and apply any confidence cap.

        Args:
            s: Agent state dict. Mutates
                ``s["competing_events_history"]`` and
                ``s["competing_events_text"]`` for downstream
                inspection.
            eval_text: Raw evaluator-model output. Implementations
                parse a ``COMPETING_EVENTS:`` block.
            conf: Current confidence value entering this hook.

        Returns:
            Possibly-capped confidence value. Repeated identical
            competing-event text is treated more leniently from
            round 3 onward to avoid thrashing.
        """
        ...

    @abstractmethod
    def extract_evidence_scores(self, s: dict, eval_text: str, conf: int) -> int:
        """Parse evidence sufficiency and consistency ratings.

        Args:
            s: Agent state dict. Mutates ``s["evidence_consistency"]``
                and ``s["evidence_sufficiency"]`` (each in ``[0, 10]``).
                When the evaluator output cannot be parsed, the fields
                stay at their current values and a
                ``_evidence_consistency_parsed`` /
                ``_evidence_sufficiency_parsed`` fallback marker is
                appended to the iteration log so the failure is
                visible.
            eval_text: Raw evaluator-model output. Implementations
                parse ``EVIDENCE_CONSISTENCY:`` and
                ``EVIDENCE_SUFFICIENCY:`` blocks.
            conf: Current confidence value. Returned unchanged unless
                a sufficiency / consistency score below threshold
                triggers a cap.

        Returns:
            Possibly-capped confidence value.
        """
        ...

    @abstractmethod
    def check_falsification(self, s: dict, conf: int, prev_conf: int) -> bool:
        """Return whether the loop should switch into falsification mode.

        Args:
            s: Agent state dict. Mutates
                ``s["falsification_active"]`` to ``True`` when the
                hook decides to enter falsification mode (drives
                planner branch in subsequent rounds).
            conf: Current confidence value after upstream hooks.
            prev_conf: Confidence value from the previous round.

        Returns:
            ``True`` when falsification mode was newly triggered in
            this round (one-shot signal — read by downstream hooks
            to suppress conflicting actions); ``False`` otherwise.
        """
        ...

    @abstractmethod
    def check_stagnation(
        self,
        s: dict,
        conf: int,
        prev_conf: int,
        n_citations: int,
        falsification_just_triggered: bool,
    ) -> tuple[int, bool]:
        """Detect low-confidence stagnation after sufficiently broad search.

        Args:
            s: Agent state dict. May emit progress messages and
                update ``s["stagnation_*"]`` bookkeeping fields.
            conf: Current confidence value.
            prev_conf: Confidence value from the previous round.
            n_citations: Number of distinct citations available so
                far. Used as a "search breadth has plateaued" signal.
            falsification_just_triggered: Whether
                :meth:`check_falsification` triggered in the same
                round (skip stagnation if so to avoid double-stop).

        Returns:
            Tuple ``(adjusted_conf, stagnation_detected)``.
        """
        ...

    @abstractmethod
    def should_suppress_utility_stop(self, s: dict) -> bool:
        """Return whether low-utility stopping should be suppressed.

        Args:
            s: Agent state dict. Implementations typically inspect
                ``enable_de_policy_bias`` flag plus the question text
                to detect German political topics where premature
                low-utility stops are particularly risky.

        Returns:
            ``True`` to suppress utility-stop for this state;
            ``False`` to allow the standard utility-stop path.
        """
        ...

    @abstractmethod
    def compute_utility(
        self, s: dict, conf: int, prev_conf: int, n_citations: int,
    ) -> tuple[float, bool]:
        """Compute per-round utility and whether it should stop the loop.

        Utility is the normalised marginal benefit of the round
        (citation gain / token cost / confidence delta). Falling
        below the configured threshold for two consecutive rounds is
        the standard low-utility stop signal.

        Args:
            s: Agent state dict. May update ``s["utility_history"]``
                and emit progress.
            conf: Current confidence value.
            prev_conf: Confidence value from the previous round.
            n_citations: Number of distinct citations after this
                round.

        Returns:
            Tuple ``(utility_value, should_stop_on_utility)``.
            ``utility_value`` is in ``[0.0, 1.0]``;
            ``should_stop_on_utility`` is ``True`` only when the
            two-round utility window crossed the stop threshold AND
            :meth:`should_suppress_utility_stop` returned ``False``.
        """
        ...

    @abstractmethod
    def check_plateau(
        self, s: dict, conf: int, prev_conf: int, stagnation_detected: bool,
    ) -> bool:
        """Detect stable-confidence plateaus that justify stopping.

        Args:
            s: Agent state dict. May update plateau-bookkeeping
                fields and emit progress.
            conf: Current confidence value.
            prev_conf: Confidence value from the previous round.
            stagnation_detected: Output of :meth:`check_stagnation`.
                Used to coordinate so the loop does not announce
                plateau and stagnation in the same round.

        Returns:
            ``True`` to stop on plateau; ``False`` otherwise. The
            standard plateau condition is "two consecutive rounds
            with confidence change ``< 1`` and confidence already
            within one point of ``confidence_stop``".
        """
        ...

    @abstractmethod
    def should_stop(self, state: dict) -> tuple[bool, str]:
        """Coarse convenience wrapper combining cascade outputs.

        Args:
            state: Agent state dict. Read-only with respect to this
                hook (the per-cascade hooks above are responsible for
                mutations).

        Returns:
            Tuple ``(stop, reason)`` where ``reason`` is a short
            machine-readable tag (``"confidence"``, ``"max_rounds"``,
            ``"plateau"``, ``"utility"``, ``"stagnation"``,
            ``"timeout"``). Used by the evaluate node when it needs a
            single-call go/no-go decision (e.g. for the
            short-circuit stop after the answer node).
        """
        ...


def _emit_progress(s: dict, message: str) -> None:
    """Send a progress update to the stream non-blockingly.

    Args:
        s: Agent state dict that may contain a ``"progress"`` queue
            (set by the streaming entry point). Plain library-mode
            runs leave it ``None`` and the helper short-circuits.
        message: User-facing German progress string.
    """
    q = s.get("progress")
    if q is not None:
        q.put(("progress", message))


class MultiSignalStopCriteria(StopCriteriaStrategy):
    """Three-phase stop heuristic for the iterative research loop.

    The evaluate node first threads confidence through LLM-parsed
    signals (contradictions, competing events, evidence scores), then
    applies deterministic guardrail caps (falsification, stagnation),
    and finally evaluates broader stop conditions (utility, plateau).

    The default implementation tracks state in the following keys
    (all under ``s``):

    - ``competing_events_history`` / ``competing_events_text``
    - ``falsification_active``
    - ``stagnation_history`` / ``stagnation_active``
    - ``utility_history``
    - ``plateau_history``

    All keys are appended to / overwritten in place; readers must
    therefore be tolerant of missing keys on the first round.
    """

    def __init__(self, settings: AgentSettings) -> None:
        """Bind the active :class:`AgentSettings` to the strategy.

        Args:
            settings: Resolved agent settings. The strategy reads
                ``confidence_stop``, ``max_rounds`` and
                ``enable_de_policy_bias``; it does not mutate the
                settings.
        """
        self._settings = settings

    @property
    def _confidence_stop(self) -> int:
        """Return the live ``confidence_stop`` from settings (proxy)."""
        return self._settings.confidence_stop

    @property
    def _max_rounds(self) -> int:
        """Return the live ``max_rounds`` from settings (proxy)."""
        return self._settings.max_rounds

    # ------------------------------------------------------------------ #
    # check_contradictions
    # ------------------------------------------------------------------ #
    def check_contradictions(self, s: dict, eval_text: str, conf: int) -> int:
        """Cap confidence on contradiction signals from the evaluator.

        Implements :meth:`StopCriteriaStrategy.check_contradictions`.
        Parses ``CONTRADICTIONS:`` from ``eval_text``; only acts when
        the line contains ``"ja"`` (German evaluator convention).
        Severe contradictions (``grundlegend``, ``fundamental``,
        ``widerspricht``, ``unvereinbar``, …) cap confidence to
        ``confidence_stop - 2``; lighter contradictions cap to
        ``confidence_stop - 1``.

        Args:
            s: Agent state. Used only for progress emission.
            eval_text: Raw evaluator-model output.
            conf: Current confidence value.

        Returns:
            ``min(conf, cap)`` based on the contradiction severity, or
            ``conf`` unchanged when no ``ja`` was detected.
        """
        m = re.search(r"CONTRADICTIONS:\s*(.+?)(?:\n|$)", eval_text)
        if not m or "ja" not in m.group(1).lower():
            return conf
        contradiction_text = m.group(1).lower()
        severe_keywords = (
            "grundlegend", "fundamental", "gegenteil",
            "widerspricht", "unvereinbar", "komplett",
            "voellig", "falsch", "inkorrekt", "gegensaetzlich",
        )
        if any(kw in contradiction_text for kw in severe_keywords):
            _emit_progress(s, "Schwere Widersprueche erkannt \u2014 Confidence stark begrenzt")
            return min(conf, self._confidence_stop - 2)
        _emit_progress(s, "Leichte Widersprueche erkannt (z.B. Datumsabweichungen)")
        return min(conf, self._confidence_stop - 1)

    # ------------------------------------------------------------------ #
    # filter_irrelevant_blocks
    # ------------------------------------------------------------------ #
    def filter_irrelevant_blocks(self, s: dict, eval_text: str) -> None:
        """Drop context blocks the evaluator flagged as irrelevant.

        Implements :meth:`StopCriteriaStrategy.filter_irrelevant_blocks`.
        Parses ``IRRELEVANT:`` from ``eval_text`` (comma-separated
        1-based block indices). Skips when the value is ``"keine"``,
        ``"none"`` or any other ``is_none_value`` form. Refuses to
        drop when it would empty out the entire context (defensive —
        prevents the evaluator from accidentally erasing all
        evidence on a parse glitch).

        Args:
            s: Agent state. Mutates ``s["context"]`` in place to
                remove the listed indices and emits a progress
                message with the dropped count.
            eval_text: Raw evaluator-model output.

        Returns:
            None. Side-effect-only.
        """
        m = re.search(r"IRRELEVANT:\s*(.+?)(?:\n|$)", eval_text)
        if not m:
            return
        irr_text = m.group(1).strip()
        if is_none_value(irr_text):
            return
        try:
            drop_indices = {
                int(x.strip()) - 1 for x in irr_text.split(",") if x.strip().isdigit()
            }
            if drop_indices and len(drop_indices) < len(s["context"]):
                filtered = [b for i, b in enumerate(s["context"]) if i not in drop_indices]
                dropped = len(s["context"]) - len(filtered)
                s["context"] = filtered
                if dropped:
                    _emit_progress(s, f"{dropped} irrelevante Quellen gefiltert")
        except (ValueError, TypeError):
            pass

    # ------------------------------------------------------------------ #
    # extract_competing_events
    # ------------------------------------------------------------------ #
    def extract_competing_events(self, s: dict, eval_text: str, conf: int) -> int:
        """Record competing explanations and cap confidence for new ambiguity.

        Repeated competing-event text is treated more leniently from round 3
        onward so the agent does not thrash forever on the same ambiguity.
        """
        m = re.search(r"COMPETING_EVENTS:\s*(.+?)(?:\n|$)", eval_text)
        if not m:
            return conf
        comp_text = m.group(1).strip()
        if is_none_value(comp_text):
            s["competing_events"] = ""
            log.info(
                "TRACE evaluate: competing_events=None (parsed '%s')",
                comp_text.lower()[:60],
            )
            return conf

        s["competing_events"] = comp_text
        log.info("TRACE evaluate: competing_events='%s'", comp_text[:200])
        _emit_progress(s, "Mehrere moegliche Erklaerungen erkannt")

        _prev_comp = s.get("prev_competing_events", "")
        _comp_is_new = (not _prev_comp) or (comp_text != _prev_comp)
        if conf >= self._confidence_stop and (_comp_is_new or s["round"] < 3):
            conf = self._confidence_stop - 1
            log.info(
                "TRACE evaluate: competing_events cap applied (is_new=%s, round=%d)",
                _comp_is_new, s["round"],
            )
        elif conf >= self._confidence_stop:
            log.info(
                "TRACE evaluate: competing_events cap SKIPPED "
                "(same text for 2+ rounds, round=%d)",
                s["round"],
            )
        return conf

    # ------------------------------------------------------------------ #
    # extract_evidence_scores
    # ------------------------------------------------------------------ #
    def extract_evidence_scores(self, s: dict, eval_text: str, conf: int) -> int:
        """Parse evidence consistency / sufficiency scores from evaluator output.

        Implements :meth:`StopCriteriaStrategy.extract_evidence_scores`.
        Looks for ``EVIDENCE_CONSISTENCY: N`` and
        ``EVIDENCE_SUFFICIENCY: N`` (each ``N`` in ``[0, 10]``).
        When either is missing, defaults to ``5`` and sets the
        corresponding ``_*_parsed`` flag to ``False`` so the iteration
        log shows the fallback.

        Sanity check: when both scores are exactly ``0`` AND
        confidence is at or above ``confidence_stop``, caps confidence
        to ``confidence_stop - 1`` (defensive — the model claimed high
        confidence with zero evidence support, which is almost always
        a parse glitch or model halluciation).

        Args:
            s: Agent state. Mutates ``s["evidence_consistency"]``,
                ``s["evidence_sufficiency"]``,
                ``s["_evidence_consistency_parsed"]``,
                ``s["_evidence_sufficiency_parsed"]``.
            eval_text: Raw evaluator-model output.
            conf: Current confidence value.

        Returns:
            Possibly-capped confidence value.
        """
        m_consistency = re.search(r"EVIDENCE_CONSISTENCY:\s*(\d+)", eval_text)
        if m_consistency:
            s["evidence_consistency"] = int(m_consistency.group(1))
            s["_evidence_consistency_parsed"] = True
        else:
            s["evidence_consistency"] = 5
            s["_evidence_consistency_parsed"] = False
            log.warning(
                "TRACE evaluate: EVIDENCE_CONSISTENCY field missing in LLM response -> default 5",
            )

        m_sufficiency = re.search(r"EVIDENCE_SUFFICIENCY:\s*(\d+)", eval_text)
        if m_sufficiency:
            s["evidence_sufficiency"] = int(m_sufficiency.group(1))
            s["_evidence_sufficiency_parsed"] = True
        else:
            s["evidence_sufficiency"] = 5
            s["_evidence_sufficiency_parsed"] = False
            log.warning(
                "TRACE evaluate: EVIDENCE_SUFFICIENCY field missing in LLM response -> default 5",
            )

        if (
            s["evidence_consistency"] == 0
            and s["evidence_sufficiency"] == 0
            and conf >= self._confidence_stop
        ):
            log.warning(
                "TRACE evaluate: evidence sanity check failed "
                "(consistency=0, sufficiency=0, conf=%d -> %d)",
                conf, self._confidence_stop - 1,
            )
            conf = self._confidence_stop - 1

        log.info(
            "TRACE evaluate: evidence_consistency=%d evidence_sufficiency=%d",
            s["evidence_consistency"], s["evidence_sufficiency"],
        )
        return conf

    # ------------------------------------------------------------------ #
    # check_falsification
    # ------------------------------------------------------------------ #
    def check_falsification(self, s: dict, conf: int, prev_conf: int) -> bool:
        """Toggle falsification mode based on consistently low confidence.

        Implements :meth:`StopCriteriaStrategy.check_falsification`.
        Two transitions:

        - **Release** (sticky-off): if the previous round had marked
          ``falsification_triggered`` and confidence has recovered to
          at least ``confidence_stop - 2``, clears the flag so the
          plan node stops injecting falsification prompt blocks.
        - **Trigger**: when round ``>= 2`` AND both ``prev_conf`` and
          ``conf`` are ``<= 4``, sets ``falsification_triggered`` and
          returns ``True`` so the plan node switches to
          falsification-style queries in the next round.

        Args:
            s: Agent state. Mutates ``s["falsification_triggered"]``
                in either direction. Emits progress on both transitions.
            conf: Current confidence value.
            prev_conf: Confidence value from the previous round.

        Returns:
            ``True`` only when falsification mode was newly triggered
            in this round (one-shot signal). ``False`` for the
            release case and for steady-state continuation.
        """
        # Reset the sticky falsification mode once confidence recovers
        # meaningfully — otherwise the plan node keeps injecting
        # "debunked / hoax / refuted" prompt blocks long after the
        # evidence picture has changed.
        if (
            s.get("falsification_triggered", False)
            and conf >= self._confidence_stop - 2
        ):
            s["falsification_triggered"] = False
            log.info(
                "TRACE evaluate: falsification mode RELEASED "
                "(conf=%d >= confidence_stop-2=%d, round=%d)",
                conf, self._confidence_stop - 2, s["round"],
            )
            _emit_progress(
                s,
                "Confidence wieder hoch \u2014 Falsifikations-Modus deaktiviert",
            )

        if (
            s["round"] >= 2
            and prev_conf > 0
            and prev_conf <= 4
            and conf <= 4
            and not s.get("falsification_triggered", False)
        ):
            s["falsification_triggered"] = True
            log.info(
                "TRACE evaluate: falsification triggered (prev=%d, curr=%d, round=%d)",
                prev_conf, conf, s["round"],
            )
            _emit_progress(s, "Niedrige Evidenz \u2014 starte Falsifikations-Recherche")
            return True
        return False

    # ------------------------------------------------------------------ #
    # check_stagnation
    # ------------------------------------------------------------------ #
    def check_stagnation(
        self,
        s: dict,
        conf: int,
        prev_conf: int,
        n_citations: int,
        falsification_just_triggered: bool,
    ) -> tuple[int, bool]:
        """Force a stop when broad search repeatedly fails to raise confidence.

        This is the negative-evidence branch described in the architecture:
        after multiple low-confidence rounds and enough citation breadth, the
        loop treats the lack of supporting evidence as informative.
        """
        if (
            s["round"] >= 2
            and prev_conf > 0
            and prev_conf <= 4
            and conf <= 4
            and abs(conf - prev_conf) <= 1
            and not falsification_just_triggered
            and (n_citations >= 30 or s.get("falsification_triggered", False))
        ):
            log.info(
                "TRACE evaluate: stagnation detected "
                "(prev=%d, curr=%d, citations=%d, falsified=%s) -> forcing stop",
                prev_conf, conf, n_citations, s.get("falsification_triggered", False),
            )
            _emit_progress(
                s,
                "Umfangreiche Recherche abgeschlossen \u2014 "
                "Praemisse der Frage wahrscheinlich falsch",
            )
            return self._confidence_stop, True
        return conf, False

    # ------------------------------------------------------------------ #
    # should_suppress_utility_stop
    # ------------------------------------------------------------------ #
    def should_suppress_utility_stop(self, s: dict) -> bool:
        """Keep searching when utility is low but policy evidence is still weak."""
        if int(s.get("round", 0) or 0) >= self._max_rounds:
            return False
        if str(s.get("query_type", "general")) == "academic":
            return False

        if not getattr(self._settings, "enable_de_policy_bias", True):
            return False
        if not is_de_policy_question(s.get("question", "") or ""):
            return False

        uncovered_n = len(s.get("uncovered_aspects", []) or [])
        if uncovered_n > 0:
            return True

        tiers = s.get("source_tier_counts", {}) or {}
        primary_n = int(tiers.get("primary", 0) or 0)
        mainstream_n = int(tiers.get("mainstream", 0) or 0)

        claim_quality = float(s.get("claim_quality_score", 0.0) or 0.0)
        claim_counts = s.get("claim_status_counts", {}) or {}
        verified = int(claim_counts.get("verified", 0) or 0)
        unverified = int(claim_counts.get("unverified", 0) or 0)

        np_total = int(s.get("claim_needs_primary_total", 0) or 0)
        np_verified = int(s.get("claim_needs_primary_verified", 0) or 0)
        if np_total > 0 and np_verified < np_total:
            return True

        if (primary_n + mainstream_n) == 0 and (unverified > verified or claim_quality < 0.35):
            return True

        return False

    # ------------------------------------------------------------------ #
    # compute_utility
    # ------------------------------------------------------------------ #
    def compute_utility(
        self,
        s: dict,
        conf: int,
        prev_conf: int,
        n_citations: int,
    ) -> tuple[float, bool]:
        """Compute marginal research utility from confidence, citations, and sufficiency.

        Returns:
            Tuple of ``(utility, should_stop)``. The stop signal is emitted
            only when two consecutive low-utility rounds occur and the policy
            suppression rules do not keep the loop alive.
        """
        _delta_conf = (conf - prev_conf) / 10.0 if prev_conf > 0 else 0.5
        _new_cit = n_citations - s.get("prev_citation_count", 0)
        _delta_cit_norm = min(1.0, _new_cit / 10.0)
        _sufficiency_norm = s.get("evidence_sufficiency", 5) / 10.0
        utility = round(
            0.4 * _delta_conf + 0.3 * _delta_cit_norm + 0.3 * _sufficiency_norm, 4,
        )
        s["utility_scores"].append(utility)
        s["prev_citation_count"] = n_citations

        log.info(
            "TRACE evaluate: utility=%.4f (delta_conf=%.2f delta_cit=%.2f suff=%.2f) scores=%s",
            utility, _delta_conf, _delta_cit_norm, _sufficiency_norm,
            [round(u, 3) for u in s["utility_scores"]],
        )

        utility_stop = False
        if len(s["utility_scores"]) >= 2 and not s["done"]:
            _last = s["utility_scores"][-1]
            _prev_u = s["utility_scores"][-2]
            if _last < 0.15 and _prev_u < 0.15:
                if self.should_suppress_utility_stop(s):
                    log.info(
                        "TRACE evaluate: utility stop SUPPRESSED "
                        "(evidence still weak) scores=%s",
                        [round(u, 3) for u in s["utility_scores"][-2:]],
                    )
                    _emit_progress(
                        s,
                        "Informationsgewinn stagniert, aber Evidenzlage "
                        "noch zu schwach \u2014 suche weiter",
                    )
                else:
                    utility_stop = True
                    s["done"] = True
                    log.info(
                        "TRACE evaluate: utility stop triggered (scores=%s)",
                        [round(u, 3) for u in s["utility_scores"][-2:]],
                    )
                    _emit_progress(s, "Informationsgewinn stagniert \u2014 beende Recherche")
        return utility, utility_stop

    # ------------------------------------------------------------------ #
    # check_plateau
    # ------------------------------------------------------------------ #
    def check_plateau(
        self,
        s: dict,
        conf: int,
        prev_conf: int,
        stagnation_detected: bool,
    ) -> bool:
        """Stop when confidence stays stable at a sufficiently high level.

        Plateau stopping is suppressed while competing events are still moving,
        unless that ambiguity has effectively stabilized across repeated rounds.
        """
        _prev_comp_for_plateau = s.get("prev_competing_events", "")
        _current_competing = s.get("competing_events", "")
        _competing_active_and_changing = (
            bool(_current_competing)
            and _current_competing != _prev_comp_for_plateau
        )

        _conf_stable_rounds = s.get("_conf_stable_rounds", 0)
        if prev_conf > 0 and conf == prev_conf:
            _conf_stable_rounds += 1
        else:
            _conf_stable_rounds = 0
        s["_conf_stable_rounds"] = _conf_stable_rounds

        if _competing_active_and_changing and _conf_stable_rounds >= 2:
            log.info(
                "TRACE evaluate: competing events override \u2014 conf stable for %d rounds, "
                "treating as non-changing for plateau check",
                _conf_stable_rounds,
            )
            _competing_active_and_changing = False

        plateau_stop = False
        if (
            s["round"] >= 2
            and prev_conf > 0
            and conf == prev_conf
            and conf >= 6
            and not stagnation_detected
            and not _competing_active_and_changing
            and not s["done"]
        ):
            plateau_stop = True
            s["done"] = True
            log.info(
                "TRACE evaluate: plateau stop triggered "
                "(conf=%d stable for 2 rounds, round=%d)",
                conf, s["round"],
            )
            _emit_progress(s, f"Confidence {conf}/10 stabil \u2014 Recherche abgeschlossen")
        elif _competing_active_and_changing and not s["done"]:
            log.info(
                "TRACE evaluate: plateau stop SUPPRESSED "
                "(competing events changed, round=%d)",
                s["round"],
            )

        s["prev_competing_events"] = _current_competing
        return plateau_stop

    # ------------------------------------------------------------------ #
    # should_stop  (combined convenience method)
    # ------------------------------------------------------------------ #
    def should_stop(self, state: dict) -> tuple[bool, str]:
        """High-level stop check combining all signals.

        Returns ``(should_stop, reason)`` where *reason* is a short tag
        describing which signal triggered the stop (empty when not stopping).

        This method is intentionally a thin wrapper.  The evaluate node
        typically calls the individual ``check_*`` / ``compute_*`` methods
        directly so it can thread confidence values between them; this
        combined entry-point is provided for simpler call-sites that only
        need a boolean answer.
        """
        if state.get("done"):
            return True, "already_done"

        conf = int(state.get("final_confidence", 0))
        if conf >= self._confidence_stop:
            return True, "confidence"

        if int(state.get("round", 0)) >= self._max_rounds:
            return True, "max_rounds"

        return False, ""
