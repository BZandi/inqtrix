"""Stop criteria strategy — evaluate whether the research loop should stop."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from inqtrix.settings import AgentSettings
from inqtrix.text import is_none_value

log = logging.getLogger("inqtrix")


class StopCriteriaStrategy(ABC):
    """Evaluate whether the research loop should stop."""

    @abstractmethod
    def check_contradictions(self, s: dict, eval_text: str, conf: int) -> int:
        """Cap confidence when the evaluator reported source contradictions."""
        ...

    @abstractmethod
    def filter_irrelevant_blocks(self, s: dict, eval_text: str) -> None:
        """Drop context blocks that the evaluator marked as irrelevant."""
        ...

    @abstractmethod
    def extract_competing_events(self, s: dict, eval_text: str, conf: int) -> int:
        """Persist competing-event signals and apply any confidence cap."""
        ...

    @abstractmethod
    def extract_evidence_scores(self, s: dict, eval_text: str, conf: int) -> int:
        """Parse evidence sufficiency and consistency ratings from evaluator output."""
        ...

    @abstractmethod
    def check_falsification(self, s: dict, conf: int, prev_conf: int) -> bool:
        """Return whether the loop should switch into falsification mode."""
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
        """Detect low-confidence stagnation after sufficiently broad search."""
        ...

    @abstractmethod
    def should_suppress_utility_stop(self, s: dict) -> bool:
        """Return whether low-utility stopping should be suppressed for this state."""
        ...

    @abstractmethod
    def compute_utility(
        self, s: dict, conf: int, prev_conf: int, n_citations: int,
    ) -> tuple[float, bool]:
        """Compute per-round utility and whether it should stop the loop."""
        ...

    @abstractmethod
    def check_plateau(
        self, s: dict, conf: int, prev_conf: int, stagnation_detected: bool,
    ) -> bool:
        """Detect stable-confidence plateaus that justify stopping."""
        ...

    @abstractmethod
    def should_stop(self, state: dict) -> tuple[bool, str]:
        """Provide a coarse convenience stop decision and reason tag."""
        ...


def _emit_progress(s: dict, message: str) -> None:
    """Send a progress update to the stream (thin helper)."""
    q = s.get("progress")
    if q is not None:
        q.put(("progress", message))


class MultiSignalStopCriteria(StopCriteriaStrategy):
    """Three-phase stop heuristic for the iterative research loop.

    The evaluate node first threads confidence through LLM-parsed signals,
    then applies deterministic guardrail caps, and finally evaluates broader
    stop conditions such as falsification, stagnation, utility decay, and
    plateau behavior.
    """

    def __init__(self, settings: AgentSettings) -> None:
        self._settings = settings

    @property
    def _confidence_stop(self) -> int:
        return self._settings.confidence_stop

    @property
    def _max_rounds(self) -> int:
        return self._settings.max_rounds

    # ------------------------------------------------------------------ #
    # check_contradictions
    # ------------------------------------------------------------------ #
    def check_contradictions(self, s: dict, eval_text: str, conf: int) -> int:
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
        m_consistency = re.search(r"EVIDENCE_CONSISTENCY:\s*(\d+)", eval_text)
        s["evidence_consistency"] = int(m_consistency.group(1)) if m_consistency else 5

        m_sufficiency = re.search(r"EVIDENCE_SUFFICIENCY:\s*(\d+)", eval_text)
        s["evidence_sufficiency"] = int(m_sufficiency.group(1)) if m_sufficiency else 5

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

        ql = (s.get("question", "") or "").lower()
        is_policyish = bool(
            re.search(
                r"\b(privatis\w*|gkv|krankenkass\w*|gesetz\w*|recht\w*|verordnung\w*"
                r"|regulier\w*|politik\w*|beitrag\w*|kosten|haushalt\w*)\b",
                ql,
            )
        )
        if not is_policyish:
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
