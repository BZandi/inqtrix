# Stop criteria

> Files: `strategies/_stop_criteria.py` (`MultiSignalStopCriteria`), `nodes.py` (`evaluate`)

## Scope

The nine-step heuristic cascade that decides whether the research loop should continue. This page is the canonical reference — the evaluate node and every derived `StopCriteriaStrategy` implementation must respect these semantics.

## Why multiple signals

A single confidence threshold is easy to game: an LLM can hallucinate certainty, or the loop can stagnate at a high-but-unwarranted score. Inqtrix defends against both by combining LLM-reported confidence with structural signals (contradictions, competing events, source quality, aspect coverage, utility delta, plateau detection) and by treating them as independent caps rather than a single linear pipeline.

## Cascade structure

The heuristics are grouped into three phases. Confidence is threaded through each phase and can be reduced but not increased by caps, except in `check_stagnation`, which can raise confidence to the stop threshold when search has been exhaustive.

```mermaid
flowchart TD
    A["LLM returns raw confidence"] --> B["Group A: LLM-parse heuristics"]
    subgraph GA["Group A — Thread confidence through"]
        B1["1. check_contradictions<br/>Severe -> cap conf-2<br/>Light -> cap conf-1"]
        B2["2. filter_irrelevant_blocks<br/>CRAG-style block dropping"]
        B3["3. extract_competing_events<br/>New events -> cap to conf_stop-1<br/>Stable events (2+ rounds) -> skip cap"]
        B4["4. extract_evidence_scores<br/>Sanity: both=0 -> cap conf_stop-1"]
        B1 --> B2 --> B3 --> B4
    end
    B --> B1
    B4 --> C["Group B: Guardrail caps"]
    subgraph GB["Group B — Independent arithmetic caps"]
        C1["5a. No citations -> cap 6"]
        C2["5b. Low >> high sources -> cap 7"]
        C3["5c. Missing primary -> cap 8"]
        C4["5d. Uncovered aspects -> cap 8"]
        C5["5e. 2+ contested claims -> cap 7"]
    end
    C --> C1
    C --> C2
    C --> C3
    C --> C4
    C --> C5
    C5 --> D["Group C: Post-LLM stop heuristics"]
    subgraph GC["Group C — Thread confidence, check stop signals"]
        D1["6. check_falsification<br/>2 rounds, 0 < prev_conf <= 4,<br/>conf <= 4, one-time trigger"]
        D2["7. check_stagnation<br/>2+ rounds, conf and prev_conf <= 4,<br/>30+ citations or falsification tried<br/>-> raise conf to threshold, stop"]
        D3["8. compute_utility<br/>0.4*delta_conf + 0.3*delta_cit + 0.3*suff<br/>Last 2 < 0.15 -> stop<br/>(unless policy suppression active)"]
        D4["9. check_plateau<br/>Stable conf >= 6 for 2+ rounds<br/>no competing events changing -> stop"]
        D1 --> D2 --> D3 --> D4
    end
    D --> D1
    D4 --> E{"Final: conf >= threshold<br/>OR round >= max?"}
    E -->|Yes| F["done=True"]
    E -->|No| G["done=False"]
```

## Key heuristic details

### Competing-events suppression

If competing events stay unchanged into round 3+ (same text as the previous round), the cap in step 3 is skipped. This prevents thrashing on the same competing explanations while still forcing at least one explicit disambiguation round.

### Falsification trigger

All conditions must be true to arm falsification: `round >= 2`, `0 < prev_conf`, `prev_conf <= 4`, `conf <= 4`, and NOT already triggered. The flag is set once and never re-triggers. Subsequent plan rounds emit debunk-style queries (see [Falsification](falsification.md)).

### Negative-evidence hinting

Injected into the evaluate prompt when `round >= 2`. If additionally `prev_conf > 0` and `prev_conf <= 4`, a stronger hint is added: after N rounds with 30+ citations and confidence still at or below 4, absence of evidence is treated as a strong signal that the premise is false (suggested confidence 7–9).

### Utility suppression for policy questions

`should_suppress_utility_stop()` prevents utility-based stopping when the question matches the policy regex AND any of the following hold:

- Uncovered aspects remain.
- Claims flagged `needs_primary` still lack primary verification.
- No quality sources found AND (unverified > verified OR `claim_quality_score < 0.35`).

This ensures policy-critical questions are not prematurely abandoned due to low marginal utility.

## Stopping rules summary

| Rule | Condition | Effect |
|------|-----------|--------|
| **Confidence** | `conf >= confidence_stop` (default 8) | Stop |
| **Max rounds** | `round >= max_rounds` (default 4) | Stop |
| **Contradictions** | Severe conflicting sources | Cap conf by −1 or −2 |
| **Competing events** | Multiple explanations (new) | Cap to threshold−1; force disambiguation |
| **Falsification** | 2+ rounds, `prev_conf` and `conf` both <= 4, one-time | Switch to debunk-style queries |
| **Stagnation** | No improvement + broad search done (30+ citations) | Raise conf to threshold, stop |
| **Utility plateau** | Last two rounds both utility < 0.15 | Stop (unless policy suppression) |
| **Confidence plateau** | Same conf >= 6 for 2+ rounds, no competing events changing | Stop |
| **Negative evidence** | Round >= 2, searched broadly, found little | Prompt hint: infer absence as evidence |

## `StopCriteriaStrategy` ABC — full method list

| Method | Signature | Returns |
|--------|-----------|---------|
| `check_contradictions` | `(s, eval_text, conf) -> int` | Modified confidence |
| `filter_irrelevant_blocks` | `(s, eval_text) -> None` | Modifies state in place |
| `extract_competing_events` | `(s, eval_text, conf) -> int` | Modified confidence |
| `extract_evidence_scores` | `(s, eval_text, conf) -> int` | Modified confidence |
| `check_falsification` | `(s, conf, prev_conf) -> bool` | Triggered flag |
| `check_stagnation` | `(s, conf, prev_conf, n_citations, falsification_just_triggered) -> tuple[int, bool]` | (conf, detected) |
| `should_suppress_utility_stop` | `(s) -> bool` | Suppress flag |
| `compute_utility` | `(s, conf, prev_conf, n_citations) -> tuple[float, bool]` | (utility, stop) |
| `check_plateau` | `(s, conf, prev_conf, stagnation_detected) -> bool` | Stop flag |
| `should_stop` | `(state) -> tuple[bool, str]` | (stop, reason) |

## Implementing your own stop strategy

Subclass `MultiSignalStopCriteria` rather than `StopCriteriaStrategy` so you inherit the default cascade and only override the specific checks you want to change. The most common overrides in practice are `compute_utility` (tighter or looser plateau) and `should_suppress_utility_stop` (domain-specific protection list).

## Related docs

- [Confidence](confidence.md)
- [Falsification](falsification.md)
- [Aspect coverage](aspect-coverage.md)
- [Claims](claims.md)
- [Source tiering](source-tiering.md)
