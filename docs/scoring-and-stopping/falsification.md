# Falsification mode

> Files: `strategies/_stop_criteria.py` (`MultiSignalStopCriteria.check_falsification`), `nodes.py` (`plan`)

## Scope

What falsification mode means in Inqtrix, when it is armed, how it changes query generation, and why it is a one-shot trigger.

## Motivation

Naive search loops keep asking the same question in slightly different words when confidence stays low. That behaviour is expensive and sycophantic — models tend to over-assert a premise they cannot disprove. Falsification mode inverts the search strategy after a few low-confidence rounds: the loop actively looks for disproof instead of looking for more supporting evidence.

The idea is inspired by the FVA-RAG line of work (*Falsification-Verification Alignment for Mitigating Sycophantic Hallucinations*).

## Trigger conditions

All of the following must be true:

- `round >= 2` (we have tried at least twice).
- `0 < prev_conf <= 4` (prior round had meaningful but low confidence).
- `conf <= 4` (current round also low).
- NOT already triggered (`falsification_triggered` is false).

The strategy records the new state by setting `falsification_triggered = True`. The flag is never cleared for the duration of the run; the mode is strictly one-shot to avoid oscillation.

Minimal state transition:

```json
{
  "round": 2,
  "prev_conf": 4,
  "confidence": 3,
  "falsification_triggered": true
}
```

## Effect on the plan node

When `falsification_triggered` is true, the plan node generates queries in a different distribution:

- 2/3 of new queries are **debunk-style** — they explicitly search for disproof, counter-examples, retractions, or absence of the claimed fact.
- 1/3 of new queries are **nearest-explanation** — they search for the actual fact that has been confused with the user's premise.

The LLM prompt for plan acknowledges the switch explicitly, so that a reasoning model does not silently revert to confirmation-seeking queries.

Example query distribution after the trigger:

```text
2 debunk-style queries:
- "<claim> counter evidence retraction"
- "<claim> no evidence official source"

1 nearest-explanation query:
- "<confused term> actual policy status official source"
```

## Interaction with stagnation

Falsification is followed shortly afterwards by `check_stagnation`. If falsification has been armed AND the round-count is sufficient AND 30+ citations have been accumulated AND confidence is still `<= 4`, stagnation raises `final_confidence` to `confidence_stop` and stops the loop. The reasoning: we have actively searched for disproof and still have no strong signal — the most honest answer is "no evidence found", which calibrates better at high confidence than at low confidence.

## Interaction with negative-evidence hinting

The evaluate-prompt hint that treats absence of evidence as a legitimate signal (see [Stop criteria](stop-criteria.md)) is injected whenever `round >= 2`. When `prev_conf > 0 AND prev_conf <= 4`, the hint is stronger, which lines up with the falsification pre-conditions. The two mechanisms are complementary: falsification changes the questions, negative-evidence hinting changes how the LLM scores the accumulated answer.

## What falsification does not do

- It does not reset the claim ledger. Accumulated claims remain available and continue to be consolidated.
- It does not lower the stop threshold. `confidence_stop` stays at its configured value; termination still requires either stagnation, plateau, max-rounds, or a confidence trajectory that crosses the threshold.
- It does not re-trigger. Even if later rounds drop back to low confidence, the plan node does not arm falsification a second time.

## Configuration

No dedicated configuration knob exists; the feature is fully driven by `confidence_stop`, `max_rounds`, and the internal trigger conditions. Custom `StopCriteriaStrategy` implementations that want to disable falsification should override `check_falsification` to always return `False` (or equivalently, return the incoming flag unchanged).

## Related docs

- [Stop criteria](stop-criteria.md)
- [Confidence](confidence.md)
- [Nodes](../architecture/nodes.md) (plan node behaviour)
- [Research foundations](../reference/research-foundations.md) (FVA-RAG reference)
