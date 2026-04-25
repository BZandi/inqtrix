# Aspect coverage

> File: `strategies/_risk_scoring.py` (`KeywordRiskScorer`)

## Scope

How Inqtrix decomposes a question into required aspects, how it checks whether those aspects are covered after each search round, and how uncovered aspects feed back into query planning and confidence capping.

## Aspect generation

Aspects are derived once, by the classify node, from the question text and the detected query type:

| Trigger | Aspects |
|---------|---------|
| Type = news / general | Status quo with date, Actor positions, Discussion direction |
| Policy keywords | Political feasibility, Proposal vs enacted distinction |
| Numeric keywords | Numbers with primary source or uncertainty |
| Type = academic | Primary publication + core claim, Methods & limitations |

Maximum 6 aspects; deduplication preserves order. The derivation is deterministic (regex plus fixed templates) — no LLM is involved at this step.

## Coverage estimation

After each search round, `estimate_aspect_coverage()` re-evaluates coverage by matching augmented tokens against the accumulated context text:

1. Extract tokens from each aspect (length > 3 characters).
2. Augment with hard-coded domain-specific synonyms:

| Aspect contains | Added synonym tokens |
|-----------------|---------------------|
| `status quo` | `status`, `stand`, `aktuell`, `derzeit` |
| `datum` | `datum`, `stand`, `heute` |
| `position` + `akteur` | `regierung`, `partei`, `verband`, `akteur` |
| `richtung` OR `diskussion` | `trend`, `debatte`, `diskussion`, `entwicklung` |
| `mehrheitslage` OR `umsetzbarkeit` | `mehrheit`, `mehrheitsfaehig`, `durchsetzbar`, `umsetzbar` |

3. Check whether any augmented token appears in the accumulated context text.
4. An aspect counts as covered if at least one token matches; uncovered if zero hits.
5. Coverage ratio = `aspects_covered / total_aspects`, rounded to 3 decimals.

The matching is deliberately fuzzy so that obvious paraphrases qualify without needing an LLM; the trade-off is that an aspect can appear "covered" while the actual statement is shallow. That trade-off is accepted because the answer node still operates on the full claim ledger and citations.

Example iteration-log shape:

```json
{
  "node": "evaluate",
  "round": 1,
  "required_aspects": [
    "Status quo with date",
    "Actor positions",
    "Discussion direction"
  ],
  "uncovered_aspects": ["Actor positions"],
  "aspect_coverage": 0.667,
  "cap_reason": "uncovered_aspects",
  "final_confidence": 8
}
```

## Consequences of uncovered aspects

Uncovered aspects have two direct effects on the control loop:

- **Confidence cap at 8.** The evaluate node guardrails final confidence to a maximum of 8 while any aspect remains uncovered. See [Stop criteria](stop-criteria.md) for the full cascade.
- **Targeted queries in subsequent plan rounds.** The plan node injects at least one query per uncovered aspect, using the same synonym list above when the aspect has a domain-specific keyword.

Typical targeted follow-up query:

```text
site:bundesgesundheitsministerium.de GKV Reform Positionen Krankenkassen Verbaende aktueller Stand
```

## Interaction with stop criteria

Two stop heuristics also consult aspect coverage:

- `should_suppress_utility_stop()` keeps the loop running on policy-critical questions if uncovered aspects remain, even when marginal utility drops below the usual plateau threshold.
- `check_plateau()` still fires when confidence stabilises high enough even with uncovered aspects — the coverage cap above ensures this only happens at `confidence >= 8`, which is an accepted policy.

## Related docs

- [Claims](claims.md)
- [Source tiering](source-tiering.md)
- [Stop criteria](stop-criteria.md)
- [Nodes](../architecture/nodes.md)
