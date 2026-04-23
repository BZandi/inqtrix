# Research foundations

## Scope

Which ideas from the research literature and open-source ecosystem influenced specific parts of the Inqtrix implementation. Inqtrix does not claim a single novel contribution — its value lies in the engineering-level integration of several established techniques into a coherent, bounded research loop.

## Related open-source systems

| Project | Shared idea | Main difference |
|---------|-------------|-----------------|
| [GPT Researcher](https://github.com/assafelovic/gpt-researcher) | Iterative search and synthesis with autonomous agent workflow | Broader multi-agent architecture; different retrieval strategy and report-generation pipeline |
| [STORM](https://github.com/stanford-oval/storm) | Multi-perspective question expansion and diverse search planning | Oriented toward long-form Wikipedia-style article generation rather than bounded Q&A with confidence tracking |
| [Open Deep Research](https://github.com/dzhng/deep-research) | Recursive search with configurable depth and breadth | Simpler stack; different trade-off between search breadth and evaluation signals |

The macro pattern — iterative search plus synthesis — is well established. What differs across systems is how they decide *when to stop*, *how to verify claims*, and *how to handle conflicting evidence*.

## Literature anchors

Each row maps a published work to the specific Inqtrix component it influenced.

| Work | Full title | Relevant idea | Inqtrix component |
|------|-----------|---------------|-------------------|
| Self-RAG | *Learning to Retrieve, Generate, and Critique through Self-Reflection* | Retrieve-critique-generate loops with self-assessment | Evaluate node's LLM self-assessment of confidence, gaps, and contradictions |
| CRAG | *Corrective Retrieval Augmented Generation* | Corrective retrieval and retrieval-quality checks | `filter_irrelevant_blocks()` — CRAG-style block dropping based on relevance scoring |
| STORM | *Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models* | Multi-perspective search planning | Perspective-diversity instruction in the plan node (round 1+), stakeholder viewpoint rotation |
| TCR | *Seeing through the Conflict: Transparent Knowledge Conflict Handling in RAG* | Explicit conflict and consistency signals | `check_contradictions()` and `extract_competing_events()` — structured conflict detection with severity grading |
| FVA-RAG | *Falsification-Verification Alignment for Mitigating Sycophantic Hallucinations* | Deliberate falsification and anti-sycophancy retrieval | Falsification mode: after 2+ rounds with low confidence, generate debunk-style queries that actively seek to disprove the premise |
| Over-Searching | *Over-Searching in Search-Augmented Large Language Models* | Search-cost and abstention trade-offs | `compute_utility()` with marginal-gain formula; `should_suppress_utility_stop()` for policy questions; plateau detection |
| Agentic-R | *Learning to Retrieve for Agentic Search* | Retrieval utility in multi-turn search | Utility-score formula (`0.4*delta_conf + 0.3*delta_cit + 0.3*sufficiency`); round-adaptive query count reduction |
| GRACE | *Reinforcement Learning for Grounded Response and Abstention under Contextual Evidence* | Evidence sufficiency and answer-vs-abstain framing | `evidence_sufficiency` score in the evaluate node; stagnation detection that treats absence of evidence as a valid high-confidence finding |

## What is distinctive in this implementation

Inqtrix is a combination of:

- A **claim ledger** with explicit consolidation status (`verified` / `contested` / `unverified`) and signature-based deduplication.
- **Domain-based source tiering** with weighted quality scores (five tiers, Germany-focused domain lists).
- **Aspect coverage** as a formal completeness signal that caps confidence and drives targeted follow-up queries.
- **Risk-based model routing** between lightweight and heavyweight models depending on question sensitivity.
- A **falsification mode** that fundamentally changes the search strategy when evidence consistently fails to materialise.
- **Multiple stopping signals** (nine-step heuristic cascade) rather than a single confidence threshold, including utility suppression for policy-critical questions.

## References

### Open source

- GPT Researcher: <https://github.com/assafelovic/gpt-researcher>
- STORM: <https://github.com/stanford-oval/storm>
- Open Deep Research: <https://github.com/dzhng/deep-research>

### Papers

- Self-RAG: <https://arxiv.org/abs/2310.11511>
- CRAG: <https://arxiv.org/abs/2401.15884>
- STORM: <https://arxiv.org/abs/2402.14207>
- TCR: <https://arxiv.org/abs/2601.06842>
- FVA-RAG: <https://arxiv.org/abs/2512.07015>
- Over-Searching in Search-Augmented LLMs: <https://arxiv.org/abs/2601.05503>
- Agentic-R: <https://arxiv.org/abs/2601.11888>
- GRACE: <https://arxiv.org/abs/2601.04525>

## Related docs

- [Stop criteria](../scoring-and-stopping/stop-criteria.md)
- [Claims](../scoring-and-stopping/claims.md)
- [Source tiering](../scoring-and-stopping/source-tiering.md)
- [Falsification](../scoring-and-stopping/falsification.md)
