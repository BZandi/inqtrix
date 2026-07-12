"""THE agent speed/depth tier table (Stufen-System).

One server-owned policy record per user-facing tier — every consumer
(intake routing, clarify cap, planner prompt + validation,
``derive_web_research_policy``, kernel budgets, synthesis/critic gating,
``/v1/capabilities`` publishing) reads THIS table, so the published
contract always equals the enforced one (the ``permission_modes``
pattern). Budgets are stated in prompts AND enforced by validators —
never prompt-only.

Deliberately NOT part of a tier: model/effort selection (the composer
model picker and operator routing own that — asserted by test) and the
autonomy/permission dimension (``strict`` keeps its gates in every
tier; permission always beats speed).

Vocabulary: the tier names reuse the Knowledge-Desk depth family
(``schnell``/``gruendlich``/``tief``). ``standard`` is deliberately
skipped — the composer already shows an autonomy preset labeled
"Standard", and two same-named controls in one composer would collide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AgentTier = Literal["schnell", "gruendlich", "tief"]

AGENT_TIERS: tuple[AgentTier, ...] = ("schnell", "gruendlich", "tief")
"""Ordered ladder, fastest first."""

DEFAULT_AGENT_TIER: AgentTier = "gruendlich"
"""The composer's default SELECTION. A request that omits ``agent_tier``
entirely runs LEGACY depth semantics (no tier budgets) — deliberate
backwards compatibility; tiers-aware clients always send a tier."""


@dataclass(frozen=True)
class TierPolicy:
    """Operational meaning of one tier (frozen — policy, not state)."""

    tier: AgentTier
    clarification_rounds: int
    """Cap on human clarification rounds; the effective cap is
    ``min(settings.max_clarification_rounds, this)``. ``0`` = never ask,
    blocking gaps degrade to visible assumptions."""
    discovery: bool
    """Whether the mission machine runs the discovery phase at all.
    ``False`` routes intake straight to planning (seconds matter more
    than probe context)."""
    plan_gate: Literal["per_autonomy", "skip_unless_strict"]
    """``per_autonomy`` keeps today's E16 behavior; ``skip_unless_strict``
    suppresses the plan interrupt except under ``strict`` autonomy
    (permission beats speed)."""
    web_research: bool
    """Whether the plan may contain multi-step ``web_research`` children
    at all. ``False`` = instant web only."""
    web_child_profile: Literal["schnell", "compact", "deep"] | None
    """Default report profile of ``web_research`` children."""
    web_child_ceiling: Literal["schnell", "compact", "deep"] | None
    """Highest child profile a user may select per task at the gate."""
    web_instant_budget: int | None
    """Cap on ``web_instant`` tasks per plan; ``None`` = uncapped. The
    plan validator enforces it (never prompt-only)."""
    rag_default_profile: Literal["schnell", "standard", "gruendlich", "tief"]
    """Default retrieval profile for ``rag_query`` tasks."""
    verify: Literal["labels", "standard", "escalating"]
    """Report-quality level: ``labels`` = citation-label validation only
    (no critic); ``standard`` = today's critic + advisory grounding;
    ``escalating`` = unverified web-quoted claims flip the critic to
    revise (P7 wires the last two)."""
    response_form: Literal["auto", "chat", "canvas"]
    """Server-side deliverable default; ``chat``/``canvas`` override the
    intake routing, ``auto`` keeps it."""
    latency_hint: str
    """Human latency expectation, published to the composer."""


TIER_POLICIES: dict[AgentTier, TierPolicy] = {
    "schnell": TierPolicy(
        tier="schnell",
        clarification_rounds=0,
        discovery=False,
        plan_gate="skip_unless_strict",
        web_research=False,
        web_child_profile=None,
        web_child_ceiling=None,
        web_instant_budget=1,
        rag_default_profile="schnell",
        verify="labels",
        response_form="chat",
        latency_hint="schnellste Stufe · unter ~3 min",
    ),
    "gruendlich": TierPolicy(
        tier="gruendlich",
        clarification_rounds=1,
        discovery=True,
        plan_gate="per_autonomy",
        web_research=True,
        web_child_profile="schnell",
        web_child_ceiling="compact",
        web_instant_budget=None,
        rag_default_profile="standard",
        verify="standard",
        response_form="auto",
        latency_hint="1-3 min",
    ),
    "tief": TierPolicy(
        tier="tief",
        clarification_rounds=2,
        discovery=True,
        plan_gate="per_autonomy",
        web_research=True,
        web_child_profile="compact",
        web_child_ceiling="deep",
        web_instant_budget=None,
        rag_default_profile="gruendlich",
        verify="escalating",
        response_form="canvas",
        latency_hint="5-15 min",
    ),
}


def resolve_tier_policy(tier: str | None) -> TierPolicy:
    """The policy of ``tier``; empty/None resolves to the default tier.

    Raises:
        ValueError: For a token outside the published ladder — admission
            validates first, so reaching this is a caller bug that must
            fail loudly instead of silently running the default budgets.
    """
    if not tier:
        return TIER_POLICIES[DEFAULT_AGENT_TIER]
    if tier not in TIER_POLICIES:
        raise ValueError(f"unknown agent tier: {tier!r}")
    return TIER_POLICIES[tier]  # type: ignore[index]


def tier_capabilities_payload(
    *, max_clarification_rounds: int | None = None
) -> list[dict[str, object]]:
    """Wire projection for ``/v1/capabilities`` — generated from
    :data:`TIER_POLICIES` so published == enforced.

    ``max_clarification_rounds`` is the operator cap the clarify node
    also applies (``min(settings, tier)``): passing it keeps the
    published round budget equal to the enforced one under a tighter
    env configuration.
    """
    return [
        {
            "id": policy.tier,
            "clarification_rounds": (
                min(max_clarification_rounds, policy.clarification_rounds)
                if max_clarification_rounds is not None
                else policy.clarification_rounds
            ),
            "plan_gate": policy.plan_gate,
            "web_research": policy.web_research,
            "web_child_profile": policy.web_child_profile,
            "web_child_ceiling": policy.web_child_ceiling,
            "web_instant_budget": policy.web_instant_budget,
            "rag_default_profile": policy.rag_default_profile,
            "verify": policy.verify,
            "response_form": policy.response_form,
            "latency_hint": policy.latency_hint,
        }
        for policy in (TIER_POLICIES[tier] for tier in AGENT_TIERS)
    ]
