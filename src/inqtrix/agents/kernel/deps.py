"""Per-segment kernel dependencies via ContextVar (plan M2 `2.2`).

The compiled kernel graph is SHARED across runs and users (E-rules:
stateless graph, thread_id separates runs) — everything one segment
needs travels through :data:`_KERNEL_DEPS` exactly like the phase
machine's ``_DEPS``. Tools and the delegating chat provider read it at
call time; the algorithm sets/resets it around each segment.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from inqtrix.core.results import SourcePolicy

if TYPE_CHECKING:
    from inqtrix.agents.control_ports import AgentControlStore
    from inqtrix.providers.base import LLMProvider
    from inqtrix.settings import AgentPlatformSettings

log = logging.getLogger("inqtrix")


SCHNELL_BLOCKED_TOOLS: frozenset[str] = frozenset(
    {"ask_user", "run_web_research", "run_deep_mission", "write_canvas"}
)
"""Kernel tools the ``schnell`` tier removes (tier_policy contract:
seconds-scale chat answers — no interrupts, no children, no canvas).
Enforced at :meth:`KernelDeps.require_tool_allowed`, never prompt-only."""


@dataclass
class KernelDeps:
    """Everything one kernel segment execution needs (never checkpointed).

    Attributes:
        run_id: The executing run (= LangGraph thread_id).
        session_id: The conversation the run belongs to; kernel
            deliverables scope to it (K2 registry), ``""`` for
            session-less API runs (then run-scoped).
        control: Control store for clarification/approval rows.
        platform: Server-side agent limits (never prompted).
        llm: The request-resolved provider (stack-aware, E-rules).
        model: Resolved model override for the ``agent_kernel`` node
            (tier resolution result), ``None`` for provider default.
        reasoning_effort: Resolved effort override, ``None`` when unset.
        timeout: Per-LLM-call timeout in seconds.
        event_sink: Structured run-event emitter; ``None`` without an
            event stream.
        capability_registry: The curated read-tool surface (M2 wave-1
            capabilities); ``None`` degrades every capability tool to a
            visible not-available result, never a crash.
        capability_context: Per-segment identity for capability calls —
            built by the ALGORITHM from the verified principal (never a
            tool argument, E-rules).
        run_service: Child-run submission seam (``None`` disables the
            child tools visibly).
        resolver: Builds child ``ResolvedAgentContext``s (E18).
        principal: The verified submitting identity, threaded into
            child submissions for attribution/quota.
        depth: The run's thoroughness (plan M4): ``normal`` or
            ``deep``. Deep forces the DEEP child research profile and
            gates the extra verification pass.
        autonomy: The run's wire autonomy mode; child missions inherit
            it.
        skill_service: Skill lookups for ``load_skill`` and the
            disclosure block (``None`` disables both visibly).
        skills: ACTIVATED skills of this segment (attached at submit +
            loaded via ``load_skill``, incl. reconstructions from
            checkpointed load markers).
        allowed_tool_union: Kernel-tool allowlist union of the
            activated skills; ``None`` = unrestricted. Enforced at the
            tool-body chokepoint (:meth:`require_tool_allowed`).
        source_policy: Effective per-run availability of web and project
            knowledge after the one-shot directive is applied.
        execution_directive: One-shot route constraining the tool surface;
            empty for normal automatic routing.
        explicit_web_research: Server-derived permission for a multi-step
            research child. True only in Deep or when the request carries
            the admitted ``web_research`` tool directive.
        web_research_profile: Server-selected profile for a permitted
            research child; ``None`` when the tool is not permitted.
        tool_use_counts: Successful source-tool invocations in this run
            segment, split into web and project knowledge for transparency.
        cancel_token: Per-run cancel event; observed before every model
            call (the kernel's node boundary — a tool loop has no other
            deterministic chokepoint).
        token_budget: Optional HARD per-run token cap
            (``RunContext.token_budget``, 0 = off). Checked against the
            RUN-cumulative total: ``usage`` restarts per segment, so
            ``prior_usage`` carries what earlier segments spent.
        prior_usage: Token totals of all EARLIER segments (read from
            the run row before this segment), budget input only.
        usage: THIS segment's token accumulator — fresh per ``run()``
            call, so the returned delta can never double-book earlier
            segments against the quota.
    """

    run_id: str
    control: "AgentControlStore"
    platform: "AgentPlatformSettings"
    llm: "LLMProvider"
    model: str | None
    reasoning_effort: str | None
    timeout: float
    session_id: str = ""
    event_sink: Any = None
    capability_registry: Any = None
    capability_context: Any = None
    run_service: Any = None
    resolver: Any = None
    principal: Any = None
    autonomy: str = "balanced"
    depth: str = "normal"
    tier: str = ""
    """Selected Agent-Desk Stufe ('' = legacy depth semantics). The
    kernel enforces its web budget via the derived research policy and
    states the remaining rules (no ask_user, chat answer) in the run's
    user message — its architecture is prompt-driven by design."""
    skill_service: Any = None
    skills: list[Any] = field(default_factory=list)
    skill_answers: dict[str, dict[str, str]] = field(default_factory=dict)
    session_history: str = ""
    artifact_registry: tuple[dict[str, Any], ...] = ()
    last_response_form: str = ""
    prior_evidence_count: int = 0
    effective_response_form: str = ""
    question: str = ""
    allowed_tool_union: set[str] | None = None
    source_policy: SourcePolicy = field(default_factory=SourcePolicy)
    execution_directive: str = ""
    explicit_web_research: bool = False
    web_research_profile: str | None = None
    tool_use_counts: dict[str, int] = field(
        default_factory=lambda: {"web": 0, "knowledge": 0}
    )
    evidence_refs: dict[str, dict[str, Any]] = field(default_factory=dict)
    _evidence_lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False
    )
    cancel_token: threading.Event | None = None
    token_budget: int = 0
    prior_usage: dict[str, int] = field(default_factory=dict)
    usage: dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0}
    )

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.event_sink is not None:
            self.event_sink(event_type, payload)

    def book_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.usage["prompt_tokens"] += max(0, int(prompt_tokens or 0))
        self.usage["completion_tokens"] += max(0, int(completion_tokens or 0))

    def activate_skill(self, record: Any) -> None:
        """Track one activated skill and widen the tool allowlist."""
        if any(existing.id == record.id for existing in self.skills):
            return
        self.skills.append(record)
        from inqtrix.agents.skills_runtime import allowed_tool_names

        self.allowed_tool_union = allowed_tool_names(self.skills)

    def require_tool_allowed(self, tool_name: str) -> None:
        """The skill, tier and source-policy dispatch chokepoint.

        Raises:
            PermissionError: The activated skills restrict tools and
                *tool_name* is outside the union, or the schnell tier
                blocks the tool — surfaced as a loud tool error the
                model must acknowledge.
        """
        if (
            self.tier == "schnell"
            and tool_name == "web_instant"
            and self.tool_use_counts.get("web", 0) >= 1
        ):
            # The published schnell budget is exactly ONE web search; the
            # checkpoint-derived counter survives park/resume, so the
            # bound holds across the whole run (never prompt-only).
            log.warning(
                "web_instant durch das schnell-Budget blockiert "
                "(bereits %d Websuche(n)).",
                self.tool_use_counts.get("web", 0),
            )
            from inqtrix.exceptions import AgentPolicyDenied

            raise AgentPolicyDenied(
                "Die Stufe 'schnell' erlaubt genau EINE Websuche — "
                "nutze die vorhandenen Ergebnisse."
            )
        if self.tier == "schnell" and tool_name in SCHNELL_BLOCKED_TOOLS:
            # The speed tier is a deterministic budget, never prompt-only:
            # no clarification interrupts, no delegated children, no
            # canvas documents — the answer belongs in the chat.
            log.warning(
                "Werkzeug %s durch die Stufe schnell blockiert.", tool_name
            )
            from inqtrix.exceptions import AgentPolicyDenied

            raise AgentPolicyDenied(
                f"Werkzeug {tool_name} ist in der Stufe 'schnell' nicht "
                "verfuegbar — antworte direkt im Chat."
            )
        union = self.allowed_tool_union
        # ask_user is the interrupt PRIMITIVE, not a work tool: skill
        # allowed_tools never scoped it (routing it through this
        # chokepoint is new with the tier enforcement), and a skill must
        # not silently remove the human gate.
        if (
            union is not None
            and tool_name not in union
            and tool_name != "ask_user"
        ):
            log.warning(
                "Werkzeug %s durch Skill-allowed_tools blockiert "
                "(erlaubt: %s).",
                tool_name,
                ", ".join(sorted(union)),
            )
            from inqtrix.exceptions import AgentPolicyDenied

            raise AgentPolicyDenied(
                f"Werkzeug {tool_name} ist durch die aktivierten Skills "
                f"nicht erlaubt (erlaubt: {', '.join(sorted(union))})."
            )
        from inqtrix.agents.source_policy import require_kernel_tool_allowed

        require_kernel_tool_allowed(
            tool_name,
            policy=self.source_policy,
            execution_directive=self.execution_directive,
        )

    def record_source_tool_use(self, source: str) -> None:
        """Increment one successful source-tool invocation."""
        if source not in ("web", "knowledge"):
            raise ValueError(f"Unknown source-tool counter: {source!r}")
        self.tool_use_counts[source] = self.tool_use_counts.get(source, 0) + 1

    @staticmethod
    def reference_id(ref: dict[str, Any]) -> str:
        """Return the stable opaque id of one canonical evidence record."""
        from inqtrix.agents.evidence import dedup_key

        key = dedup_key(ref)
        if not key:
            return ""
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        return f"ref_{digest}"

    def hydrate_evidence(self) -> None:
        """Restore the run-local evidence ledger after park/resume."""
        from inqtrix.agents.control_ports import ArtifactNotFound

        artifact_id = f"art_{self.run_id[-12:]}_evidence"
        try:
            record, _ = run_coro(
                self.control.get_artifact(self.run_id, artifact_id)
            )
        except ArtifactNotFound:
            return
        changed = False
        with self._evidence_lock:
            for raw in record.refs:
                ref = dict(raw)
                ref_id = str(ref.get("reference_id") or self.reference_id(ref))
                if ref_id:
                    ref["reference_id"] = ref_id
                    changed = (
                        self._ensure_reference_label(ref_id, ref) or changed
                    )
                    self.evidence_refs[ref_id] = ref
            if changed:
                run_coro(
                    self.control.upsert_artifact(
                        run_id=self.run_id,
                        kind="evidence_bundle",
                        session_id=None,
                        title="Kernel evidence",
                        status="ready",
                        content_markdown="",
                        payload={},
                        refs=list(self.evidence_refs.values()),
                        updated_by="agent",
                        artifact_id=artifact_id,
                    )
                )

    def register_references(
        self, refs: list[dict[str, Any]] | tuple[dict[str, Any], ...]
    ) -> list[dict[str, Any]]:
        """Merge trusted tool evidence and persist the canonical ledger."""
        from inqtrix.agents.evidence import merge_missing_evidence_fields

        registered: list[dict[str, Any]] = []
        changed = False
        with self._evidence_lock:
            for raw in refs:
                ref = dict(raw)
                ref_id = str(ref.get("reference_id") or self.reference_id(ref))
                if not ref_id:
                    continue
                ref["reference_id"] = ref_id
                existing = self.evidence_refs.get(ref_id)
                if existing is None:
                    self._ensure_reference_label(ref_id, ref)
                    self.evidence_refs[ref_id] = ref
                    existing = ref
                    changed = True
                else:
                    changed = (
                        merge_missing_evidence_fields(existing, ref) or changed
                    )
                    changed = (
                        self._ensure_reference_label(ref_id, existing)
                        or changed
                    )
                registered.append(dict(existing))
            if changed:
                run_coro(
                    self.control.upsert_artifact(
                        run_id=self.run_id,
                        kind="evidence_bundle",
                        session_id=None,
                        title="Kernel evidence",
                        status="ready",
                        content_markdown="",
                        payload={},
                        refs=list(self.evidence_refs.values()),
                        updated_by="agent",
                        artifact_id=f"art_{self.run_id[-12:]}_evidence",
                    )
                )
        return registered

    def _ensure_reference_label(
        self, ref_id: str, ref: dict[str, Any]
    ) -> bool:
        """Assign one stable K/W label inside the locked kernel ledger."""
        import re

        label = str(ref.get("label") or "")
        owners = {
            str(item.get("label") or ""): existing_id
            for existing_id, item in self.evidence_refs.items()
        }
        if re.fullmatch(r"[KW][1-9][0-9]*", label) and owners.get(
            label, ref_id
        ) == ref_id:
            return False
        prefix = "K" if ref.get("document_id") is not None else "W"
        numbers = [
            int(match.group(1))
            for known in owners
            if (match := re.fullmatch(fr"{prefix}([1-9][0-9]*)", known))
        ]
        next_number = max(numbers, default=0) + 1
        ref["label"] = f"{prefix}{next_number}"
        return ref["label"] != label

    def resolve_reference_ids(
        self, reference_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Resolve model-selected ids against trusted evidence only."""
        with self._evidence_lock:
            unknown = [
                item for item in reference_ids if item not in self.evidence_refs
            ]
            if unknown:
                raise ValueError(
                    "Unbekannte reference_id(s): " + ", ".join(unknown)
                )
            return [dict(self.evidence_refs[item]) for item in reference_ids]

    def check_abort(self) -> None:
        """Stop the loop at the model-call boundary (cancel or budget).

        Mirrors the research graph's node-boundary check
        (``state.raise_if_cancelled``): best-effort — an in-flight
        provider HTTP call is not interrupted, the NEXT model turn is.

        Raises:
            AgentCancelled: The cancel event is set, or the configured
                token budget is reached by the run-cumulative total.
        """
        from inqtrix.exceptions import AgentCancelled, AgentTokenBudgetExceeded

        if self.cancel_token is not None and self.cancel_token.is_set():
            raise AgentCancelled("Lauf vom Client abgebrochen.")
        if self.token_budget:
            used = sum(
                int(self.prior_usage.get(key, 0) or 0)
                + int(self.usage.get(key, 0) or 0)
                for key in ("prompt_tokens", "completion_tokens")
            )
            if used >= self.token_budget:
                log.warning(
                    "Kernel-Lauf wegen Token-Budget gestoppt: %d/%d "
                    "Tokens (INQTRIX_QUOTA_MAX_TOKENS_PER_RUN).",
                    used,
                    self.token_budget,
                )
                raise AgentTokenBudgetExceeded(
                    "Lauf wegen Token-Budget (max_tokens_per_run) gestoppt."
                )


_KERNEL_DEPS: contextvars.ContextVar[KernelDeps | None] = (
    contextvars.ContextVar("inqtrix_kernel_deps", default=None)
)


def kernel_deps() -> KernelDeps:
    deps = _KERNEL_DEPS.get()
    if deps is None:  # pragma: no cover - graph never runs undepped
        raise RuntimeError("Kernel-Deps sind fuer dieses Segment nicht gesetzt.")
    return deps


def set_kernel_deps(deps: KernelDeps | None) -> None:
    _KERNEL_DEPS.set(deps)


def run_coro(coro: Any) -> Any:
    """Drive one control-store coroutine from the sync worker thread."""
    return asyncio.run(coro)
