"""Per-segment kernel dependencies via ContextVar.

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


CONTEXT_ARCHIVE_SECTION_CHARS = 20_000
"""Hard cap per context-archive section, aligned with the read_canvas
display limit: a section must stay FULLY readable through its archive
pointer — an uncapped 100k web result would truncate on read-back and
make the pointer promise false. Overflow is cut visibly (in-body marker
plus warning), and the evidence ledger — not the archive — remains the
citation-bearing store, so the cap can never break a citation."""


SCHNELL_BLOCKED_TOOLS: frozenset[str] = frozenset(
    {
        "ask_user",
        "run_web_research",
        "run_deep_mission",
        "delegate_batch",
        "write_canvas",
    }
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
        depth: The run's thoroughness: ``normal`` or
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
        web_research_allowed: Server-derived permission for a multi-step
            research child. The normal Kernel may choose the shared compact
            Research-Desk workflow adaptively; speed tiers can still disable
            it deterministically.
        web_research_profile: Server-selected profile for a permitted
            research child; ``None`` when the tool is not permitted.
        tool_use_counts: Successful source-tool invocations in this run
            segment, split into web and project knowledge for transparency.
        tool_call_limit: Effective run-wide tool-call allowance after folding
            explicit, durable extension decisions into the configured base.
        tool_call_ceiling: Operator maximum for that allowance; models and
            clients cannot widen it.
        step_limit: Effective cumulative checkpointed graph-step allowance.
        step_ceiling: Operator maximum for an explicit step extension.
        checkpointed_steps: Steps already committed before this segment.
        tool_calls_used: All model-requested tool calls committed in the
            checkpoint, including non-source tools.
        cancel_token: Per-run cancel event; observed before every model
            call (the kernel's node boundary — a tool loop has no other
            deterministic chokepoint).
        token_budget: Optional HARD per-run token cap
            (``RunContext.token_budget``, 0 = off). Checked against the
            RUN-cumulative total: ``usage`` restarts per segment, so
            ``prior_usage`` carries what earlier segments spent.
        prior_usage: Token totals of all EARLIER segments, reconstructed
            from the ``usage_metadata`` stamped on checkpointed AI
            messages (``_checkpointed_usage``) — the checkpoint itself is
            the bookkeeping channel, not the run row. Budget input only.
        usage: THIS segment's token accumulator — fresh per ``run()``
            call, so the returned delta can never double-book earlier
            segments against the quota.
        context_trigger_tokens: Per-run compaction threshold in tokens
            (0 = compaction disabled for the segment). Resolved at
            deps-build time from the platform pin or the resolved
            model's card; the shared compiled graph reads it through
            the deps ContextVar, so one graph serves every model.
        context_offload_chars: Bulk tool results above this many
            characters are archived in full and replaced in-context by
            a digest plus their citable reference lines (0 = off).
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
    schnell rules are enforced deterministically at the tool chokepoint
    (``require_tool_allowed``: one-web-search budget, blocked tools) —
    the run's user message merely RESTATES them for the model."""
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
    web_research_allowed: bool = False
    web_research_profile: str | None = None
    tool_use_counts: dict[str, int] = field(
        default_factory=lambda: {"web": 0, "knowledge": 0}
    )
    tool_call_limit: int = 0
    tool_call_ceiling: int = 0
    step_limit: int = 0
    step_ceiling: int = 0
    checkpointed_steps: int = 0
    tool_calls_used: int = 0
    emitted_tool_start_ids: set[str] = field(default_factory=set)
    emitted_tool_finish_ids: set[str] = field(default_factory=set)
    """Logical tool invocations already represented in the durable event
    stream. Seeded from the checkpoint before every segment so approval,
    clarification, and child-wait resumes do not replay start/finish events."""
    evidence_refs: dict[str, dict[str, Any]] = field(default_factory=dict)
    web_search_ledger: dict[str, Any] = field(
        default_factory=lambda: {
            "schema_version": 1,
            "kind": "web_search_ledger",
            "searches": {},
        }
    )
    _evidence_lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False
    )
    cancel_token: threading.Event | None = None
    token_budget: int = 0
    prior_usage: dict[str, int] = field(default_factory=dict)
    usage: dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0}
    )
    context_trigger_tokens: int = 0
    context_offload_chars: int = 0
    memory: Any = None
    """The shared AgentMemoryService (None = feature not wired)."""
    memory_opt_in: bool = False
    """Per-user opt-in resolved once per segment; a read error degrades
    to False visibly (memory is never silently enabled)."""
    memory_briefing: str = ""
    """Recalled non-evidentiary briefing (K5) — context, never citable."""
    memory_recalled: bool = False
    """Once-per-segment recall latch (deep verify rebuilds the user
    message and must not trigger a second briefing round-trip)."""
    stack_name: str = ""
    """Provider stack of THIS run ("" = default) — threaded into child
    submissions so a parent on stack X never fans out on default (F7c)."""
    memory_status: str = "disabled"
    """disabled | used | empty | unavailable — mirrors the mission
    engine's vocabulary so both engines report memory identically."""

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
        authority_check = getattr(
            self.capability_context, "authority_check", None
        )
        if authority_check is not None:
            authority_check()
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
            raw_ledger = record.payload.get("web_search_ledger")
            if isinstance(raw_ledger, dict):
                from inqtrix.evidence import merge_web_search_ledgers

                self.web_search_ledger = merge_web_search_ledgers(
                    [self.web_search_ledger, raw_ledger]
                )
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
                self._persist_evidence_artifact(artifact_id=artifact_id)

    def register_references(
        self, refs: list[dict[str, Any]] | tuple[dict[str, Any], ...]
    ) -> list[dict[str, Any]]:
        """Merge trusted tool evidence and persist the canonical ledger."""
        from inqtrix.agents.evidence import merge_missing_evidence_fields
        from inqtrix.urls import safe_public_url_identity

        registered: list[dict[str, Any]] = []
        changed = False
        with self._evidence_lock:
            for raw in refs:
                ref = dict(raw)
                raw_url = str(ref.get("url") or "").strip()
                if raw_url:
                    if raw_url.casefold().startswith(
                        ("http://", "https://")
                    ):
                        try:
                            ref["url"] = safe_public_url_identity(raw_url).url
                        except ValueError:
                            continue
                    elif not raw_url.casefold().startswith("inqtrix://"):
                        continue
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
                self._persist_evidence_artifact()
        return registered

    def _persist_evidence_artifact(self, *, artifact_id: str = "") -> None:
        """Persist trusted refs and provider-returned search lineage."""

        run_coro(
            self.control.upsert_artifact(
                run_id=self.run_id,
                kind="evidence_bundle",
                session_id=None,
                title="Kernel evidence",
                status="ready",
                content_markdown="",
                payload={
                    "schema_version": 1,
                    "web_search_ledger": dict(self.web_search_ledger),
                },
                refs=list(self.evidence_refs.values()),
                updated_by="agent",
                artifact_id=artifact_id
                or f"art_{self.run_id[-12:]}_evidence",
            )
        )

    def register_web_search_ledger(
        self,
        ledger: dict[str, Any],
        references: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    ) -> list[dict[str, Any]]:
        """Persist provider search output and link its references to it.

        The ledger is a read-only projection for audit and Canvas display. It
        never fetches a source and never determines whether information may be
        used in an answer.
        """

        from inqtrix.evidence import (
            attach_web_search_lineage,
            merge_web_search_ledgers,
        )

        with self._evidence_lock:
            merged = merge_web_search_ledgers(
                [self.web_search_ledger, ledger]
            )
            changed = merged != self.web_search_ledger
            self.web_search_ledger = merged
            if changed:
                self._persist_evidence_artifact()
        if not references:
            return []
        projected = attach_web_search_lineage(
            [dict(reference) for reference in references],
            self.web_search_ledger,
        )
        return self.register_references(projected)

    def register_instant_web_search(self, output: Any) -> list[dict[str, Any]]:
        """Register one ``web.search.instant`` result without re-fetching it."""

        from inqtrix.agents.evidence import (
            GROUNDED_SUPPORT_MAX_CHARS,
            enrich_instant_evidence,
        )
        from inqtrix.evidence import build_instant_web_search_ledger

        query_id = str(output.query_id)
        query = str(output.query)
        provider_answer = str(output.answer or "")
        sources = [
            (
                source.model_dump(mode="json")
                if hasattr(source, "model_dump")
                else dict(source)
            )
            for source in list(getattr(output, "sources", []) or [])
        ]
        ledger = build_instant_web_search_ledger(
            run_id=self.run_id,
            query_id=query_id,
            query=query,
            provider=str(output.provider),
            answer=provider_answer,
            sources=sources,
            parameters=dict(output.parameters or {}),
            started_at=str(output.started_at or ""),
            finished_at=str(output.finished_at or ""),
            duration_ms=int(getattr(output, "duration_ms", 0) or 0),
            prompt_tokens=int(output.prompt_tokens or 0),
            completion_tokens=int(output.completion_tokens or 0),
        )
        references = enrich_instant_evidence(
            provider_answer,
            [
                {
                    "url": str(source.get("url") or ""),
                    "title": str(source.get("title") or "") or None,
                    "provider_snippet": (
                        str(source.get("snippet") or "") or None
                    ),
                    "source_rank": int(source.get("rank") or index),
                    "origin": str(source.get("origin") or "provider"),
                }
                for index, source in enumerate(sources, start=1)
                if source.get("url")
            ],
        )
        if not references and provider_answer.strip():
            # Azure may return one coherent grounded answer without exposing
            # individual URL rows. Preserve that search result as a citable
            # ledger pointer instead of making the evidence disappear. The
            # full, unmodified answer remains in ``web_search_ledger``; this
            # bounded field is only the compact synthesis projection.
            support = " ".join(provider_answer.split())
            if len(support) > GROUNDED_SUPPORT_MAX_CHARS:
                support = (
                    support[: GROUNDED_SUPPORT_MAX_CHARS - 1].rstrip()
                    + "…"
                )
            identity = hashlib.sha1(
                f"web-search:{self.run_id}:{query_id}".encode("utf-8")
            ).hexdigest()[:16]
            references = [
                {
                    "reference_id": f"ref_{identity}",
                    "query_id": query_id,
                    "query_ids": [query_id],
                    "source_run_id": self.run_id,
                    "source_run_ids": [self.run_id],
                    "title": query.strip() or "Web search result",
                    "grounded_support": support,
                    "evidence_kind": "provider_search_answer",
                    "provider": str(output.provider),
                }
            ]
        return self.register_web_search_ledger(ledger, references)

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
        if label:
            # Collision renumber (typically a child-ledger merge): never
            # silent — a copied child citation like [W3] now means the
            # NEW label here (the child tool output surfaces the mapping).
            log.info(
                "Beleg-Label neu vergeben (Kollision im Kernel-Ledger): "
                "run_id=%s reference_id=%s vorhandene_labels=%d",
                self.run_id,
                ref_id,
                len(owners),
            )
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

    @property
    def visible_to(self) -> Any:
        """The owner's per-run visibility context (UserContext or None).

        Child runs are created with the parent's principal, so reading
        a child row/result/outcome must present the SAME visibility the
        store authorizes against — under RLS a ``visible_to=None`` read
        only sees anonymous rows (``created_by_user_id IS NULL``), so an
        authenticated parent projecting its child would be denied
        (``RunNotFound``). Resolved once per segment on the capability
        context; mirrors the mission engine's ``deps.visible_to``.
        """
        return getattr(self.capability_context, "visible_to", None)

    @property
    def context_archive_prefix(self) -> str:
        """Shared id prefix of the run's archive SECTION artifacts.

        ``read_canvas`` on the bare prefix lists every section; each
        :meth:`append_context_archive` call returns the concrete section
        id (``<prefix>_<sha1[:8]>``) for the in-context pointer.
        """
        return f"art_{self.run_id[-12:]}_ctx"

    def append_context_archive(self, section_title: str, text: str) -> str:
        """Write one archive section as its OWN run-local artifact.

        One artifact per section (content-hash id) instead of one
        growing aggregate: an append never re-reads or re-writes earlier
        sections — the aggregate form snapshotted the FULL body into the
        append-only revision table on every append (O(N^2) bytes and IO
        on the hot tool path, all under the evidence lock). A
        redelivered segment re-appends the identical content and upserts
        the same artifact (idempotent, replay-safe). Sections are capped
        at :data:`CONTEXT_ARCHIVE_SECTION_CHARS` so every pointer stays
        FULLY readable through ``read_canvas`` — overflow is cut with a
        visible marker and a warning, never silently on read-back.
        Returns the section artifact id.
        """
        body = text.strip()
        if len(body) > CONTEXT_ARCHIVE_SECTION_CHARS:
            dropped = len(body) - CONTEXT_ARCHIVE_SECTION_CHARS
            body = body[:CONTEXT_ARCHIVE_SECTION_CHARS] + (
                f"\n\n[... {dropped} Zeichen gekappt — Sektionslimit "
                f"{CONTEXT_ARCHIVE_SECTION_CHARS} Zeichen ...]"
            )
            log.warning(
                "Archiv-Sektion '%s' gekappt: %d Zeichen ueber dem "
                "Sektionslimit (%d) verworfen.",
                section_title,
                dropped,
                CONTEXT_ARCHIVE_SECTION_CHARS,
            )
        digest = hashlib.sha1(
            f"{section_title}\n{body}".encode("utf-8")
        ).hexdigest()[:8]
        artifact_id = f"{self.context_archive_prefix}_{digest}"
        run_coro(
            self.control.upsert_artifact(
                run_id=self.run_id,
                kind="context_archive",
                session_id=None,
                title=section_title,
                status="ready",
                content_markdown=f"## {section_title}\n\n{body}\n",
                payload={},
                refs=[],
                updated_by="agent",
                artifact_id=artifact_id,
            )
        )
        return artifact_id

    def append_context_archive_chunked(
        self, section_title: str, text: str
    ) -> list[str]:
        """Archive TEXT losslessly across as many capped sections as needed.

        The single-section writer truncates at the section limit — a
        compaction that evicts far more than one section silently lost
        the overflow (>90% of a long transcript). Chunking preserves
        everything: cuts prefer paragraph boundaries, an open ``` fence
        at a cut is closed and reopened so every section renders as
        valid markdown through ``read_canvas``, and multi-chunk titles
        carry ``(i/n)``. Returns the section artifact ids in order.
        """
        body = text.strip()
        if not body:
            return []
        # Headroom for the reopened fence marker at a chunk head.
        limit = CONTEXT_ARCHIVE_SECTION_CHARS - 16
        chunks: list[str] = []
        remaining = body
        while remaining:
            if len(remaining) <= limit:
                chunks.append(remaining)
                break
            cut = remaining.rfind("\n\n", limit // 2, limit)
            if cut == -1:
                cut = limit
            chunks.append(remaining[:cut])
            remaining = remaining[cut:].lstrip("\n")
        normalized: list[str] = []
        reopen = ""
        for chunk in chunks:
            chunk_text = f"{reopen}{chunk}"
            if chunk_text.count("```") % 2 == 1:
                chunk_text += "\n```"
                reopen = "```\n"
            else:
                reopen = ""
            normalized.append(chunk_text)
        if len(normalized) == 1:
            return [self.append_context_archive(section_title, normalized[0])]
        total = len(normalized)
        return [
            self.append_context_archive(
                f"{section_title} ({index}/{total})", chunk
            )
            for index, chunk in enumerate(normalized, start=1)
        ]

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
