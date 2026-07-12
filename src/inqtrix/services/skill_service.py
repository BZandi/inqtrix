"""Skill CRUD with the owned-resource visibility rule (plan M3 `3.1`).

Same enforcement shape as prompt templates: the router resolves the
caller's shared-in grants into ``also_visible``, the service decides per
record via :func:`~inqtrix.auth.permissions.grant_for_owned_resource`.
Reads need view, updates need edit, deletion stays owner-only.
Conflict policy is optimistic concurrency over ``updated_at``.

The skill-specific validation lives here: the ``/``-label shape, the
enum fields, the clarification-point sanitizer (deterministic ids,
author input validated HARD — unlike LLM-proposed questions there is no
soft-drop path), and the placeholder coupling rule: every ``{{name}}``
in the instructions must be declared as a point (a silent hole in the
substitution would surface mid-run, plan `3.4`); points WITHOUT a
placeholder stay allowed (context the agent needs verbatim-free).
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any, Mapping

from inqtrix.auth.permissions import (
    SharePermission,
    grant_for_owned_resource,
)
from inqtrix.content.skills import (
    MAX_CLARIFICATION_POINTS,
    MAX_POINT_OPTIONS,
    SKILL_ALLOWED_TOOLS,
    SKILL_DELIVERABLES,
    SKILL_INVOCATIONS,
    SKILL_REQUIRES_PLAN,
    SkillConflict,
    SkillNotFound,
    SkillRecord,
    SkillRepository,
    new_skill_id,
)

__all__ = [
    "SkillService",
    "SkillValidationError",
    "SkillConflict",
    "extract_placeholders",
    "skill_access",
]

if TYPE_CHECKING:
    from inqtrix.auth.principal import UserContext


class SkillValidationError(ValueError):
    """Raised for client-payload problems (maps to HTTP 400)."""


_LABEL_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([a-zA-Z0-9_-]+)\s*\}\}")

_EFFORT_TOKENS = ("", "none", "minimal", "low", "medium", "high", "xhigh")
_TIER_TOKENS = ("", "high", "mid", "fast")


def extract_placeholders(instructions_markdown: str) -> list[str]:
    """The ordered, de-duplicated ``{{name}}`` tokens of a skill body.

    Shared by the save validation here, the library editor (point
    scaffolding), and the runtime substitution — one definition of what
    counts as a placeholder (Designprinzip 4).
    """
    seen: list[str] = []
    for match in _PLACEHOLDER_PATTERN.finditer(instructions_markdown):
        name = match.group(1)
        if name not in seen:
            seen.append(name)
    return seen


def skill_access(
    record: SkillRecord,
    visible_to: "UserContext | None",
    also_visible: "Mapping[str, SharePermission] | None" = None,
) -> SharePermission | None:
    """The caller's grant on *record*; raises the indistinct 404.

    ``None`` means full access (unscoped caller, ownerless skill, or
    the owner); a permission means shared-in access at that level.
    """
    visible, shared = grant_for_owned_resource(
        owner_sub=record.owner_sub,
        resource_tenant_id=record.tenant_id,
        resource_id=record.id,
        visible_to=visible_to,
        also_visible=also_visible,
    )
    if not visible:
        raise SkillNotFound(record.id)
    return shared


def _sanitized_points(raw: Any) -> tuple[dict[str, Any], ...]:
    """Author-declared clarification points with deterministic ids.

    Positional ids (``p1``, ``p1_o1``) follow the M1 sanitizer
    convention so the runtime can map answers without trusting any
    client-minted id. Author input fails HARD (400) — there is no
    LLM-garbage soft path here.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise SkillValidationError(
            "Feld 'clarification_points' muss eine Liste sein"
        )
    if len(raw) > MAX_CLARIFICATION_POINTS:
        raise SkillValidationError(
            f"Hoechstens {MAX_CLARIFICATION_POINTS} Klaerungspunkte erlaubt"
        )
    points: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise SkillValidationError(
                f"Klaerungspunkt {index} muss ein Objekt sein"
            )
        question = str(item.get("question", "") or "").strip()
        if not question:
            raise SkillValidationError(
                f"Klaerungspunkt {index} braucht eine Frage"
            )
        name = str(item.get("name", "") or "").strip()
        raw_options = item.get("options") or []
        if not isinstance(raw_options, list):
            raise SkillValidationError(
                f"Klaerungspunkt {index}: 'options' muss eine Liste sein"
            )
        if len(raw_options) > MAX_POINT_OPTIONS:
            raise SkillValidationError(
                f"Klaerungspunkt {index}: hoechstens {MAX_POINT_OPTIONS} "
                "Optionen erlaubt"
            )
        options: list[dict[str, str]] = []
        for opt_index, option in enumerate(raw_options, start=1):
            if not isinstance(option, dict):
                raise SkillValidationError(
                    f"Klaerungspunkt {index}: Option {opt_index} muss ein "
                    "Objekt sein"
                )
            label = str(option.get("label", "") or "").strip()
            if not label:
                raise SkillValidationError(
                    f"Klaerungspunkt {index}: Option {opt_index} braucht "
                    "ein Label"
                )
            options.append(
                {
                    "id": f"p{index}_o{opt_index}",
                    "label": label[:60],
                    "description": str(
                        option.get("description", "") or ""
                    ).strip()[:120],
                }
            )
        required = item.get("required", False)
        if not isinstance(required, bool):
            raise SkillValidationError(
                f"Klaerungspunkt {index}: 'required' muss ein Boolean sein"
            )
        points.append(
            {
                "id": f"p{index}",
                "name": name,
                "question": question[:500],
                "options": options,
                "required": required,
                "default_assumption": str(
                    item.get("default_assumption", "") or ""
                ).strip()[:300],
            }
        )
    return tuple(points)


def _validated_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the writable skill fields."""
    label = str(payload.get("label", "") or "").strip().lower()
    if not _LABEL_PATTERN.match(label):
        raise SkillValidationError(
            "Feld 'label' muss ein /-Kuerzel aus a-z, 0-9 und '-' sein "
            "(1-64 Zeichen, beginnend mit Buchstabe/Ziffer)"
        )
    title = str(payload.get("title", "") or "").strip()
    if not title:
        raise SkillValidationError("Feld 'title' ist erforderlich")
    instructions = str(payload.get("instructions_markdown", "") or "")
    if not instructions.strip():
        raise SkillValidationError(
            "Feld 'instructions_markdown' ist erforderlich"
        )
    for enum_field, allowed in (
        ("deliverable", SKILL_DELIVERABLES),
        ("requires_plan", SKILL_REQUIRES_PLAN),
        ("invocation", SKILL_INVOCATIONS),
        ("model_tier", _TIER_TOKENS),
        ("effort", _EFFORT_TOKENS),
    ):
        value = payload.get(enum_field)
        if value is not None and str(value) not in allowed:
            raise SkillValidationError(
                f"Feld '{enum_field}' muss eines von "
                f"{', '.join(repr(item) for item in allowed)} sein"
            )
    raw_tools = payload.get("allowed_tools") or []
    if not isinstance(raw_tools, list) or any(
        not isinstance(tool, str) or not tool.strip() for tool in raw_tools
    ):
        raise SkillValidationError(
            "Feld 'allowed_tools' muss eine Liste nicht-leerer Strings sein"
        )
    unknown_tools = [
        tool for tool in raw_tools if tool not in SKILL_ALLOWED_TOOLS
    ]
    if unknown_tools:
        raise SkillValidationError(
            "Feld 'allowed_tools' enthaelt unbekannte Werkzeuge: "
            + ", ".join(repr(tool) for tool in unknown_tools)
            + " (erlaubt: "
            + ", ".join(SKILL_ALLOWED_TOOLS)
            + ")"
        )
    include = payload.get("include_in_autocomplete", True)
    if not isinstance(include, bool):
        raise SkillValidationError(
            "Feld 'include_in_autocomplete' muss ein Boolean sein"
        )
    points = _sanitized_points(payload.get("clarification_points"))
    # Placeholder coupling (plan `3.4`): every {{name}} must map onto a
    # declared point, or the runtime substitution would leave a visible
    # hole mid-run. Points without a placeholder stay allowed.
    placeholders = extract_placeholders(instructions)
    point_names = {point["name"] for point in points if point["name"]}
    missing = [name for name in placeholders if name not in point_names]
    if missing:
        raise SkillValidationError(
            "Jeder {{Platzhalter}} braucht einen Klaerungspunkt mit "
            f"gleichem Namen; ohne Punkt: {', '.join(missing)}"
        )
    return {
        "label": label,
        "title": title,
        "description": str(payload.get("description", "") or "").strip(),
        "when_to_use": str(payload.get("when_to_use", "") or "").strip(),
        "instructions_markdown": instructions,
        "clarification_points": points,
        "deliverable": str(payload.get("deliverable", "") or ""),
        "allowed_tools": tuple(
            tool.strip() for tool in raw_tools
        ),
        "requires_plan": str(payload.get("requires_plan", "auto") or "auto"),
        "invocation": str(
            payload.get("invocation", "user_only") or "user_only"
        ),
        "argument_hint": str(
            payload.get("argument_hint", "") or ""
        ).strip()[:120],
        "model_tier": str(payload.get("model_tier", "") or ""),
        "effort": str(payload.get("effort", "") or ""),
        "include_in_autocomplete": include,
    }


class SkillService:
    """Application service over the skill repository.

    Args:
        repository: The persistence backend.
        durable: Whether *repository* survives server restarts; drives
            ``features.skills`` (same volatile-store rule as prompt
            templates — a browser must not sync against a store that
            reads as "everything deleted" after a bounce).
    """

    def __init__(
        self,
        *,
        repository: SkillRepository,
        durable: bool = True,
    ) -> None:
        self._repository = repository
        self._durable = durable

    @property
    def durable(self) -> bool:
        """Whether skills survive a server restart."""
        return self._durable

    async def create(
        self,
        payload: Mapping[str, Any],
        *,
        tenant_id: str,
        owner_sub: str | None,
    ) -> SkillRecord:
        """Create one skill; scoped principals own what they create."""
        now = time.time()
        record = SkillRecord(
            id=new_skill_id(),
            tenant_id=tenant_id,
            owner_sub=owner_sub,
            created_at=now,
            updated_at=now,
            **_validated_fields(payload),
        )
        return await self._repository.create(record)

    async def get_visible(
        self,
        skill_id: str,
        *,
        tenant_id: str,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> tuple[SkillRecord, SharePermission | None]:
        """One visible skill with its shared-in grant (loud 404)."""
        record = await self._repository.get(skill_id, tenant_id=tenant_id)
        shared = skill_access(record, visible_to, also_visible)
        return record, shared

    async def get_admitted(
        self, skill_id: str, *, tenant_id: str
    ) -> SkillRecord:
        """One skill WITHOUT a visibility check — for admitted runs only.

        The runs router already enforced visibility (incl. shared-in
        grants) at submission; the worker segment merely re-loads what
        was admitted and must not re-derive grants it cannot resolve.
        Every other caller uses :meth:`get_visible`.
        """
        return await self._repository.get(skill_id, tenant_id=tenant_id)

    async def list_visible(
        self,
        *,
        tenant_id: str,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> list[tuple[SkillRecord, SharePermission | None]]:
        """Visible skills with their shared-in grant (annotation)."""
        visible: list[tuple[SkillRecord, SharePermission | None]] = []
        for record in await self._repository.list_for_tenant(
            tenant_id=tenant_id
        ):
            try:
                shared = skill_access(record, visible_to, also_visible)
            except SkillNotFound:
                continue
            visible.append((record, shared))
        return visible

    async def update(
        self,
        skill_id: str,
        payload: Mapping[str, Any],
        *,
        tenant_id: str,
        expected_updated_at: float | None = None,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> SkillRecord:
        """Replace the writable fields (edit grant required)."""
        current = await self._repository.get(skill_id, tenant_id=tenant_id)
        shared = skill_access(current, visible_to, also_visible)
        if shared is not None and not shared.at_least(SharePermission.EDIT):
            raise SkillNotFound(skill_id)
        updated = SkillRecord(
            id=current.id,
            tenant_id=current.tenant_id,
            owner_sub=current.owner_sub,
            created_at=current.created_at,
            updated_at=time.time(),
            **_validated_fields(payload),
        )
        return await self._repository.update(
            updated, expected_updated_at=expected_updated_at
        )

    async def delete(
        self,
        skill_id: str,
        *,
        tenant_id: str,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> None:
        """Delete one skill (owner-only; shares never delete)."""
        current = await self._repository.get(skill_id, tenant_id=tenant_id)
        shared = skill_access(current, visible_to, also_visible)
        if shared is not None:
            raise SkillNotFound(skill_id)
        await self._repository.delete(skill_id, tenant_id=tenant_id)

    async def owner_sub(self, tenant_id: str, skill_id: str) -> str | None:
        """Owner lookup for the share layer (``None`` = unshareable)."""
        try:
            record = await self._repository.get(skill_id, tenant_id=tenant_id)
        except SkillNotFound:
            return None
        return record.owner_sub

    async def title(self, tenant_id: str, skill_id: str) -> str | None:
        """Title lookup for the share surface (``None`` = absent)."""
        try:
            record = await self._repository.get(skill_id, tenant_id=tenant_id)
        except SkillNotFound:
            return None
        return record.title
