"""Resolution of a result requirement against the caller's own catalog.

The requirement can be set in two places — at submit time, before the run
starts, and at the plan gate — and both must resolve attached library
rules the SAME way. Two copies of this logic would drift, and the drift
would be a security one: the whole point of resolving server-side is that
a client sends ids and nothing else.

The caller supplies free text and template ids; label, revision and body
come from the caller's own visible templates. A client-supplied body
would be an unchecked text channel straight into the writing prompts, and
a client-supplied label would let one rule impersonate another in the
origin markers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from inqtrix.agents.report_requirement import (
    MAX_COMPOSED_REQUIREMENT_CHARS,
    compose_report_requirement,
    composed_requirement_is_oversized,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from inqtrix.auth.context import UserContext


class ReportRequirementError(Exception):
    """A requirement that cannot be honored as asked.

    Never a silent drop: an unknown rule or an oversized composition
    fails loudly, because running under requirements the user believes
    are in force is the exact failure this feature exists to prevent.
    """

    def __init__(self, messages: "Sequence[str]") -> None:
        self.messages = list(messages)
        super().__init__("; ".join(self.messages))


async def resolve_report_rules(
    template_ids: "Sequence[str]",
    *,
    prompt_templates: Any,
    visible_to: "UserContext | None",
) -> list[tuple[str, str, int, str]]:
    """``(template_id, label, revision, body)`` per attached rule.

    Raises:
        ReportRequirementError: The library is inactive, or an id is not
            visible to this caller.
    """
    wanted = [str(item).strip() for item in template_ids if str(item).strip()]
    if not wanted:
        return []
    if prompt_templates is None:
        raise ReportRequirementError(
            ["Die Prompt-Bibliothek ist in dieser Instanz nicht aktiv."]
        )
    tenant_id = (
        visible_to.principal.tenant_id if visible_to is not None else "default"
    )
    visible = await prompt_templates.list_visible(
        tenant_id=tenant_id, visible_to=visible_to
    )
    by_id = {str(record.id): record for record, _access in visible}
    missing = [item for item in wanted if item not in by_id]
    if missing:
        raise ReportRequirementError(
            ["Unbekannte oder nicht sichtbare Regel: " + ", ".join(missing)]
        )
    return [
        (
            str(by_id[item].id),
            str(by_id[item].label),
            int(by_id[item].revision),
            str(by_id[item].content_markdown),
        )
        for item in wanted
    ]


async def resolve_report_requirement(
    *,
    free_text: str,
    template_ids: "Sequence[str]",
    prompt_templates: Any,
    visible_to: "UserContext | None",
) -> tuple[str, list[dict[str, Any]]]:
    """The composed requirement plus the parts it was composed from.

    Returns:
        ``(composed, parts)`` — ``composed`` is what the writing prompts
        see, ``parts`` is what the user chose, kept for the read-back so
        a surface can name the rules without re-reading the catalog.

    Raises:
        ReportRequirementError: Unresolvable rule, or a composition over
            the visible ceiling.
    """
    rules = await resolve_report_rules(
        template_ids, prompt_templates=prompt_templates, visible_to=visible_to
    )
    composed = compose_report_requirement(
        free_text=free_text,
        rules=[(label, content) for _, label, _, content in rules],
    )
    if composed_requirement_is_oversized(composed):
        raise ReportRequirementError(
            [
                "Die Ergebnisvorgabe ist mit "
                f"{len(composed)} Zeichen zu lang (Grenze "
                f"{MAX_COMPOSED_REQUIREMENT_CHARS}). Haenge weniger oder "
                "kuerzere Regeln an."
            ]
        )
    parts = [
        {"template_id": template_id, "label": label, "revision": revision}
        for template_id, label, revision, _content in rules
    ]
    return composed, parts
