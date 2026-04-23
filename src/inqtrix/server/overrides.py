"""Per-request override surface for ``/v1/chat/completions``.

This module hosts the whitelist that lets HTTP callers (Streamlit
sliders, OpenAI-SDK clients with extra-body, integration tests)
override a small, deliberate set of behavioural fields per request.
The override travels in the JSON body under the optional
``"agent_overrides"`` key; OpenAI-SDK clients that do not know about
the field simply omit it, so the schema stays drop-in compatible.

How to add a new override field
===============================

The merge logic is generic: every new whitelist field flows through
:func:`apply_overrides` and on into ``agent_run`` without touching
``routes.py`` or the LangGraph code. Adding a new field is a
five-step recipe:

1. **Precondition** — the field must already exist on
   :class:`~inqtrix.settings.AgentSettings`. Practically every
   behavioural field there is fair game (``min_rounds``,
   ``high_risk_*``, ``enable_de_policy_bias``, ``search_cache_*``,
   ``summarize_timeout``, ``reasoning_timeout``, ``search_timeout``,
   ``answer_prompt_citations_max`` and friends).

   Forbidden without a fresh ADR: provider selection, model names,
   ``testing_mode``, anything on
   :class:`~inqtrix.settings.ServerSettings` (``max_concurrent``,
   ``api_key``, ``cors_origins``, ``tls_*``, ``session_*``) and any
   strategy slot. These are deployment / operator concerns, not
   request concerns.

2. **AgentOverridesRequest** — add a single field with sane range
   validation::

       new_field: int | None = Field(default=None, ge=..., le=...)

3. ``apply_overrides`` — leave it alone. The implementation calls
   ``base.model_copy(update=overrides.model_dump(exclude_none=True))``
   followed by ``with_report_profile_defaults(...)`` and is generic.

4. ``routes.chat_completions`` — leave it alone. The route hands the
   whole Pydantic model through.

5. ``tests/test_server_overrides.py`` — add at least one test for the
   new field (range validation + an end-to-end test against a mocked
   ``agent_run``).

The same recipe lives as a convention entry in
``.cursor/memory/conventions.md`` so follow-up agents do not have to
reverse-engineer the path.

Profile-switch semantics (ADR-WS-6, scenarios A/B/C)
====================================================

When a caller flips ``report_profile`` per request, the profile preset
fans out across other settings via
:meth:`AgentSettings.with_report_profile_defaults`. To keep both the
operator's explicit defaults *and* the caller's explicit overrides
authoritative, :func:`apply_overrides` builds the ``explicit_fields``
union from two sources:

* ``base.model_fields_set`` — fields the operator set explicitly
  (server-side ``AgentConfig`` / env / YAML).
* The keys present in the request override.

The profile defaults only fill in fields that **neither** side
specified, which preserves operator intent and request intent at
the same time.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from inqtrix.report_profiles import ReportProfile
from inqtrix.settings import AgentSettings

log = logging.getLogger("inqtrix")


class AgentOverridesRequest(BaseModel):
    """Whitelist of behavioural fields a caller may override per request.

    Mirrors a deliberate subset of :class:`~inqtrix.settings.AgentSettings`.
    Unknown keys are rejected with a Pydantic ``ValidationError`` so the
    server can return a clean ``400 invalid_request_error`` instead of
    silently ignoring typos.

    Args:
        max_rounds: Hard upper bound for the research loop. ``1``-``10``.
            See :attr:`AgentConfig.max_rounds` for tuning guidance.
        min_rounds: Lower bound for the research loop. ``1``-``10`` and
            must not exceed ``max_rounds`` when both are present in the
            same request.
        confidence_stop: Minimum evaluator confidence (1-10) at which
            the stop cascade may emit ``done``.
        report_profile: ``"compact"`` or ``"deep"``. Triggers the
            profile preset cascade for non-explicit fields, see module
            docstring for the merge semantics.
        max_total_seconds: Wall-clock deadline for the entire run, in
            seconds. ``30``-``1800``.
        first_round_queries: Number of broad queries generated in
            Round 0 by the plan node. ``1``-``20``.
        max_context: Maximum number of context blocks retained between
            rounds. ``1``-``50``.
    """

    model_config = ConfigDict(extra="forbid")

    max_rounds: int | None = Field(default=None, ge=1, le=10)
    """Hard upper bound for the research loop."""

    min_rounds: int | None = Field(default=None, ge=1, le=10)
    """Lower bound for the research loop."""

    confidence_stop: int | None = Field(default=None, ge=1, le=10)
    """Minimum evaluator confidence (1-10) before the stop cascade may emit done."""

    report_profile: ReportProfile | None = None
    """Report style preset (``compact`` or ``deep``)."""

    max_total_seconds: int | None = Field(default=None, ge=30, le=1800)
    """Wall-clock deadline for the entire run, in seconds."""

    first_round_queries: int | None = Field(default=None, ge=1, le=20)
    """Number of broad queries generated in Round 0."""

    max_context: int | None = Field(default=None, ge=1, le=50)
    """Maximum number of context blocks retained between rounds."""

    skip_search: bool | None = Field(default=None)
    """When ``True`` the request skips plan/search/evaluate and hits the LLM
    directly. See :attr:`AgentSettings.skip_search` for semantics."""

    enable_de_policy_bias: bool | None = Field(default=None)
    """Toggle the German health- and social-policy heuristics
    (quality-site injection, utility-stop suppression, risk-score
    bonus). See :attr:`AgentSettings.enable_de_policy_bias` for the
    full contract and the two strategies it influences
    (:mod:`inqtrix.strategies._risk_scoring`,
    :mod:`inqtrix.strategies._stop_criteria`)."""


def parse_overrides_payload(payload: Any) -> AgentOverridesRequest | None:
    """Validate the raw ``agent_overrides`` body slice into the model.

    Args:
        payload: The raw value of ``body["agent_overrides"]``. ``None``
            (the field was absent) returns ``None`` without raising.

    Returns:
        The validated :class:`AgentOverridesRequest`, or ``None`` when
        no overrides were supplied.

    Raises:
        HTTPException: 400 ``invalid_request_error`` when the payload is
            not a dict, contains unknown keys, or fails range validation.
    """
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "agent_overrides muss ein Objekt sein",
                    "type": "invalid_request_error",
                }
            },
        )
    try:
        overrides = AgentOverridesRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Ungueltige agent_overrides: {exc.errors()}",
                    "type": "invalid_request_error",
                }
            },
        ) from exc
    return overrides


def apply_overrides(
    base: AgentSettings,
    overrides: AgentOverridesRequest | None,
) -> AgentSettings:
    """Merge per-request overrides into a base ``AgentSettings`` instance.

    Returns ``base`` unchanged when no overrides are present so that the
    common no-override path stays allocation-free.

    The merge is generic and treats every whitelist field uniformly.
    For the ``report_profile`` cascade see the module docstring.

    Args:
        base: The ``AgentSettings`` resolved from server-side
            ``AgentConfig`` / env / YAML for this server process.
        overrides: Validated request-side overrides. ``None`` returns
            ``base`` verbatim.

    Returns:
        A new :class:`AgentSettings` with the overrides applied and the
        ``report_profile``-driven defaults expanded for every field
        neither the operator nor the caller set explicitly.

    Side Effects:
        Emits an ``INFO`` log line for each applied override field so
        the audit trail tracks which slider moved which value.
    """
    if overrides is None:
        return base
    update = overrides.model_dump(exclude_none=True)
    if not update:
        return base
    for field_name, value in update.items():
        log.info("Per-request override: %s=%r", field_name, value)
    # Combine the two "explicit" sources so profile presets neither
    # overwrite operator-set fields nor caller-set fields. See module
    # docstring (scenarios A / B / C) for the rationale.
    explicit_fields = set(base.model_fields_set) | set(update.keys())
    patched = base.model_copy(update=update)
    return patched.with_report_profile_defaults(explicit_fields=explicit_fields)
