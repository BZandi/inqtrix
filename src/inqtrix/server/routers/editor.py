"""Editor endpoints (``/v1/editor/suggest``, ``/v1/editor/instruct``).

Single-LLM-call orchestration moved verbatim from the monolithic
route factory. Model/effort routing follows the composer-selected
tier or the explicitly picked model (Designprinzip 6: the provider's
own metadata decides, with a loud warning when none exists).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from inqtrix.auth.principal import Principal
from inqtrix.exceptions import AgentStructuredOutputError
from inqtrix.model_routing import resolve_effort, resolve_model
from inqtrix.quota.models import QuotaDimension
from inqtrix.server.editor_instructions import (
    EDITOR_INSTRUCT_SCHEMA,
    EDITOR_INSTRUCT_SCHEMA_NAME,
    EditorInstructError,
    build_editor_instruct_prompt,
    parse_editor_instruct_payload,
    parse_editor_instruct_response,
    result_from_parsed as result_from_instruction_parsed,
    validate_editor_instruct_result,
)
from inqtrix.server.editor_suggestions import (
    EDITOR_SUGGEST_SCHEMA,
    EDITOR_SUGGEST_SCHEMA_NAME,
    EditorSuggestError,
    build_editor_suggest_prompt,
    clamp_background,
    parse_editor_suggest_payload,
    parse_editor_suggest_response,
    result_from_parsed,
    validate_editor_suggest_result,
    warnings_for_validation_issues,
)
from inqtrix.server.reference_documents import (
    clamp_reference_documents,
    render_reference_documents,
)
from inqtrix.server.routers import (
    quota_admission,
    quota_record,
    stack_error_response,
)
from inqtrix.services.agent_context import StackResolutionError
from inqtrix.services.request_parsing import (
    editor_wait_seconds,
    error_response,
    workspace_id_from_request,
)
from inqtrix.urls import sanitize_error

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer

log = logging.getLogger("inqtrix")

EDITOR_INSTRUCT_CONTEXT_FLOOR_TOKENS = 128_000
EDITOR_INSTRUCT_RESERVED_PROMPT_TOKENS = 6_000


def _usage_accumulator() -> dict[str, int]:
    """A fresh ``state`` dict for one ``llm.complete`` token tally.

    ``inqtrix.state.track_tokens`` accumulates with ``+=``, so the two
    counters must pre-exist. Passed as ``state=`` so the raw-completion
    path captures the provider's REAL usage (the structured path reads
    it off the response object directly).
    """
    return {"total_prompt_tokens": 0, "total_completion_tokens": 0}


def _accumulated_tokens(usage_state: dict[str, int]) -> int:
    """Total tokens a :func:`_usage_accumulator` captured (prompt + completion)."""
    return int(usage_state.get("total_prompt_tokens", 0) or 0) + int(
        usage_state.get("total_completion_tokens", 0) or 0
    )


def _structured_tokens(structured: object) -> int:
    """Real prompt+completion tokens off a structured-output response.

    Defensive ``getattr`` so a provider/test double without usage fields
    contributes 0 rather than raising.
    """
    return int(getattr(structured, "prompt_tokens", 0) or 0) + int(
        getattr(structured, "completion_tokens", 0) or 0
    )


def _resolve_editor_model(resolved, llm) -> tuple[str | None, str | None, list[str]]:
    """Resolve model/effort for one editor call, mirroring the chat picker.

    Returns:
        ``(model, effort, warnings)`` where a UI-picked concrete model
        wins, tier metadata resolves next, and a custom provider
        without metadata falls back to its own default with a visible
        warning (Designprinzip 1).
    """
    provider_models = getattr(llm, "models", None)
    requested_tier = resolved.agent_settings.model_tier or None
    requested_model = (resolved.agent_settings.model or "").strip()
    requested_effort = (resolved.agent_settings.effort or "").strip()
    warnings: list[str] = []
    if requested_model:
        # The UI picked a concrete model -> use it directly (surfaced as
        # explicit_request); works even without published model metadata.
        return requested_model, requested_effort or None, warnings
    if provider_models is not None:
        model = resolve_model("direct_chat", provider_models, requested_tier) or None
        effort = resolve_effort("direct_chat", provider_models, requested_tier) or None
        return model, effort, warnings
    warnings.append("Modellauswahl ueber Provider-Default (keine Tier-Metadaten).")
    return None, None, warnings


def build_router(container: "AppContainer") -> APIRouter:
    """Bind both editor routes against the container."""
    router = APIRouter()
    resolver = container.resolver
    quota_service = container.quota_service

    @router.post("/v1/editor/suggest")
    async def editor_suggest(
        req: Request,
        principal: Principal = Depends(container.principal_dependency),
    ):
        """Rewrite one editor paragraph with the LLM (no agent run).

        Serves the editor's Direkt single-paragraph edit and each
        per-paragraph call of a Sammeln global run. Unlike
        ``/v1/text/improvements`` it routes by the composer-selected
        tier (``agent_overrides.model_tier`` via the ``direct_chat``
        node) and prefers native structured output, with a visible
        prompt-JSON fallback.
        """
        try:
            body = await req.json()
        except Exception:
            return error_response(400, "Ungueltiger JSON-Body", "invalid_request_error")

        try:
            workspace_id_from_request(req, body)
            resolved = resolver.resolve(body)
            suggest_request = parse_editor_suggest_payload(
                body,
                max_text_chars=resolved.agent_settings.max_question_length,
                max_background_chars=400_000,
            )
        except ValueError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        except StackResolutionError as exc:
            return stack_error_response(exc)

        denied = await quota_admission(
            quota_service, principal, QuotaDimension.LLM_TOKENS
        )
        if denied is not None:
            return denied

        sem = container.semaphore_factory()
        if sem.locked():
            return error_response(
                429,
                "Zu viele gleichzeitige Anfragen. Bitte warten.",
                "rate_limit_error",
            )

        llm = resolved.providers.llm
        model, effort, warnings = _resolve_editor_model(resolved, llm)

        context_window = getattr(llm, "context_window_tokens", None)
        budget_tokens = max(2000, (context_window or 16000) - 6000)
        background, truncated = clamp_background(
            suggest_request.background,
            suggest_request.block_text,
            max_chars=budget_tokens * 3,
        )
        if truncated:
            # Read-only report context is windowed around the edited paragraph.
            # For a single-paragraph rewrite the excerpt is sufficient, so this
            # is informational, not a failure — phrased neutrally so it does not
            # read as an error, and logged for visibility (No Silent Fallbacks).
            log.warning(
                "Editor suggest: report context windowed to the edited passage."
            )
            warnings.append(
                "Fuer diesen Vorschlag wurde nur der relevante Ausschnitt des "
                "Berichts beruecksichtigt."
            )
        suggest_request = replace(suggest_request, background=background)

        reference_budget = max(0, budget_tokens * 3 - len(background))
        reference_docs, reference_truncated = clamp_reference_documents(
            suggest_request.reference_documents,
            max_chars=reference_budget,
        )
        warnings.extend(suggest_request.reference_warnings)
        if reference_truncated:
            warnings.append("Reference documents truncated (context budget).")
            log.warning(
                "Editor suggestion reference documents truncated to fit the context budget."
            )
        reference_block = render_reference_documents(reference_docs)

        # Editor suggest/instruct are full reasoning+generation calls (a whole
        # paragraph or document, large attached context), not the tight
        # per-search-hit claim extraction. They run under the dedicated
        # editor_assistant_timeout (default 120s, raise it via
        # EDITOR_ASSISTANT_TIMEOUT for long instructions with large attachments)
        # so editor work can be lengthened without inflating research reasoning
        # calls. wait_timeout wraps the executor with the matching HTTP grace.
        timeout = resolved.agent_settings.editor_assistant_timeout
        wait_timeout = editor_wait_seconds(resolved.agent_settings)
        async with sem:
            loop = asyncio.get_running_loop()

            async def complete_editor_prompt(
                prompt: str,
                prompt_warnings: list[str],
            ) -> tuple[object, int]:
                """Run one suggest call; return ``(result, tokens_consumed)``.

                Tokens come from the provider's REAL usage — the
                structured response carries it directly; the raw path
                reads it back from the ``state`` accumulator
                (:func:`inqtrix.state.track_tokens`). This includes any
                reasoning tokens the char estimate would miss.
                """
                if llm.supports_structured_output(model=model):
                    structured = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            partial(
                                llm.complete_structured,
                                prompt,
                                schema=EDITOR_SUGGEST_SCHEMA,
                                schema_name=EDITOR_SUGGEST_SCHEMA_NAME,
                                model=model,
                                reasoning_effort=effort,
                                max_output_tokens=4000,
                                timeout=timeout,
                            ),
                        ),
                        timeout=wait_timeout,
                    )
                    return (
                        result_from_parsed(
                            structured.parsed, warnings=prompt_warnings
                        ),
                        _structured_tokens(structured),
                    )
                usage_state = _usage_accumulator()
                raw_response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        partial(
                            llm.complete,
                            prompt,
                            model=model,
                            reasoning_effort=effort,
                            max_output_tokens=4000,
                            timeout=timeout,
                            state=usage_state,
                        ),
                    ),
                    timeout=wait_timeout,
                )
                return (
                    parse_editor_suggest_response(
                        raw_response, warnings=prompt_warnings
                    ),
                    _accumulated_tokens(usage_state),
                )

            try:
                prompt = build_editor_suggest_prompt(
                    suggest_request,
                    reference_block=reference_block,
                )
                result, consumed = await complete_editor_prompt(prompt, warnings)
                validation_issues = validate_editor_suggest_result(suggest_request, result)
                if validation_issues:
                    log.warning(
                        "Editor-Vorschlag verletzt Editiervertrag: %s",
                        ", ".join(issue.code for issue in validation_issues),
                    )
                    retry_prompt = build_editor_suggest_prompt(
                        suggest_request,
                        previous_result=result.rewritten_text,
                        validation_issues=validation_issues,
                        reference_block=reference_block,
                    )
                    retry_result, retry_tokens = await complete_editor_prompt(
                        retry_prompt, warnings
                    )
                    # The retry is a second full LLM call — its tokens
                    # add to the booked spend (the estimate billed it as
                    # one).
                    result = retry_result
                    consumed += retry_tokens
                    validation_issues = validate_editor_suggest_result(
                        suggest_request, result
                    )
                    if validation_issues:
                        result = replace(
                            result,
                            warnings=[
                                *result.warnings,
                                *warnings_for_validation_issues(
                                    validation_issues,
                                    locale=suggest_request.locale,
                                ),
                            ],
                        )
            except asyncio.TimeoutError:
                log.warning(
                    "Editor-Vorschlag Timeout nach %ss "
                    "(EDITOR_ASSISTANT_TIMEOUT/MAX_TOTAL_SECONDS erhoehen?)",
                    wait_timeout,
                )
                return error_response(504, "Editor-Vorschlag Timeout", "timeout_error")
            except (EditorSuggestError, AgentStructuredOutputError) as exc:
                log.warning("Editor-Vorschlag konnte nicht geparst werden: %s", exc)
                return error_response(502, str(exc), "server_error")
            except Exception as exc:  # noqa: BLE001 — provider failures map to 502
                log.error("Editor-Vorschlag Fehler: %s", exc)
                return error_response(
                    502,
                    f"Editor-Vorschlag Fehler: {sanitize_error(exc)}",
                    "server_error",
                )

        await quota_record(
            quota_service, principal, QuotaDimension.LLM_TOKENS, consumed
        )
        return result.to_payload()

    @router.post("/v1/editor/instruct")
    async def editor_instruct(
        req: Request,
        principal: Principal = Depends(container.principal_dependency),
    ):
        """Turn one free-form editor instruction into anchored document edits."""
        try:
            body = await req.json()
        except Exception:
            return error_response(400, "Ungueltiger JSON-Body", "invalid_request_error")

        try:
            workspace_id_from_request(req, body)
            resolved = resolver.resolve(body)
            instruct_request = parse_editor_instruct_payload(
                body,
                max_instruction_chars=resolved.agent_settings.max_question_length,
                max_document_chars=400_000,
            )
        except ValueError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        except StackResolutionError as exc:
            return stack_error_response(exc)

        denied = await quota_admission(
            quota_service, principal, QuotaDimension.LLM_TOKENS
        )
        if denied is not None:
            return denied

        sem = container.semaphore_factory()
        if sem.locked():
            return error_response(
                429,
                "Zu viele gleichzeitige Anfragen. Bitte warten.",
                "rate_limit_error",
            )

        llm = resolved.providers.llm
        context_window = getattr(llm, "context_window_tokens", None)
        if not isinstance(context_window, int) or context_window <= 0:
            context_window = None
        effective_context_window = max(
            context_window or 0,
            EDITOR_INSTRUCT_CONTEXT_FLOOR_TOKENS,
        )
        budget_chars = max(
            2000,
            effective_context_window - EDITOR_INSTRUCT_RESERVED_PROMPT_TOKENS,
        ) * 3
        if len(instruct_request.document_markdown) > budget_chars:
            return error_response(
                400,
                "Dokument zu groß für eine Komplettüberarbeitung.",
                "invalid_request_error",
            )

        model, effort, warnings = _resolve_editor_model(resolved, llm)

        reference_budget = max(0, budget_chars - len(instruct_request.document_markdown))
        reference_docs, reference_truncated = clamp_reference_documents(
            instruct_request.reference_documents,
            max_chars=reference_budget,
        )
        warnings.extend(instruct_request.reference_warnings)
        if reference_truncated:
            warnings.append("Reference documents truncated (context budget).")
            log.warning(
                "Editor instruction reference documents truncated to fit the context budget."
            )
        reference_block = render_reference_documents(reference_docs)

        # Editor suggest/instruct are full reasoning+generation calls (a whole
        # paragraph or document, large attached context), not the tight
        # per-search-hit claim extraction. They run under the dedicated
        # editor_assistant_timeout (default 120s, raise it via
        # EDITOR_ASSISTANT_TIMEOUT for long instructions with large attachments)
        # so editor work can be lengthened without inflating research reasoning
        # calls. wait_timeout wraps the executor with the matching HTTP grace.
        timeout = resolved.agent_settings.editor_assistant_timeout
        wait_timeout = editor_wait_seconds(resolved.agent_settings)
        async with sem:
            loop = asyncio.get_running_loop()

            async def complete_instruction_prompt(
                prompt: str,
            ) -> tuple[object, int]:
                """Run one instruct call; return ``(result, tokens_consumed)``.

                Real provider usage, as in the suggest path — structured
                response carries it; the raw path reads it back from the
                ``state`` accumulator.
                """
                if llm.supports_structured_output(model=model):
                    structured = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            partial(
                                llm.complete_structured,
                                prompt,
                                schema=EDITOR_INSTRUCT_SCHEMA,
                                schema_name=EDITOR_INSTRUCT_SCHEMA_NAME,
                                model=model,
                                reasoning_effort=effort,
                                max_output_tokens=8000,
                                timeout=timeout,
                            ),
                        ),
                        timeout=wait_timeout,
                    )
                    return (
                        result_from_instruction_parsed(structured.parsed),
                        _structured_tokens(structured),
                    )
                usage_state = _usage_accumulator()
                raw_response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        partial(
                            llm.complete,
                            prompt,
                            model=model,
                            reasoning_effort=effort,
                            max_output_tokens=8000,
                            timeout=timeout,
                            state=usage_state,
                        ),
                    ),
                    timeout=wait_timeout,
                )
                return (
                    parse_editor_instruct_response(raw_response),
                    _accumulated_tokens(usage_state),
                )

            try:
                prompt = build_editor_instruct_prompt(
                    instruct_request,
                    reference_block=reference_block,
                )
                result, consumed = await complete_instruction_prompt(prompt)
                if warnings:
                    result = replace(result, warnings=[*warnings, *result.warnings])
                result = validate_editor_instruct_result(instruct_request, result)
            except asyncio.TimeoutError:
                log.warning(
                    "Editor-Anweisung Timeout nach %ss "
                    "(EDITOR_ASSISTANT_TIMEOUT/MAX_TOTAL_SECONDS erhoehen?)",
                    wait_timeout,
                )
                return error_response(504, "Editor-Anweisung Timeout", "timeout_error")
            except (EditorInstructError, AgentStructuredOutputError) as exc:
                log.warning("Editor-Anweisung konnte nicht geparst werden: %s", exc)
                return error_response(502, str(exc), "server_error")
            except Exception as exc:  # noqa: BLE001 — provider failures map to 502
                log.error("Editor-Anweisung Fehler: %s", exc)
                return error_response(
                    502,
                    f"Editor-Anweisung Fehler: {sanitize_error(exc)}",
                    "server_error",
                )

        await quota_record(
            quota_service, principal, QuotaDimension.LLM_TOKENS, consumed
        )
        return result.to_payload()

    return router
