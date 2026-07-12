"""Editor endpoints (``/v1/editor/suggest``, ``/v1/editor/instruct``).

Thin HTTP layer over the shared orchestration cores in
:mod:`inqtrix.services.editor_assist_service` (extracted for the M7
workspace-agent patch flow). The router keeps request parsing, workspace
resolution, quota admission/record, the concurrency semaphore, and the
exception -> error-envelope mapping; the SYNC cores run in the executor
under the same ``editor_wait_seconds`` HTTP grace as before, so the wire
behavior is unchanged.
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from inqtrix.auth.principal import Principal
from inqtrix.exceptions import AgentStructuredOutputError
from inqtrix.quota.models import QuotaDimension
from inqtrix.server.editor_instructions import (
    EditorInstructError,
    parse_editor_instruct_payload,
)
from inqtrix.server.editor_suggestions import (
    EditorSuggestError,
    parse_editor_suggest_payload,
)
from inqtrix.server.routers import (
    quota_admission,
    quota_record,
    stack_error_response,
)
from inqtrix.services.agent_context import StackResolutionError
from inqtrix.services.editor_assist_service import (
    EditorDocumentTooLarge,
    resolve_editor_model,
    run_editor_instruct,
    run_editor_suggest,
)
from inqtrix.services.request_parsing import (
    editor_wait_seconds,
    error_response,
    workspace_id_from_request,
)
from inqtrix.urls import sanitize_error

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer

log = logging.getLogger("inqtrix")


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
        model, effort, warnings = resolve_editor_model(resolved, llm)

        # Editor suggest/instruct are full reasoning+generation calls (a whole
        # paragraph or document, large attached context), not the tight
        # per-search-hit claim extraction. They run under the dedicated
        # editor_assistant_timeout (default 600s, tune it via
        # EDITOR_ASSISTANT_TIMEOUT for long instructions with large attachments)
        # so editor work can be lengthened without inflating research reasoning
        # calls. wait_timeout wraps the executor with the matching HTTP grace.
        timeout = resolved.agent_settings.editor_assistant_timeout
        wait_timeout = editor_wait_seconds(resolved.agent_settings)
        async with sem:
            loop = asyncio.get_running_loop()
            try:
                result, consumed = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        partial(
                            run_editor_suggest,
                            suggest_request,
                            llm=llm,
                            model=model,
                            reasoning_effort=effort,
                            structured_supported=llm.supports_structured_output(
                                model=model
                            ),
                            timeout_seconds=timeout,
                            base_warnings=warnings,
                        ),
                    ),
                    timeout=wait_timeout,
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
        model, effort, warnings = resolve_editor_model(resolved, llm)

        # Same dedicated editor budget as the suggest route (see above).
        timeout = resolved.agent_settings.editor_assistant_timeout
        wait_timeout = editor_wait_seconds(resolved.agent_settings)
        async with sem:
            loop = asyncio.get_running_loop()
            try:
                result, consumed = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        partial(
                            run_editor_instruct,
                            instruct_request,
                            llm=llm,
                            model=model,
                            reasoning_effort=effort,
                            structured_supported=llm.supports_structured_output(
                                model=model
                            ),
                            timeout_seconds=timeout,
                            base_warnings=warnings,
                        ),
                    ),
                    timeout=wait_timeout,
                )
            except asyncio.TimeoutError:
                log.warning(
                    "Editor-Anweisung Timeout nach %ss "
                    "(EDITOR_ASSISTANT_TIMEOUT/MAX_TOTAL_SECONDS erhoehen?)",
                    wait_timeout,
                )
                return error_response(504, "Editor-Anweisung Timeout", "timeout_error")
            except EditorDocumentTooLarge as exc:
                return error_response(400, str(exc), "invalid_request_error")
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
