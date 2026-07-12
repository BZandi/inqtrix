"""Wave-1 web capability: a single grounded search call (no research graph).

``web.search.instant`` is the smallest external-information tool: ONE
:class:`~inqtrix.providers.base.SearchProvider` call, returning the
provider's grounded answer and sources as citable references — no
LangGraph research run, no child run. It is the discovery-preview and
small-gap tool; a full report goes through ``research`` (E18).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from inqtrix.capabilities.contracts import (
    CapabilityContext,
    CapabilityDefinition,
    CapabilityError,
    Effect,
)
from inqtrix.search_result import GroundedSearchResult
from inqtrix.providers.base import observe_provider_retries

if TYPE_CHECKING:
    from inqtrix.providers.base import SearchProvider

_RECENCY = {"", "day", "week", "month", "year"}


class WebInstantInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    recency: str = Field("", description="'', 'day', 'week', 'month', 'year'")
    max_sources: int = Field(8, ge=1, le=20)


class WebSource(BaseModel):
    url: str
    title: str
    snippet: str
    date: str
    rank: int


class WebInstantOutput(BaseModel):
    query: str
    answer: str
    sources: list[WebSource]
    prompt_tokens: int = Field(0, ge=0)
    completion_tokens: int = Field(0, ge=0)


def build_web_capabilities(
    search_provider: "SearchProvider",
) -> list[CapabilityDefinition]:
    """Build the wave-1 web capability bound to *search_provider*."""

    async def _instant(
        payload: WebInstantInput, context: CapabilityContext
    ) -> WebInstantOutput:
        if payload.recency not in _RECENCY:
            raise CapabilityError(
                "invalid_input",
                "recency muss '', 'day', 'week', 'month' oder 'year' sein.",
                http_status=400,
            )
        # SearchProvider.search is synchronous (blocking HTTP); keep it off
        # the event loop.
        def _search_with_notice() -> tuple[GroundedSearchResult, object]:
            callback = None
            if context.on_provider_retry is not None:
                callback = lambda notice: context.on_provider_retry({
                    **notice,
                    "operation": "web.search.instant",
                    "query": payload.query,
                })
            with observe_provider_retries(search_provider, callback):
                result = search_provider.search(
                    payload.query,
                    search_context_size="low",
                    recency_filter=payload.recency or None,
                )
            consumer = getattr(
                search_provider, "consume_nonfatal_notice_detail", None
            )
            notice = consumer() if callable(consumer) else None
            return result, notice

        result, notice = await asyncio.to_thread(_search_with_notice)
        if isinstance(notice, dict) and notice.get("code"):
            raise CapabilityError(
                str(notice["code"]),
                str(notice.get("message") or "Websuche fehlgeschlagen."),
                http_status=int(notice.get("http_status") or 502),
            )
        sources = [
            WebSource(
                url=source.url,
                title=source.title,
                snippet=source.snippet,
                date=source.date,
                rank=source.rank or index,
            )
            for index, source in enumerate(result.sources[: payload.max_sources], start=1)
        ]
        return WebInstantOutput(
            query=payload.query,
            answer=result.answer,
            sources=sources,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )

    return [
        CapabilityDefinition(
            id="web.search.instant",
            summary="Run one grounded web search (no research graph).",
            input_model=WebInstantInput,
            output_model=WebInstantOutput,
            effect=Effect.READ,
            idempotent=True,
            handler=_instant,
        ),
    ]
