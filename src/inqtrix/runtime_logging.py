"""Helpers for structured runtime logging.

Keeps provider/model metadata extraction and iteration-entry formatting in
one place so normal runtime logs and testing-mode iteration logs stay
semantically aligned.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from inqtrix.settings import AgentSettings
from inqtrix.urls import sanitize_error

log = logging.getLogger("inqtrix")


def _serialize_payload(payload: dict[str, Any]) -> str:
    """Render a payload as stable JSON for debug logs.

    Values are pre-sanitised via :func:`_sanitize_value` **before** JSON
    serialisation so that the ``_RedactSecretsFilter`` in
    ``logging_config.py`` does not corrupt the JSON structure.  The
    filter's ``sanitize_error`` regex ``https?://[^\\s]+`` can consume
    trailing quotes, turning ``"url": "https://x"`` into ``"url": "[URL]``
    (missing closing quote).  Pre-sanitising replaces URLs while the
    value is still a plain string, keeping JSON delimiters intact.
    """
    return json.dumps(_sanitize_value(payload), ensure_ascii=False, sort_keys=True, default=str)


def _sanitize_value(value: Any) -> Any:
    """Recursively sanitize values before JSON serialization."""
    if isinstance(value, dict):
        return {key: _sanitize_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, (str, Exception)):
        return sanitize_error(value)
    return value


def _unwrap_provider(provider: object) -> object:
    """Return the wrapped provider for ConfiguredLLMProvider-like adapters."""
    if type(provider).__name__ == "ConfiguredLLMProvider" and hasattr(provider, "_provider"):
        return getattr(provider, "_provider")
    return provider


def _clean_value(value: Any) -> Any:
    if value in (None, "", [], {}):
        return None
    return value


def describe_llm_provider(provider: object) -> dict[str, Any]:
    """Extract human-readable runtime metadata for the active LLM provider."""
    resolved = _unwrap_provider(provider)
    metadata: dict[str, Any] = {
        "provider": type(resolved).__name__,
    }

    models = getattr(provider, "models", None)
    if models is not None:
        metadata["reasoning_model"] = str(getattr(models, "reasoning_model", "") or "")
        metadata["classify_model"] = str(
            getattr(models, "effective_classify_model", "")
            or getattr(models, "classify_model", "")
            or getattr(models, "reasoning_model", "")
            or ""
        )
        metadata["summarize_model"] = str(
            getattr(models, "effective_summarize_model", "")
            or getattr(models, "summarize_model", "")
            or getattr(models, "reasoning_model", "")
            or ""
        )
        metadata["evaluate_model"] = str(
            getattr(models, "effective_evaluate_model", "")
            or getattr(models, "evaluate_model", "")
            or getattr(models, "reasoning_model", "")
            or ""
        )

    for attr_name in (
        "_default_max_tokens",
        "_summarize_max_tokens",
        "_token_budget_parameter",
        "_temperature",
    ):
        cleaned = _clean_value(getattr(resolved, attr_name, None))
        if cleaned is not None:
            metadata[attr_name.lstrip("_")] = cleaned

    thinking = getattr(resolved, "_thinking", None)
    if isinstance(thinking, dict) and thinking:
        metadata["thinking"] = dict(thinking)

    return metadata


def describe_search_provider(provider: object) -> dict[str, Any]:
    """Extract human-readable runtime metadata for the active search provider."""
    resolved = _unwrap_provider(provider)
    metadata: dict[str, Any] = {
        "provider": type(resolved).__name__,
    }

    model = _clean_value(getattr(resolved, "_model", None))
    agent_name = _clean_value(getattr(resolved, "_agent_name", None))
    agent_version = _clean_value(getattr(resolved, "_agent_version", None))
    agent_id = _clean_value(getattr(resolved, "_agent_id", None))

    if model is not None:
        metadata["engine"] = str(model)
    elif agent_name is not None:
        engine = str(agent_name)
        if agent_version is not None:
            engine = f"{engine}@{agent_version}"
        metadata["engine"] = engine
        metadata["agent_name"] = str(agent_name)
        if agent_version is not None:
            metadata["agent_version"] = str(agent_version)
    elif agent_id is not None:
        metadata["engine"] = str(agent_id)
        metadata["agent_id"] = str(agent_id)
    elif type(resolved).__name__ == "BraveSearch":
        metadata["engine"] = "brave-web-search"

    return metadata


def build_run_metadata(
    *,
    question: str,
    history: str,
    prev_session: dict[str, Any] | None,
    providers: Any,
    settings: AgentSettings,
    run_mode: str = "run",
) -> dict[str, Any]:
    """Build a structured payload for the start of a research run."""
    return {
        "event": "run_start",
        "run_mode": run_mode,
        "question_length": len(question or ""),
        "history_length": len(history or ""),
        "seeded_session": bool(prev_session),
        "llm": describe_llm_provider(getattr(providers, "llm", None)),
        "search": describe_search_provider(getattr(providers, "search", None)),
        "settings": {
            "report_profile": str(settings.report_profile),
            "max_rounds": settings.max_rounds,
            "confidence_stop": settings.confidence_stop,
            "max_total_seconds": settings.max_total_seconds,
            "testing_mode": settings.testing_mode,
        },
    }


def log_run_start(
    *,
    question: str,
    history: str,
    prev_session: dict[str, Any] | None,
    providers: Any,
    settings: AgentSettings,
    run_mode: str = "run",
) -> None:
    """Write a compact start banner plus structured debug metadata."""
    metadata = build_run_metadata(
        question=question,
        history=history,
        prev_session=prev_session,
        providers=providers,
        settings=settings,
        run_mode=run_mode,
    )

    llm = metadata["llm"]
    search = metadata["search"]
    run_settings = metadata["settings"]

    log.info(
        "RUN start: mode=%s profile=%s llm=%s reasoning=%s classify=%s evaluate=%s summarize=%s default_max_tokens=%s search=%s engine=%s max_rounds=%d confidence_stop=%d max_total_seconds=%d testing_mode=%s question_len=%d history_len=%d seeded_session=%s",
        run_mode,
        run_settings.get("report_profile") or "compact",
        llm.get("provider") or "unknown",
        llm.get("reasoning_model") or "-",
        llm.get("classify_model") or "-",
        llm.get("evaluate_model") or "-",
        llm.get("summarize_model") or "-",
        llm.get("default_max_tokens") if llm.get("default_max_tokens") is not None else "-",
        search.get("provider") or "unknown",
        search.get("engine") or "-",
        run_settings["max_rounds"],
        run_settings["confidence_stop"],
        run_settings["max_total_seconds"],
        run_settings["testing_mode"],
        metadata["question_length"],
        metadata["history_length"],
        metadata["seeded_session"],
    )
    log.debug("RUN metadata: %s", _serialize_payload(metadata))


def log_iteration_entry(entry: dict[str, Any]) -> None:
    """Write a structured iteration payload to DEBUG logs."""
    if not log.isEnabledFor(logging.DEBUG):
        return
    node = str(entry.get("node", "unknown"))
    log.debug("ITERATION %s: %s", node, _serialize_payload(entry))
