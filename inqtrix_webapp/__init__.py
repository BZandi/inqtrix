"""Streamlit webapp helpers for talking to the Inqtrix HTTP server.

This package lives next to ``webapp.py`` at the repository root and
provides a thin, synchronous HTTP client for the OpenAI-compatible
Inqtrix server (``src/inqtrix/server``). It intentionally has no
dependency on the ``inqtrix`` package itself so the UI stays a pure
HTTP consumer.
"""

from inqtrix_webapp.client import (
    call_chat,
    fetch_health,
    fetch_models_fallback,
    fetch_stacks,
    get_api_key,
    get_base_url,
    stream_chat,
)

__all__ = [
    "call_chat",
    "fetch_health",
    "fetch_models_fallback",
    "fetch_stacks",
    "get_api_key",
    "get_base_url",
    "stream_chat",
]
