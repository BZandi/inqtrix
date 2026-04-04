""" Custom exceptions for the research agent."""

from __future__ import annotations


class AgentTimeout(Exception):
    """Raised when MAX_TOTAL_SECONDS is exceeded."""


class AgentRateLimited(Exception):
    """Raised on 429 rate-limit or daily token limit."""

    def __init__(self, model: str, original: Exception):
        self.model = model
        self.original = original
        super().__init__(f"Rate-Limit erreicht fuer Modell '{model}': {original}")
