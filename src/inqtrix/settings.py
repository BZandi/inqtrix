"""Pydantic Settings for type-safe configuration."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


_SETTINGS_MODEL_CONFIG = {
    "env_prefix": "",
    "extra": "ignore",
    "populate_by_name": True,
    "env_file": ".env",
    "env_file_encoding": "utf-8",
}


class ModelSettings(BaseSettings):
    """LLM model configuration."""

    model_config = _SETTINGS_MODEL_CONFIG

    reasoning_model: str = Field("claude-opus-4.6-agent", alias="REASONING_MODEL")
    search_model: str = Field("perplexity-sonar-pro-agent", alias="SEARCH_MODEL")
    classify_model: str = Field("", alias="CLASSIFY_MODEL")
    summarize_model: str = Field("", alias="SUMMARIZE_MODEL")
    evaluate_model: str = Field("", alias="EVALUATE_MODEL")

    @property
    def effective_classify_model(self) -> str:
        return self.classify_model or self.reasoning_model

    @property
    def effective_summarize_model(self) -> str:
        return self.summarize_model or self.reasoning_model

    @property
    def effective_evaluate_model(self) -> str:
        return self.evaluate_model or self.reasoning_model


class AgentSettings(BaseSettings):
    """Agent behavior configuration."""

    model_config = _SETTINGS_MODEL_CONFIG

    max_rounds: int = Field(4, alias="MAX_ROUNDS")
    confidence_stop: int = Field(8, alias="CONFIDENCE_STOP")
    max_context: int = Field(12, alias="MAX_CONTEXT")
    first_round_queries: int = Field(6, alias="FIRST_ROUND_QUERIES")
    answer_prompt_citations_max: int = Field(60, alias="ANSWER_PROMPT_CITATIONS_MAX")

    reasoning_timeout: int = Field(120, alias="REASONING_TIMEOUT")
    search_timeout: int = Field(60, alias="SEARCH_TIMEOUT")
    summarize_timeout: int = Field(60, alias="SUMMARIZE_TIMEOUT")
    max_total_seconds: int = Field(300, alias="MAX_TOTAL_SECONDS")
    max_question_length: int = Field(10_000, alias="MAX_QUESTION_LENGTH")

    high_risk_score_threshold: int = Field(4, alias="HIGH_RISK_SCORE_THRESHOLD")
    high_risk_classify_escalate: bool = Field(True, alias="HIGH_RISK_CLASSIFY_ESCALATE")
    high_risk_evaluate_escalate: bool = Field(True, alias="HIGH_RISK_EVALUATE_ESCALATE")

    search_cache_maxsize: int = Field(256, alias="SEARCH_CACHE_MAXSIZE")
    search_cache_ttl: int = Field(3600, alias="SEARCH_CACHE_TTL")

    testing_mode: bool = Field(False, alias="TESTING_MODE")


class ServerSettings(BaseSettings):
    """FastAPI server configuration."""

    model_config = _SETTINGS_MODEL_CONFIG

    litellm_base_url: str = Field("http://litellm-proxy:4000/v1", alias="LITELLM_BASE_URL")
    litellm_api_key: str = Field("sk-placeholder", alias="LITELLM_API_KEY")
    max_concurrent: int = Field(3, alias="MAX_CONCURRENT")
    max_messages_history: int = Field(20, alias="MAX_MESSAGES_HISTORY")
    session_ttl_seconds: int = Field(1800, alias="SESSION_TTL_SECONDS")
    session_max_count: int = Field(20, alias="SESSION_MAX_COUNT")
    session_max_context_blocks: int = Field(8, alias="SESSION_MAX_CONTEXT_BLOCKS")
    session_max_claim_ledger: int = Field(50, alias="SESSION_MAX_CLAIM_LEDGER")


class Settings(BaseSettings):
    """Root settings container."""

    models: ModelSettings = Field(default_factory=ModelSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
