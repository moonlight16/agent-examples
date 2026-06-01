"""Configuration settings for Kagenti Chat."""

import json
import os
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default=os.getenv("LOG_LEVEL", "INFO"),  # type: ignore[arg-type]
        description="Application log level",
    )

    A2A_HOST: str = Field(
        default=os.getenv("A2A_HOST", "0.0.0.0"),
        description="Host address for the A2A server",
    )
    A2A_PORT: int = Field(
        default=int(os.getenv("A2A_PORT", "8000")),
        description="Port for the A2A server",
    )
    A2A_PUBLIC_URL: Optional[str] = Field(
        default=os.getenv("A2A_PUBLIC_URL"),
        description="Publicly routable A2A base URL advertised in the AgentCard",
    )

    LLM_MODEL: str = Field(
        default=os.getenv("LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
        description="LLM model name (must match what the LLM endpoint advertises)",
    )
    LLM_API_KEY: Optional[str] = Field(
        default=os.getenv("LLM_API_KEY", "dummy"),
        description="API key for the LLM provider (vLLM/llm-d typically accept any value)",
    )
    # Accept both LLM_BASE_URL (simple_generalist heritage) and LLM_API_BASE
    # (the convention used elsewhere in this repo's sample-environments.yaml).
    LLM_BASE_URL: Optional[str] = Field(
        default=os.getenv("LLM_BASE_URL") or os.getenv("LLM_API_BASE"),
        description="Base URL of the OpenAI-compatible LLM endpoint",
        validation_alias=AliasChoices("LLM_BASE_URL", "LLM_API_BASE"),
    )
    LLM_TEMPERATURE: float = Field(
        default=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        description="Sampling temperature",
    )
    EXTRA_HEADERS: dict[str, str] = Field(
        default_factory=dict,
        description="Extra HTTP headers for the LLM API (JSON object)",
    )

    @field_validator("EXTRA_HEADERS", mode="before")
    @classmethod
    def _parse_extra_headers(cls, v: Any) -> dict[str, str]:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return {}
            return json.loads(v)
        if v is None:
            return {}
        return v

    OTEL_CONSOLE_TRACING: bool = Field(
        default=os.getenv("OTEL_CONSOLE_TRACING", "false").lower() in ("true", "1", "yes"),
        description="Print OpenTelemetry traces to stdout when no OTLP endpoint is configured",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def load_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
