"""Kagenti Chat agent - a general-purpose AG2 chat agent (no tools)."""

import logging
import os
from typing import Any, Callable

from autogen import ConversableAgent
from autogen.opentelemetry import instrument_llm_wrapper, instrument_agent
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from kagenti_chat.agent.prompts import CHAT_AGENT_PROMPT
from kagenti_chat.config import Settings

logger = logging.getLogger(__name__)

_SERVICE_NAME = "kagenti_chat_agent"
_AGENT_IDS: dict[str, str] = {"chat_agent": "kagenti-chat-001"}


class AgentIdSpanProcessor(SpanProcessor):
    """Inject gen_ai.agent.id on spans that carry gen_ai.agent.name (kagenti-friendly)."""

    def __init__(self, agent_ids: dict[str, str]) -> None:
        self._agent_ids = agent_ids

    def on_start(self, span: ReadableSpan, parent_context=None) -> None:
        agent_name = span.attributes.get("gen_ai.agent.name") if span.attributes else None
        if agent_name and agent_name in self._agent_ids:
            span.set_attribute("gen_ai.agent.id", self._agent_ids[agent_name])

    def on_end(self, span: ReadableSpan) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


_tracer_provider: TracerProvider | None = None
_tracing_initialized = False


def _init_tracing() -> TracerProvider:
    global _tracer_provider, _tracing_initialized
    if _tracing_initialized:
        return _tracer_provider  # type: ignore[return-value]

    resource = Resource.create(attributes={"service.name": _SERVICE_NAME})
    _tracer_provider = TracerProvider(resource=resource)
    _tracer_provider.add_span_processor(AgentIdSpanProcessor(_AGENT_IDS))

    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        _tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        logger.info("OpenTelemetry tracing enabled (OTLP: %s)", os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"])
    elif os.environ.get("OTEL_CONSOLE_TRACING", "").lower() in ("true", "1", "yes"):
        _tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        logger.info("OpenTelemetry tracing enabled (console exporter)")

    trace.set_tracer_provider(_tracer_provider)
    instrument_llm_wrapper(tracer_provider=_tracer_provider)
    _tracing_initialized = True
    return _tracer_provider


class ChatAgent:
    """A simple, stateless chat agent using AG2's ConversableAgent.

    One LLM call per request. Conversation memory across requests is handled by the
    A2A layer's task/context store, not here.
    """

    def __init__(
        self,
        settings: Settings,
        event_callback: Callable[[str, bool], Any] | None = None,
    ):
        self.settings = settings
        self.event_callback = event_callback
        self._tracer_provider = _init_tracing()
        self._init_ag2_agent()

    def _init_ag2_agent(self) -> None:
        llm_config: dict[str, Any] = {
            "api_type": "openai",
            "model": self.settings.LLM_MODEL,
            "temperature": self.settings.LLM_TEMPERATURE,
        }
        if self.settings.LLM_API_KEY:
            llm_config["api_key"] = self.settings.LLM_API_KEY
        if self.settings.LLM_BASE_URL:
            llm_config["base_url"] = self.settings.LLM_BASE_URL
        if self.settings.EXTRA_HEADERS:
            llm_config["default_headers"] = self.settings.EXTRA_HEADERS

        self.agent = ConversableAgent(
            name="chat_agent",
            system_message=CHAT_AGENT_PROMPT,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        instrument_agent(self.agent, tracer_provider=self._tracer_provider)
        logger.info("Chat agent initialized (model=%s)", self.settings.LLM_MODEL)

    async def _emit_event(self, message: str, final: bool = False) -> None:
        if self.event_callback:
            try:
                await self.event_callback(message, final)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error in event callback: %s", exc)

    async def run_task(self, instruction: str) -> dict[str, Any]:
        """Run a single chat turn and return the assistant's reply."""
        logger.info("Starting chat turn (chars=%d)", len(instruction))
        await self._emit_event("Thinking...")

        try:
            reply = await self.agent.a_generate_reply(
                messages=[{"role": "user", "content": instruction}],
            )
            if isinstance(reply, dict):
                answer = reply.get("content") or ""
            else:
                answer = reply or ""
            if not answer:
                answer = "(no response generated)"
            return {"answer": answer, "error": False}

        except Exception as exc:  # noqa: BLE001
            logger.error("Error during chat turn: %s", exc, exc_info=True)
            return {"answer": f"Error: {exc}", "error": True}
