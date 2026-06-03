"""A2A server implementation for Kagenti Chat."""

import logging
import traceback
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Part,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message, new_task

from kagenti_chat.a2a_server.auth import BearerAuthMiddleware
from kagenti_chat.agent import ChatAgent
from kagenti_chat.config import Settings

logger = logging.getLogger(__name__)


def get_agent_card(settings: Settings) -> AgentCard:
    """Create the AgentCard advertised at /.well-known/agent.json."""
    capabilities = AgentCapabilities(streaming=True)

    skill = AgentSkill(
        id="kagenti_chat",
        name="Kagenti Chat",
        description=(
            "A general-purpose conversational assistant. Answers questions, explains "
            "concepts, helps with writing and code, and discusses arbitrary topics."
        ),
        tags=["chat", "general", "assistant", "conversation"],
        examples=[
            "Explain how Kubernetes pod scheduling works.",
            "Write a Python function that deduplicates a list while preserving order.",
            "Summarize the key tradeoffs between gRPC and REST.",
        ],
    )

    agent_url = settings.A2A_PUBLIC_URL
    if not agent_url:
        if settings.A2A_HOST == "0.0.0.0":
            agent_url = f"http://localhost:{settings.A2A_PORT}/"
        else:
            agent_url = f"http://{settings.A2A_HOST}:{settings.A2A_PORT}/"

    return AgentCard(
        name="Kagenti Chat",
        description="General-purpose chat agent powered by AG2 and an OpenAI-compatible LLM (e.g. llm-d).",
        url=agent_url,
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        skills=[skill],
    )


class KagentiChatExecutor(AgentExecutor):
    """A2A executor for Kagenti Chat."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore[arg-type]
            await event_queue.enqueue_event(task)

        task_updater = TaskUpdater(event_queue, task.id, task.context_id)

        async def event_callback(message: str, final: bool = False) -> None:
            if final:
                parts = [Part(root=TextPart(text=message))]
                await task_updater.add_artifact(parts)
                await task_updater.complete()
            else:
                await task_updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        message,
                        task_updater.context_id,
                        task_updater.task_id,
                    ),
                )

        async def error_callback(message: str) -> None:
            parts = [Part(root=TextPart(text=message))]
            await task_updater.add_artifact(parts)
            await task_updater.failed()

        user_input = context.get_user_input()
        logger.info("Processing chat request (chars=%d)", len(user_input or ""))

        try:
            agent = ChatAgent(settings=self.settings, event_callback=event_callback)
            result = await agent.run_task(user_input)
            answer = result.get("answer", "(no response)")
            if result.get("error"):
                await error_callback(answer)
            else:
                await event_callback(answer, final=True)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            logger.error("Error executing task: %s", exc, exc_info=True)
            await error_callback(f"I encountered an error while processing your request: {exc}")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Task cancellation is not supported")


def create_app(settings: Settings) -> Any:
    """Build the A2A Starlette application."""
    agent_card = get_agent_card(settings)
    request_handler = DefaultRequestHandler(
        agent_executor=KagentiChatExecutor(settings),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
    app = server.build()

    if settings.API_KEY:
        app.add_middleware(BearerAuthMiddleware, api_key=settings.API_KEY)
        logger.info("Bearer token authentication enabled")
    else:
        logger.warning("API_KEY not set — server is running without authentication")

    logger.info("A2A server application created")
    return app
