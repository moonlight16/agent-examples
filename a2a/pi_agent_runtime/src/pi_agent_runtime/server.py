"""A2A server exposing an instruction-free Pi coding agent."""

from __future__ import annotations

import os

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TextPart
from a2a.utils import new_task

from pi_agent_runtime.runtime import PiConfig, PiRunner


class PiExecutor(AgentExecutor):
    def __init__(self, runner: PiRunner):
        self.runner = runner

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task(context.message)
        if context.current_task is None:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            response = await self.runner.run(context.get_user_input())
            await updater.add_artifact([TextPart(text=response)])
            await updater.complete()
        except Exception as exc:
            await updater.add_artifact([TextPart(text=f"Pi runtime error: {exc}")])
            await updater.failed()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancellation is not implemented")


def build_app() -> object:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    endpoint = os.getenv("AGENT_ENDPOINT", f"http://{host}:{port}").rstrip("/") + "/"
    card = AgentCard(
        name="Pi Agent Runtime",
        description="A generic Pi coding agent operating in a persistent workspace.",
        url=endpoint,
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(
                id="general-coding-agent",
                name="General coding agent",
                description="Reads, writes, and reasons about files in its workspace.",
                tags=["pi", "coding", "workspace"],
                examples=[],
            )
        ],
    )
    handler = DefaultRequestHandler(
        agent_executor=PiExecutor(PiRunner(PiConfig.from_env())),
        task_store=InMemoryTaskStore(),
    )
    return A2AStarletteApplication(agent_card=card, http_handler=handler).build()


def main() -> None:
    uvicorn.run(build_app(), host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
