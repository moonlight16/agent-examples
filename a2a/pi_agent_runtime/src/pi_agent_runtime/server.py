"""A2A server exposing an instruction-free Pi coding agent."""

from __future__ import annotations

import os

import uvicorn
from a2a.helpers import new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from starlette.applications import Starlette

from pi_agent_runtime.runtime import PiConfig, PiRunner


class PiExecutor(AgentExecutor):
    def __init__(self, runner: PiRunner):
        self.runner = runner

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.message is None:
            raise ValueError("request does not contain a message")
        task = context.current_task or new_task_from_user_message(context.message)
        if context.current_task is None:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            response = await self.runner.run(context.get_user_input())
            await updater.add_artifact([new_text_part(response)])
            await updater.complete()
        except Exception as exc:
            await updater.add_artifact([new_text_part(f"Pi runtime error: {exc}")])
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
        version="0.1.0",
        supported_interfaces=[
            AgentInterface(
                url=endpoint,
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            )
        ],
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
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
        agent_card=card,
    )
    return Starlette(
        routes=[
            *create_agent_card_routes(card),
            *create_jsonrpc_routes(handler, rpc_url="/"),
        ]
    )


def main() -> None:
    uvicorn.run(build_app(), host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
