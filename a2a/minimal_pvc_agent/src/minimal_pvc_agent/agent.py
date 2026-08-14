"""A2A server entrypoint for the minimal PVC agent.

Wires the LangGraph to the A2A protocol using the harness's own
persistence primitives (LangGraph AsyncSqliteSaver, a2a-sdk
DatabaseTaskStore over SQLite). Both save to files on /shared; when
Kagenti mounts a PVC at /shared, agent state survives pod restarts.

The agent code itself never handles persistence — that's the harness's
job. The only knobs are two env vars (CHECKPOINT_PATH, TASK_STORE_PATH)
that point the two components at files.
"""

import logging
import os
from pathlib import Path
from textwrap import dedent

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TaskState, TextPart
from a2a.utils import new_agent_text_message, new_task
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from minimal_pvc_agent.configuration import Configuration
from minimal_pvc_agent.graph import build_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_CONFIG = Configuration()
_GRAPH = None
_CHECKPOINTER_CM = None
_CHECKPOINTER = None


async def _ensure_graph():
    """Lazy-initialize the LangGraph with the right checkpointer."""
    global _GRAPH, _CHECKPOINTER, _CHECKPOINTER_CM
    if _GRAPH is not None:
        return _GRAPH

    path = _CONFIG.checkpoint_path
    if path:
        # Ensure parent dir exists (PVC mount is typically /shared with rwx)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        _CHECKPOINTER_CM = AsyncSqliteSaver.from_conn_string(path)
        _CHECKPOINTER = await _CHECKPOINTER_CM.__aenter__()
        logger.info("LangGraph checkpointer: AsyncSqliteSaver at %s", path)
    else:
        _CHECKPOINTER = MemorySaver()
        logger.info("LangGraph checkpointer: MemorySaver (CHECKPOINT_PATH unset)")

    _GRAPH = build_graph(_CONFIG, _CHECKPOINTER)
    return _GRAPH


def get_agent_card(host: str, port: int) -> AgentCard:
    capabilities = AgentCapabilities(streaming=True)
    skill = AgentSkill(
        id="minimal_pvc_chat",
        name="Minimal PVC Chat",
        description="Minimal A2A chat agent demonstrating harness-native persistence on a Kagenti PVC.",
        tags=["minimal", "pvc", "persistence", "demo"],
        examples=["hello", "what did I just say?"],
    )
    return AgentCard(
        name="Minimal PVC Agent",
        description=dedent(
            """\
            Minimal one-node LangGraph agent that demonstrates Kagenti
            persistent context across pod restarts using the harness's
            own persistence primitives (SQLite files on /shared).

            ## What it does
            - LangGraph AsyncSqliteSaver at CHECKPOINT_PATH (default
              /shared/checkpoints.db) persists conversation state.
            - A2A DatabaseTaskStore at TASK_STORE_PATH (default
              /shared/tasks.db) persists task metadata.
            - Both files live on the PVC Kagenti mounts at /shared, so
              deleting the pod and letting the StatefulSet recreate it
              preserves the conversation history and task records.
            - GET /checkpoint?context_id=... returns the LangGraph state
              for a given context, useful for inspection.
            """
        ),
        url=os.getenv("AGENT_ENDPOINT", f"http://{host}:{port}").rstrip("/") + "/",
        version="0.3.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        skills=[skill],
    )


class MinimalPVCExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)

        user_input = context.get_user_input()
        logger.info("turn: context=%s task=%s input=%s", task.context_id, task.id, user_input)

        graph = await _ensure_graph()
        thread_cfg = {"configurable": {"thread_id": task.context_id}}

        # Emit an initial "thinking" status update so streaming clients see
        # progress before the graph starts producing events.
        await task_updater.update_status(
            TaskState.working,
            new_agent_text_message(
                "thinking...",
                task_updater.context_id,
                task_updater.task_id,
            ),
        )

        reply = ""
        try:
            async for event in graph.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=thread_cfg,
                stream_mode="updates",
            ):
                # Each event is a dict of {node_name: state_update}. Summarize
                # each node's update as a working status message, capped at
                # 256 chars to avoid unwieldy status messages.
                summary = "\n".join(
                    f"{key}: {str(value)[:256] + '...' if len(str(value)) > 256 else str(value)}"
                    for key, value in event.items()
                )
                await task_updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        summary,
                        task_updater.context_id,
                        task_updater.task_id,
                    ),
                )
                # Track the final reply from the last node update.
                for value in event.values():
                    if isinstance(value, dict) and value.get("messages"):
                        last_msg = value["messages"][-1]
                        reply = getattr(last_msg, "content", str(last_msg))
        except Exception as exc:
            logger.exception("graph invocation failed: %s", exc)
            await task_updater.update_status(
                TaskState.failed,
                new_agent_text_message(
                    f"agent error: {exc}",
                    task_updater.context_id,
                    task_updater.task_id,
                ),
            )
            return

        await task_updater.add_artifact([TextPart(text=reply)])
        await task_updater.update_status(
            TaskState.input_required,
            new_agent_text_message(reply, task_updater.context_id, task_updater.task_id),
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")


async def health(request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "checkpoint_path": _CONFIG.checkpoint_path,
        "task_store_path": _CONFIG.task_store_path,
    })


async def agent_card_compat(request: Request) -> JSONResponse:
    card = get_agent_card(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
    return JSONResponse(card.model_dump(mode="json", exclude_none=True))


async def checkpoint(request: Request) -> JSONResponse:
    context_id = request.query_params.get("context_id")
    if not context_id:
        return JSONResponse({"error": "context_id is required"}, status_code=400)
    graph = await _ensure_graph()
    state = await graph.aget_state({"configurable": {"thread_id": context_id}})
    messages = state.values.get("messages", []) if state else []
    out = [
        {"type": getattr(m, "type", type(m).__name__), "content": getattr(m, "content", str(m))}
        for m in messages
    ]
    return JSONResponse({
        "context_id": context_id,
        "checkpointer": "sqlite" if _CONFIG.checkpoint_path else "memory",
        "checkpoint_path": _CONFIG.checkpoint_path,
        "messages": out,
    })


def _build_task_store():
    """DatabaseTaskStore over SQLite when TASK_STORE_PATH is set, else InMemoryTaskStore."""
    path = _CONFIG.task_store_path
    if not path:
        logger.info("A2A task store: InMemoryTaskStore (TASK_STORE_PATH unset)")
        return InMemoryTaskStore()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    from a2a.server.tasks import DatabaseTaskStore
    from sqlalchemy.ext.asyncio import create_async_engine
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    logger.info("A2A task store: DatabaseTaskStore (sqlite) at %s", path)
    return DatabaseTaskStore(engine=engine)


def run():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    agent_card = get_agent_card(host=host, port=port)

    request_handler = DefaultRequestHandler(
        agent_executor=MinimalPVCExecutor(),
        task_store=_build_task_store(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    app = server.build()

    app.routes.insert(0, Route("/health", health, methods=["GET"]))
    app.routes.insert(0, Route("/checkpoint", checkpoint, methods=["GET"]))
    app.routes.insert(0, Route("/.well-known/agent-card.json", agent_card_compat, methods=["GET"]))

    uvicorn.run(app, host=host, port=port)
