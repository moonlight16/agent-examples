"""A2A server entrypoint for the minimal PVC agent.

Wires the LangGraph to the A2A protocol, persists per-turn JSONL to
/shared, and exposes GET /history and GET /checkpoint endpoints so
persistence can be verified without kubectl exec.
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
from minimal_pvc_agent.persistence import append_turn, read_history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_CONFIG = Configuration()
_GRAPH = None
_CHECKPOINTER_CM = None  # async context manager when using AsyncPostgresSaver
_CHECKPOINTER = None


async def _ensure_graph():
    """Lazy-initialize the LangGraph with the right checkpointer."""
    global _GRAPH, _CHECKPOINTER, _CHECKPOINTER_CM
    if _GRAPH is not None:
        return _GRAPH

    if _CONFIG.checkpoint_db_url:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        _CHECKPOINTER_CM = AsyncPostgresSaver.from_conn_string(_CONFIG.checkpoint_db_url)
        _CHECKPOINTER = await _CHECKPOINTER_CM.__aenter__()
        try:
            await _CHECKPOINTER.setup()
        except Exception as exc:
            logger.warning("checkpointer setup failed: %s", exc)
        logger.info("Using AsyncPostgresSaver checkpointer")
    else:
        _CHECKPOINTER = MemorySaver()
        logger.info("Using MemorySaver checkpointer (no CHECKPOINT_DB_URL set)")

    _GRAPH = build_graph(_CONFIG, _CHECKPOINTER)
    return _GRAPH


def get_agent_card(host: str, port: int) -> AgentCard:
    capabilities = AgentCapabilities(streaming=False)
    skill = AgentSkill(
        id="minimal_pvc_chat",
        name="Minimal PVC Chat",
        description="Minimal A2A chat agent for proving Kagenti persistence (PVC + Postgres).",
        tags=["minimal", "pvc", "persistence", "demo"],
        examples=["hello", "what did I just say?"],
    )
    return AgentCard(
        name="Minimal PVC Agent",
        description=dedent(
            """\
            Minimal one-node LangGraph agent that demonstrates Kagenti
            persistent context across pod restarts.

            ## What it does
            - Writes one JSONL line per turn to /shared/<context_id>.jsonl
            - Uses LangGraph's checkpointer for conversation state
              (MemorySaver by default, AsyncPostgresSaver if CHECKPOINT_DB_URL is set)
            - GET /history?context_id=... returns the JSONL turn log
            - GET /checkpoint?context_id=... returns the LangGraph state
            """
        ),
        url=os.getenv("AGENT_ENDPOINT", f"http://{host}:{port}").rstrip("/") + "/",
        version="0.1.0",
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

        try:
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=thread_cfg,
            )
            reply_msg = result["messages"][-1]
            reply = getattr(reply_msg, "content", str(reply_msg))
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

        append_turn(
            Path(_CONFIG.context_dir),
            task.context_id,
            task.id,
            user_input,
            reply,
        )

        await task_updater.add_artifact([TextPart(text=reply)])
        await task_updater.update_status(
            TaskState.input_required,
            new_agent_text_message(reply, task_updater.context_id, task_updater.task_id),
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "context_dir": _CONFIG.context_dir})


async def agent_card_compat(request: Request) -> JSONResponse:
    card = get_agent_card(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
    return JSONResponse(card.model_dump(mode="json", exclude_none=True))


async def history(request: Request) -> JSONResponse:
    context_id = request.query_params.get("context_id")
    if not context_id:
        return JSONResponse({"error": "context_id is required"}, status_code=400)
    records = read_history(Path(_CONFIG.context_dir), context_id)
    return JSONResponse({"context_id": context_id, "turns": records})


async def checkpoint(request: Request) -> JSONResponse:
    context_id = request.query_params.get("context_id")
    if not context_id:
        return JSONResponse({"error": "context_id is required"}, status_code=400)
    graph = await _ensure_graph()
    state = await graph.aget_state({"configurable": {"thread_id": context_id}})
    # state.values contains a MessagesState dict; messages are LangChain objects
    messages = state.values.get("messages", []) if state else []
    out = [
        {"type": getattr(m, "type", type(m).__name__), "content": getattr(m, "content", str(m))}
        for m in messages
    ]
    return JSONResponse({
        "context_id": context_id,
        "checkpointer": "postgres" if _CONFIG.checkpoint_db_url else "memory",
        "messages": out,
    })


def _build_task_store():
    """Return a DatabaseTaskStore when TASK_STORE_DB_URL is set, else InMemoryTaskStore."""
    dsn = _CONFIG.task_store_db_url
    if not dsn:
        logger.info("Using InMemoryTaskStore (no TASK_STORE_DB_URL set)")
        return InMemoryTaskStore()
    from a2a.server.tasks import DatabaseTaskStore
    from sqlalchemy.ext.asyncio import create_async_engine
    engine = create_async_engine(dsn)
    logger.info("Using DatabaseTaskStore with DSN host=%s", dsn.split("@")[-1].split("/")[0])
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

    # Add custom routes
    app.routes.insert(0, Route("/health", health, methods=["GET"]))
    app.routes.insert(0, Route("/history", history, methods=["GET"]))
    app.routes.insert(0, Route("/checkpoint", checkpoint, methods=["GET"]))
    app.routes.insert(0, Route("/.well-known/agent-card.json", agent_card_compat, methods=["GET"]))

    uvicorn.run(app, host=host, port=port)
