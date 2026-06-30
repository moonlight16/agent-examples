"""One-node LangGraph for the minimal PVC agent.

The graph has a single 'respond' node that either calls the configured
LLM (when a valid API key is available) or runs in echo mode (returns
the user input plus a turn counter). When CHECKPOINT_DB_URL is set,
the graph uses AsyncPostgresSaver so state survives pod restarts;
otherwise it uses MemorySaver.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from minimal_pvc_agent.configuration import Configuration

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a minimal demonstration agent for a Kagenti persistence "
    "experiment. Be concise (one or two sentences) and acknowledge that "
    "you remember earlier turns when the user references them."
)


async def get_checkpointer(config: Configuration):
    """Return a LangGraph checkpointer. Postgres if CHECKPOINT_DB_URL set, else MemorySaver."""
    if config.checkpoint_db_url:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        cp = AsyncPostgresSaver.from_conn_string(config.checkpoint_db_url)
        # AsyncPostgresSaver.from_conn_string returns an async context manager;
        # the caller must use it as such. We return the cm so caller can manage.
        return cp
    return MemorySaver()


def _echo_reply(state: MessagesState) -> dict[str, Any]:
    """Generate an echo reply when no LLM is configured."""
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    n = len(user_messages)
    last = user_messages[-1].content if user_messages else ""
    return {"messages": [AIMessage(content=f"echo: {last} (turn {n})")]}


def build_graph(config: Configuration, checkpointer):
    """Compile the one-node graph with the given checkpointer."""

    if config.has_valid_api_key:
        llm = ChatOpenAI(
            model=config.llm_model,
            openai_api_key=config.llm_api_key,
            openai_api_base=config.llm_api_base,
            temperature=0,
        )

        async def respond(state: MessagesState) -> dict[str, Any]:
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
            result = await llm.ainvoke(messages)
            return {"messages": [result]}
    else:
        async def respond(state: MessagesState) -> dict[str, Any]:
            return _echo_reply(state)

    builder = StateGraph(MessagesState)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    return builder.compile(checkpointer=checkpointer)
