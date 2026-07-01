"""One-node LangGraph for the minimal PVC agent.

The graph has a single 'respond' node that either calls the configured
LLM (when a valid API key is available) or runs in echo mode (returns
the user input plus a turn counter).

The checkpointer is chosen from Configuration.checkpoint_path:
  - a filesystem path (default /shared/checkpoints.db) -> AsyncSqliteSaver
    with state persisted to that file. Point that path at a PVC mount
    for durability across pod restarts.
  - empty string -> MemorySaver (for tests / echo-only smoke checks).
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
