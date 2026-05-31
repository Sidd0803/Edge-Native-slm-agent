from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.tools import check_stock, lookup_part

MODEL = "qwen2.5:3b"

SYSTEM_PROMPT = """\
You are a parts triage agent for an industrial distributor. \
Help warehouse staff find parts and check inventory.

You have two tools:
- lookup_part: use when the request describes a part in words \
(e.g. "do we have any 3/8 inch hydraulic fittings?")
- check_stock: use when the request gives a specific SKU \
(e.g. "check SKU HF-4821-B")

Response rules:
- Part in stock → SKU, description, quantity, unit price.
- Part out of stock → SKU, description, supplier name, lead time in days.
- Part not found → say so clearly, suggest searching by category.
- Ambiguous request → ask one clarifying question before calling any tool.

Be concise and structured.\
"""

TOOLS = [lookup_part, check_stock]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _reason(state: AgentState) -> AgentState:
    llm = ChatOllama(model=MODEL).bind_tools(TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "call_tool"
    return END


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("reason", _reason)
    graph.add_node("call_tool", ToolNode(TOOLS))
    graph.set_entry_point("reason")
    graph.add_conditional_edges(
        "reason",
        _should_continue,
        {"call_tool": "call_tool", END: END},
    )
    graph.add_edge("call_tool", "reason")
    return graph.compile()
