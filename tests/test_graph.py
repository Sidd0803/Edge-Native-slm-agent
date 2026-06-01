import pytest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from agent.graph import AgentState, _should_continue, _reason, build_graph


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------

def test_graph_compiles():
    graph = build_graph()
    assert graph is not None


# ---------------------------------------------------------------------------
# _should_continue routing
# ---------------------------------------------------------------------------

def test_should_continue_routes_to_tool_when_tool_calls_present():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "check_stock", "args": {"sku": "HF-001"}, "id": "t1"}],
    )
    assert _should_continue({"messages": [msg]}) == "call_tool"


def test_should_continue_routes_to_end_when_no_tool_calls():
    msg = AIMessage(content="The part is in stock.")
    assert _should_continue({"messages": [msg]}) == END


def test_should_continue_routes_to_end_on_empty_tool_calls_list():
    msg = AIMessage(content="Done.", tool_calls=[])
    assert _should_continue({"messages": [msg]}) == END


# ---------------------------------------------------------------------------
# _reason node
# ---------------------------------------------------------------------------

def test_reason_node_returns_single_message():
    mock_response = AIMessage(content="Here is the answer.")
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = MagicMock(invoke=MagicMock(return_value=mock_response))

    with patch("agent.graph.ChatOllama", return_value=mock_llm):
        result = _reason({"messages": [HumanMessage(content="test")]})

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "Here is the answer."


def test_reason_node_prepends_system_prompt():
    mock_response = AIMessage(content="ok")
    mock_bound = MagicMock(invoke=MagicMock(return_value=mock_response))
    mock_llm = MagicMock(bind_tools=MagicMock(return_value=mock_bound))

    with patch("agent.graph.ChatOllama", return_value=mock_llm):
        _reason({"messages": [HumanMessage(content="test")]})

    invoked_messages = mock_bound.invoke.call_args[0][0]
    from langchain_core.messages import SystemMessage
    assert isinstance(invoked_messages[0], SystemMessage)


# ---------------------------------------------------------------------------
# Recursion limit
# ---------------------------------------------------------------------------

def test_graph_raises_on_recursion_limit():
    """
    A model that always emits a tool call would loop forever without a
    recursion_limit. Verify the graph surfaces an error rather than hanging.
    GraphRecursionError extends RecursionError in LangGraph.
    """
    always_tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "check_stock", "args": {"sku": "HF-001"}, "id": "t1"}],
    )
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = MagicMock(
        invoke=MagicMock(return_value=always_tool_call)
    )

    with patch("agent.graph.ChatOllama", return_value=mock_llm), \
         patch("agent.tools._get_memory") as mock_get_mem:
        mock_get_mem.return_value.get_by_sku.return_value = None
        graph = build_graph()
        with pytest.raises(RecursionError):
            graph.invoke(
                {"messages": [HumanMessage(content="check SKU HF-001")]},
                {"recursion_limit": 3},
            )
