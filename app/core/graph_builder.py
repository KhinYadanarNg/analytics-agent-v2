"""
LangGraph builder for analytics processing workflow.
"""
import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from app.config import OPENAI_MODEL, USE_LLM, SYSTEM, MAX_AGENT_LOOPS
from app.utils.formatting import format_error_message, format_basic_message
from app.utils.context import extract_context_insights, get_conversation_context, create_interpretation_prompt
from app.core.tools_agent import get_success_rate_by_file_name_tool

logger = logging.getLogger(__name__)


def build_app():
    """
    Build the LangGraph application for analytics processing.
    """
    if not USE_LLM:
        raise SystemExit("OPENAI_API_KEY missing. Add it to .env to run the chat agent.")

    # Import AnalyticsService here to avoid circular imports
    from app.core.analytics_service import AnalyticsService

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0).bind_tools(AnalyticsService.TOOLS)

    graph = StateGraph(MessagesState)

    def assistant(state: MessagesState):
        messages = state["messages"]

        # Enhanced loop protection with context awareness
        loop_count = state.get("_loop_count", 0)
        if loop_count >= MAX_AGENT_LOOPS:
            error_msg = format_error_message(
                "Maximum tool call cycles reached",
                "The query is too complex or requires too many data operations. Try breaking it into simpler parts.",
                f"Query attempted {loop_count} tool calls"
            )
            return {"messages": [AIMessage(content=error_msg)]}

        state["_loop_count"] = loop_count + 1

        try:
            # Process tool results with enhanced interpretation
            has_tool_results = False
            tool_results = []
            context_insights = []

            for msg in messages:
                if isinstance(msg, ToolMessage):
                    has_tool_results = True
                    try:
                        tool_data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                        tool_results.append(tool_data)

                        # Extract context insights from tool results
                        if isinstance(tool_data, dict) and "data" in tool_data:
                            insights = extract_context_insights(tool_data["data"])
                            context_insights.extend(insights)
                    except Exception as e:
                        logger.warning(f"Failed to parse tool result: {e}")
                        tool_results.append({"content": msg.content, "error": str(e)})

            # If we have tool results, use enhanced interpretation
            if has_tool_results and tool_results:
                try:
                    # Get conversation history for context
                    conversation_context = get_conversation_context(messages)

                    # Create enhanced interpretation prompt
                    interpretation_messages = [
                        SystemMessage(content=SYSTEM),
                        *conversation_context,
                        HumanMessage(content=create_interpretation_prompt(tool_results, context_insights))
                    ]

                    # Use LLM for intelligent interpretation
                    interpretation_response = llm.invoke(interpretation_messages)
                    return {"messages": [interpretation_response]}

                except Exception as e:
                    logger.error(f"Interpretation failed: {e}")
                    # Fallback to basic formatting with correct parameters
                    # Extract parameters from tool_results for fallback
                    file_name = None
                    row_count = 0
                    chart_type = "bar"
                    report_type = "both"

                    # Try to extract parameters from tool results
                    for result in tool_results:
                        if isinstance(result, dict):
                            if "file_name" in result:
                                file_name = result["file_name"]
                            if "row_count" in result:
                                row_count = result["row_count"]
                            if "chart_type" in result:
                                chart_type = result["chart_type"]

                    fallback_content = format_basic_message(
                        chart_data=[],  # No chart data in fallback
                        file_name=file_name,
                        row_count=row_count,
                        chart_type=chart_type,
                        report_type=report_type,
                        date_filter_used=None
                    )
                    return {"messages": [AIMessage(content=fallback_content)]}

            # Handle initial queries and follow-ups without tool results
            elif not has_tool_results:
                # Add system prompt to ensure consistent behavior
                enhanced_messages = [SystemMessage(content=SYSTEM)] + list(messages)
                response = llm.invoke(enhanced_messages)
                return {"messages": [response]}

            # Fallback for edge cases
            else:
                fallback_msg = format_basic_message(
                    chart_data=[],
                    file_name=None,
                    row_count=0,
                    chart_type="bar",
                    report_type="both",
                    date_filter_used=None
                )
                return {"messages": [AIMessage(content=fallback_msg)]}

        except Exception as e:
            logger.error(f"Assistant function error: {e}")
            error_content = format_error_message(
                "Processing Error",
                "An unexpected error occurred while processing your request. Please try again.",
                f"Technical details: {str(e)}"
            )
            return {"messages": [AIMessage(content=error_content)]}

    tool_node = ToolNode(AnalyticsService.TOOLS)

    graph.add_node("assistant", assistant)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")

    return graph.compile()