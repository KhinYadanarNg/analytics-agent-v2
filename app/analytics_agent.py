"""
Enhanced analytics query processing service with report type detection.
"""
import json
import logging
import asyncio
import re
from typing import Tuple, Optional, Dict, Any, List
from app.config import OPENAI_MODEL, USE_LLM, DEBUG
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from app.tools_agent import (
    get_success_rate_by_file_name_tool
)
from app.chart_generator import chart_generator

class AnalyticsService:
    """Handles LLM tool selection, execution, chart generation, and interpretation."""

    TOOLS = [
        get_success_rate_by_file_name_tool
    ]

    @staticmethod
    def detect_report_type(prompt: str) -> str:
        """
        Detect what type of report the user is asking for.
        
        Returns:
            "success" - only success data
            "failure" - only failure data  
            "both" - both success and failure data (default)
        """
        prompt_lower = prompt.lower()
        
        # Check for success-only indicators
        success_only_patterns = [
            r'\bonly\s+success',
            r'\bsuccess\s+only', 
            r'\bjust\s+success',
            r'\bsuccess\s+rate\s+only',
            r'\bsuccessful\s+only',
            r'\bshow\s+me\s+success',
            r'\bwhat.*success.*rate',
            r'\bhow\s+many\s+succeeded',
            r'\bpercentage\s+of\s+success',
            r'\bsuccess\s+percentage'
        ]
        
        # Check for failure-only indicators
        failure_only_patterns = [
            r'\bonly\s+fail',
            r'\bfail\s+only',
            r'\bjust\s+fail',
            r'\bfailure\s+only',
            r'\bfail\s+rate\s+only',
            r'\bshow\s+me\s+fail',
            r'\bwhat.*fail.*rate',
            r'\bhow\s+many\s+failed',
            r'\bpercentage\s+of\s+fail',
            r'\bfailure\s+percentage',
            r'\berror\s+rate'
        ]
        
        # Check for both indicators (override single type)
        both_patterns = [
            r'\bboth\s+success\s+and\s+fail',
            r'\bsuccess\s+and\s+fail',
            r'\boverall\s+analysis',
            r'\bcomplete\s+report',
            r'\bfull\s+analysis',
            r'\ball\s+results'
        ]
        
        # Check patterns in order of specificity
        if any(re.search(pattern, prompt_lower) for pattern in both_patterns):
            return "both"
        elif any(re.search(pattern, prompt_lower) for pattern in success_only_patterns):
            return "success"
        elif any(re.search(pattern, prompt_lower) for pattern in failure_only_patterns):
            return "failure"
        else:
            # Default to both if ambiguous
            return "both"

    @staticmethod
    def filter_chart_data_by_report_type(chart_data: List[Dict], report_type: str) -> List[Dict]:
        """
        Filter chart data based on report type.
        
        Args:
            chart_data: Original chart data with both success and failure
            report_type: "success", "failure", or "both"
            
        Returns:
            Filtered chart data
        """
        if report_type == "both":
            return chart_data
        
        filtered_data = []
        for item in chart_data:
            status = item.get('status', '').lower()
            if report_type == "success" and status == 'success':
                filtered_data.append(item)
            elif report_type == "failure" and status in ['fail', 'failure']:
                filtered_data.append(item)
        
        return filtered_data

    @staticmethod
    async def process_query(prompt: str, session_id: str = None, conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build the agent graph, invoke it, generate chart, 
        then have LLM interpret the results for a natural response.

        Args:
            prompt: The user's query
            session_id: Optional session identifier for tracking
            conversation_history: Optional conversation history for context

        Returns a dict with keys: success (bool), message (str), chart_image (str base64)
        """
        try:
            # Detect chart type and report type from prompt
            chart_type = chart_generator.detect_chart_type_from_prompt(prompt)
            report_type = AnalyticsService.detect_report_type(prompt)
            
            # Log detected chart type for debugging
            logger = logging.getLogger("analytics_agent")
            logger.info(f"Detected chart type: '{chart_type}' for prompt: '{prompt[:100]}...'")
            logger.info(f"Detected report type: '{report_type}'")
            
            # build_app may raise if USE_LLM is False or config missing
            app_graph = build_app()
        except Exception as e:
            return {"success": False, "message": str(e), "chart_image": None}

        # prepare initial graph state with chart type and report type context
        messages = [SystemMessage(content=SYSTEM)]
        
        # Add conversation history if available (previous context)
        if conversation_history:
            for interaction in conversation_history[-3:]:  # Use last 3 interactions for context
                if interaction.get("user_message"):
                    messages.append(HumanMessage(content=interaction["user_message"]))
                if interaction.get("assistant_response"):
                    # Extract the main response text
                    response_text = interaction["assistant_response"]
                    if isinstance(response_text, dict):
                        response_text = response_text.get("message", str(response_text))
                    messages.append(AIMessage(content=response_text))
        
        # Add the current prompt
        messages.append(HumanMessage(content=prompt))
        
        state = {
            "messages": messages,
            "chart_type": chart_type,
            "report_type": report_type,
            "session_id": session_id  # Store session_id in state for tools if needed
        }

        loop = asyncio.get_running_loop()
        try:
            # First pass: Run the graph to get data
            compiled_result = await loop.run_in_executor(None, lambda: app_graph.invoke(state))
        except Exception as e:
            return {"success": False, "message": str(e), "chart_image": None}

        # Extract tool results and generate chart
        tool_results = []
        chart_data = []
        file_name = None
        row_count = 0
        
        for m in compiled_result.get("messages", []):
            if isinstance(m, ToolMessage):
                try:
                    tool_data = json.loads(m.content) if isinstance(m.content, str) else m.content
                    tool_results.append(tool_data)
                    
                    # Extract chart data from tool results
                    if tool_data.get("success") and tool_data.get("chart_data"):
                        chart_data = tool_data.get("chart_data", [])
                        file_name = tool_data.get("file_name")
                        row_count = tool_data.get("row_count", 0)
                        # Keep the user's requested chart_type - don't override with tool result
                        # The chart_type was already detected from user prompt and should be preserved
                except:
                    tool_results.append(m.content)

        # Filter chart data based on report type
        original_chart_data = chart_data.copy()
        filtered_chart_data = AnalyticsService.filter_chart_data_by_report_type(chart_data, report_type)

        # Generate chart image if we have data
        chart_image = None
        chart_generated = False
        if filtered_chart_data:
            try:
                chart_image = chart_generator.generate_chart(
                    chart_data=filtered_chart_data,
                    chart_type=chart_type,
                    file_name=file_name,
                    report_type=report_type  # Pass report type to chart generator
                )
                chart_generated = True
            except Exception as e:
                logger = logging.getLogger("analytics_agent")
                logger.exception(f"Failed to generate chart: {e}")

        # Now, have the LLM interpret the results with chart context
        interpretation = await get_llm_interpretation(
            prompt=prompt,
            chart_data=filtered_chart_data,
            original_chart_data=original_chart_data,
            chart_type=chart_type,
            chart_generated=chart_generated,
            file_name=file_name,
            row_count=row_count,
            report_type=report_type,
            conversation_history=conversation_history
        )
        
        result = {
            "success": True,
            "message": interpretation,
            "chart_image": chart_image
        }
        
        # Add additional data if DEBUG mode
        if DEBUG:
            result["chart_data"] = filtered_chart_data
            result["original_chart_data"] = original_chart_data
            result["tool_results"] = tool_results

        return result

async def get_llm_interpretation(prompt: str, chart_data: List[Dict], 
                                original_chart_data: List[Dict],
                                chart_type: str, chart_generated: bool,
                                file_name: str, row_count: int, 
                                report_type: str, conversation_history: List[Dict[str, Any]] = None) -> str:
    """
    Have the LLM interpret the results and provide a natural language response.
    """
    if not USE_LLM:
        # Fallback to basic message if LLM not available
        return format_basic_message(chart_data, file_name, row_count, chart_type, report_type)
    
    try:
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)
        
        # Prepare context for interpretation based on report type
        context_parts = []
        
        # Add conversation context if available
        if conversation_history:
            recent_files = []
            for interaction in conversation_history[-2:]:  # Last 2 interactions
                if interaction.get("user_message"):
                    # Extract file references from previous conversations
                    import re
                    file_matches = re.findall(r'(\w+\.csv)', interaction["user_message"])
                    recent_files.extend(file_matches)
            
            if recent_files:
                context_parts.append(f"Previously discussed files: {', '.join(set(recent_files))}")
        
        if chart_data:
            context_parts.append(f"File analyzed: {file_name}")
            context_parts.append(f"Total records: {row_count}")
            context_parts.append(f"Report type requested: {report_type}")
            
            # Extract metrics based on report type
            if report_type == "success":
                success_data = next((item for item in chart_data if item.get('status', '').lower() == 'success'), None)
                if success_data:
                    context_parts.append(f"Success: {success_data['percentage']:.1f}% ({success_data['count']} records)")
                else:
                    context_parts.append("No successful records found")
            elif report_type == "failure":
                fail_data = next((item for item in chart_data if item.get('status', '').lower() == 'fail'), None)
                if fail_data:
                    context_parts.append(f"Failure: {fail_data['percentage']:.1f}% ({fail_data['count']} records)")
                else:
                    context_parts.append("No failed records found")
            else:  # both
                success_data = next((item for item in original_chart_data if item.get('status', '').lower() == 'success'), None)
                fail_data = next((item for item in original_chart_data if item.get('status', '').lower() == 'fail'), None)
                
                if success_data:
                    context_parts.append(f"Success: {success_data['percentage']:.1f}% ({success_data['count']} records)")
                if fail_data:
                    context_parts.append(f"Failure: {fail_data['percentage']:.1f}% ({fail_data['count']} records)")
            
            if chart_generated:
                report_desc = {
                    "success": "success-only",
                    "failure": "failure-only", 
                    "both": "comprehensive"
                }
                context_parts.append(f"Generated {report_desc[report_type]} {chart_type} chart")
        else:
            context_parts.append(f"No {report_type} data found for file: {file_name}")
        
        # Create interpretation prompt with conversation context
        conversation_context = ""
        if conversation_history:
            conversation_context = "\nConversation Context:\n"
            for i, interaction in enumerate(conversation_history[-2:], 1):  # Last 2 interactions
                if interaction.get("user_message"):
                    conversation_context += f"Previous Request {i}: {interaction['user_message']}\n"
                if interaction.get("assistant_response"):
                    response = interaction["assistant_response"]
                    if isinstance(response, dict):
                        response = response.get("message", "Response provided")
                    conversation_context += f"Previous Response {i}: {response[:100]}...\n"
        
        interpretation_prompt = f"""
Based on the user's current request: "{prompt}"
{conversation_context}
Current Results obtained:
{chr(10).join(context_parts)}

The user specifically requested a {report_type} report, so focus your response accordingly.

Please provide a natural, conversational response that:
1. Directly answers what the user asked for
2. Focuses on the {report_type} metrics they requested
3. Highlights key insights from the filtered data
4. If there are concerning patterns (based on the report type), mention them
5. Mentions that a {report_type}-focused {chart_type} chart has been generated (if chart was generated)
6. Be concise but informative
7. Reference previous context if the current request relates to it (e.g., "that file" referring to previously mentioned files)

Do not use bullet points or numbered lists. Provide a flowing, natural response.
"""
        
        # Get LLM interpretation
        messages = [
            SystemMessage(content="You are a data analyst providing insights from quality metrics. Be direct and insightful. Focus on what the user specifically requested."),
            HumanMessage(content=interpretation_prompt)
        ]
        
        response = await asyncio.get_running_loop().run_in_executor(
            None, 
            lambda: llm.invoke(messages)
        )
        
        return response.content
        
    except Exception as e:
        logger = logging.getLogger("analytics_agent")
        logger.exception(f"Failed to get LLM interpretation: {e}")
        # Fallback to basic message
        return format_basic_message(chart_data, file_name, row_count, chart_type, report_type)

def format_basic_message(chart_data: List[Dict], file_name: str, row_count: int, 
                        chart_type: str, report_type: str) -> str:
    """Fallback message formatter when LLM interpretation fails."""
    if not chart_data:
        return f"No {report_type} data found for file: {file_name}" if file_name else f"No {report_type} data available"
    
    success_data = next((item for item in chart_data if item.get('status', '').lower() == 'success'), None)
    fail_data = next((item for item in chart_data if item.get('status', '').lower() == 'fail'), None)
    
    message_parts = []
    
    if file_name:
        message_parts.append(f"Analysis complete for {file_name}.")
    
    if row_count > 0:
        message_parts.append(f"Analyzed {row_count} records.")
    
    # Format message based on report type
    if report_type == "success" and success_data:
        if success_data['percentage'] > 90:
            message_parts.append(f"Excellent success rate: {success_data['percentage']:.1f}% ({success_data['count']} successful records).")
        else:
            message_parts.append(f"Success rate: {success_data['percentage']:.1f}% ({success_data['count']} successful records).")
    elif report_type == "failure" and fail_data:
        if fail_data['percentage'] > 50:
            message_parts.append(f"High failure rate detected: {fail_data['percentage']:.1f}% ({fail_data['count']} failed records).")
        else:
            message_parts.append(f"Failure rate: {fail_data['percentage']:.1f}% ({fail_data['count']} failed records).")
    elif report_type == "both":
        if success_data and fail_data:
            if fail_data['percentage'] > 50:
                message_parts.append(f"High failure rate detected: {fail_data['percentage']:.1f}% of records failed.")
            elif success_data['percentage'] > 90:
                message_parts.append(f"Excellent success rate: {success_data['percentage']:.1f}% of records succeeded.")
            else:
                message_parts.append(f"Mixed results: {success_data['percentage']:.1f}% success, {fail_data['percentage']:.1f}% failure.")
    
    report_desc = {
        "success": "success-focused",
        "failure": "failure-focused",
        "both": ""
    }
    chart_desc = f"A {report_desc[report_type]} {chart_type} chart".strip()
    message_parts.append(f"{chart_desc} has been generated for visualization.")
    
    return " ".join(message_parts)

# --- System prompt -----------------------------------------------------------------

SYSTEM = (
    "You are the Analytics Agent for data quality and data accuracy.\n"
    "- You specialize in retrieving analytics from DynamoDB via provided tools.\n"
    "- You analyze success and failure rates for data processing tasks.\n"
    "- Pay attention to the user's specific request: they may want only success data, only failure data, or both.\n"
    "- When you receive tool results, interpret them thoughtfully based on what the user asked for.\n"
    "- Highlight concerning patterns like high failure rates when relevant.\n"
    "- Be direct and insightful in your analysis.\n"
    "- NEVER fabricate data - only report what the tools return.\n"
    "- Focus your response on the type of report the user requested (success-only, failure-only, or both).\n"
    "- Use conversation history to understand file references like 'that file', 'this file', or 'the same file'.\n"
    "- When a user refers to 'that file' or similar, look at the conversation history to identify which file they mean.\n"
)

# maximum number of assistant->tool cycles before we force-stop the agent
MAX_AGENT_LOOPS = 10

def build_app():
    if not USE_LLM:
        raise SystemExit("OPENAI_API_KEY missing. Add it to .env to run the chat agent.")
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0).bind_tools(AnalyticsService.TOOLS)

    graph = StateGraph(MessagesState)

    def assistant(state: MessagesState):
        messages = state["messages"]
        
        # Manual loop protection
        loop_count = state.get("_loop_count", 0)
        if loop_count >= MAX_AGENT_LOOPS:
            return {"messages": [AIMessage(content="Maximum number of tool calls reached. Please try a simpler query.")]}
        
        state["_loop_count"] = loop_count + 1
        
        # Check if we have tool results to process
        has_tool_results = False
        tool_results = []
        
        for msg in messages:
            if isinstance(msg, ToolMessage):
                has_tool_results = True
                try:
                    tool_data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    tool_results.append(tool_data)
                except:
                    tool_results.append({"content": msg.content})

        # If we have tool results, provide a final answer
        if has_tool_results:
            # Store results but don't format final answer here
            # We'll use LLM interpretation instead
            return {"messages": [AIMessage(content="Results obtained")]}
        
        # Otherwise, let the LLM decide what to do
        response = llm.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(AnalyticsService.TOOLS)

    graph.add_node("assistant", assistant)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")

    return graph.compile()