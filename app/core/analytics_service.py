"""
Core analytics service for processing queries and generating responses.
"""
import asyncio
import json
import logging
from typing import List, Dict, Any
from app.core.tools_agent import get_success_rate_by_file_name_tool, get_success_rate_by_domain_name_tool
from app.llm.classification import get_report_type_from_llm
from app.utils.sanitization import sanitize_filename

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Handles LLM tool selection, execution, chart generation, and interpretation."""

    TOOLS = [
        get_success_rate_by_file_name_tool,
        get_success_rate_by_domain_name_tool
    ]

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
    async def process_query_with_report_type(prompt: str, report_type: str, session_id: str = None,
                                           conversation_history: List[Dict[str, Any]] = None,
                                           org_id: str = None) -> Dict[str, Any]:
        """
        Process analytics query with pre-determined report type.

        Args:
            prompt: The user's query
            report_type: Pre-determined report type ("success", "failure", or "both")
            session_id: Optional session identifier for tracking
            conversation_history: Optional conversation history for context
            org_id: Organization ID for filtering data

        Returns a dict with keys: success (bool), message (str), chart_image (str base64)
        """
        try:
            # Log detected parameters for debugging
            logger.info(f"Processing prompt: '{prompt[:100]}...'")
            logger.info(f"Report type: '{report_type}'")

            # Import here to avoid circular imports
            from .graph_builder import build_app
            from app.llm.interpretation import get_llm_interpretation
            from app.generators.chart_generator import chart_generator
            from app.utils.sanitization import sanitize_text_input, create_safe_context_message
            from app.utils.context import extract_context_insights, get_conversation_context
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
            from datetime import date
            from app.config import SYSTEM

            # build_app may raise if USE_LLM is False or config missing
            app_graph = build_app()

        except Exception as e:
            return {"success": False, "message": str(e), "chart_image": None}

        # Get current date for context
        current_date = date.today().strftime('%Y-%m-%d')

        # Prepare system message with current date context
        system_message = SYSTEM.format(current_date=current_date)
        messages = [SystemMessage(content=system_message)]

        # Enhanced conversation history with context awareness
        if conversation_history:
            # Analyze recent context for better understanding
            recent_interactions = conversation_history[-3:]
            context_insights = []

            for interaction in recent_interactions:
                if interaction.get("response_summary", {}).get("file_name"):
                    safe_filename = sanitize_filename(interaction['response_summary']['file_name'])
                    context_insights.append(f"Previously analyzed: {safe_filename}")
                if interaction.get("response_summary", {}).get("report_type"):
                    safe_report_type = sanitize_text_input(interaction['response_summary']['report_type'], 20)
                    context_insights.append(f"Previous focus: {safe_report_type} metrics")

            # Add context-aware system message with sanitized content
            safe_context = create_safe_context_message(context_insights)
            context_message = f"""
RECENT CONTEXT:
{safe_context}

Use this context to provide more relevant and personalized responses.
"""
            messages.append(SystemMessage(content=context_message))

            # Add conversation history with sanitized inputs
            for interaction in recent_interactions:
                if interaction.get("user_prompt"):
                    safe_prompt = sanitize_text_input(interaction['user_prompt'], 200)
                    messages.append(HumanMessage(content=f"Previous query: {safe_prompt}"))
                if interaction.get("response_summary", {}).get("message"):
                    # Extract key insights from previous response
                    prev_response = interaction["response_summary"]["message"]
                    # Truncate if too long but keep key metrics
                    if len(prev_response) > 150:
                        # Try to keep percentage/rate information
                        import re
                        rates = re.findall(r'\d+\.?\d*%', prev_response)
                        if rates:
                            from app.utils.sanitization import sanitize_numeric_value
                            safe_rates = [sanitize_numeric_value(rate) for rate in rates[:2]]
                            prev_response = f"Previous analysis showed rates: {', '.join(safe_rates)}"
                        else:
                            prev_response = sanitize_text_input(prev_response[:150], 150) + "..."
                    else:
                        prev_response = sanitize_text_input(prev_response, 150)
                    messages.append(AIMessage(content=f"Previous analysis: {prev_response}"))

        # Add the current prompt with sanitization
        safe_prompt = sanitize_text_input(prompt, 300)
        messages.append(HumanMessage(content=safe_prompt))

        # Prepare state with all required parameters
        state = {
            "messages": messages,
            "report_type": report_type,
            "session_id": session_id,
            "org_id": org_id if org_id is not None else None  # Ensure org_id is properly handled
        }

        loop = asyncio.get_running_loop()
        try:
            # First pass: Run the graph to get data
            compiled_result = await loop.run_in_executor(None, lambda: app_graph.invoke(state))
        except Exception as e:
            # Enhanced error handling with context preservation
            logger.exception(f"Query processing failed: {e}")

            # Provide context-aware error message
            error_context = {
                "report_type": report_type,
                "session_id": session_id[:8] if session_id else None,
                "has_conversation_history": bool(conversation_history),
                "error_type": type(e).__name__
            }

            error_message = f"""
I encountered an issue processing your {report_type} analysis request.

ERROR CONTEXT:
├── Report Type: {error_context['report_type']}
├── Session: {error_context['session_id'] or 'New session'}
├── Conversation History: {'Available' if error_context['has_conversation_history'] else 'None'}
├── Error Type: {error_context['error_type']}

This might be due to:
- Database connectivity issues
- Invalid file references
- Authentication problems
- System resource constraints

Please try your request again, or contact support if the issue persists.
"""

            return {
                "success": False,
                "error": str(e),
                "message": error_message.strip(),
                "context": error_context,
                "chart_image": None
            }

        # Extract tool results and generate chart
        tool_results = []
        chart_data = []
        file_name = None
        row_count = 0
        date_filter_used = None
        chart_type = "bar"  # Default chart type

        for m in compiled_result.get("messages", []):
            if isinstance(m, ToolMessage):
                try:
                    tool_data = json.loads(m.content) if isinstance(m.content, str) else m.content
                    tool_results.append(tool_data)

                    # Extract chart data from tool results
                    if tool_data.get("success") and "chart_data" in tool_data:
                        chart_data = tool_data.get("chart_data", [])
                        file_name = tool_data.get("file_name")
                        row_count = tool_data.get("row_count", 0)

                        # Get the chart type specified by the LLM
                        if tool_data.get("chart_type_requested"):
                            chart_type = tool_data.get("chart_type", "bar")
                            logger.info(f"LLM specified chart type: {chart_type}")

                        # Check if date filters were used
                        if tool_data.get("date_filter"):
                            date_filter_used = tool_data.get("date_filter")
                except:
                    tool_results.append(m.content)

            # Also check AIMessage for tool calls to see what dates and chart type were used
            if isinstance(m, AIMessage) and hasattr(m, 'tool_calls'):
                for tool_call in m.tool_calls:
                    if tool_call.get('args'):
                        args = tool_call['args']
                        if args.get('start_date') or args.get('end_date'):
                            date_filter_used = {
                                'start_date': args.get('start_date'),
                                'end_date': args.get('end_date')
                            }
                        # Get chart type from tool call arguments
                        if args.get('chart_type'):
                            chart_type = args.get('chart_type')
                            logger.info(f"LLM called tool with chart_type: {chart_type}")

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
                    chart_type=chart_type,  # Use LLM-specified chart type
                    file_name=file_name,
                    report_type=report_type
                )
                chart_generated = True
                logger.info(f"Generated {chart_type} chart for {file_name}")
            except Exception as e:
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
            date_filter_used=date_filter_used,
            conversation_history=conversation_history
        )

        result = {
            "success": True,
            "message": interpretation,
            "chart_image": chart_image
        }

        # Add date filters if they were used
        # if date_filter_used:
        #     result["date_filters"] = date_filter_used

        # Add additional data if DEBUG mode
        from app.config import DEBUG
        if DEBUG:
            result["chart_data"] = filtered_chart_data
            result["original_chart_data"] = original_chart_data
            result["tool_results"] = tool_results
            result["chart_type"] = chart_type

        return result

    @staticmethod
    async def process_query(prompt: str, session_id: str = None, conversation_history: List[Dict[str, Any]] = None, org_id: str = None) -> Dict[str, Any]:
        """
        Main entry point for processing analytics queries.
        Detects report type using LLM and then processes the query.

        Args:
            prompt: The user's query
            session_id: Optional session identifier for tracking
            conversation_history: Optional conversation history for context
            org_id: Organization ID for filtering data

        Returns a dict with keys: success (bool), message (str), chart_image (str base64)
        """
        try:
            # Detect report type from prompt using LLM
            report_type = await get_report_type_from_llm(prompt)

            # Process the query with the detected report type
            return await AnalyticsService.process_query_with_report_type(
                prompt=prompt,
                report_type=report_type,
                session_id=session_id,
                conversation_history=conversation_history,
                org_id=org_id
            )

        except Exception as e:
            logger.exception(f"Query processing failed: {e}")
            return {"success": False, "message": str(e), "chart_image": None}