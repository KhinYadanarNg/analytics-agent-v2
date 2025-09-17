"""
LLM interpretation utilities for analytics processing.
"""
import asyncio
import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import OPENAI_MODEL, USE_LLM
from app.utils.sanitization import sanitize_text_input, sanitize_filename, sanitize_numeric_value
from app.utils.formatting import format_basic_message
from app.utils.context import create_interpretation_prompt

logger = logging.getLogger(__name__)


async def get_llm_interpretation(prompt: str, chart_data: List[Dict],
                                original_chart_data: List[Dict],
                                chart_type: str, chart_generated: bool,
                                file_name: str, row_count: int,
                                report_type: str, date_filter_used: Dict[str, str],
                                conversation_history: List[Dict[str, Any]] = None) -> str:
    """
    Have the LLM interpret the results and provide a natural language response.
    """
    if not USE_LLM:
        # Fallback to basic message if LLM not available
        return format_basic_message(chart_data, file_name, row_count, chart_type, report_type, date_filter_used)

    try:
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)

        # Prepare context for interpretation based on report type
        context_parts = []

        if chart_data:
            context_parts.append(f"File analyzed: {sanitize_filename(file_name)}")
            context_parts.append(f"Total records: {sanitize_numeric_value(row_count)}")
            context_parts.append(f"Report type requested: {sanitize_text_input(report_type, 20)}")

            # Add date context if filters were applied
            if date_filter_used:
                if date_filter_used.get("start_date") and date_filter_used.get("end_date"):
                    if date_filter_used["start_date"] == date_filter_used["end_date"]:
                        context_parts.append(f"Date filter: {sanitize_text_input(date_filter_used['start_date'], 20)}")
                    else:
                        context_parts.append(f"Date range: {sanitize_text_input(date_filter_used['start_date'], 20)} to {sanitize_text_input(date_filter_used['end_date'], 20)}")
                elif date_filter_used.get("start_date"):
                    context_parts.append(f"From date: {sanitize_text_input(date_filter_used['start_date'], 20)}")
                elif date_filter_used.get("end_date"):
                    context_parts.append(f"Until date: {sanitize_text_input(date_filter_used['end_date'], 20)}")

            # Extract metrics based on report type
            if report_type == "success":
                success_data = next((item for item in chart_data if item.get('status', '').lower() == 'success'), None)
                if success_data:
                    context_parts.append(f"Success: {sanitize_numeric_value(success_data['percentage'])}% ({sanitize_numeric_value(success_data['count'])} records)")
                else:
                    context_parts.append("No successful records found")
            elif report_type == "failure":
                fail_data = next((item for item in chart_data if item.get('status', '').lower() == 'fail'), None)
                if fail_data:
                    context_parts.append(f"Failure: {sanitize_numeric_value(fail_data['percentage'])}% ({sanitize_numeric_value(fail_data['count'])} records)")
                else:
                    context_parts.append("No failed records found")
            else:  # both
                success_data = next((item for item in original_chart_data if item.get('status', '').lower() == 'success'), None)
                fail_data = next((item for item in original_chart_data if item.get('status', '').lower() == 'fail'), None)

                if success_data:
                    context_parts.append(f"Success: {sanitize_numeric_value(success_data['percentage'])}% ({sanitize_numeric_value(success_data['count'])} records)")
                if fail_data:
                    context_parts.append(f"Failure: {sanitize_numeric_value(fail_data['percentage'])}% ({sanitize_numeric_value(fail_data['count'])} records)")

            if chart_generated:
                report_desc = {
                    "success": "success-only",
                    "failure": "failure-only",
                    "both": "comprehensive"
                }
                context_parts.append(f"Generated {sanitize_text_input(report_desc[report_type], 20)} {sanitize_text_input(chart_type, 20)} chart")
        else:
            context_parts.append(f"No {sanitize_text_input(report_type, 20)} data found for file: {sanitize_filename(file_name)}")
            if date_filter_used:
                context_parts.append("Note: Date filters were applied which may have limited the results")

        interpretation_prompt = f"""
CONTEXT ANALYSIS:
├── User Request: "{sanitize_text_input(prompt, 200)}"
├── Report Type: {report_type}
├── Chart Type: {chart_type}
├── Date Filters: {date_filter_used or 'None'}
├── Data Available: {'Yes' if chart_data else 'No'}

DATA SUMMARY:
{chr(10).join(f"├── {sanitize_text_input(item['status'].title(), 20)}: {sanitize_numeric_value(item['percentage'])}% ({sanitize_numeric_value(item['count'])} records)" for item in chart_data) if chart_data else "├── No data available"}

RESPONSE REQUIREMENTS:
├── Answer the user's specific question directly
├── Focus on {report_type} metrics as requested
├── Include date context if filters were applied
├── Highlight key insights and patterns
├── Mention chart generation when applicable
├── Use natural, conversational language
├── Be concise but informative
├── Celebrate excellent performance (e.g., 0% failure rate)
├── Flag concerning patterns (e.g., high failure rates)

OUTPUT FORMAT:
- Single flowing paragraph (no bullet points)
- Professional but approachable tone
- Data-driven insights
- Actionable observations
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
        return format_basic_message(chart_data, file_name, row_count, chart_type, report_type, date_filter_used)