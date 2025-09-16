"""
Simplified analytics query processing service where LLM handles date extraction.
"""
import json
import logging
import asyncio
import re
from typing import Tuple, Optional, Dict, Any, List
from datetime import datetime, date
from app.config import OPENAI_MODEL, USE_LLM, DEBUG
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from app.tools_agent import (
    get_success_rate_by_file_name_tool
)
from app.chart_generator import chart_generator

# Setup logger
logger = logging.getLogger(__name__)

def format_error_message(error_type: str, user_message: str, technical_details: str = "") -> str:
    """Format error messages with consistent structure and user-friendly language."""
    formatted = f"❌ **{error_type}**\n\n{user_message}"
    if technical_details:
        formatted += f"\n\n🔧 *Technical Details:* {technical_details}"
    formatted += "\n\n💡 *Suggestion:* Try rephrasing your query or contact support if the issue persists."
    return formatted

def extract_context_insights(data: List[Dict[str, Any]]) -> List[str]:
    """Extract key insights from tool result data for context awareness."""
    insights = []
    
    if not data:
        return insights
    
    try:
        # Calculate basic metrics
        total_records = len(data)
        insights.append(f"📊 Dataset contains {total_records} records")
        
        # Extract success rates if available
        success_rates = []
        for record in data:
            if "success_rate" in record:
                try:
                    rate = float(record["success_rate"])
                    success_rates.append(rate)
                except (ValueError, TypeError):
                    continue
        
        if success_rates:
            avg_success = sum(success_rates) / len(success_rates)
            min_success = min(success_rates)
            max_success = max(success_rates)
            
            insights.append(f"📈 Average success rate: {avg_success:.1f}%")
            insights.append(f"🎯 Success rate range: {min_success:.1f}% - {max_success:.1f}%")
            
            # Add performance insights
            if avg_success >= 95:
                insights.append("🌟 Excellent performance across the board")
            elif avg_success >= 80:
                insights.append("✅ Good overall performance")
            elif avg_success >= 60:
                insights.append("⚠️ Moderate performance - room for improvement")
            else:
                insights.append("🚨 Poor performance - requires immediate attention")
    
    except Exception as e:
        logger.warning(f"Failed to extract context insights: {e}")
    
    return insights

def get_conversation_context(messages: List) -> List:
    """Extract recent conversation context for better continuity."""
    context_messages = []
    
    # Get last 5 messages for context (excluding current message)
    recent_messages = messages[-6:-1] if len(messages) > 1 else []
    
    for msg in recent_messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            context_messages.append(msg)
    
    return context_messages

def create_interpretation_prompt(tool_results: List[Dict[str, Any]], context_insights: List[str]) -> str:
    """Create an enhanced interpretation prompt for the LLM."""
    
    # Build the interpretation prompt
    prompt_parts = []
    
    # Add context insights
    if context_insights:
        prompt_parts.append("CONTEXT INSIGHTS:")
        for insight in context_insights:
            prompt_parts.append(f"├── {insight}")
        prompt_parts.append("")
    
    # Add tool results summary
    prompt_parts.append("TOOL RESULTS:")
    for i, result in enumerate(tool_results):
        prompt_parts.append(f"├── Result {i+1}:")
        if isinstance(result, dict):
            for key, value in result.items():
                if key == "data" and isinstance(value, list):
                    prompt_parts.append(f"│   ├── {key}: {len(value)} records")
                    if value and len(value) > 0:
                        # Show sample of first record
                        sample = value[0]
                        if isinstance(sample, dict):
                            sample_keys = list(sample.keys())[:3]  # Show first 3 keys
                            prompt_parts.append(f"│   │   ├── Sample keys: {', '.join(sample_keys)}")
                else:
                    prompt_parts.append(f"│   ├── {key}: {value}")
        else:
            prompt_parts.append(f"│   ├── Raw content: {str(result)[:100]}...")
        prompt_parts.append("")
    
    # Add interpretation instructions
    prompt_parts.append("INTERPRETATION TASK:")
    prompt_parts.append("├── Analyze the data and provide insights")
    prompt_parts.append("├── Highlight key patterns, trends, or anomalies")
    prompt_parts.append("├── Provide actionable recommendations if applicable")
    prompt_parts.append("├── Use clear, conversational language")
    prompt_parts.append("└── Focus on the most important findings")
    
    return "\n".join(prompt_parts)

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
        
        # Check if user explicitly wants both
        both_indicators = [
            r'\bboth\b',
            r'\bsuccess\s+and\s+fail',
            r'\bsuccess\s+and\s+failure',
            r'\ball\s+data',
            r'\bcomplete\s+analysis',
            r'\bfull\s+report',
            r'\boverall',
            r'\btotal',
            r'\beverything'
        ]
        
        # Check for success-only indicators
        success_only_patterns = [
            r'\bonly\s+success',
            r'\bsuccess\s+only',
            r'\bjust\s+success',
            r'\bsuccessful\s+only'
        ]
        
        # Check for failure-only indicators
        failure_only_patterns = [
            r'\bonly\s+fail',
            r'\bfail\s+only',
            r'\bfailure\s+only',
            r'\bjust\s+fail',
            r'\berror\s+only',
            r'\bissue\s+only'
        ]
        
        # Check if both success and failure are mentioned (without "only")
        has_success = any(word in prompt_lower for word in ['success', 'successful', 'succeeded', 'pass', 'passed'])
        has_failure = any(word in prompt_lower for word in ['fail', 'failure', 'failed', 'error', 'issue'])
        
        # Decision logic
        # First check for explicit "only" patterns
        if any(re.search(pattern, prompt_lower) for pattern in success_only_patterns):
            return "success"
        elif any(re.search(pattern, prompt_lower) for pattern in failure_only_patterns):
            return "failure"
        # Then check for explicit "both" indicators
        elif any(re.search(pattern, prompt_lower) for pattern in both_indicators):
            return "both"
        # If both success and failure mentioned without "only", show both
        elif has_success and has_failure:
            return "both"
        # If only success mentioned
        elif has_success and not has_failure:
            return "success"
        # If only failure mentioned
        elif has_failure and not has_success:
            return "failure"
        # Default to both if no clear indication
        else:
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
    async def process_query(prompt: str, session_id: str = None, conversation_history: List[Dict[str, Any]] = None, org_id: str = None) -> Dict[str, Any]:
        """
        Build the agent graph, invoke it, generate chart, 
        then have LLM interpret the results for a natural response.
        The LLM will handle date extraction and chart type detection from the prompt.

        Args:
            prompt: The user's query
            session_id: Optional session identifier for tracking
            conversation_history: Optional conversation history for context
            org_id: Organization ID for filtering data

        Returns a dict with keys: success (bool), message (str), chart_image (str base64)
        """
        try:
            # Detect report type from prompt (success/failure/both)
            report_type = AnalyticsService.detect_report_type(prompt)
            
            # Set org_id for tools to use as fallback
            # if org_id:
            #     from app.tools_agent import set_tools_org_id
            #     set_tools_org_id(org_id)
            
            # Log detected parameters for debugging
            logger = logging.getLogger("analytics_agent")
            logger.info(f"Processing prompt: '{prompt[:100]}...'")
            logger.info(f"Detected report type: '{report_type}'")
            
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
                    context_insights.append(f"Previously analyzed: {interaction['response_summary']['file_name']}")
                if interaction.get("response_summary", {}).get("report_type"):
                    context_insights.append(f"Previous focus: {interaction['response_summary']['report_type']} metrics")

            # Add context-aware system message
            context_message = f"""
RECENT CONTEXT:
{chr(10).join(f"├── {insight}" for insight in context_insights) if context_insights else "├── No recent context available"}

Use this context to provide more relevant and personalized responses.
"""
            messages.append(SystemMessage(content=context_message))

            # Add conversation history with better formatting
            for interaction in recent_interactions:
                if interaction.get("user_prompt"):
                    messages.append(HumanMessage(content=f"Previous query: {interaction['user_prompt']}"))
                if interaction.get("response_summary", {}).get("message"):
                    # Extract key insights from previous response
                    prev_response = interaction["response_summary"]["message"]
                    # Truncate if too long but keep key metrics
                    if len(prev_response) > 150:
                        # Try to keep percentage/rate information
                        import re
                        rates = re.findall(r'\d+\.?\d*%', prev_response)
                        if rates:
                            prev_response = f"Previous analysis showed rates: {', '.join(rates[:2])}"
                        else:
                            prev_response = prev_response[:150] + "..."
                    messages.append(AIMessage(content=f"Previous analysis: {prev_response}"))
        
        # Add the current prompt
        messages.append(HumanMessage(content=prompt))
        
        state = {
            "messages": messages,
            "report_type": report_type,
            "session_id": session_id,
            "org_id": org_id  # Pass org_id through state
        }

        loop = asyncio.get_running_loop()
        try:
            # First pass: Run the graph to get data
            compiled_result = await loop.run_in_executor(None, lambda: app_graph.invoke(state))
        except Exception as e:
            # Enhanced error handling with context preservation
            logger = logging.getLogger("analytics_agent")
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
        if DEBUG:
            result["chart_data"] = filtered_chart_data
            result["original_chart_data"] = original_chart_data
            result["tool_results"] = tool_results
            result["chart_type"] = chart_type

        return result

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
            context_parts.append(f"File analyzed: {file_name}")
            context_parts.append(f"Total records: {row_count}")
            context_parts.append(f"Report type requested: {report_type}")
            
            # Add date context if filters were applied
            if date_filter_used:
                if date_filter_used.get("start_date") and date_filter_used.get("end_date"):
                    if date_filter_used["start_date"] == date_filter_used["end_date"]:
                        context_parts.append(f"Date filter: {date_filter_used['start_date']}")
                    else:
                        context_parts.append(f"Date range: {date_filter_used['start_date']} to {date_filter_used['end_date']}")
                elif date_filter_used.get("start_date"):
                    context_parts.append(f"From date: {date_filter_used['start_date']}")
                elif date_filter_used.get("end_date"):
                    context_parts.append(f"Until date: {date_filter_used['end_date']}")
            
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
            if date_filter_used:
                context_parts.append("Note: Date filters were applied which may have limited the results")
        
        interpretation_prompt = f"""
CONTEXT ANALYSIS:
├── User Request: "{prompt}"
├── Report Type: {report_type}
├── Chart Type: {chart_type}
├── Date Filters: {date_filter_used or 'None'}
├── Data Available: {'Yes' if chart_data else 'No'}

DATA SUMMARY:
{chr(10).join(f"├── {item['status'].title()}: {item['percentage']:.1f}% ({item['count']} records)" for item in chart_data) if chart_data else "├── No data available"}

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

def format_basic_message(chart_data: List[Dict], file_name: str, row_count: int,
                        chart_type: str, report_type: str, date_filter_used: Dict[str, str]) -> str:
    """Enhanced fallback message formatter with intelligent insights."""

    if not chart_data:
        base_msg = f"No {report_type} data found"
        if file_name:
            base_msg += f" for file: {file_name}"
        if date_filter_used:
            base_msg += f" within the specified date range"
        return base_msg

    success_data = next((item for item in chart_data if item.get('status', '').lower() == 'success'), None)
    fail_data = next((item for item in chart_data if item.get('status', '').lower() == 'fail'), None)

    message_parts = []

    # File and date context
    if file_name:
        message_parts.append(f"Analysis of {file_name} complete.")
    if date_filter_used:
        if date_filter_used.get("start_date") and date_filter_used.get("end_date"):
            if date_filter_used["start_date"] == date_filter_used["end_date"]:
                message_parts.append(f"Data from {date_filter_used['start_date']}.")
            else:
                message_parts.append(f"Data from {date_filter_used['start_date']} to {date_filter_used['end_date']}.")
    if row_count > 0:
        message_parts.append(f"Processed {row_count:,} records.")

    # Intelligent insights based on data patterns
    if report_type == "success" and success_data:
        percentage = success_data['percentage']
        if percentage >= 95:
            message_parts.append(f"🎉 Exceptional success rate: {percentage:.1f}% - outstanding performance!")
        elif percentage >= 90:
            message_parts.append(f"✅ Excellent success rate: {percentage:.1f}% - very good results.")
        elif percentage >= 80:
            message_parts.append(f"👍 Good success rate: {percentage:.1f}% - solid performance.")
        else:
            message_parts.append(f"⚠️ Success rate: {percentage:.1f}% - room for improvement.")

    elif report_type == "failure" and fail_data:
        percentage = fail_data['percentage']
        if percentage == 0:
            message_parts.append(f"🎉 Perfect! Zero failure rate - all records processed successfully.")
        elif percentage <= 5:
            message_parts.append(f"✅ Excellent! Very low failure rate: {percentage:.1f}%.")
        elif percentage <= 10:
            message_parts.append(f"⚠️ Moderate failure rate: {percentage:.1f}% - worth investigating.")
        else:
            message_parts.append(f"🚨 High failure rate detected: {percentage:.1f}% - requires attention.")

    elif report_type == "both":
        if success_data and fail_data:
            success_pct = success_data['percentage']
            fail_pct = fail_data['percentage']

            # Overall assessment
            if fail_pct == 0:
                message_parts.append(f"🎉 Perfect performance! 100% success rate with zero failures.")
            elif success_pct >= 90:
                message_parts.append(f"✅ Strong performance: {success_pct:.1f}% success, {fail_pct:.1f}% failure.")
            elif success_pct >= 80:
                message_parts.append(f"👍 Good performance: {success_pct:.1f}% success, {fail_pct:.1f}% failure.")
            else:
                message_parts.append(f"⚠️ Needs improvement: {success_pct:.1f}% success, {fail_pct:.1f}% failure.")

    # Chart information
    chart_descriptions = {
        "bar": "bar chart",
        "pie": "pie chart",
        "donut": "donut chart",
        "line": "line chart",
        "stacked": "stacked bar chart"
    }
    chart_desc = chart_descriptions.get(chart_type, f"{chart_type} chart")
    message_parts.append(f"Generated {report_type}-focused {chart_desc} for visualization.")

    return " ".join(message_parts)

# --- Enhanced System Prompt with Modular Design -----------------------------------------------------------------

SYSTEM_CORE = """You are the Analytics Agent for data quality and data accuracy.
Today's date is: {current_date}

CORE CAPABILITIES:
- Retrieve analytics from DynamoDB via get_success_rate_by_file_name tool
- Analyze success/failure rates for data processing tasks
- Filter data by 'created_date' column in database
- Generate various visualizations based on user preferences
- Handle multi-tenant data with organization-level isolation"""

REPORT_TYPE_INSTRUCTIONS = """
REPORT TYPE DETECTION:
├── SUCCESS-ONLY: "success rate", "successful", "pass rate", "only success"
├── FAILURE-ONLY: "fail rate", "failure rate", "error rate", "only failures"
├── COMBINED: "analyze", "both", "complete analysis", "overall"
└── DEFAULT: Show both success and failure metrics when unclear"""

CHART_TYPE_INSTRUCTIONS = """
CHART TYPE MAPPING:
├── PIE: "pie chart", "pie graph", "circular", "proportion"
├── DONUT: "donut", "doughnut", "ring", "modern pie"
├── LINE: "line chart", "trend", "progression", "over time"
├── STACKED: "stacked", "horizontal bar", "composition", "breakdown"
└── BAR: "bar chart", "column", "bars" (DEFAULT)"""

DATE_HANDLING_INSTRUCTIONS = """
DATE EXTRACTION RULES:
├── "from DATE" → start_date=DATE, end_date=today
├── "since DATE" → start_date=DATE, end_date=today
├── "on DATE" → start_date=DATE, end_date=DATE
├── "between DATE1 and DATE2" → start_date=DATE1, end_date=DATE2
├── "last N days" → start_date=N_days_ago, end_date=today
└── If no dates mentioned → do NOT include start_date or end_date parameters"""

TOOL_USAGE_GUIDELINES = """
TOOL CALL REQUIREMENTS:
├── ALWAYS specify chart_type parameter (default: "bar")
├── ONLY include start_date and end_date when dates are explicitly mentioned in the user query
├── If no dates are mentioned, do NOT include start_date or end_date parameters
├── Extract file names and clean extra quotes/spaces
├── Filter data by created_date column
└── Use org_id for multi-tenant isolation"""

SYSTEM = f"""{SYSTEM_CORE}

{REPORT_TYPE_INSTRUCTIONS}

{CHART_TYPE_INSTRUCTIONS}

{DATE_HANDLING_INSTRUCTIONS}

{TOOL_USAGE_GUIDELINES}

RESPONSE PRINCIPLES:
- Be direct and data-driven in analysis
- NEVER fabricate data - only report what tools return
- Focus responses on user's specific requests
- Provide actionable insights from the data
- Use natural, conversational language
- Highlight concerning patterns or excellent performance
- Only apply date filters when explicitly mentioned in the user query
- If no dates are specified, analyze all available data without date restrictions"""

# maximum number of assistant->tool cycles before we force-stop the agent
MAX_AGENT_LOOPS = 10

def build_app():
    if not USE_LLM:
        raise SystemExit("OPENAI_API_KEY missing. Add it to .env to run the chat agent.")
    
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
                    # Fallback to basic formatting
                    fallback_content = format_basic_message(tool_results, context_insights)
                    return {"messages": [AIMessage(content=fallback_content)]}
            
            # Handle initial queries and follow-ups without tool results
            elif not has_tool_results:
                # Add system prompt to ensure consistent behavior
                enhanced_messages = [SystemMessage(content=SYSTEM)] + list(messages)
                response = llm.invoke(enhanced_messages)
                return {"messages": [response]}
            
            # Fallback for edge cases
            else:
                fallback_msg = format_basic_message([], [])
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