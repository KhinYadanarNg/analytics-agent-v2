"""
Context and conversation handling utilities.
"""
import re
from typing import List, Dict, Any
from .sanitization import sanitize_text_input, sanitize_filename, sanitize_numeric_value


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
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to extract context insights: {e}")

    return insights


def get_conversation_context(messages: List) -> List:
    """Extract recent conversation context for better continuity."""
    from langchain_core.messages import HumanMessage, AIMessage

    context_messages = []

    # Get last 5 messages for context (excluding current message)
    recent_messages = messages[-6:-1] if len(messages) > 1 else []

    for msg in recent_messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            context_messages.append(msg)

    return context_messages


def create_interpretation_prompt(tool_results: List[Dict[str, Any]], context_insights: List[str]) -> str:
    """Create an enhanced interpretation prompt for the LLM with sanitized inputs."""

    # Build the interpretation prompt
    prompt_parts = []

    # Add context insights with sanitization
    if context_insights:
        prompt_parts.append("CONTEXT INSIGHTS:")
        for insight in context_insights:
            safe_insight = sanitize_text_input(insight, 100)
            if safe_insight:
                prompt_parts.append(f"├── {safe_insight}")
        prompt_parts.append("")

    # Add tool results summary with sanitization
    prompt_parts.append("TOOL RESULTS:")
    for i, result in enumerate(tool_results):
        prompt_parts.append(f"├── Result {i+1}:")
        if isinstance(result, dict):
            for key, value in result.items():
                safe_key = sanitize_text_input(str(key), 50)
                if key == "data" and isinstance(value, list):
                    prompt_parts.append(f"│   ├── {safe_key}: {len(value)} records")
                    if value and len(value) > 0:
                        # Show sample of first record with sanitization
                        sample = value[0]
                        if isinstance(sample, dict):
                            sample_keys = list(sample.keys())[:3]  # Show first 3 keys
                            safe_sample_keys = [sanitize_text_input(str(k), 30) for k in sample_keys]
                            prompt_parts.append(f"│   │   ├── Sample keys: {', '.join(safe_sample_keys)}")
                else:
                    safe_value = sanitize_text_input(str(value), 100)
                    prompt_parts.append(f"│   ├── {safe_key}: {safe_value}")
        else:
            safe_content = sanitize_text_input(str(result), 100)
            prompt_parts.append(f"│   ├── Raw content: {safe_content}...")
        prompt_parts.append("")

    # Add interpretation instructions
    prompt_parts.append("INTERPRETATION TASK:")
    prompt_parts.append("├── Analyze the data and provide insights")
    prompt_parts.append("├── Highlight key patterns, trends, or anomalies")
    prompt_parts.append("├── Provide actionable recommendations if applicable")
    prompt_parts.append("├── Use clear, conversational language")
    prompt_parts.append("└── Focus on the most important findings")

    return "\n".join(prompt_parts)