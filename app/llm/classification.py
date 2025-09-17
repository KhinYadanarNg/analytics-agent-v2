"""
LLM-based classification utilities for analytics processing.
"""
import asyncio
import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import OPENAI_MODEL, USE_LLM
from app.utils.sanitization import sanitize_text_input

logger = logging.getLogger(__name__)


async def get_report_type_from_llm(prompt: str) -> str:
    """
    Use LLM to determine what type of report the user is asking for.

    Returns:
        "success" - only success data
        "failure" - only failure data
        "both" - both success and failure data (default)
    """
    if not USE_LLM:
        # Fallback to regex-based detection if LLM not available
        return detect_report_type_regex(prompt)

    try:
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)  # Low temperature for consistent classification

        classification_prompt = f"""
Analyze the user's query and determine what type of analytics report they want.

USER QUERY: "{sanitize_text_input(prompt, 200)}"

CLASSIFICATION RULES:
- "success": User wants ONLY success metrics (e.g., "success rate", "successful records", "only success")
- "failure": User wants ONLY failure/error metrics (e.g., "failure rate", "error rate", "only failures")
- "both": User wants combined analysis or mentions both success and failure (e.g., "analyze", "both", "complete report")

Respond with ONLY one word: "success", "failure", or "both"
"""

        messages = [
            SystemMessage(content="You are a classification assistant. Analyze user queries for analytics report types. Respond with exactly one word."),
            HumanMessage(content=classification_prompt)
        ]

        response = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: llm.invoke(messages)
        )

        result = response.content.strip().lower()

        # Validate the response
        if result in ["success", "failure", "both"]:
            return result
        else:
            logger.warning(f"LLM returned invalid report type: {result}, falling back to regex")
            return detect_report_type_regex(prompt)

    except Exception as e:
        logger.warning(f"Failed to get report type from LLM: {e}, falling back to regex")
        return detect_report_type_regex(prompt)


def detect_report_type_regex(prompt: str) -> str:
    """
    Detect what type of report the user is asking for using regex patterns.

    Returns:
        "success" - only success data
        "failure" - only failure data
        "both" - both success and failure data (default)
    """
    import re

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