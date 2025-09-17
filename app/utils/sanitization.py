"""
Input sanitization utilities for security and data integrity.
"""
import re
import logging
from typing import Any, List, Dict

logger = logging.getLogger(__name__)


def sanitize_text_input(text: str, max_length: int = 500) -> str:
    """Sanitize user input to prevent prompt injection attacks while preserving query intent."""
    if not text:
        return ""

    # Remove or escape potentially dangerous characters/patterns
    sanitized = text.strip()

    # More targeted filtering - preserve core query while blocking injection
    dangerous_patterns = [
        r'###.*?(?=\s|$)',  # Section separators with content
        r'---.*?(?=\s|$)',  # Dividers with content
        r'System:.*?$(?=\n|$)',  # System message overrides (multiline)
        r'Assistant:.*?$(?=\n|$)',  # Assistant role overrides (multiline)
        r'User:.*?$(?=\n|$)',  # User role overrides (multiline)
        r'Ignore\s+previous.*?$(?=\n|$)',  # Common injection phrase
        r'Forget\s+previous.*?$(?=\n|$)',  # Common injection phrase
        r'Disregard.*?$(?=\n|$)',  # Common injection phrase
        r'You\s+are\s+now.*?$(?=\n|$)',  # Role override attempts
        r'Your\s+role\s+is.*?$(?=\n|$)',  # Role override attempts
        r'Act\s+as.*?$(?=\n|$)',  # Role override attempts
    ]

    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

    # Additional single-word patterns that might be embedded
    single_patterns = [
        r'\bSystem\b(?!\w)',  # Word boundary to avoid false positives
        r'\bAssistant\b(?!\w)',
        r'\bUser\b(?!\w)',
    ]

    for pattern in single_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)

    # Clean up extra whitespace and normalize
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()

    # Limit length to prevent extremely long inputs
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."

    return sanitized


def sanitize_filename(filename: str) -> str:
    """Sanitize filename inputs to prevent directory traversal and injection."""
    if not filename:
        return ""

    # Remove path traversal attempts
    sanitized = re.sub(r'[./\\]', '', filename)

    # Only allow alphanumeric, dots, hyphens, and underscores
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '', sanitized)

    # Limit length
    return sanitized[:100]


def sanitize_numeric_value(value: Any) -> str:
    """Safely convert values to strings for prompt inclusion."""
    try:
        if isinstance(value, (int, float)):
            # Format numbers safely
            if isinstance(value, float):
                return f"{value:.2f}"
            return str(int(value))
        elif isinstance(value, str):
            # Remove any non-numeric characters except decimal point
            return re.sub(r'[^0-9.]', '', value)[:20]
        else:
            return str(value)[:50]  # Generic fallback with length limit
    except:
        return "[INVALID_VALUE]"


def create_safe_context_message(context_insights: List[str]) -> str:
    """Create a safe context message with sanitized inputs."""
    if not context_insights:
        return "├── No recent context available"

    safe_insights = []
    for insight in context_insights:
        # Sanitize each insight
        safe_insight = sanitize_text_input(insight, 100)
        if safe_insight:
            safe_insights.append(f"├── {safe_insight}")

    return "\n".join(safe_insights)