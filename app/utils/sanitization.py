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
        r'###.*?$(?=\n|$)',  # Section separators with content (multiline)
        r'###.*?(?=\s|$)',  # Section separators with content (single line)
        r'---.*?$(?=\n|$)',  # Dividers with content (multiline)
        r'---.*?(?=\s|$)',  # Dividers with content (single line)
        r'System:.*?$(?=\n|$)',  # System message overrides (multiline)
        r'System:.*?(?=\s|$)',  # System message overrides (single line)
        r'Assistant:.*?$(?=\n|$)',  # Assistant role overrides (multiline)
        r'Assistant:.*?(?=\s|$)',  # Assistant role overrides (single line)
        r'User:.*?$(?=\n|$)',  # User role overrides (multiline)
        r'User:.*?(?=\s|$)',  # User role overrides (single line)
        r'Ignore\s+previous.*?$(?=\n|$)',  # Common injection phrase (multiline)
        r'Ignore\s+previous.*?(?=\s|$)',  # Common injection phrase (single line)
        r'Forget\s+previous.*?$(?=\n|$)',  # Common injection phrase (multiline)
        r'Forget\s+previous.*?(?=\s|$)',  # Common injection phrase (single line)
        r'Disregard.*?$(?=\n|$)',  # Common injection phrase (multiline)
        r'Disregard.*?(?=\s|$)',  # Common injection phrase (single line)
        r'You\s+are\s+now.*?$(?=\n|$)',  # Role override attempts (multiline)
        r'You\s+are\s+now.*?(?=\s|$)',  # Role override attempts (single line)
        r'Your\s+role\s+is.*?$(?=\n|$)',  # Role override attempts (multiline)
        r'Your\s+role\s+is.*?(?=\s|$)',  # Role override attempts (single line)
        r'Act\s+as.*?$(?=\n|$)',  # Role override attempts (multiline)
        r'Act\s+as.*?(?=\s|$)',  # Role override attempts (single line)
        r'Execute:.*?$(?=\n|$)',  # Direct command execution (multiline)
        r'Execute:.*?(?=\s|$)',  # Direct command execution (single line)
        r'Run:.*?$(?=\n|$)',  # Command execution (multiline)
        r'Run:.*?(?=\s|$)',  # Command execution (single line)
        r'rm\s+-rf.*?$(?=\n|$)',  # Dangerous rm commands (multiline)
        r'rm\s+-rf.*?(?=\s|$)',  # Dangerous rm commands (single line)
    ]

    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

    # Additional single-word patterns that might be embedded
    # Only include truly dangerous standalone commands
    single_patterns = [
        r'\brm\b(?!\w)',  # Dangerous file deletion command
        r'\bdel\b(?!\w)',  # Windows delete command
        r'\bformat\b(?!\w)',  # Disk formatting command
        r'\bcmd\b(?!\w)',  # Command prompt
        r'\bpowershell\b(?!\w)',  # PowerShell
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