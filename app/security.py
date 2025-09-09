import re
from typing import Tuple, List

INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"forget (all )?previous instructions",
    r"disregard (all )?previous instructions",
    r"ignore this message",
    r"follow only the instructions",
    r"override system prompt",
    r"system:",
    r"assistant should",
    r"don't follow the system",
]


def detect_prompt_injection(text: str) -> Tuple[bool, List[str]]:
    """Return (True, reasons) if any suspicious patterns are found."""
    lower = text.lower()
    reasons = []
    for p in INJECTION_PATTERNS:
        if re.search(p, lower):
            reasons.append(p)
    return (len(reasons) > 0, reasons)


def sanitize_input(text: str) -> Tuple[str, List[str]]:
    """Sanitize a user prompt by removing or redacting suspicious instructions.

    Returns (clean_text, warnings). Warnings list contains matched patterns.
    This is intentionally conservative: we redact suspicious substrings rather than attempt
    to fully rewrite intent.
    """
    clean = text
    warnings = []

    # Normalize whitespace
    clean = re.sub(r"\s+", " ", clean).strip()

    detected, reasons = detect_prompt_injection(clean)
    if detected:
        warnings.extend(reasons)
        # Redact matched phrases
        for p in reasons:
            try:
                clean = re.sub(p, "[REDACTED_INJECTION]", clean, flags=re.IGNORECASE)
            except re.error:
                # fallback: naive replacement
                clean = clean

    # Remove suspicious lines starting with common injection tokens
    lines = []
    for line in clean.split("\n"):
        if re.match(r"^\s*(system:|assistant:|user:)", line.lower()):
            warnings.append("line_redacted")
            continue
        lines.append(line)
    clean = "\n".join(lines)

    return clean, list(set(warnings))
