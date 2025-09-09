import re
from typing import Tuple, List
import base64
import urllib.parse
import logging

logger = logging.getLogger("security")

# Enhanced injection patterns with severity levels
CRITICAL_INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"forget (all )?previous instructions", 
    r"disregard (all )?previous instructions",
    r"override system prompt",
    r"you are now",
    r"from now on",
    r"new instructions:",
    r"</system>",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
]

HIGH_RISK_PATTERNS = [
    r"ignore this message",
    r"follow only the instructions",
    r"system:",
    r"assistant should",
    r"don't follow the system",
    r"act as if you are",
    r"pretend you are",
    r"roleplay as",
    r"simulate",
    r"bypass",
]

MEDIUM_RISK_PATTERNS = [
    r"can you help me with",
    r"instead of",
    r"actually",
    r"however",
    r"but really",
    r"what I really want",
    r"```.*?```",  # Code blocks
    r"<.*?>.*?</.*?>",  # HTML/XML tags
]

# Encoding-based evasion patterns
ENCODING_PATTERNS = [
    r"base64:",
    r"hex:",
    r"url:",
    r"unicode:",
    r"\\x[0-9a-fA-F]{2}",  # Hex encoding
    r"&#[0-9]+;",  # HTML entities
    r"%[0-9a-fA-F]{2}",  # URL encoding
]

# Secret extraction patterns
SECRET_PATTERNS = [
    r"api[_\s-]?key",
    r"secret[_\s-]?key",
    r"password",
    r"token",
    r"private[_\s-]?key",
    r"ssh[_\s-]?key",
    r"openai[_\s-]?key",
    r"sk-[a-zA-Z0-9]+",  # OpenAI API key pattern
]


def detect_prompt_injection(text: str) -> Tuple[bool, List[str]]:
    """Enhanced prompt injection detection with severity levels."""
    lower = text.lower()
    detected_patterns = []
    risk_score = 0
    
    # Check critical patterns (high risk score)
    for pattern in CRITICAL_INJECTION_PATTERNS:
        if re.search(pattern, lower):
            detected_patterns.append(f"CRITICAL: {pattern}")
            risk_score += 10
    
    # Check high risk patterns
    for pattern in HIGH_RISK_PATTERNS:
        if re.search(pattern, lower):
            detected_patterns.append(f"HIGH: {pattern}")
            risk_score += 5
    
    # Check medium risk patterns
    for pattern in MEDIUM_RISK_PATTERNS:
        if re.search(pattern, lower):
            detected_patterns.append(f"MEDIUM: {pattern}")
            risk_score += 2
    
    # Check encoding evasion attempts
    for pattern in ENCODING_PATTERNS:
        if re.search(pattern, lower):
            detected_patterns.append(f"ENCODING: {pattern}")
            risk_score += 7
    
    # Check secret extraction attempts
    for pattern in SECRET_PATTERNS:
        if re.search(pattern, lower):
            detected_patterns.append(f"SECRET: {pattern}")
            risk_score += 15
    
    # Additional checks for encoded content
    if _check_encoded_payloads(text):
        detected_patterns.append("ENCODED_PAYLOAD")
        risk_score += 8
    
    # Log security events
    if detected_patterns:
        logger.warning(
            "Prompt injection detected: patterns=%s, risk_score=%d, text_length=%d",
            detected_patterns, risk_score, len(text)
        )
    
    # Return True if risk score exceeds threshold
    return (risk_score >= 5, detected_patterns)


def _check_encoded_payloads(text: str) -> bool:
    """Check for base64, URL, or hex encoded suspicious content."""
    try:
        # Check for base64 encoded content
        if len(text) > 20:
            try:
                decoded = base64.b64decode(text, validate=True).decode('utf-8', errors='ignore')
                if any(pattern in decoded.lower() for pattern in ["system:", "ignore", "instructions"]):
                    return True
            except:
                pass
        
        # Check for URL encoded content  
        try:
            decoded = urllib.parse.unquote(text)
            if decoded != text and any(pattern in decoded.lower() for pattern in ["system:", "ignore"]):
                return True
        except:
            pass
            
        return False
    except:
        return False


def sanitize_input(text: str) -> Tuple[str, List[str]]:
    """Enhanced input sanitization with multiple security layers."""
    clean = text
    warnings = []

    # Normalize whitespace and control characters
    clean = re.sub(r"\s+", " ", clean).strip()
    clean = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", clean)  # Remove control chars
    
    # Check for injection patterns
    detected, patterns = detect_prompt_injection(clean)
    if detected:
        warnings.extend(patterns)
        
        # Redact critical and high-risk patterns more aggressively
        for pattern_info in patterns:
            if pattern_info.startswith("CRITICAL:") or pattern_info.startswith("SECRET:"):
                # Block entirely for critical violations
                return "[BLOCKED_CRITICAL_VIOLATION]", warnings
            
            pattern = pattern_info.split(": ", 1)[1] if ": " in pattern_info else pattern_info
            try:
                clean = re.sub(pattern, "[REDACTED]", clean, flags=re.IGNORECASE)
            except re.error:
                continue

    # Remove suspicious role-play indicators
    role_patterns = [
        r"^\s*(system|assistant|user)\s*:",
        r"<\|.*?\|>",
        r"</?(system|assistant|user)>",
        r"```.*?```",
    ]
    
    for pattern in role_patterns:
        if re.search(pattern, clean, re.IGNORECASE | re.DOTALL):
            warnings.append(f"role_indicator: {pattern}")
            clean = re.sub(pattern, "[REDACTED_ROLE]", clean, flags=re.IGNORECASE | re.DOTALL)

    # Remove suspicious lines
    lines = []
    for line in clean.split("\n"):
        line_lower = line.lower().strip()
        
        # Skip lines that look like system prompts
        if any(line_lower.startswith(prefix) for prefix in ["system:", "assistant:", "user:", "ai:", "human:"]):
            warnings.append("suspicious_line_removed")
            continue
            
        # Skip lines with excessive special characters (potential obfuscation)
        special_char_ratio = len(re.findall(r"[^a-zA-Z0-9\s.,!?]", line)) / max(len(line), 1)
        if special_char_ratio > 0.3 and len(line) > 10:
            warnings.append("obfuscated_line_removed")
            continue
            
        lines.append(line)
    
    clean = "\n".join(lines)
    
    # Final length and content checks
    if len(clean) > 2000:  # Prevent extremely long inputs
        warnings.append("truncated_length")
        clean = clean[:2000] + "..."
    
    # Check for repeated characters (potential DoS)
    if re.search(r"(.)\1{50,}", clean):
        warnings.append("repeated_chars_removed")
        clean = re.sub(r"(.)\1{10,}", r"\1" * 10, clean)
    
    return clean, list(set(warnings))
