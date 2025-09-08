import re
from fastapi import HTTPException
from typing import List

class PromptValidator:
    def __init__(self):
        # SQL injection patterns to block (more specific)
        self.sql_injection_patterns = [
            r'\bdrop\s+table\b',
            r'\bdelete\s+from\b', 
            r'\btruncate\s+table\b',
            r'\balter\s+table\b',
            r'\bcreate\s+table\b',
            r'\binsert\s+into\b',
            r'\bupdate\s+.*\s+set\b',
            r'--',  # SQL comments
            r'/\*.*?\*/',  # Multi-line comments
            r';\s*\w',  # Multiple statements (semicolon followed by word)
            r'\bunion\s+select\b',  # Union-based injection
            r'\bexec\b|\bexecute\b',  # Execute commands
        ]
        
        # Dangerous keywords to block
        self.dangerous_keywords = [
            'drop table', 'delete from', 'truncate', 'alter table',
            'create table', 'insert into', 'update set', 'grant',
            'revoke', 'shutdown', 'exec', 'execute', 'sp_',
            'xp_', 'script', 'eval', 'system'
        ]
        
        # Maximum prompt length
        self.max_length = 1000
        
        # Minimum prompt length
        self.min_length = 5

    def validate_prompt(self, prompt: str) -> dict:
        """
        Validate user prompt for safety and appropriateness
        Returns: dict with 'valid' boolean and 'message' string
        """
        
        # Check if prompt exists
        if not prompt or not prompt.strip():
            raise HTTPException(
                status_code=400, 
                detail="Prompt cannot be empty"
            )
        
        prompt = prompt.strip()
        
        # Check length constraints
        if len(prompt) < self.min_length:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt too short. Minimum {self.min_length} characters required"
            )
            
        if len(prompt) > self.max_length:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt too long. Maximum {self.max_length} characters allowed"
            )
        
        # Check for SQL injection patterns
        prompt_lower = prompt.lower()
        
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                raise HTTPException(
                    status_code=400,
                    detail="Prompt contains potentially dangerous SQL patterns"
                )
        
        # Check for dangerous keywords
        for keyword in self.dangerous_keywords:
            if keyword in prompt_lower:
                raise HTTPException(
                    status_code=400,
                    detail="Prompt contains restricted keywords"
                )
        
        # Block common non-analytics greetings and casual conversation
        casual_patterns = [
            r'^\s*(hi|hello|hey|good\s+morning|good\s+afternoon|good\s+evening)\s*$',
            r'^\s*(how\s+are\s+you|how\s+do\s+you\s+do|what\'s\s+up|wassup)\s*\??$',
            r'^\s*(thank\s+you|thanks|bye|goodbye|see\s+you)\s*$',
            r'^\s*(yes|no|ok|okay|sure)\s*$',
            r'^\s*(who\s+are\s+you|what\s+is\s+your\s+name)\s*\??$'
        ]
        
        # Check if prompt is casual conversation
        for pattern in casual_patterns:
            if re.search(pattern, prompt_lower):
                raise HTTPException(
                    status_code=400,
                    detail="This is an analytics agent. Please ask questions about data, files, success rates, or request charts/reports."
                )
        
        # Check for analytics-related intent using multiple approaches
        analytics_patterns = [
            # File and data analysis patterns
            r'\b(show|display|get|find|analyze|report|list)\b.*\b(data|records?|files?|csv)\b',
            r'\b(success|fail|error|failure)\b.*\b(rate|percentage|ratio|count)\b',
            r'\b(chart|graph|visualization|dashboard)\b',
            r'\b(calculate|compute|sum|count|average|total)\b',
            r'\b(file|filename|csv)\b.*\b(status|records?|data)\b',
            # Specific to your system
            r'customer_sample_values\.csv',
            r'\b(master\s*data|tracker|header)\b',
            # General analytics terms
            r'\b(metrics|kpi|performance|statistics|insights|trends|analysis)\b'
        ]
        
        # Analytics question patterns (more specific)
        analytics_questions = [
            r'\b(what|how\s+many|how\s+much|which)\b.*\b(data|records?|files?|success|fail|rate|percentage)\b',
            r'\b(show\s+me|give\s+me|can\s+you)\b.*\b(chart|graph|data|records?|success\s*rate)\b'
        ]
        
        # Check for analytics intent
        has_analytics_pattern = any(re.search(pattern, prompt_lower) for pattern in analytics_patterns)
        has_analytics_question = any(re.search(pattern, prompt_lower) for pattern in analytics_questions)
        
        # For testing purposes, allow "test" messages to pass
        is_test_message = 'test' in prompt_lower
        
        # Combined validation - must have clear analytics intent
        has_analytics_intent = has_analytics_pattern or has_analytics_question or is_test_message
        
        if not has_analytics_intent:
            raise HTTPException(
                status_code=400,
                detail="This is an analytics agent. Please ask questions about data files, success rates, records, or request charts/reports. Example: 'Show me success rate for file customer_sample_values.csv'"
            )
        
        return {
            "valid": True,
            "message": "Prompt validation successful",
            "cleaned_prompt": prompt.strip()
        }

# Initialize validator
prompt_validator = PromptValidator()
