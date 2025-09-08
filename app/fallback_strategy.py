from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import re

class FallbackTrigger(Enum):
    LLM_SERVICE_DOWN = "llm_service_down"
    DATABASE_ERROR = "database_error"
    CHART_GENERATION_FAILED = "chart_generation_failed"
    TOOL_SELECTION_FAILED = "tool_selection_failed"
    VALIDATION_FAILED = "validation_failed"
    UNKNOWN_QUERY = "unknown_query"

class FallbackStrategy:
    """
    Comprehensive fallback strategies for the analytics agent.
    Provides graceful degradation and alternative execution paths.
    """
    
    def __init__(self):
        self.fallback_rules = self._initialize_fallback_rules()
        self.pattern_based_tools = self._initialize_pattern_tools()
        self.mock_responses = self._initialize_mock_responses()
        
    def _initialize_fallback_rules(self) -> Dict[FallbackTrigger, Dict[str, Any]]:
        """Initialize fallback rules for different failure scenarios."""
        return {
            FallbackTrigger.LLM_SERVICE_DOWN: {
                "strategy": "pattern_matching",
                "fallback_method": self._pattern_based_tool_selection,
                "degraded_mode": True,
                "user_message": "Using simplified mode. Please be specific with your requests."
            },
            FallbackTrigger.DATABASE_ERROR: {
                "strategy": "mock_data",
                "fallback_method": self._provide_mock_data,
                "degraded_mode": True,
                "user_message": "Database temporarily unavailable. Showing sample data."
            },
            FallbackTrigger.CHART_GENERATION_FAILED: {
                "strategy": "data_only",
                "fallback_method": self._provide_data_without_chart,
                "degraded_mode": True,
                "user_message": "Chart generation failed. Returning data in table format."
            },
            FallbackTrigger.TOOL_SELECTION_FAILED: {
                "strategy": "guided_selection",
                "fallback_method": self._guided_tool_selection,
                "degraded_mode": False,
                "user_message": "Please clarify your request with specific examples."
            },
            FallbackTrigger.VALIDATION_FAILED: {
                "strategy": "safe_mode",
                "fallback_method": self._safe_mode_response,
                "degraded_mode": True,
                "user_message": "Request blocked for security. Please rephrase your analytics question."
            },
            FallbackTrigger.UNKNOWN_QUERY: {
                "strategy": "suggestion_mode",
                "fallback_method": self._provide_suggestions,
                "degraded_mode": False,
                "user_message": "I can help with analytics. Here are some examples:"
            }
        }
    
    def _initialize_pattern_tools(self) -> Dict[str, Dict[str, Any]]:
        """Initialize pattern-based tool matching for LLM fallback."""
        return {
            "success_rate_patterns": {
                "patterns": [
                    r'\b(success|failure)\s+(rate|percentage)\b',
                    r'\b(chart|graph)\b.*\b(success|fail)\b',
                    r'\bshow\b.*\b(performance|analytics)\b'
                ],
                "tool": "get_success_rate_by_file_name",
                "confidence": 0.8
            },
            "data_retrieval_patterns": {
                "patterns": [
                    r'\b(show|get|list|find)\b.*\b(records?|data)\b',
                    r'\b(success|fail)\b.*\b(records?|entries)\b',
                    r'\bstatus\b.*\b(data|records?)\b'
                ],
                "tool": "get_records_by_status",
                "confidence": 0.7
            }
        }
    
    def _initialize_mock_responses(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mock responses for database fallback."""
        return {
            "get_records_by_status": {
                "success": True,
                "data": [
                    {"id": "sample_1", "status": "success", "file_id": "mock_file"},
                    {"id": "sample_2", "status": "fail", "file_id": "mock_file"}
                ],
                "row_count": 2,
                "mock_data": True,
                "message": "This is sample data. Database connection unavailable."
            },
            "get_success_rate_by_file_name": {
                "success": True,
                "chart_data": [
                    {"status": "success", "percentage": 75.0, "count": 150},
                    {"status": "fail", "percentage": 25.0, "count": 50}
                ],
                "row_count": 200,
                "mock_data": True,
                "message": "This is sample data. Database connection unavailable."
            }
        }
    
    def execute_fallback(self, trigger: FallbackTrigger, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute appropriate fallback strategy based on trigger."""
        if trigger not in self.fallback_rules:
            return self._default_fallback(context)
        
        rule = self.fallback_rules[trigger]
        fallback_method = rule["fallback_method"]
        
        try:
            result = fallback_method(context)
            result.update({
                "fallback_triggered": True,
                "fallback_strategy": rule["strategy"],
                "degraded_mode": rule["degraded_mode"],
                "user_message": rule["user_message"]
            })
            return result
        except Exception as e:
            return self._emergency_fallback(trigger, str(e), context)
    
    def _pattern_based_tool_selection(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback tool selection using pattern matching."""
        user_prompt = context.get("user_prompt", "").lower()
        
        best_match = None
        best_confidence = 0.0
        
        for tool_category, tool_info in self.pattern_based_tools.items():
            for pattern in tool_info["patterns"]:
                if re.search(pattern, user_prompt, re.IGNORECASE):
                    if tool_info["confidence"] > best_confidence:
                        best_confidence = tool_info["confidence"]
                        best_match = tool_info
        
        if best_match:
            # Extract file name if possible
            file_name = self._extract_file_name_simple(user_prompt)
            status = self._extract_status_simple(user_prompt)
            
            return {
                "success": True,
                "tool_calls": [{
                    "function": {
                        "name": best_match["tool"],
                        "arguments": {
                            "file_name": file_name or "customer_sample_values.csv",
                            "status": status
                        }
                    }
                }],
                "confidence": best_confidence,
                "method": "pattern_matching"
            }
        
        return {
            "success": False,
            "error": "No matching patterns found",
            "suggestions": self._get_usage_suggestions()
        }
    
    def _provide_mock_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide mock data when database is unavailable."""
        # Try to determine which tool was intended
        user_prompt = context.get("user_prompt", "").lower()
        
        if "chart" in user_prompt or "rate" in user_prompt:
            mock_response = self.mock_responses["get_success_rate_by_file_name"].copy()
        else:
            mock_response = self.mock_responses["get_records_by_status"].copy()
        
        return mock_response
    
    def _provide_data_without_chart(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide data without chart when chart generation fails."""
        return {
            "success": True,
            "chart_data": context.get("chart_data", []),
            "chart_image_base64": None,
            "chart_generation_failed": True,
            "alternative_format": "table",
            "message": "Chart generation unavailable. Data provided in table format."
        }
    
    def _guided_tool_selection(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide guided tool selection when automatic selection fails."""
        return {
            "success": False,
            "error": "Could not determine appropriate tool",
            "guided_options": [
                {
                    "option": "Data Retrieval",
                    "description": "Get records by status",
                    "example": "Show me success records for customer_sample_values.csv",
                    "tool": "get_records_by_status"
                },
                {
                    "option": "Success Rate Analysis",
                    "description": "Calculate success/fail rates with chart",
                    "example": "Show me success rate for customer_sample_values.csv",
                    "tool": "get_success_rate_by_file_name"
                }
            ],
            "message": "Please specify your request using one of the examples above."
        }
    
    def _safe_mode_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide safe mode response for validation failures."""
        return {
            "success": False,
            "error": "Request blocked by security validation",
            "safe_mode": True,
            "allowed_operations": [
                "Analytics queries about data files",
                "Success rate calculations",
                "Chart generation requests",
                "Record retrieval by status"
            ],
            "examples": [
                "Show me success rate for customer_sample_values.csv",
                "Get success records for my_file.csv",
                "Create a chart for data analysis"
            ]
        }
    
    def _provide_suggestions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide suggestions for unknown queries."""
        return {
            "success": False,
            "error": "Query not understood",
            "suggestions": self._get_usage_suggestions(),
            "examples": [
                "Show me success rate for customer_sample_values.csv",
                "Get fail records for data_file.csv", 
                "Create a success/fail chart for my_data.csv"
            ],
            "capabilities": [
                "Retrieve records by file and status",
                "Calculate success/failure rates",
                "Generate charts and visualizations",
                "Analyze data file performance"
            ]
        }
    
    def _extract_file_name_simple(self, prompt: str) -> Optional[str]:
        """Simple file name extraction for fallback."""
        patterns = [r'(\w+\.csv)', r'(\w+\.\w+)']
        for pattern in patterns:
            match = re.search(pattern, prompt)
            if match:
                return match.group(1)
        return None
    
    def _extract_status_simple(self, prompt: str) -> Optional[str]:
        """Simple status extraction for fallback."""
        if re.search(r'\b(success|successful)\b', prompt, re.IGNORECASE):
            return "success"
        elif re.search(r'\b(fail|failed|failure)\b', prompt, re.IGNORECASE):
            return "fail"
        return None
    
    def _get_usage_suggestions(self) -> List[str]:
        """Get general usage suggestions."""
        return [
            "Specify a file name (e.g., 'customer_sample_values.csv')",
            "Ask for success rates, charts, or record retrieval",
            "Use clear terms like 'show', 'get', 'analyze', or 'chart'",
            "Include status filters like 'success' or 'fail' if needed"
        ]
    
    def _default_fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Default fallback for unknown triggers."""
        return {
            "success": False,
            "error": "Service temporarily unavailable",
            "fallback_triggered": True,
            "fallback_strategy": "default",
            "message": "Please try again in a moment or contact support if the issue persists."
        }
    
    def _emergency_fallback(self, trigger: FallbackTrigger, error: str, 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Emergency fallback when primary fallback fails."""
        return {
            "success": False,
            "error": f"Emergency fallback activated due to {trigger.value}",
            "original_error": error,
            "emergency_mode": True,
            "message": "System is experiencing issues. Basic functionality may be limited.",
            "support_contact": "Please contact system administrator"
        }

# Initialize fallback strategy
fallback_strategy = FallbackStrategy()
