"""
Analytics query processing service.
"""
import json
import logging
from typing import Tuple, Optional, Dict, Any, List

from app.llm_service import llm_service
from app.plan_executor import execute_tool_with_coordination
from app.agent import plan_and_execute
from app.memory_service import memory_service

logger = logging.getLogger("analytics_agent")

class AnalyticsService:
    """Handles LLM tool selection and execution."""
    
    @staticmethod
    async def process_query(prompt: str, session_context: Dict[str, Any], 
                          workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process analytics query through LLM or planner."""
        
        # Try LLM first
        tool_calls, llm_result, llm_error = await AnalyticsService._try_llm(prompt, workflow_context)
        
        if llm_error:
            # Fallback to planner
            logger.info("LLM failed, trying planner fallback")
            result = await AnalyticsService._try_planner(prompt, session_context, workflow_context)
            if result:
                return result
        
        # Handle LLM validation failures
        if llm_result and llm_result.get("success") == False and "message" in llm_result:
            return {
                "success": False,
                "message": llm_result["message"]
            }
        
        # Handle no tool calls
        if not tool_calls:
            return {
                "success": False,
                "error": "No tool call detected",
                "message": "I couldn't understand your request. Please try rephrasing your question about the data analysis."
            }
        
        # Execute tool
        return await AnalyticsService._execute_tool(tool_calls[0], workflow_context)
    
    @staticmethod
    async def _try_llm(prompt: str, workflow_context: Dict[str, Any]) -> Tuple[List, Dict, Optional[Exception]]:
        """Try LLM for tool selection."""
        try:
            llm_result = llm_service.extractPrompt(prompt)
            tool_calls = llm_result.get("tool_calls", [])
            logger.info("LLM service call successful")
            return tool_calls, llm_result, None
        except Exception as error:
            logger.error(f"LLM service failed: {error}")
            return [], None, error
    
    @staticmethod
    async def _try_planner(prompt: str, session_context: Dict[str, Any], 
                         workflow_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Try planner as fallback."""
        try:
            agent_out = await plan_and_execute(prompt, session_context, workflow_context)
            if agent_out.get("success") and agent_out.get("execution_result"):
                result = agent_out["execution_result"]
                result["coordination_log"] = agent_out.get("coordination_log")
                return result
        except Exception as error:
            logger.error(f"Planner failed: {error}")
        return None
    
    @staticmethod
    async def _execute_tool(tool_call, workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call."""
        try:
            tool_name, tool_args = AnalyticsService._parse_tool_call(tool_call)
            result = await execute_tool_with_coordination(tool_name, tool_args, workflow_context)
            result["workflow_completed"] = True
            return result
        except Exception as error:
            logger.exception(f"Tool execution failed: {error}")
            return {
                "success": False,
                "error": str(error),
                "message": "Database operation failed. Please try again later."
            }
    
    @staticmethod
    def _parse_tool_call(tool_call) -> Tuple[str, Dict[str, Any]]:
        """Parse and validate a tool call."""
        if hasattr(tool_call, 'function'):
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
        elif isinstance(tool_call, dict) and 'function' in tool_call:
            tool_name = tool_call['function']['name']
            tool_args = tool_call['function']['arguments']
        elif isinstance(tool_call, dict):
            tool_name = tool_call.get('name', tool_call.get('tool_name', 'unknown'))
            tool_args = tool_call.get('arguments', tool_call.get('args', {}))
        else:
            raise ValueError(f"Invalid tool_call format: {type(tool_call)}")
        
        if isinstance(tool_args, str):
            tool_args = json.loads(tool_args)
        
        logger.info(f"Parsed tool: {tool_name} with args: {tool_args}")
        return tool_name, tool_args
