"""
Request handlers for the analytics agent.
Separated from main.py to improve maintainability and testability.
"""
import json
import logging
import time
from typing import Tuple, Optional, Dict, Any, List

from fastapi import Request, Response
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel

from app.auth import validate_jwt_token
from app.memory_service import memory_service
from app.llm_service import llm_service
from app.plan_executor import execute_tool_with_coordination
from app.agent import plan_and_execute

logger = logging.getLogger("analytics_agent")

class PromptRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None

class SessionManager:
    """Handles session creation, validation, and context management."""
    
    @staticmethod
    async def setup_session(request: PromptRequest, credentials: HTTPAuthorizationCredentials, 
                           http_request: Request) -> Tuple[str, Dict[str, Any], Dict[str, Any], str]:
        """Setup user session and workflow context."""
        
        # JWT validation
        user = validate_jwt_token(credentials)
        org_id = user.get("orgId")
        user_id = user.get("userId")
        
        # Session resolution
        session_id = SessionManager._resolve_session_id(request, http_request, user_id)
        
        # Get session context
        session_context = memory_service.get_session_context(session_id)
        
        # Process and resolve file references in prompt
        cleaned_prompt = SessionManager._process_prompt(request.prompt, session_id)
        
        # Create workflow context
        workflow_context = {
            "user_prompt": cleaned_prompt,
            "session_id": session_id,
            "user_id": user_id,
            "org_id": org_id
        }
        
        # Log activity
        logger.info(
            "User activity: user_id=%s, session_id=%s, prompt_length=%d", 
            user_id, session_id[:8], len(cleaned_prompt)
        )
        
        return session_id, session_context, workflow_context, cleaned_prompt
    
    @staticmethod
    def _resolve_session_id(request: PromptRequest, http_request: Request, user_id: str) -> str:
        """Resolve session ID from request body, cookie, or create new."""
        
        # Priority: 1. Request body, 2. Cookie, 3. Create new
        if request.session_id:
            session_id = request.session_id
            logger.info(f"Using session_id from request body: {session_id}")
        else:
            cookie_session_id = http_request.cookies.get("analytics_session_id")
            if cookie_session_id:
                session_id = cookie_session_id
                logger.info(f"Using session_id from cookie: {session_id}")
            else:
                session_id = memory_service.create_session(user_id)
                logger.info(f"Created new session_id: {session_id}")
        
        # Ensure session exists
        if session_id not in memory_service.sessions:
            session_id = memory_service.create_session(user_id)
            logger.info(f"Session not found, created new session_id: {session_id}")
        
        return session_id
    
    @staticmethod
    def _process_prompt(prompt: str, session_id: str) -> str:
        """Extract file references and resolve file context in prompt."""
        
        # Store file references
        import re
        file_pattern = r"['\"]([^'\"]*\.csv)['\"]|(\w+\.csv)"
        file_matches = re.findall(file_pattern, prompt)
        
        for match in file_matches:
            file_name = match[0] or match[1]
            if file_name:
                memory_service.store_file_reference(session_id, file_name)
                logger.info(f"Stored file reference: {file_name} for session {session_id}")
        
        # Resolve file references
        resolved_prompt = memory_service.resolve_file_reference(session_id, prompt)
        
        if resolved_prompt != prompt:
            logger.info(f"Resolved prompt: '{prompt}' -> '{resolved_prompt}'")
        
        return resolved_prompt
    
    @staticmethod
    def set_session_cookie(response: Response, session_id: str) -> None:
        """Set session cookie for frontend."""
        response.set_cookie(
            key="analytics_session_id",
            value=session_id,
            max_age=3600,  # 1 hour
            httponly=True,
            secure=True,  # Use in production with HTTPS
            samesite="lax"
        )

class ToolProcessor:
    """Handles LLM tool selection and execution."""
    
    @staticmethod
    async def try_llm_tool_selection(prompt: str, workflow_context: Dict[str, Any]) -> Tuple[List, Dict, Optional[Exception]]:
        """Try LLM for tool selection."""
        try:
            llm_result = llm_service.extractPrompt(prompt)
            tool_calls = llm_result.get("tool_calls", [])
            logger.info("LLM Service call successful")
            return tool_calls, llm_result, None
        except Exception as llm_error:
            logger.error("LLM Service failed: %s", llm_error)
            return [], None, llm_error
    
    @staticmethod
    async def try_planner_fallback(prompt: str, session_context: Dict[str, Any], 
                                 workflow_context: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[Dict], Optional[str]]:
        """Try planner as fallback when LLM fails."""
        try:
            agent_out = await plan_and_execute(prompt, session_context, workflow_context)
            coordination_log = agent_out.get("coordination_log") if isinstance(agent_out, dict) else None

            if agent_out.get("success") and agent_out.get("execution_result"):
                return agent_out["execution_result"], coordination_log, None
                
            return None, coordination_log, "Planner did not return successful result"
        except Exception as planner_error:
            logger.error("Planner failed: %s", planner_error)
            return None, None, str(planner_error)
    
    @staticmethod
    def parse_tool_call(tool_call) -> Tuple[Optional[str], Optional[Dict], Optional[Dict]]:
        """Parse and validate a single tool call."""
        
        def _extract_tool_info(tc):
            if hasattr(tc, 'function'):
                return tc.function.name, tc.function.arguments
            if isinstance(tc, dict) and 'function' in tc:
                return tc['function']['name'], tc['function']['arguments']
            if isinstance(tc, dict):
                return tc.get('name', tc.get('tool_name', 'unknown')), tc.get('arguments', tc.get('args', {}))
            raise ValueError(f"Invalid tool_call format: {type(tc)}")

        try:
            tool_name, tool_args = _extract_tool_info(tool_call)
            logger.info("Extracted tool: %s with args: %s", tool_name, tool_args)
            
            if isinstance(tool_args, str):
                tool_args = json.loads(tool_args)
                
            return tool_name, tool_args, None
        except Exception as error:
            logger.exception("Error parsing tool call: %s", error)
            return None, None, {
                "success": False,
                "error": f"Error parsing tool call: {str(error)}",
                "message": "Internal error processing tool selection",
                "debug_info": {
                    "tool_call": str(tool_call),
                    "parse_error": str(error)
                }
            }

class ResponseBuilder:
    """Builds standardized API responses."""
    
    @staticmethod
    def success_response(session_id: str, result: Dict[str, Any], coordination_log: Optional[Dict] = None) -> Dict[str, Any]:
        """Build success response."""
        result["session_id"] = session_id
        result["workflow_completed"] = True
        if coordination_log:
            result["coordination_log"] = coordination_log
        return result
    
    @staticmethod
    def error_response(session_id: str, error: str, message: str, **kwargs) -> Dict[str, Any]:
        """Build error response."""
        return {
            "success": False,
            "error": error,
            "message": message,
            "session_id": session_id,
            **kwargs
        }
    
    @staticmethod
    def validation_error_response(session_id: str, message: str) -> Dict[str, Any]:
        """Build validation error response."""
        return {
            "success": False,
            "message": message,
            "session_id": session_id
        }
    
    @staticmethod
    def fallback_response(session_id: str) -> Dict[str, Any]:
        """Build fallback response when all services fail."""
        return {
            "success": False,
            "error": "Both LLM and planner services unavailable",
            "message": "Our analytics service is currently experiencing issues. Please try again later.",
            "fallback_triggered": True,
            "session_id": session_id
        }

class QueryHandler:
    """Main query processing orchestrator."""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.tool_processor = ToolProcessor()
        self.response_builder = ResponseBuilder()
    
    async def process_query(self, request: PromptRequest, http_request: Request, 
                          response: Response, credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
        """Process analytics query with proper error handling and fallbacks."""
        
        session_id = None
        start_time = time.time()
        
        try:
            # Setup session and context
            session_id, session_context, workflow_context, cleaned_prompt = await self.session_manager.setup_session(
                request, credentials, http_request
            )
            
            # Set session cookie
            self.session_manager.set_session_cookie(response, session_id)
            
            # Try LLM first
            tool_calls, llm_result, llm_error = await self.tool_processor.try_llm_tool_selection(
                cleaned_prompt, workflow_context
            )
            
            coordination_log = None
            
            # Handle LLM failure with planner fallback
            if llm_error:
                return await self._handle_llm_failure(
                    session_id, cleaned_prompt, session_context, workflow_context
                )
            
            # Handle LLM validation rejection
            if llm_result and llm_result.get("success") == False and "message" in llm_result:
                return self._handle_validation_rejection(session_id, cleaned_prompt, llm_result)
            
            # Handle no tool calls
            if not tool_calls:
                return self._handle_no_tool_calls(session_id, cleaned_prompt)
            
            # Process tool execution
            return await self._execute_tool(session_id, cleaned_prompt, tool_calls[0], workflow_context)
            
        except Exception as e:
            logger.exception("Unexpected error in process_query: %s", e)
            if session_id:
                memory_service.store_interaction(session_id, request.prompt, "error", {"error": str(e)})
            
            return self.response_builder.error_response(
                session_id or "unknown",
                str(e),
                "An error occurred while processing your request. Please try again with a valid analytics question."
            )
    
    async def _handle_llm_failure(self, session_id: str, prompt: str, session_context: Dict, 
                                workflow_context: Dict) -> Dict[str, Any]:
        """Handle LLM failure with planner fallback."""
        logger.info("Falling back to planner due to LLM failure")
        
        execution_result, coordination_log, planner_error = await self.tool_processor.try_planner_fallback(
            prompt, session_context, workflow_context
        )
        
        if execution_result:
            memory_service.store_interaction(session_id, prompt, "plan_execution", execution_result)
            return self.response_builder.success_response(session_id, execution_result, coordination_log)
        
        if planner_error:
            return self.response_builder.fallback_response(session_id)
        
        return self.response_builder.fallback_response(session_id)
    
    def _handle_validation_rejection(self, session_id: str, prompt: str, llm_result: Dict) -> Dict[str, Any]:
        """Handle LLM validation rejection."""
        logger.info("LLM rejected non-analytics request: %s", llm_result.get("message"))
        memory_service.store_interaction(session_id, prompt, "validation_rejected", llm_result)
        return self.response_builder.validation_error_response(session_id, llm_result["message"])
    
    def _handle_no_tool_calls(self, session_id: str, prompt: str) -> Dict[str, Any]:
        """Handle case where no tool calls were generated."""
        memory_service.store_interaction(session_id, prompt, "none", {"error": "No tool calls generated"})
        return self.response_builder.error_response(
            session_id,
            "No tool call detected",
            "I couldn't understand your request. Please try rephrasing your question about the data analysis.",
            workflow_completed=True
        )
    
    async def _execute_tool(self, session_id: str, prompt: str, tool_call, workflow_context: Dict) -> Dict[str, Any]:
        """Execute the selected tool."""
        # Parse tool call
        tool_name, tool_args, parse_error = self.tool_processor.parse_tool_call(tool_call)
        
        if parse_error:
            parse_error["session_id"] = session_id
            return parse_error
        
        # Execute tool
        try:
            result = await execute_tool_with_coordination(tool_name, tool_args, workflow_context)
            memory_service.store_interaction(session_id, prompt, tool_name, result)
            return self.response_builder.success_response(session_id, result)
            
        except Exception as tool_error:
            logger.exception("Tool execution failed: %s", tool_error)
            
            error_result = {
                "success": False,
                "error": str(tool_error),
                "message": "Database operation failed. Please try again later.",
                "tool_name": tool_name
            }
            
            memory_service.store_interaction(session_id, prompt, tool_name, error_result)
            return self.response_builder.success_response(session_id, error_result)

# Create global query handler instance
query_handler = QueryHandler()
