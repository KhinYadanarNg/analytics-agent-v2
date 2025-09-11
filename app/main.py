
import base64
import json
import logging
import time
logging.basicConfig(level=logging.INFO)

from typing import Optional
from fastapi import FastAPI, Depends, Request, Response
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel
from app.auth import validate_jwt_token, bearer_scheme
from app.validator import prompt_validator
from app.llm_service import llm_service
from app.memory_service import memory_service
from app.communication_coordinator import communication_coordinator, ComponentStatus
from app.fallback_strategy import fallback_strategy, FallbackTrigger
from app.plan_executor import execute_tool_with_coordination
from app.agent import plan_and_execute

# Enhanced security imports

app = FastAPI()

# Logger setup
logger = logging.getLogger("analytics_agent")

class PromptRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None


async def setup_session_and_context(request: PromptRequest, credentials: HTTPAuthorizationCredentials, http_request: Request):
    """Setup user session and workflow context with enhanced security."""
    
    # JWT validation
    user = validate_jwt_token(credentials)
    org_id = user.get("orgId")
    user_id = user.get("userId")
    
    # Session management with cookie support
    session_id = None
    
    # Priority order: 1. Request body, 2. Cookie, 3. Create new
    if request.session_id:
        session_id = request.session_id
        logger.info(f"Using session_id from request body: {session_id}")
    else:
        # Try to get session_id from cookie
        cookie_session_id = http_request.cookies.get("analytics_session_id")
        if cookie_session_id:
            session_id = cookie_session_id
            logger.info(f"Using session_id from cookie: {session_id}")
        else:
            # Create new session
            session_id = memory_service.create_session(user_id)
            logger.info(f"Created new session_id: {session_id}")
    
    # Ensure session exists in memory
    if session_id not in memory_service.sessions:
        session_id = memory_service.create_session(user_id)
        logger.info(f"Session not found, created new session_id: {session_id}")
    
    # Get session context from memory service
    session_context = memory_service.get_session_context(session_id)

    # Enhanced prompt validation
    try:
        validation_result = prompt_validator.validate_prompt(request.prompt)
        cleaned_prompt = validation_result["cleaned_prompt"]
    except Exception as validation_error:
        logger.warning(
            "Prompt validation failed for user %s: %s", 
            user_id, str(validation_error)
        )
        raise

    # Store file references from the original prompt
    import re
    file_pattern = r"['\"]([^'\"]*\.csv)['\"]|(\w+\.csv)"
    file_matches = re.findall(file_pattern, cleaned_prompt)
    
    for match in file_matches:
        file_name = match[0] or match[1]  # Get non-empty group
        if file_name:
            memory_service.store_file_reference(session_id, file_name)
            logger.info(f"Stored file reference: {file_name} for session {session_id}")

    # Resolve file references in the prompt
    resolved_prompt = memory_service.resolve_file_reference(session_id, cleaned_prompt)
    
    if resolved_prompt != cleaned_prompt:
        logger.info(f"Resolved prompt: '{cleaned_prompt}' -> '{resolved_prompt}'")
        cleaned_prompt = resolved_prompt

    # Log user activity (simplified)
    logger.info(
        "User activity: user_id=%s, session_id=%s, prompt_length=%d", 
        user_id, session_id[:8] if session_id else "none", len(cleaned_prompt)
    )

    workflow_context = {
        "user_prompt": cleaned_prompt,
        "session_id": session_id,
        "user_id": user_id,
        "org_id": org_id
    }
    
    return session_id, session_context, workflow_context, cleaned_prompt


async def try_llm_tool_selection(cleaned_prompt: str, workflow_context: dict):
    """Try LLM for tool selection, return tool_calls and result."""
    try:
        llm_result = llm_service.extractPrompt(cleaned_prompt)
        tool_calls = llm_result.get("tool_calls", [])
        
        # Update component status - LLM working
        communication_coordinator.update_component_status("llm_service", ComponentStatus.HEALTHY)
        
        return tool_calls, llm_result, None
        
    except Exception as llm_error:
        logger.error("LLM Service failed: %s", llm_error)
        communication_coordinator.handle_component_error("llm_service", llm_error, workflow_context)
        return [], None, llm_error


async def try_planner_fallback(cleaned_prompt: str, session_context: dict, workflow_context: dict):
    """Try planner as fallback when LLM fails."""
    try:
        agent_out = await plan_and_execute(cleaned_prompt, session_context, workflow_context)
        coordination_log = agent_out.get("coordination_log") if isinstance(agent_out, dict) else None

        if agent_out.get("success") and agent_out.get("execution_result"):
            return agent_out["execution_result"], coordination_log, None
            
        return None, coordination_log, "Planner did not return successful result"
        
    except Exception as planner_error:
        logger.error("Planner failed: %s", planner_error)
        return None, None, planner_error


async def handle_final_fallback(workflow_context: dict):
    """Handle final fallback when both LLM and planner fail."""
    fallback_result = fallback_strategy.execute_fallback(FallbackTrigger.LLM_SERVICE_DOWN, workflow_context)
    
    if fallback_result.get("success"):
        return fallback_result.get("tool_calls", []), fallback_result
    
    return [], {
        "success": False,
        "error": "Both LLM and planner services unavailable",
        "message": fallback_result.get("user_message", "Service temporarily unavailable"),
        "fallback_triggered": True
    }


def parse_and_validate_tool_call(tool_call):
    """Parse and validate a single tool call."""
    def _parse_tool_call(tc):
        if hasattr(tc, 'function'):
            return tc.function.name, tc.function.arguments
        if isinstance(tc, dict) and 'function' in tc:
            return tc['function']['name'], tc['function']['arguments']
        if isinstance(tc, dict):
            return tc.get('name', tc.get('tool_name', 'unknown')), tc.get('arguments', tc.get('args', {}))
        raise ValueError(f"Invalid tool_call format: {type(tc)}")

    try:
        tool_name, tool_args = _parse_tool_call(tool_call)
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


@app.post("/query")
async def receive_prompt(
    request: PromptRequest,
    http_request: Request,
    response: Response,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    session_id = None
    start_time = time.time()
    
    try:
        # Setup session and context with enhanced security
        session_id, session_context, workflow_context, cleaned_prompt = await setup_session_and_context(
            request, credentials, http_request
        )
        
        # Set session cookie for frontend
        response.set_cookie(
            key="analytics_session_id",
            value=session_id,
            max_age=3600,  # 1 hour
            httponly=True,
            secure=True,  # Use in production with HTTPS
            samesite="lax"
        )

        # Try LLM-first approach for tool selection
        tool_calls, llm_result, llm_error = await try_llm_tool_selection(cleaned_prompt, workflow_context)
        coordination_log = None

        # If LLM failed, try planner fallback
        if llm_error:
            logger.info("Falling back to planner due to LLM failure")
            execution_result, coordination_log, planner_error = await try_planner_fallback(
                cleaned_prompt, session_context, workflow_context
            )
            
            # If planner succeeded, return immediately
            if execution_result:
                memory_service.store_interaction(session_id, cleaned_prompt, "plan_execution", execution_result)
                execution_result["session_id"] = session_id
                execution_result["workflow_completed"] = True
                execution_result["coordination_log"] = coordination_log
                return execution_result
            
            # Both LLM and planner failed, try final fallback
            if planner_error:
                tool_calls, final_fallback = await handle_final_fallback(workflow_context)
                if not tool_calls:
                    final_fallback["session_id"] = session_id
                    return final_fallback
                llm_result = final_fallback

        # Handle case where no tool calls were generated
        if not tool_calls:
            fallback_result = fallback_strategy.execute_fallback(
                FallbackTrigger.TOOL_SELECTION_FAILED, workflow_context
            )
            memory_service.store_interaction(session_id, cleaned_prompt, "none", fallback_result)
            return {
                "success": False,
                "error": "No tool call detected",
                "message": fallback_result.get("user_message", "Unable to understand your request"),
                "session_id": session_id,
                "suggestions": fallback_result.get("suggestions", []),
                "llm_result": llm_result
            }

        # Parse and validate the first tool call
        tool_call = tool_calls[0]
        tool_name, tool_args, parse_error = parse_and_validate_tool_call(tool_call)
        
        if parse_error:
            parse_error["session_id"] = session_id
            return parse_error

        # Execute the tool
        try:
            result = await execute_tool_with_coordination(tool_name, tool_args, workflow_context)
            memory_service.store_interaction(session_id, cleaned_prompt, tool_name, result)
            
            result["session_id"] = session_id
            result["workflow_completed"] = True
            result["coordination_log"] = coordination_log
            return result

        except Exception as tool_error:
            logger.exception("Tool execution failed: %s", tool_error)
            
            error_context = {**workflow_context, "tool_name": tool_name, "tool_args": tool_args}
            communication_coordinator.handle_component_error("database_service", tool_error, error_context)

            # Try fallback based on error type
            trigger = FallbackTrigger.DATABASE_ERROR if "database" in str(tool_error).lower() else FallbackTrigger.TOOL_SELECTION_FAILED
            fallback_result = fallback_strategy.execute_fallback(trigger, error_context)
            
            memory_service.store_interaction(session_id, cleaned_prompt, tool_name, fallback_result)
            fallback_result["session_id"] = session_id
            return fallback_result

    except Exception as e:
        logger.exception("Unexpected error in receive_prompt: %s", e)
        error_context = {"session_id": session_id, "error": str(e)}

        if session_id:
            memory_service.store_interaction(session_id, request.prompt, "error", {"error": str(e)})

        return {
            "success": False,
            "error": str(e),
            "message": "An error occurred while processing your request. Please try again with a valid analytics question.",
            "session_id": session_id
        }





# Executor functions have been moved to `app.plan_executor` for separation of concerns.

@app.get("/health")
async def health_check():
    system_health = communication_coordinator.get_system_health()
    return {
        "status": "healthy" 
    }

# Debug endpoints for testing file reference resolution
@app.get("/debug/memory/{session_id}")
async def debug_memory(session_id: str):
    """Debug endpoint to check session memory"""
    return memory_service.get_session_context(session_id)

@app.post("/debug/resolve")
async def debug_resolve(request: dict):
    """Debug endpoint to test prompt resolution"""
    session_id = request.get("session_id")
    prompt = request.get("prompt")
    
    # Store a test file reference first if provided
    test_file = request.get("test_file")
    if test_file:
        memory_service.store_file_reference(session_id, test_file)
    
    resolved = memory_service.resolve_file_reference(session_id, prompt)
    return {
        "original": prompt,
        "resolved": resolved,
        "session_context": memory_service.get_session_context(session_id)
    }
