
import base64
import json
import logging
logging.basicConfig(level=logging.INFO)

from typing import Optional
from fastapi import FastAPI, Depends
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel
from app.auth import validate_jwt_token, bearer_scheme
from app.validator import prompt_validator
from app.llm_service import llm_service
from app.memory_service import memory_service
from app.communication_coordinator import communication_coordinator, ComponentStatus
from app.fallback_strategy import fallback_strategy, FallbackTrigger
from app.plan_executor import execute_plan, execute_tool_with_coordination
from app.agent import plan_and_execute

app = FastAPI()

# Logger setup
logger = logging.getLogger("analytics_agent")

class PromptRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None


@app.post("/query")
async def receive_prompt(
    request: PromptRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    session_id = None
    try:
        # Step 1: Validate JWT token and setup session
        user = validate_jwt_token(credentials)
        # Extract organization id from token if present (check common claim names and nested claims)
        org_id = user.get("orgId")
        

        # Log a minimal, non-sensitive view of the validated token for observability
        safe_user_log = {"sub": user.get("sub"), "org_id": org_id}
        logger.info("Authenticated user: %s", safe_user_log)
        user_id = user.get("sub", "anonymous")

        # Create or get session
        if request.session_id:
            session_id = request.session_id
        else:
            session_id = memory_service.create_session(user_id)

        # Get session context for reasoning
        session_context = memory_service.get_session_context(session_id)

        # Step 2: Validate user prompt
        validation_result = prompt_validator.validate_prompt(request.prompt)
        cleaned_prompt = validation_result["cleaned_prompt"]

        # Build a minimal workflow context to pass around
        workflow_context = {
            "user_prompt": cleaned_prompt,
            "session_id": session_id,
            "user_id": user_id,
            "org_id": org_id
        }

        # Try plan-first via agent wrapper
        agent_out = await plan_and_execute(cleaned_prompt, session_context, workflow_context)
        coordination_log = agent_out.get("coordination_log") if isinstance(agent_out, dict) else None

        if agent_out.get("success") and agent_out.get("execution_result"):
            execution_result = agent_out["execution_result"]
            memory_service.store_interaction(session_id, cleaned_prompt, "plan_execution", execution_result)
            execution_result["session_id"] = session_id
            execution_result["workflow_completed"] = True
            execution_result["coordination_log"] = coordination_log
            return execution_result

        # Fall back to LLM tool selection
        fallback_needed = bool(agent_out.get("fallback_needed", True))

        if fallback_needed:
            try:
                llm_result = llm_service.extractPrompt(cleaned_prompt)
                tool_calls = llm_result.get("tool_calls")

                # Update component status - LLM working
                communication_coordinator.update_component_status("llm_service", ComponentStatus.HEALTHY)

            except Exception as llm_error:
                # Handle LLM failure with fallback
                logger.error("LLM Service failed: %s", llm_error)
                communication_coordinator.handle_component_error("llm_service", llm_error, workflow_context)

                fallback_result = fallback_strategy.execute_fallback(FallbackTrigger.LLM_SERVICE_DOWN, workflow_context)

                if fallback_result.get("success"):
                    tool_calls = fallback_result.get("tool_calls")
                    llm_result = fallback_result
                else:
                    return {
                        "success": False,
                        "error": "LLM service unavailable and fallback failed",
                        "message": fallback_result.get("user_message", "Service temporarily unavailable"),
                        "session_id": session_id,
                        "fallback_triggered": True
                    }

        else:
            tool_calls = []

        logger.debug("tool_calls from LLM: %s", tool_calls)
        logger.debug("tool_calls type: %s", type(tool_calls))
        if tool_calls:
            logger.debug("First tool_call type: %s", type(tool_calls[0]))
            logger.debug("First tool_call content: %s", tool_calls[0])

        if not tool_calls:
            # Handle no tool calls with fallback
            fallback_result = fallback_strategy.execute_fallback(
                FallbackTrigger.TOOL_SELECTION_FAILED, workflow_context
            )

            # Store interaction in memory
            memory_service.store_interaction(session_id, cleaned_prompt, "none", fallback_result)

            return {
                "success": False,
                "error": "No tool call detected",
                "message": fallback_result.get("user_message", "Unable to understand your request"),
                "session_id": session_id,
                "suggestions": fallback_result.get("suggestions", []),
                "llm_result": llm_result
            }

        # Only handle the first tool call for now
        tool_call = tool_calls[0]

        # Normalize tool_call to (tool_name, tool_args)
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
        except Exception as tool_parse_error:
            logger.exception("Error parsing tool call: %s", tool_parse_error)
            return {
                "success": False,
                "error": f"Error parsing tool call: {str(tool_parse_error)}",
                "message": "Internal error processing tool selection",
                "session_id": session_id,
                "debug_info": {
                    "tool_call": str(tool_call),
                    "parse_error": str(tool_parse_error)
                }
            }

        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError as json_error:
                logger.error("JSON decode error in tool args: %s", json_error)
                return {
                    "success": False,
                    "error": f"Invalid JSON in tool arguments: {str(json_error)}",
                    "message": "Internal error processing tool parameters",
                    "session_id": session_id
                }

        # Step 6: Execute tools with error handling and coordination
        try:
            result = await execute_tool_with_coordination(tool_name, tool_args, workflow_context)

            # Store successful interaction in memory
            memory_service.store_interaction(session_id, cleaned_prompt, tool_name, result)

            # Add session context to response
            result["session_id"] = session_id
            result["workflow_completed"] = True
            result["coordination_log"] = coordination_log

            return result

        except Exception as tool_error:
            # Handle tool execution error
            logger.exception("Tool execution failed: %s", tool_error)

            error_context = {**workflow_context, "tool_name": tool_name, "tool_args": tool_args}
            coordination_response = communication_coordinator.handle_component_error(
                "database_service", tool_error, error_context
            )

            # Try fallback
            if "database" in str(tool_error).lower():
                fallback_result = fallback_strategy.execute_fallback(
                    FallbackTrigger.DATABASE_ERROR, error_context
                )
            else:
                fallback_result = {
                    "success": False,
                    "error": str(tool_error),
                    "message": "Tool execution failed"
                }

            # Store failed interaction in memory
            memory_service.store_interaction(session_id, cleaned_prompt, tool_name, fallback_result)

            fallback_result["session_id"] = session_id
            return fallback_result
    except Exception as e:
        error_context = {"session_id": session_id, "error": str(e)}

        # Store error in memory if session exists
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
