"""
Enhanced query coordinator that passes conversation history to analytics agent.
"""
import logging
import time
from typing import Dict, Any

from fastapi import Request, Response
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel

from app.analytics_agent import AnalyticsService
from app.memory_service import memory_service
from app.auth import validate_jwt_token
from app.request_context import current_org_id, current_user_id

logger = logging.getLogger("analytics_agent")

class PromptRequest(BaseModel):
    prompt: str
    session_id: str = None

class QueryCoordinator:
    """Enhanced coordinator that provides conversation context to analytics service."""
    
    async def process_query(self, request: PromptRequest, http_request: Request,
                          response: Response, credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
        """Process analytics query with session management and conversation context."""
        session_id = None
        
        try:
            # JWT validation and session setup
            user = validate_jwt_token(credentials)
            org_id = user.get("orgId")
            user_id = user.get("sub")
            
            # Resolve session ID
            if request.session_id:
                session_id = request.session_id
            else:
                cookie_session_id = http_request.cookies.get("analytics_session_id")
                session_id = cookie_session_id if cookie_session_id else memory_service.create_session(user_id)
            
            # Ensure session exists
            if session_id not in memory_service.sessions:
                session_id = memory_service.create_session(user_id)
            
            # Get session context and conversation history
            session_context = memory_service.get_session_context(session_id)
            conversation_history = memory_service.get_conversation_history(session_id)
            
            # Resolve file references like "this file", "that file" to previous file
            resolved_prompt = memory_service.resolve_file_reference(session_id, request.prompt)
            
            # Store file references from the prompt
            import re
            file_pattern = r"['\"]([^'\"]*\.csv)['\"]|(\w+\.csv)"
            file_matches = re.findall(file_pattern, resolved_prompt)
            
            for match in file_matches:
                file_name = match[0] or match[1]  # Get non-empty group
                if file_name:
                    memory_service.store_file_reference(session_id, file_name)
                    logger.info(f"Stored file reference: {file_name} for session {session_id[:8]}")
            
            # Set session cookie
            response.set_cookie(
                key="analytics_session_id",
                value=session_id,
                max_age=3600,
                httponly=True,
                secure=True,
                samesite="lax"
            )
            
            # Set per-request context so tools can access org/user info when the LLM doesn't include them
            if org_id:
                current_org_id.set(org_id)
            if user_id:
                current_user_id.set(user_id)

            # Process analytics query with conversation history
            result = await AnalyticsService.process_query(
                prompt=resolved_prompt,
                session_id=session_id,
                conversation_history=conversation_history
            )
            
            # Store interaction and add session info
            memory_service.store_interaction(
                session_id, 
                resolved_prompt, 
                result.get("tool_name", "unknown"), 
                result
            )
            
            #result["session_id"] = session_id
            
            # Log conversation flow for debugging
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Session {session_id[:8]} - Original prompt: '{request.prompt}'")
                logger.debug(f"Session {session_id[:8]} - Resolved prompt: '{resolved_prompt}'")
                logger.debug(f"Session {session_id[:8]} - Report type: {result.get('report_type', 'unknown')}")
                
            return result
            
        except Exception as error:
            logger.exception(f"Query processing failed: {error}")
            
            if session_id:
                memory_service.store_interaction(session_id, request.prompt, "error", {"error": str(error)})
            
            return {
                "success": False,
                "error": str(error),
                "message": "An error occurred while processing your request."
            }