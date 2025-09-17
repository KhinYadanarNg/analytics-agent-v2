"""
Enhanced query coordinator that passes conversation history to analytics agent.
Optimized for performance and maintainability.
"""
import logging
import re
from typing import Dict, Any, Optional, Tuple

from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, ValidationError

from app.core.analytics_service import AnalyticsService
from app.services.memory_service import memory_service
from app.auth import validate_jwt_token
from app.utils.request_context import set_current_org_id, reset_current_org_id, get_current_org_id

# Pre-compile regex patterns for better performance
FILE_PATTERN = re.compile(r"['\"]([^'\"]*\.csv)['\"]|(\w+\.csv)")

# Constants
SESSION_COOKIE_MAX_AGE = 3600
SESSION_ID_PREFIX_LENGTH = 8

logger = logging.getLogger("analytics_agent")

class PromptRequest(BaseModel):
    prompt: str
    session_id: str | None = None

class QueryCoordinator:
    """Enhanced coordinator that provides conversation context to analytics service."""

    def __init__(self):
        """Initialize the coordinator with optimized settings."""
        self._analytics_service = AnalyticsService()

    async def process_query(
        self,
        request: PromptRequest,
        http_request: Request,
        response: Response,
        credentials: HTTPAuthorizationCredentials
    ) -> Dict[str, Any]:
        """Process analytics query with session management and conversation context."""

        session_id = None
        context_token = None

        try:
            # 1) JWT validation and org_id extraction
            user = validate_jwt_token(credentials)
            user_id = user.get("sub")
            org_id = user.get("orgId")

            # 2) Validate and ensure context consistency
            context_token = self._ensure_context_consistency(org_id)

            # 3) Resolve and validate session
            session_id = await self._resolve_session_id(request, http_request, user_id)

            # 4) Get session data efficiently
            session_data = await self._get_session_data(session_id)

            # 5) Process prompt and extract file references
            processed_prompt = self._process_prompt(session_id, request.prompt, session_data)

            # 6) Set session cookie
            self._set_session_cookie(response, session_id)

            # 7) Execute analytics query
            result = await self._execute_analytics_query(
                processed_prompt, session_id, session_data["conversation_history"], org_id
            )

            # 8) Store interaction result
            await self._store_interaction(session_id, processed_prompt, result)

            # 9) Log debug information if enabled
            self._log_debug_info(session_id, request.prompt, processed_prompt, result)

            return result

        except (ValidationError, ValueError) as e:
            logger.warning(f"Validation error: {e}")
            return self._create_error_response("Invalid request data", str(e))

        except HTTPException:
            # Re-raise HTTP exceptions (like auth failures)
            raise

        except Exception as error:
            logger.exception(f"Query processing failed: {error}")
            if session_id:
                await self._store_error_interaction(session_id, request.prompt, error)
            return self._create_error_response("Processing failed", str(error))

        finally:
            # Cleanup context
            self._cleanup_context(context_token)

    def _ensure_context_consistency(self, org_id: Optional[str]) -> Optional[object]:
        """Ensure the request context has the correct org_id."""
        if not org_id:
            return None

        current_org_id = get_current_org_id()

        if current_org_id is None or current_org_id != org_id:
            logger.warning(f"Context org_id mismatch: context={current_org_id}, jwt={org_id}")
            return set_current_org_id(org_id)
        return None

    async def _resolve_session_id(
        self,
        request: PromptRequest,
        http_request: Request,
        user_id: str
    ) -> str:
        """Resolve session ID from request or create new one."""
        # Check provided session_id first
        if request.session_id:
            session_id = request.session_id
        else:
            # Check cookie
            cookie_session_id = http_request.cookies.get("analytics_session_id")
            session_id = cookie_session_id or memory_service.create_session(user_id)

        # Ensure session exists
        if session_id not in memory_service.sessions:
            session_id = memory_service.create_session(user_id)

        return session_id

    async def _get_session_data(self, session_id: str) -> Dict[str, Any]:
        """Get session context and conversation history efficiently."""
        # Batch these calls if memory_service supports it
        session_context = memory_service.get_session_context(session_id)
        conversation_history = memory_service.get_conversation_history(session_id)

        return {
            "context": session_context,
            "conversation_history": conversation_history
        }

    def _process_prompt(
        self,
        session_id: str,
        original_prompt: str,
        session_data: Dict[str, Any]
    ) -> str:
        """Process prompt with file reference resolution and extraction."""
        # Resolve file references
        resolved_prompt = memory_service.resolve_file_reference(session_id, original_prompt)

        # Extract and store file references
        self._extract_file_references(session_id, resolved_prompt)

        return resolved_prompt

    def _extract_file_references(self, session_id: str, prompt: str) -> None:
        """Extract and store file references from prompt."""
        for match in FILE_PATTERN.finditer(prompt):
            file_name = match.group(1) or match.group(2)
            if file_name:
                memory_service.store_file_reference(session_id, file_name)
                logger.info(f"Stored file reference: {file_name} for session {session_id[:SESSION_ID_PREFIX_LENGTH]}")

    def _set_session_cookie(self, response: Response, session_id: str) -> None:
        """Set session cookie with optimized settings."""
        response.set_cookie(
            key="analytics_session_id",
            value=session_id,
            max_age=SESSION_COOKIE_MAX_AGE,
            httponly=True,
            secure=True,
            samesite="lax",
        )

    async def _execute_analytics_query(
        self,
        prompt: str,
        session_id: str,
        conversation_history: list,
        org_id: Optional[str]
    ) -> Dict[str, Any]:
        """Execute the analytics query with error handling."""
        return await self._analytics_service.process_query(
            prompt=prompt,
            session_id=session_id,
            conversation_history=conversation_history,
            org_id=org_id
        )

    async def _store_interaction(
        self,
        session_id: str,
        prompt: str,
        result: Dict[str, Any]
    ) -> None:
        """Store successful interaction in memory service."""
        tool_name = result.get("tool_name", "unknown")
        memory_service.store_interaction(session_id, prompt, tool_name, result)

    async def _store_error_interaction(
        self,
        session_id: str,
        prompt: str,
        error: Exception
    ) -> None:
        """Store error interaction in memory service."""
        memory_service.store_interaction(session_id, prompt, "error", {"error": str(error)})

    def _log_debug_info(
        self,
        session_id: str,
        original_prompt: str,
        resolved_prompt: str,
        result: Dict[str, Any]
    ) -> None:
        """Log debug information if debug logging is enabled."""
        if logger.isEnabledFor(logging.DEBUG):
            session_prefix = session_id[:SESSION_ID_PREFIX_LENGTH]
            logger.debug(f"Session {session_prefix} - Original prompt: '{original_prompt}'")
            logger.debug(f"Session {session_prefix} - Resolved prompt: '{resolved_prompt}'")
            logger.debug(f"Session {session_prefix} - Report type: {result.get('report_type', 'unknown')}")

    def _create_error_response(self, error_type: str, details: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "success": False,
            "error": error_type,
            "message": f"An error occurred: {details}",
        }

    def _cleanup_context(self, token: Optional[object]) -> None:
        """Clean up request context safely."""
        if token is not None:
            try:
                reset_current_org_id(token)
            except Exception:
                logger.debug("Context cleanup failed", exc_info=True)