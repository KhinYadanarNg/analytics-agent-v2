"""
Enhanced query coordinator that passes conversation history to analytics agent.
"""
import logging
from typing import Dict, Any

from fastapi import Request, Response
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel

from app.analytics_agent import AnalyticsService
from app.memory_service import memory_service
from app.auth import validate_jwt_token
from app.request_context import set_current_org_id, reset_current_org_id, get_current_org_id  # <- keep
# If you still want user_id context, mirror this with a user_id contextvar module.

logger = logging.getLogger("analytics_agent")

class PromptRequest(BaseModel):
    prompt: str
    session_id: str | None = None

class QueryCoordinator:
    """Enhanced coordinator that provides conversation context to analytics service."""

    async def process_query(
        self,
        request: PromptRequest,
        http_request: Request,
        response: Response,
        credentials: HTTPAuthorizationCredentials
    ) -> Dict[str, Any]:
        """Process analytics query with session management and conversation context."""

        session_id = None
        token = None  # <- IMPORTANT: define upfront so finally can check it

        try:
            # 1) JWT validation
            user = validate_jwt_token(credentials)
            org_id = user.get("orgId")
            user_id = user.get("sub")

            # Context is already set by middleware, just validate it
            current_org_id = get_current_org_id()
            if current_org_id != org_id:
                logger.warning(f"Context org_id mismatch: context={current_org_id}, jwt={org_id}")
                # Reset context to match JWT
                if token:
                    reset_current_org_id(token)
                token = set_current_org_id(org_id)

            # 3) Resolve session id (cookie -> provided -> create)
            if request.session_id:
                session_id = request.session_id
            else:
                cookie_session_id = http_request.cookies.get("analytics_session_id")
                session_id = cookie_session_id or memory_service.create_session(user_id)

            # 4) Ensure session exists
            if session_id not in memory_service.sessions:
                session_id = memory_service.create_session(user_id)

            # 5) Pull session context + history
            session_context = memory_service.get_session_context(session_id)
            conversation_history = memory_service.get_conversation_history(session_id)

            # 6) Resolve file references ("this file", etc.)
            resolved_prompt = memory_service.resolve_file_reference(session_id, request.prompt)

            # 7) Extract/store file references found in the prompt
            import re
            file_pattern = r"['\"]([^'\"]*\.csv)['\"]|(\w+\.csv)"
            for m in re.findall(file_pattern, resolved_prompt):
                file_name = m[0] or m[1]
                if file_name:
                    memory_service.store_file_reference(session_id, file_name)
                    logger.info(f"Stored file reference: {file_name} for session {session_id[:8]}")

            # 8) Set/update session cookie
            response.set_cookie(
                key="analytics_session_id",
                value=session_id,
                max_age=3600,
                httponly=True,
                secure=True,
                samesite="lax",
            )

            # 9) Call analytics service (pass org_id if the service needs it)
            result = await AnalyticsService.process_query(
                prompt=resolved_prompt,
                session_id=session_id,
                conversation_history=conversation_history
            )

            # 10) Persist the interaction
            memory_service.store_interaction(
                session_id,
                resolved_prompt,
                result.get("tool_name", "unknown"),
                result,
            )

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
                "message": "An error occurred while processing your request.",
            }

        finally:
            # 11) ALWAYS clear per-request org_id to avoid leaking it to the next request
            if token is not None:
                try:
                    reset_current_org_id(token)
                except Exception:
                    # Never let cleanup throw
                    logger.debug("reset_current_org_id failed during teardown", exc_info=True)