from typing import Optional
from contextvars import ContextVar

# Context variables to carry per-request metadata (org_id, user_id) into tool execution
current_org_id: ContextVar[Optional[str]] = ContextVar("current_org_id", default=None)
current_user_id: ContextVar[Optional[str]] = ContextVar("current_user_id", default=None)
