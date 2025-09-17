import contextvars
from typing import Optional

# A context variable that's isolated per request/task
_current_org_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_org_id", default=None
)

def set_current_org_id(org_id: Optional[str]) -> contextvars.Token:
    """Set org_id for the current request; returns a token you should reset later."""
    return _current_org_id.set(org_id)

def get_current_org_id() -> Optional[str]:
    """Get org_id for the current request (or None)."""
    return _current_org_id.get()

def reset_current_org_id(token: contextvars.Token) -> None:
    """Reset to the previous value when the request ends."""
    _current_org_id.reset(token)