"""
Enhanced request context with async-safe context management
"""
import threading
from typing import Optional
import logging
import contextvars

logger = logging.getLogger("request_context")

# Use thread-local storage
_thread_local = threading.local()

# Also use context variables for async support
_org_id_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('org_id', default=None)
_user_id_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('user_id', default=None)

def set_request_context(org_id: str, user_id: str = None):
    """Set the request context in both thread-local storage and context variables."""
    # Set in thread-local storage
    _thread_local.org_id = org_id
    _thread_local.user_id = user_id
    
    # Set in context variables (async-safe)
    _org_id_context.set(org_id)
    _user_id_context.set(user_id)
    
    logger.info(f"Set context: org_id={org_id}, user_id={user_id}")

def get_org_id() -> Optional[str]:
    """Get org_id from context variables first, then thread-local storage."""
    # Try context variables first (async-safe)
    try:
        org_id = _org_id_context.get()
        if org_id:
            logger.debug(f"Retrieved org_id from context vars: {org_id}")
            return org_id
    except Exception as e:
        logger.debug(f"Context vars not available: {e}")
    
    # Fallback to thread-local storage
    org_id = getattr(_thread_local, 'org_id', None)
    logger.debug(f"Retrieved org_id from thread-local: {org_id}")
    return org_id

def get_user_id() -> Optional[str]:
    """Get user_id from context variables first, then thread-local storage."""
    # Try context variables first (async-safe)
    try:
        user_id = _user_id_context.get()
        if user_id:
            return user_id
    except Exception:
        pass
    
    # Fallback to thread-local storage
    return getattr(_thread_local, 'user_id', None)

def clear_request_context():
    """Clear the request context."""
    if hasattr(_thread_local, 'org_id'):
        delattr(_thread_local, 'org_id')
    if hasattr(_thread_local, 'user_id'):
        delattr(_thread_local, 'user_id')
    
    # Context variables will be cleared automatically when context exits
    logger.debug("Cleared request context")
