"""
Clean FastAPI application with separated concerns.
Main.py only handles routing and basic app setup.
"""
import logging
import time
from typing import Dict, Any

from fastapi import FastAPI, Depends, Request, Response
from fastapi.security import HTTPAuthorizationCredentials

from app.auth import bearer_scheme
from app.query_coordinator import QueryCoordinator, PromptRequest
from app.memory_service import memory_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analytics_agent")

# Create FastAPI app
app = FastAPI(
    title="Analytics Agent API",
    description="Scalable analytics agent with session management",
    version="2.0.0"
)

# Initialize query coordinator
coordinator = QueryCoordinator()

@app.post("/query", response_model=Dict[str, Any])
async def receive_prompt(
    request: PromptRequest,
    http_request: Request,
    response: Response,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> Dict[str, Any]:
    """
    Process analytics query with authentication and session management.
    """
    return await coordinator.process_query(request, http_request, response, credentials)

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint with session storage status."""
    session_health = memory_service.health_check()
    
    return {
        "status": "healthy",
        "session_storage": session_health,
        "timestamp": time.time()
    }

