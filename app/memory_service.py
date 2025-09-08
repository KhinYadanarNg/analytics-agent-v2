from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

class ConversationMemory:
    """
    Memory and state management for the analytics agent.
    Tracks conversation history, user context, and session state.
    """
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.max_session_history = 10  # Keep last 10 interactions per session
    
    def create_session(self, user_id: str) -> str:
        """Create a new session for a user."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "interactions": [],
            "context": {
                "last_file_queried": None,
                "preferred_chart_type": "bar",
                "session_focus": "analytics"
            }
        }
        return session_id
    
    def store_interaction(self, session_id: str, user_prompt: str, 
                         tool_used: str, response: Dict[str, Any]):
        """Store an interaction in session memory."""
        if session_id not in self.sessions:
            return False
        
        interaction = {
            "timestamp": datetime.now(),
            "user_prompt": user_prompt,
            "tool_used": tool_used,
            "response_summary": {
                "success": response.get("success", False),
                "tool": response.get("tool"),
                "file_name": response.get("file_name"),
                "row_count": response.get("row_count", 0)
            }
        }
        
        # Update context based on interaction
        if response.get("file_name"):
            self.sessions[session_id]["context"]["last_file_queried"] = response.get("file_name")
        
        # Add interaction and maintain history limit
        self.sessions[session_id]["interactions"].append(interaction)
        if len(self.sessions[session_id]["interactions"]) > self.max_session_history:
            self.sessions[session_id]["interactions"].pop(0)
        
        return True
    
    def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session context for reasoning and planning."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        recent_interactions = session["interactions"][-3:]  # Last 3 interactions
        
        return {
            "user_id": session["user_id"],
            "session_age": (datetime.now() - session["created_at"]).seconds,
            "recent_interactions": recent_interactions,
            "context": session["context"],
            "interaction_count": len(session["interactions"])
        }
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for the session."""
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id]["interactions"]
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up sessions older than specified hours."""
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, session_data in self.sessions.items():
            age_hours = (current_time - session_data["created_at"]).total_seconds() / 3600
            if age_hours > max_age_hours:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
        
        return len(sessions_to_remove)

# Initialize memory service
memory_service = ConversationMemory()
