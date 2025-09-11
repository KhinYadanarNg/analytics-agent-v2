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
    
    def store_file_reference(self, session_id: str, file_name: str):
        """Store file reference in session context."""
        if session_id not in self.sessions:
            return False
        
        clean_file_name = file_name.strip("'\"")
        self.sessions[session_id]["context"]["last_file_queried"] = clean_file_name
        return True
    
    def resolve_file_reference(self, session_id: str, prompt: str) -> str:
        """Resolve file references like 'that file', 'the file', etc."""
        if session_id not in self.sessions:
            print(f"DEBUG: Session {session_id} not found in memory")
            return prompt
        
        last_file = self.sessions[session_id]["context"].get("last_file_queried")
        print(f"DEBUG: Session {session_id} last_file_queried: {last_file}")
        
        if not last_file:
            print(f"DEBUG: No last file found for session {session_id}")
            return prompt
        
        import re
        
        # Enhanced patterns to match file references with better context awareness
        file_reference_patterns = [
            # Direct file references
            (r'\bthat file\b', f"'{last_file}'"),
            (r'\bthe file\b', f"'{last_file}'"),
            (r'\bthis file\b', f"'{last_file}'"),
            (r'\bsame file\b', f"'{last_file}'"),
            (r'\bprevious file\b', f"'{last_file}'"),
            (r'\blast file\b', f"'{last_file}'"),
            
            # Contextual references
            (r'\bit\b(?=.*(?:rate|analysis|data|success|fail|record))', f"'{last_file}'"),  # 'it' in data context
            (r'\bthat\b(?=.*(?:csv|data|dataset|file))', f"'{last_file}'"),  # 'that' referring to data
            
            # Pattern for "for that file" or similar
            (r'\bfor that\b(?!\s+file)', f"for '{last_file}'"),  # "show me success rate for that" 
            
            # More specific patterns
            (r'\bthe same\b(?=.*(?:csv|data|dataset))', f"'{last_file}'"),
            (r'\bagain\b(?=.*(?:file|csv|data))', f"'{last_file}' again"),
        ]
        
        updated_prompt = prompt
        replacements_made = []
        
        for pattern, replacement in file_reference_patterns:
            if re.search(pattern, updated_prompt, re.IGNORECASE):
                old_prompt = updated_prompt
                updated_prompt = re.sub(pattern, replacement, updated_prompt, flags=re.IGNORECASE)
                if old_prompt != updated_prompt:
                    replacements_made.append((pattern, replacement))
                    print(f"DEBUG: Resolved '{pattern}' -> '{replacement}' in session {session_id}")
        
        # If no direct pattern matches but the prompt seems to be asking about data
        # and doesn't contain a specific file name, try a more general approach
        if updated_prompt == prompt and not re.search(r'\.csv|file\s+[\'"]', prompt, re.IGNORECASE):
            # Check if this looks like a data analysis request without explicit file reference
            data_analysis_indicators = [
                r'success\s+rate', r'analysis', r'data', r'records?', r'show\s+me',
                r'analyze', r'chart', r'graph', r'statistics'
            ]
            
            if any(re.search(indicator, prompt, re.IGNORECASE) for indicator in data_analysis_indicators):
                # Add the file reference to the end of the prompt
                updated_prompt = f"{prompt} for '{last_file}'"
                replacements_made.append(("implicit_context", f"for '{last_file}'"))
                print(f"DEBUG: Added implicit file context to prompt in session {session_id}")
        
        if replacements_made:
            print(f"DEBUG: Final resolved prompt: '{updated_prompt}'")
        else:
            print(f"DEBUG: No file references found in prompt: '{prompt}'")
        
        return updated_prompt

# Initialize memory service
memory_service = ConversationMemory()
