from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

class ConversationMemory:
    """
    Enhanced memory and state management for the analytics agent.
    Tracks conversation history, user context, and session state with better
    conversation context for LLM interactions.
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
                "last_report_type": "both",
                "preferred_chart_type": "bar",
                "session_focus": "analytics"
            }
        }
        return session_id
    
    def store_interaction(self, session_id: str, user_prompt: str, 
                         tool_used: str, response: Dict[str, Any]):
        """Store an interaction in session memory with enhanced context."""
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
                "row_count": response.get("row_count", 0),
                "report_type": response.get("report_type", "both"),
                "message": response.get("message", "")[:200] + "..." if len(response.get("message", "")) > 200 else response.get("message", "")  # Truncate long messages
            }
        }
        
        # Update context based on interaction
        if response.get("file_name"):
            self.sessions[session_id]["context"]["last_file_queried"] = response.get("file_name")
        
        if response.get("report_type"):
            self.sessions[session_id]["context"]["last_report_type"] = response.get("report_type")
        
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
    
    def get_conversation_messages_for_llm(self, session_id: str, max_interactions: int = 3) -> List[Dict[str, str]]:
        """
        Get formatted conversation history suitable for LLM context.
        Returns list of dicts with 'role' and 'content' keys.
        """
        if session_id not in self.sessions:
            return []
        
        interactions = self.sessions[session_id]["interactions"][-max_interactions:]
        messages = []
        
        for interaction in interactions:
            # Add user message
            messages.append({
                "role": "user",
                "content": interaction["user_prompt"]
            })
            
            # Add assistant response summary
            response_summary = interaction.get("response_summary", {})
            if response_summary.get("success"):
                file_name = response_summary.get("file_name", "file")
                row_count = response_summary.get("row_count", 0)
                report_type = response_summary.get("report_type", "both")
                
                assistant_msg = f"I analyzed {file_name} "
                if report_type != "both":
                    assistant_msg += f"for {report_type} data "
                assistant_msg += f"and found {row_count} records."
                
                # Add brief summary of findings if available
                if response_summary.get("message"):
                    summary_snippet = response_summary["message"][:150]
                    if len(summary_snippet) < len(response_summary["message"]):
                        summary_snippet += "..."
                    assistant_msg += f" {summary_snippet}"
                
                messages.append({
                    "role": "assistant", 
                    "content": assistant_msg
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": "I encountered an issue processing that request."
                })
        
        return messages
    
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
        """
        Enhanced file reference resolution with better context awareness.
        Resolves references like 'that file', 'the file', etc.
        """
        if session_id not in self.sessions:
            return prompt
        
        last_file = self.sessions[session_id]["context"].get("last_file_queried")
        
        if not last_file:
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
            
            # Specific pattern for "for this file" - should come before general patterns
            (r'\bfor this file\b', f"for '{last_file}'"),
            (r'\bfor that file\b', f"for '{last_file}'"),
            (r'\bfor the file\b', f"for '{last_file}'"),
            
            # Contextual references
            (r'\bit\b(?=.*(?:rate|analysis|data|success|fail|record))', f"'{last_file}'"),  # 'it' in data context
            (r'\bthat\b(?=.*(?:csv|data|dataset|file))', f"'{last_file}'"),  # 'that' referring to data
            
            # Pattern for "for that" or similar (more general)
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
        
        # If no direct pattern matches but the prompt seems to be asking about data
        # and doesn't contain a specific file name, try a more general approach
        if updated_prompt == prompt and not re.search(r'\.csv|file\s+[\'"]', prompt, re.IGNORECASE):
            # Check if this looks like a data analysis request without explicit file reference
            data_analysis_indicators = [
                r'success\s+rate', r'analysis', r'data', r'records?', r'show\s+me',
                r'analyze', r'chart', r'graph', r'statistics', r'failure\s+rate'
            ]
            
            if any(re.search(indicator, prompt, re.IGNORECASE) for indicator in data_analysis_indicators):
                # Add the file reference to the end of the prompt
                updated_prompt = f"{prompt} for '{last_file}'"
                replacements_made.append(("implicit_context", f"for '{last_file}'"))
        
        return updated_prompt
    
    def get_last_interaction_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of the last interaction for context."""
        if session_id not in self.sessions or not self.sessions[session_id]["interactions"]:
            return None
        
        last_interaction = self.sessions[session_id]["interactions"][-1]
        return {
            "user_prompt": last_interaction["user_prompt"],
            "response_summary": last_interaction["response_summary"],
            "timestamp": last_interaction["timestamp"]
        }

# Initialize enhanced memory service
memory_service = ConversationMemory()