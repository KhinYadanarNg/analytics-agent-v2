from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import asyncio
from datetime import datetime

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    NOTIFICATION = "notification"

class ComponentStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"

class CommunicationCoordinator:
    """
    Communication and coordination logic for the analytics agent.
    Manages inter-component communication, error propagation, and system coordination.
    """
    
    def __init__(self):
        self.components = {
            "auth": {"status": ComponentStatus.HEALTHY, "last_check": datetime.now()},
            "llm_service": {"status": ComponentStatus.HEALTHY, "last_check": datetime.now()},
            "database_service": {"status": ComponentStatus.HEALTHY, "last_check": datetime.now()},
            "chart_generator": {"status": ComponentStatus.HEALTHY, "last_check": datetime.now()},
            "memory_service": {"status": ComponentStatus.HEALTHY, "last_check": datetime.now()},
            "reasoning_engine": {"status": ComponentStatus.HEALTHY, "last_check": datetime.now()}
        }
        
        self.message_queue = []
        self.error_handlers = {}
        self.coordination_rules = {}
        
        # Setup default coordination rules
        self._setup_coordination_rules()
    
    def _setup_coordination_rules(self):
        """Setup default coordination rules between components."""
        self.coordination_rules = {
            "llm_failure": {
                "fallback_sequence": ["reasoning_engine"],
                "degraded_mode": True,
                "user_notification": "Using fallback mode due to LLM service issues"
            },
            "database_failure": {
                "fallback_sequence": [],
                "degraded_mode": False,
                "user_notification": "Database service unavailable. Please try again later."
            },
            "chart_generator_failure": {
                "fallback_sequence": [],
                "degraded_mode": True,
                "user_notification": "Chart generation unavailable. Returning data only."
            }
        }
    
    def register_error_handler(self, component: str, handler: Callable):
        """Register error handler for a component."""
        self.error_handlers[component] = handler
    
    def send_message(self, from_component: str, to_component: str, 
                    message_type: MessageType, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send message between components with logging."""
        message = {
            "id": f"{from_component}_{to_component}_{datetime.now().timestamp()}",
            "timestamp": datetime.now(),
            "from": from_component,
            "to": to_component,
            "type": message_type,
            "payload": payload
        }
        
        self.message_queue.append(message)
        
        # Keep only last 100 messages
        if len(self.message_queue) > 100:
            self.message_queue.pop(0)
        
        return message
    
    def handle_component_error(self, component: str, error: Exception, 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle component errors with coordination logic."""
        # Update component status
        self.components[component]["status"] = ComponentStatus.FAILED
        self.components[component]["last_error"] = {
            "error": str(error),
            "timestamp": datetime.now(),
            "context": context
        }
        
        # Log error message
        error_message = self.send_message(
            component, "coordinator", MessageType.ERROR,
            {"error": str(error), "context": context}
        )
        
        # Apply coordination rules
        coordination_response = self._apply_coordination_rules(component, error, context)
        
        # Call registered error handler if exists
        if component in self.error_handlers:
            try:
                self.error_handlers[component](error, context)
            except Exception as handler_error:
                print(f"Error handler for {component} failed: {handler_error}")
        
        return coordination_response
    
    def _apply_coordination_rules(self, failed_component: str, error: Exception, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply coordination rules for component failures."""
        rule_key = f"{failed_component}_failure"
        
        if rule_key in self.coordination_rules:
            rule = self.coordination_rules[rule_key]
            
            return {
                "fallback_available": len(rule["fallback_sequence"]) > 0,
                "fallback_components": rule["fallback_sequence"],
                "degraded_mode": rule["degraded_mode"],
                "user_message": rule["user_notification"],
                "recovery_action": "automatic" if rule["fallback_sequence"] else "manual"
            }
        
        # Default coordination response
        return {
            "fallback_available": False,
            "fallback_components": [],
            "degraded_mode": True,
            "user_message": f"Service temporarily unavailable due to {failed_component} error",
            "recovery_action": "manual"
        }
    
    def orchestrate_workflow(self, workflow_plan: Dict[str, Any], 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate workflow execution across components."""
        execution_log = {
            "workflow_id": f"workflow_{datetime.now().timestamp()}",
            "start_time": datetime.now(),
            "steps": [],
            "status": "in_progress"
        }
        
        try:
            # Step 1: Validation coordination
            validation_result = self._coordinate_validation(workflow_plan, context)
            execution_log["steps"].append({
                "step": "validation",
                "status": "completed" if validation_result["success"] else "failed",
                "result": validation_result
            })
            
            if not validation_result["success"]:
                execution_log["status"] = "failed"
                return execution_log
            
            # Step 2: Tool selection coordination
            tool_selection_result = self._coordinate_tool_selection(workflow_plan, context)
            execution_log["steps"].append({
                "step": "tool_selection",
                "status": "completed" if tool_selection_result["success"] else "failed",
                "result": tool_selection_result
            })
            
            if not tool_selection_result["success"]:
                execution_log["status"] = "failed"
                return execution_log
            
            # Step 3: Tool execution coordination
            execution_result = self._coordinate_tool_execution(
                tool_selection_result["tool"], 
                tool_selection_result["parameters"], 
                context
            )
            execution_log["steps"].append({
                "step": "tool_execution",
                "status": "completed" if execution_result["success"] else "failed",
                "result": execution_result
            })
            
            execution_log["status"] = "completed" if execution_result["success"] else "failed"
            execution_log["end_time"] = datetime.now()
            
            return execution_log
            
        except Exception as e:
            execution_log["status"] = "error"
            execution_log["error"] = str(e)
            execution_log["end_time"] = datetime.now()
            return execution_log
    
    def _coordinate_validation(self, workflow_plan: Dict[str, Any], 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate validation step - now handled by LLM."""
        return {
            "success": True,
            "component": "llm_service",
            "message": "Validation handled by LLM service"
        }
    
    def _coordinate_tool_selection(self, workflow_plan: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate tool selection step."""
        primary_tool = workflow_plan.get("primary_tool")
        parameters = workflow_plan.get("parameters", {})
        
        if not primary_tool:
            return {
                "success": False,
                "error": "No primary tool specified in workflow plan"
            }
        
        return {
            "success": True,
            "tool": primary_tool,
            "parameters": parameters,
            "component": "reasoning_engine"
        }
    
    def _coordinate_tool_execution(self, tool_name: str, parameters: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate tool execution step."""
        # This would integrate with actual tool execution
        # For now, return coordination metadata
        return {
            "success": True,
            "tool": tool_name,
            "parameters": parameters,
            "component": "database_service",
            "coordination_complete": True
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        healthy_components = sum(1 for comp in self.components.values() 
                               if comp["status"] == ComponentStatus.HEALTHY)
        total_components = len(self.components)
        
        overall_status = ComponentStatus.HEALTHY
        if healthy_components < total_components:
            if healthy_components >= total_components * 0.7:
                overall_status = ComponentStatus.DEGRADED
            else:
                overall_status = ComponentStatus.FAILED
        
        return {
            "overall_status": overall_status.value,
            "healthy_components": healthy_components,
            "total_components": total_components,
            "components": {name: comp["status"].value for name, comp in self.components.items()},
            "recent_messages": len(self.message_queue),
            "last_check": datetime.now()
        }
    
    def update_component_status(self, component: str, status: ComponentStatus):
        """Update component status."""
        if component in self.components:
            self.components[component]["status"] = status
            self.components[component]["last_check"] = datetime.now()

# Initialize communication coordinator
communication_coordinator = CommunicationCoordinator()
