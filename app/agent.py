import logging
from typing import Any, Dict

from app.reasoning_engine import reasoning_engine
from app.plan_executor import execute_plan
from app.communication_coordinator import communication_coordinator

logger = logging.getLogger("agent")


async def plan_and_execute(user_prompt: str, session_context: Dict[str, Any], workflow_context: Dict[str, Any]) -> Dict[str, Any]:
    """Compose planning and execution into a single async entrypoint.

    Returns a dict containing either a successful execution_result under
    `execution_result` or flags indicating a fallback is needed.
    """
    # Analyze and produce a plan
    analysis = reasoning_engine.analyze_query(user_prompt, session_context)
    plan = reasoning_engine.plan_execution(analysis)

    # Let the communication coordinator observe the plan
    try:
        coordination_log = communication_coordinator.orchestrate_workflow(plan, workflow_context)
    except Exception as e:
        logger.exception("Coordinator failed to orchestrate workflow: %s", e)
        coordination_log = {"error": str(e)}

    # Plan-first decision heuristic
    PLAN_CONFIDENCE_THRESHOLD = 0.4
    plan_steps = getattr(plan, "steps", None) or []
    plan_conf = analysis.get("confidence", 0)
    requires_decomp = analysis.get("requires_decomposition", False)

    prefer_plan = bool(plan_steps) and (plan_conf >= PLAN_CONFIDENCE_THRESHOLD or not requires_decomp)

    if not prefer_plan:
        return {
            "success": False,
            "fallback_needed": True,
            "query_analysis": analysis,
            "plan": plan,
            "coordination_log": coordination_log,
        }

    # Try to execute the plan
    try:
        execution_result = await execute_plan(plan, workflow_context)
        return {
            "success": True,
            "execution_result": execution_result,
            "query_analysis": analysis,
            "plan": plan,
            "coordination_log": coordination_log,
        }
    except Exception as e:
        logger.exception("Plan execution failed: %s", e)
        try:
            communication_coordinator.handle_component_error("reasoning_engine", e, workflow_context)
        except Exception:
            logger.debug("Coordinator handler raised during plan execution failure")
        return {
            "success": False,
            "fallback_needed": True,
            "error": str(e),
            "query_analysis": analysis,
            "plan": plan,
            "coordination_log": coordination_log,
        }
