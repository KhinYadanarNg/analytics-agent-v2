import base64
import logging
from typing import Dict, Any
from app.database_service import db_service
from app.chart_generator import chart_generator
from app.communication_coordinator import communication_coordinator, ComponentStatus
from app.fallback_strategy import fallback_strategy, FallbackTrigger
from app.memory_service import memory_service

logger = logging.getLogger("plan_executor")


async def execute_tool_with_coordination(tool_name: str, tool_args: dict, context: dict) -> dict:
    """Execute tools with coordination and error handling."""
    # Data retrieval
    if tool_name == "get_records_by_status":
        file_name = tool_args.get("file_name")
        status = tool_args.get("status")
        file_id = await db_service.get_file_id_by_name(file_name)
        if not file_id:
            return {
                "success": False,
                "error": "File not found in header table",
                "message": f"The file '{file_name}' was not found in the database.",
                "file_name": file_name
            }
        db_result = await db_service.get_records_by_status(file_id=file_id, status=status)
        return {
            "success": True,
            "user_prompt": context.get("user_prompt"),
            "tool": tool_name,
            "file_name": file_name,
            "status": status,
            "data": db_result.get("data", []),
            "row_count": db_result.get("row_count", 0)
        }

    # Success rate -> chart
    if tool_name == "get_success_rate_by_file_name":
        file_name = tool_args.get("file_name")
        file_id = await db_service.get_file_id_by_name(file_name)
        if not file_id:
            return {
                "success": False,
                "error": "File not found in header table",
                "message": f"The file '{file_name}' was not found in the database.",
                "file_name": file_name
            }
        org_id = context.get("org_id")
        chart_result = await db_service.get_success_rate_by_file_id(file_id=file_id, org_id=org_id)

        # Generate chart image using matplotlib with error handling
        try:
            chart_base64 = chart_generator.generate_bar_chart_base64(
                chart_data=chart_result.get("chart_data", []),
                title=f"Success/Fail Rate for {file_name}"
            )
            # Optionally persist a local copy for debugging
            try:
                with open("test_chart.png", "wb") as f:
                    f.write(base64.b64decode(chart_base64))
                logger.debug("Chart saved as 'test_chart.png'")
            except Exception:
                logger.debug("Could not write debug chart to disk")

            # Update component status - chart generator working
            communication_coordinator.update_component_status("chart_generator", ComponentStatus.HEALTHY)

        except Exception as chart_error:
            logger.exception("Chart generation failed: %s", chart_error)
            communication_coordinator.handle_component_error("chart_generator", chart_error, context)

            fallback_result = fallback_strategy.execute_fallback(
                FallbackTrigger.CHART_GENERATION_FAILED,
                {**context, "chart_data": chart_result.get("chart_data", [])}
            )

            chart_base64 = None
            chart_result.update(fallback_result)

        return {
            "success": True,
            "user_prompt": context.get("user_prompt"),
            "tool": tool_name,
            "file_name": file_name,
            "chart_data": chart_result.get("chart_data", []),
            "chart_image_base64": chart_base64,
            "row_count": chart_result.get("row_count", 0),
            "chart_generation_failed": chart_base64 is None
        }

    # List files
    if tool_name == "list_available_files":
        try:
            files_result = await db_service.list_available_files()
            return {
                "success": True,
                "tool": tool_name,
                "files": files_result.get("files", []),
                "count": files_result.get("count", 0)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    return {
        "success": False,
        "error": f"Unknown tool called: {tool_name}",
        "message": f"The requested tool '{tool_name}' is not supported.",
        "tool_args": tool_args
    }


async def execute_plan(plan, context: Dict[str, Any], max_iterations: int = 3) -> Dict[str, Any]:
    """Execute a Plan object produced by the ReasoningEngine.

    Executes each Step in order, validates postconditions, and applies fallbacks or refinements.
    Returns a structured result which may include a final chart image when available.
    """
    from app.reasoning_engine import Step

    iteration = 0
    last_result = None
    while iteration < max_iterations:
        iteration += 1
        context["plan_iteration"] = iteration

        for step in list(plan.steps):
            tool = step.tool
            params = step.params or {}

            # If file_name missing and context has last_file_queried, fill in
            if params.get("file_name") is None and context.get("last_file_queried"):
                params["file_name"] = context.get("last_file_queried")

            # Execute mapped tool
            try:
                result = await execute_tool_with_coordination(tool, params, context)
            except Exception as e:
                logger.exception("Step execution error for %s: %s", tool, e)
                fb = fallback_strategy.execute_fallback(
                    FallbackTrigger.DATABASE_ERROR if "database" in str(e).lower() else FallbackTrigger.UNKNOWN_QUERY,
                    {**context, **{"failed_step": step.tool}}
                )
                if fb.get("success") and fb.get("tool_calls"):
                    fc = fb["tool_calls"][0]
                    if isinstance(fc, dict) and fc.get("function"):
                        fname = fc["function"]["name"]
                        fargs = fc["function"].get("arguments", {})
                    else:
                        fname = fc.get("name") or fc.get("tool_name")
                        fargs = fc.get("arguments") or {}

                    try:
                        result = await execute_tool_with_coordination(fname, fargs, context)
                    except Exception as e2:
                        logger.exception("Fallback execution also failed for %s: %s", fname, e2)
                        result = {"success": False, "error": str(e2)}
                else:
                    result = {"success": False, "error": str(e), "fallback": fb}

            # Reflect: check postconditions if defined
            postconditions = getattr(step, "postconditions", None) or []
            post_ok = True
            for cond in postconditions:
                if cond == "non_empty":
                    if not result.get("data"):
                        post_ok = False
                        break
                if cond.startswith("rate_in_"):
                    rate_list = result.get("chart_data") or []
                    if not rate_list:
                        post_ok = False
                        break

            # Update memory/context when useful
            if result.get("file_name"):
                context["last_file_queried"] = result.get("file_name")

            last_result = result

            if not post_ok or not result.get("success"):
                # Try to refine plan: if missing file, add list_available_files
                if any("file" in str(x).lower() for x in (params.keys())) and not context.get("last_file_queried"):
                    plan.steps.insert(0, Step("list_available_files", {}, "file_list"))
                    break  # break the for-loop to restart plan
                else:
                    return {
                        "success": False,
                        "error": "Step failed or postcondition not met",
                        "failed_step": step.tool,
                        "step_result": result,
                        "iteration": iteration
                    }

        else:
            # Completed all steps without break
            if last_result and last_result.get("chart_image_base64"):
                return {"success": True, "result": last_result, "iterations": iteration}
            return {"success": True, "result": last_result, "iterations": iteration}

    return {"success": False, "error": "Max plan iterations reached", "last_result": last_result, "iterations": iteration}
