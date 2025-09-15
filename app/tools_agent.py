from app.database_service import DatabaseService
from langchain_core.tools import tool
from typing import Optional, Dict, Any
from app.request_context import current_org_id, current_user_id
import asyncio

db_service = DatabaseService()


@tool("get_success_rate_by_file_name", return_direct=False)
def get_success_rate_by_file_name_tool(
    file_name: Optional[str] = None,
    org_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    chart_type: Optional[str] = "bar"  # LLM will specify chart type
) -> Dict[str, Any]:
    """Calculate success/failure percentage for a file and specify visualization type.

    If org_id is not provided by the LLM prompt, fall back to the org_id supplied in the
    request's JWT token via the request context.
    
    Args:
        file_name: Name of the file to analyze (e.g., 'customer_sample_values.csv')
        org_id: Organization ID to filter by
        start_date: Start date for filtering (YYYY-MM-DD format)
        end_date: End date for filtering (YYYY-MM-DD format)
        chart_type: Type of chart to generate ('bar', 'pie', 'donut', 'line', 'stacked'). Default is 'bar'.
    """
    # Validate chart_type
    valid_chart_types = ['bar', 'pie', 'donut', 'line', 'stacked']
    if chart_type not in valid_chart_types:
        chart_type = 'bar'  # Default to bar if invalid type provided
    
    # fallback to request context org_id/user_id when the LLM didn't supply them
    if not org_id:
        org_id = current_org_id.get()

    # Handle the async database call
    def run_async_db_call():
        """Run the async database call in a new event loop"""
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(
                db_service.get_success_rate_by_file_name(file_name, org_id, start_date, end_date)
            )
        finally:
            new_loop.close()

    try:
        # Try to get the current event loop
        current_loop = asyncio.get_running_loop()
        # If we're in an async context, run in a thread to avoid blocking
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = executor.submit(run_async_db_call).result()
    except RuntimeError:
        # No event loop running, safe to run directly
        result = run_async_db_call()
    
    # Ensure the result is properly formatted
    if isinstance(result, dict):
        # Add the chart_type to the result
        result["chart_type"] = chart_type
        result["chart_type_requested"] = True  # Flag to indicate LLM specified the type
        
        # Add stop flag to prevent infinite loops
        result.setdefault("stop", True)
        return result
    else:
        # Handle unexpected result format
        return {
            "success": False,
            "message": f"Unexpected result format from database: {type(result)}",
            "raw_result": str(result),
            "chart_type": chart_type,
            "stop": True
        }