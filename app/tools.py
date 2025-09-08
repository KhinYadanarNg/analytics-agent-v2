from app.database_service import db_service
from fastapi import HTTPException

async def get_records_by_status_tool(file_name: str, status: str):
    # Step 1: Get file_id from file_name
    file_id = await db_service.get_file_id_by_name(file_name)
    if not file_id:
        raise HTTPException(status_code=404, detail=f"File '{file_name}' not found.")
    # Step 2: Get records by status
    result = await db_service.get_records_by_status(file_id, status)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    return result
