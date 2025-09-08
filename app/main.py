
import base64
from fastapi import FastAPI, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from app.auth import validate_jwt_token, bearer_scheme
from app.validator import prompt_validator
from app.llm_service import llm_service
from app.database_service import db_service
from app.chart_generator import chart_generator

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str


@app.post("/query")
async def receive_prompt(
    request: PromptRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    try:
        # Step 1: Validate JWT token
        user = validate_jwt_token(credentials)
        
        # Step 2: Validate user prompt
        validation_result = prompt_validator.validate_prompt(request.prompt)
        cleaned_prompt = validation_result["cleaned_prompt"]
        
        # Step 3: Use LLM to determine which tool to call and with what parameters
        llm_result = llm_service.generate_sql_query(cleaned_prompt)
        tool_calls = llm_result.get("tool_calls")
        print(f"🔧 tool_calls from LLM: {tool_calls}")

        if not tool_calls:
            return {
                "success": False,
                "error": "No tool call detected from LLM",
                "message": "Unable to understand your request. Please ask analytics-related questions about data files, success rates, or request charts.",
                "llm_result": llm_result
            }

        # Only handle the first tool call for now
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        import json
        if isinstance(tool_args, str):
            tool_args = json.loads(tool_args)

        # Map tool name to backend function
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
                "user_prompt": cleaned_prompt,
                "tool": tool_name,
                "file_name": file_name,
                "status": status,
                "data": db_result["data"],
                "row_count": db_result["row_count"],
                "workflow_completed": True
            }
        elif tool_name == "get_success_rate_by_file_name":
            file_name = tool_args.get("file_name")
            file_id = await db_service.get_file_id_by_name(file_name)
            if not file_id:
                return {
                    "success": False,
                    "error": "File not found in header table",
                    "message": f"The file '{file_name}' was not found in the database.",
                    "file_name": file_name
                }
            chart_result = await db_service.get_success_rate_by_file_id(file_id)
            
            # Generate chart image using matplotlib and encode as base64
            chart_base64 = chart_generator.generate_bar_chart_base64(
                chart_data=chart_result.get("chart_data", []),
                title=f"Success/Fail Rate for {file_name}"
            )

            # try:
            #     with open("test_chart.png", "wb") as f:
            #         f.write(base64.b64decode(chart_base64))
            #     print("💾 Chart saved as 'test_chart.png'")
            # except Exception as e:
            #     print(f"❌ Error saving chart: {e}")
            
            return {
                "success": True,
                "user_prompt": cleaned_prompt,
                "tool": tool_name,
                "file_name": file_name,
                "chart_data": chart_result.get("chart_data", []),
                "chart_image_base64": chart_base64,
                "row_count": chart_result.get("row_count", 0),
                "workflow_completed": True
            }
        else:
            return {
                "success": False,
                "error": f"Unknown tool called: {tool_name}",
                "message": f"The requested tool '{tool_name}' is not supported.",
                "tool_args": tool_args
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "An error occurred while processing your request. Please try again with a valid analytics question."
        }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/test-query")
async def test_receive_prompt(request: PromptRequest):
    try:
        # Test endpoint without authentication
        print(f"🧪 Test request received: {request.prompt}")
        
        # Test the full workflow without JWT
        validation_result = prompt_validator.validate_prompt(request.prompt)
        cleaned_prompt = validation_result["cleaned_prompt"]
        print(f"✅ Validation passed: {cleaned_prompt}")
        
        # Test LLM service
        llm_result = llm_service.generate_sql_query(cleaned_prompt)
        print(f"🤖 LLM Result: {llm_result}")
        
        return {
            "success": True,
            "received_prompt": request.prompt, 
            "status": "success",
            "llm_result": llm_result
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Validation failed. Please ask analytics-related questions about data files, success rates, or request charts.",
            "received_prompt": request.prompt
        }
