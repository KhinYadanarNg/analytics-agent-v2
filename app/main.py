
from fastapi import FastAPI, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from app.auth import validate_jwt_token, bearer_scheme
from app.validator import prompt_validator

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str


@app.post("/query")
async def receive_prompt(
    request: PromptRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    # Validate JWT token
    user = validate_jwt_token(credentials)
    
    # Validate user prompt
    validation_result = prompt_validator.validate_prompt(request.prompt)
    cleaned_prompt = validation_result["cleaned_prompt"]
    
    # TODO: Call LLM to generate SQL query from cleaned_prompt
    # TODO: Execute SQL query against database
    # TODO: Generate chart from results
    
    return {
        "received_prompt": cleaned_prompt,
        "user": user,
        "validation": validation_result["message"],
        "next_steps": ["generate_sql", "execute_query", "create_chart"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/test-query")
async def test_receive_prompt(request: PromptRequest):
    # Test endpoint without authentication
    return {"received_prompt": request.prompt, "status": "success"}
