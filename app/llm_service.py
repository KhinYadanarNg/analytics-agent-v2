import openai
from typing import Dict, Any
import json
import os
from dotenv import load_dotenv
from app.tool_schema import tools
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger("llm_service")


class LLMService:
    def __init__(self):
        # Check if OpenAI API key is available
        logger.info("LLMService __init__ called")

        # Database schema context for DynamoDB
        self.database_schema = """
            DynamoDB Tables:

            Table: MasterDataHeaderSIT
                - id (Primary Key, String)
                - domain_name (String)
                - file_name (String, GSI: file_name-index)
                - file_status (String)
                   
            Table: MasterDataTaskTrackerSIT
                - id (Primary Key, String)
                - file_id (String, GSI: file_id-final_status-index)
                - final_status (String)
                - organization_id (String)
                - rule_status (String)
            """
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                self.client = openai.OpenAI(api_key=api_key)
                self.use_llm = True
                logger.info("OpenAI API key found; LLM client initialized.")
            except Exception as e:
                self.client = None
                self.use_llm = False
                logger.exception("Failed to initialize OpenAI client: %s", e)
        else:
            self.client = None
            self.use_llm = False
            logger.warning("OpenAI API key not found. Using mock responses for demo.")

    def extractPrompt(self, user_prompt: str) -> Dict[str, Any]:
        """Convert user prompt to tool calls using LLM or provide a mock response.

        Returns a dict with keys: success (bool), tool_calls (list) and llm_response (str) on success.
        """
        # Defensive check for database_schema
        if not getattr(self, "database_schema", None):
            raise AttributeError("LLMService instance is missing 'database_schema'. Ensure __init__ was called.")

        if not self.use_llm or not self.client:
            logger.debug("LLM client unavailable; returning mock fallback tool call.")
            # Simple mock: instruct to call list_available_files when user asks generically
            return {
                "success": True,
                "tool_calls": [{"name": "list_available_files", "arguments": {}}],
                "llm_response": "mock: list available files"
            }

        try:
            system_prompt = f"""
You are an expert analytics agent that helps users analyze data from DynamoDB tables. Your ONLY job is to understand the user's request and call the appropriate backend tool.

Database Schema:
{self.database_schema}

Available Tools:
1. get_records_by_status: Retrieves records by file name and status.
    - Parameters: file_name (str), status (str)
    - Use when user asks for records with a specific status (e.g., "show me success records for file X")

2. get_success_rate_by_file_name: Calculates success/fail rates for a file.
    - Parameters: file_name (str)
    - Use when user asks for success rate, fail rate, percentage, or chart for a file

CRITICAL RULES:
- ALWAYS call a tool when the user asks for data, records, rates, or charts
- NEVER try to generate SQL queries or charts yourself
- NEVER return JSON responses - only call tools
- If the user's request matches a tool's purpose, call that tool immediately
- Extract the file name from the user's request and pass it as a parameter

Examples:
- "Show success records for customer.csv" → Call get_records_by_status with file_name="customer.csv", status="success"
- "Show success rate for customer_sample_values.csv" → Call get_success_rate_by_file_name with file_name="customer_sample_values.csv"
- "Create chart for file X" → Call get_success_rate_by_file_name with file_name="X"

Your response should ONLY be tool calls, nothing else.
"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{user_prompt}"}
                ],
                temperature=0.1,
                tools=tools
            )

            # Parse the response
            content = response.choices[0].message.content
            tool_calls = getattr(response.choices[0].message, "tool_calls", None)
            logger.debug("Parsed LLM content: %s", content)
            logger.debug("tool_calls: %s", tool_calls)

            if tool_calls:
                # Return the tool_calls as-is for the orchestration layer to handle
                return {
                    "success": True,
                    "tool_calls": tool_calls,
                    "llm_response": content
                }
            else:
                logger.warning("No tool calls detected from LLM.")
                return {
                    "success": False,
                    "error": "LLM did not call any tools",
                    "llm_response": content,
                    "message": "Unable to understand your request. Please ask for data records or success rates for a specific file."
                }

        except Exception as e:
            logger.exception("Error while calling LLM: %s", e)
            return {
                "success": False,
                "error": str(e),
                "fallback_response": "Error processing the request."
            }

    def respond_with_chart(self, query_result: Dict[str, Any]) -> Dict[str, Any]:
        """Accepts the query result from the database and validates it for chart generation.

        Ensures chart_type and chart_data are present for downstream charting components.
        """
        if "chart_type" not in query_result or "chart_data" not in query_result:
            return {
                "success": False,
                "error": "Invalid query result format.",
                "fallback_response": "Error processing the request."
            }
        return query_result


# Initialize LLM service
llm_service = LLMService()
