import openai
from typing import Dict, Any
import json
import os
from dotenv import load_dotenv
from app.tool_schema import tools

# Load environment variables
load_dotenv()

class LLMService:
    def __init__(self):
        # Check if OpenAI API key is available
        print("LLMService __init__ called")

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
        if api_key is not None:
            self.client = openai.OpenAI(api_key=api_key)
            self.use_llm = True
            print("⚠️  OpenAI API key found.")
        else:
            self.client = None
            self.use_llm = False
            print("⚠️  OpenAI API key not found. Using mock responses for demo.")


    def generate_sql_query(self, user_prompt: str) -> Dict[str, Any]:
        # Defensive check for database_schema
        if not hasattr(self, "database_schema") or not self.database_schema:
            raise AttributeError("LLMService instance is missing 'database_schema'. Ensure you are using the instance and __init__ is called.")
        """
        Convert user prompt to SQL query using LLM or mock responses
        """
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
            print(f"📝 Parsed LLM content: {content}")
            print(f"🔧 tool_calls: {tool_calls}")

            if tool_calls:
                print("LLM tool calls detected:")
                for call in tool_calls:
                    print(f"Tool called: {call.function.name}")
                    print(f"Arguments: {call.function.arguments}")
                return {
                    "success": True,
                    "tool_calls": tool_calls,
                    "llm_response": content
                }
            else:
                print("📝 No tool calls detected from LLM.")
                return {
                    "success": False,
                    "error": "LLM did not call any tools",
                    "llm_response": content,
                    "message": "Unable to understand your request. Please ask for data records or success rates for a specific file."
                }
        
        except Exception as e:
            print(f"📝 Parsed error: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_response": "Error processing the request."
            }
        
        def respond_with_chart(self, query_result: dict) -> dict:
           """
        Accepts the query result from the database and returns it to the LLM for chart generation.
        Ensures chart_type and chart_data are present for LLM to generate a chart.
        """
        # You can add any additional formatting or validation here if needed
        if "chart_type" not in query_result or "chart_data" not in query_result:
            return {
                "success": False,
                "error": "Invalid query result format.",
                "fallback_response": "Error processing the request."
            }
        return query_result

        return query_result

# Initialize LLM service
llm_service = LLMService()
