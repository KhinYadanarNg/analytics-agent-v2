import openai
from typing import Dict, Any
import json
import os
from dotenv import load_dotenv
from app.tool_schema import tools

# Load environment variables
load_dotenv()

class LLMService:
    def create_chart_from_data(self, chart_result: dict) -> dict:
        """
        Sends chart data to the LLM and returns the generated chart (image, JSON, or summary).
        """
        if not self.use_llm:
            # Mock response for demo/testing
            return {
                "success": True,
                "chart_type": chart_result.get("chart_type", "bar"),
                "chart_data": chart_result.get("chart_data", []),
                "message": "Mock chart generated (LLM not enabled)."
            }
        # Compose prompt for chart generation
        chart_type = chart_result.get("chart_type", "bar")
        chart_data = chart_result.get("chart_data", [])
        prompt = f"Generate a {chart_type} chart as an image using this data: {chart_data}. Return only the image (base64 or URL), no text."
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a chart generation assistant. Always return a chart image (base64 or URL), never text or JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        # Parse the response for image (base64 or URL)
        image_data = response.choices[0].message.content
        print(f"📝 image_data1: {response}")
        return {
            "success": True,
            "chart_type": chart_type,
            "chart_data": chart_data,
            "chart_image": image_data
        }

    def __init__(self):
        # Check if OpenAI API key is available
        print("LLMService __init__ called")

        # Database schema context for DynamoDB
        self.database_schema = """
            DynamoDB Tables:

            Table: MasterDataHeaderSIT
                import re
                match = re.search(r'!\[.*?\]\((.*?)\)', image_data)
                if match:
                    image_url = match.group(1)
                else:
                    image_url = None
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
You are an expert analytics agent. You answer questions about data and call backend tools to retrieve or analyze data from DynamoDB tables.

Database Schema:
{self.database_schema}

Available Tools:
1. get_records_by_status: Retrieves records by file name and status.
    - Parameters: file_name (str), status (str)
    - Use when the user asks for records with a specific status (e.g., success, failed) for a file.
2. get_success_rate_by_file_name: Calculates the success and fail rate for a file and returns chart data.
    - Parameters: file_name (str)
    - Use when the user asks for success/fail rate, percentage, or chart for a file.

CRITICAL RULES:
- ALWAYS call a tool if the user's request matches a tool's description.
- NEVER generate SQL queries. Only use the available tools for all data retrieval and analytics.
- ALWAYS create visual charts - NEVER return "table" as chart_type.
- For ANY sales data request: use "bar" chart with GROUP BY aggregation.
- For time-based data: use "line" chart.
- For distribution/percentage data: use "pie" chart.
- The chart_type must be one of: "bar", "line", "pie" (NEVER "table").

Examples:
- "Show me all success records for file 'customer.csv'" → Call get_records_by_status
- "Show me success rate for file 'customer_sample_values.csv'" → Call get_success_rate_by_file_name

If you call a tool, use the correct parameters and do not generate a SQL query.
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
                # Return tool call info directly, do not parse content
                return {
                    "success": True,
                    "tool_calls": tool_calls,
                    "llm_response": content
                }
            elif content:
                result = json.loads(content)
                print(f"📝 Parsed LLM result: {result}")
                chart_type = result.get("chart_type", "bar")
                if chart_type == "table":
                    chart_type = "bar"  # Default to bar chart for better visualization
                return {
                    "success": True,
                    "sql_query": result.get("sql_query"),
                    "chart_type": chart_type,
                    "explanation": result.get("explanation"),
                    "llm_response": content,
                    "tool_calls": None
                }
            else:
                print("📝 LLM response content and tool_calls are both None.")
                return {
                    "success": False,
                    "error": "LLM response content and tool_calls are both None.",
                    "llm_response": None
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
