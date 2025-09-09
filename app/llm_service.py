import openai
from typing import Dict, Any
import json
import os
from dotenv import load_dotenv
from app.tool_schema import tools, function_schemas as TOOL_FUNCTION_SCHEMAS
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

        # Use tool_schema as single source of truth for required params
        self.function_schemas = TOOL_FUNCTION_SCHEMAS

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
            # Include a small machine-readable function schema block to help the model
            function_schema_block = json.dumps({
                "get_records_by_status": {"required": ["file_name", "status"]},
                "get_success_rate_by_file_name": {"required": ["file_name"]},
                "list_available_files": {"required": []}
            }, indent=2)

            system_prompt = f"""
You are an expert analytics agent whose job is to map user requests to backend tool calls only.

Database Schema:
{self.database_schema}

Function Schemas (machine-readable):
{function_schema_block}

Available Tools:
- get_records_by_status(file_name, status)
- get_success_rate_by_file_name(file_name)
- list_available_files()

CRITICAL RULES:
- ALWAYS call a tool when the user asks for data, records, rates, or charts.
- If required parameters are missing, ASK ONE precise clarifying question instead of guessing.
- NEVER generate charts or SQL yourself; call the appropriate tool.

Examples:
- "Show success records for customer.csv" → Call get_records_by_status with file_name="customer.csv", status="success"
- "Show success rate for customer_sample_values.csv" → Call get_success_rate_by_file_name with file_name="customer_sample_values.csv"

If you call a tool, return only the tool call (function name and arguments) in the SDK's tool-calling format.
"""

            # Flexible client invocation: support patterns where `chat` is a callable or attribute,
            # where `completions` is a property, or where `create` exists at different levels.
            call_kwargs = dict(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{user_prompt}"}
                ],
                temperature=0.1,
                tools=tools
            )

            def _call_create(kwargs):
                # pattern: client.chat.completions.create(...)
                chat_attr = getattr(self.client, "chat", None)
                if chat_attr:
                    chat_obj = chat_attr() if callable(chat_attr) else chat_attr
                    completions = getattr(chat_obj, "completions", None)
                    if completions and hasattr(completions, "create"):
                        return completions.create(**kwargs)

                # pattern: client.completions.create(...)
                completions = getattr(self.client, "completions", None)
                if completions and hasattr(completions, "create"):
                    return completions.create(**kwargs)

                # pattern: client.create(...)
                create_fn = getattr(self.client, "create", None)
                if create_fn and callable(create_fn):
                    return create_fn(**kwargs)

                raise AttributeError("LLM client does not expose a compatible create(...) method")

            response = _call_create(call_kwargs)

            # Parse the response
            content = response.choices[0].message.content
            raw_tool_calls = getattr(response.choices[0].message, "tool_calls", None)
            logger.debug("Parsed LLM content: %s", content)
            logger.debug("raw_tool_calls: %s", raw_tool_calls)

            # Normalize tool call shapes into canonical dicts: {name: str, arguments: dict}
            def _normalize_tool_call(tc):
                # OpenAI function-like shape
                if isinstance(tc, dict) and tc.get("function"):
                    fn = tc["function"]
                    return {"name": fn.get("name"), "arguments": fn.get("arguments", {})}

                # Flat dict shape
                if isinstance(tc, dict) and tc.get("name"):
                    return {"name": tc.get("name"), "arguments": tc.get("arguments", tc.get("args", {}))}

                # Some SDKs return objects with attributes
                if hasattr(tc, "function"):
                    fn = tc.function
                    args = getattr(fn, "arguments", {})
                    name = getattr(fn, "name", None)
                    return {"name": name, "arguments": args}

                # Fallback: try to coerce
                try:
                    return {"name": tc["name"], "arguments": tc.get("arguments", {})}
                except Exception:
                    return {"name": str(tc), "arguments": {}}

            tool_calls = []
            if raw_tool_calls:
                for rtc in raw_tool_calls:
                    tool_calls.append(_normalize_tool_call(rtc))

            # Validator: ensure required params are present for each tool
            def _validate_tool_calls(tcs):
                missing = []
                for tc in tcs:
                    name = tc.get("name")
                    schema = self.function_schemas.get(name, {})
                    required = schema.get("required", [])
                    args = tc.get("arguments") or {}
                    for r in required:
                        if r not in args or args.get(r) in (None, ""):
                            missing.append((name, r))
                if missing:
                    # Return a single precise clarifying question for the first missing param
                    name, param = missing[0]
                    question = f"Which {param} do you mean for {name}?"
                    return False, question
                return True, None

            valid, clarify_question = _validate_tool_calls(tool_calls)
            if not valid:
                logger.info("Tool call missing required params, asking clarifying question: %s", clarify_question)
                return {
                    "success": False,
                    "clarify": clarify_question,
                    "tool_calls": tool_calls,
                    "llm_response": content
                }

            if tool_calls:
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
