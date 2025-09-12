# llm_service.py
from __future__ import annotations

import json
import os
import re
import time
import logging
from typing import Dict, Any, List, Optional, TypedDict

from dotenv import load_dotenv
from openai import OpenAI

from app.tool_schema import  tools as TOOL_FUNCTION_SCHEMAS

# -------------------------------------------------------------------
# Initialization
# -------------------------------------------------------------------

load_dotenv()
logger = logging.getLogger("llm_service")
logger.setLevel(logging.INFO)

# -------------------------------------------------------------------
# Types
# -------------------------------------------------------------------

class ExtractResponse(TypedDict, total=False):
    success: bool
    tool_calls: List[Dict[str, Any]]
    llm_response: str
    clarify: Optional[str]
    error: Optional[str]
    message: Optional[str]

# -------------------------------------------------------------------
# Guardrails (minimal, targeted)
# -------------------------------------------------------------------

INSTRUCTION_OVERRIDE = re.compile(
    r"^(?:please\s+)?(?:ignore|disregard|override)\s+(?:all\s+)?(?:previous|above)\s+(?:instructions|messages)",
    re.IGNORECASE,
)

ROLE_REASSIGN = re.compile(r"^(?:system:|you are now\s|act as\s)", re.IGNORECASE)

def is_instruction_override(text: str) -> bool:
    """Block only clear attempts to override system instructions."""
    t = text.strip()
    return bool(INSTRUCTION_OVERRIDE.search(t) or ROLE_REASSIGN.search(t))

# Optional: cheap keyword check for when LLM is unavailable
ANALYTICS_HINTS = (
    "analytics", "data", "chart", "graph", "success rate", "fail rate",
    "csv", "table", "record", "statistic", "metric", "report",
    "list files", "available files", "show data", "show chart", "count", "total"
)

def looks_like_analytics(text: str) -> bool:
    t = text.lower()
    return any(hint in t for hint in ANALYTICS_HINTS)

# -------------------------------------------------------------------
# Main Service
# -------------------------------------------------------------------

class LLMService:
    def __init__(self):
        logger.info("LLMService __init__ called")

        # Database schema context (for the prompt)
        self.database_schema = """
DynamoDB Tables:

Table: MasterDataHeaderSIT
  - id (PK, String)
  - domain_name (String)
  - file_name (String, GSI: file_name-index)
  - file_status (String)

Table: MasterDataTaskTrackerSIT
  - id (PK, String)
  - file_id (String, GSI: file_id-final_status-index)
  - final_status (String)
  - organization_id (String)
  - rule_status (String)
""".strip()

        # OpenAI client (optional)
        api_key = os.getenv("OPENAI_API_KEY")
        try:
            self.client = OpenAI(api_key=api_key) if api_key else None
            self.use_llm = bool(self.client)
            if self.use_llm:
                logger.info("OpenAI client initialized.")
            else:
                logger.warning("OPENAI_API_KEY not set. Falling back to local mock routing if needed.")
        except Exception as e:
            logger.exception("Failed to initialize OpenAI client: %s", e)
            self.client = None
            self.use_llm = False

        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # configurable default
        self.function_schemas = TOOL_FUNCTION_SCHEMAS          # single source of truth

    # -------------------------------
    # Helpers
    # -------------------------------

    def _error(self, msg: str) -> ExtractResponse:
        return {"success": False, "error": msg, "llm_response": ""}

    def _system_prompt(self) -> str:
        # NOTE: No duplicate schema here; rely solely on TOOL_FUNCTION_SCHEMAS for validation
        return f"""You are a specialized analytics agent.

You MUST do one of two things:
1) If the user's request is about analytics (data analysis, charts, success/fail rates, file data, statistics, visualizations),
   call exactly one or more of the available tools with correct parameters.
2) If the request is NOT analytics, reply EXACTLY with this JSON (no extra text):
{{"success": false, "message": "This is a specialized analytics agent. Please ask questions about data analysis, success rates, charts, or file data."}}

Parameter rules:
- If user asks "success rate" (singular) → show_only="success"
- If user asks "fail rate" → show_only="fail"
- If user asks "rates" (plural) or "both" → show_only="both"

Tool selection logic:
- If user asks for success/fail rates for a specific file → use get_success_rate_by_file_name

Database Schema Context (read-only):
{self.database_schema}
"""

    def _validate_tool_calls(self, calls: List[Dict[str, Any]]) -> ExtractResponse | None:
        """
        Validates tool calls against TOOL_FUNCTION_SCHEMAS.
        Returns an error/clarify response if invalid, or None if OK.
        """
        # Convert tools list to a dict for easier lookup
        function_schemas = {}
        for tool in self.function_schemas:
            if tool.get("type") == "function" and "function" in tool:
                func_def = tool["function"]
                function_schemas[func_def["name"]] = func_def["parameters"]
        
        for c in calls:
            name = c.get("name")
            if not name or name not in function_schemas:
                return {"success": False, "clarify": f"Tool '{name}' is not allowed.", "tool_calls": calls, "llm_response": ""}

            required = function_schemas[name].get("required", [])
            args = c.get("arguments") or {}
            # If args are a JSON string, parse safely here as well
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                    c["arguments"] = args
                except json.JSONDecodeError:
                    return {"success": False, "clarify": f"Invalid arguments for '{name}' (must be JSON).", "tool_calls": calls, "llm_response": ""}

            for r in required:
                if r not in args or args[r] in (None, ""):
                    return {"success": False, "clarify": f"Which {r} do you mean for {name}?", "tool_calls": calls, "llm_response": ""}

            # Optional: basic suspicious payload guard (lightweight)
            for k, v in args.items():
                if isinstance(v, str):
                    lowered = v.lower()
                    if any(tok in lowered for tok in ("rm -rf", "drop table", "shutdown", "format c:", "curl ")):
                        return {"success": False, "clarify": f"Argument '{k}' for tool '{name}' looks unsafe.", "tool_calls": calls, "llm_response": ""}

        return None  # Valid

    # -------------------------------
    # Public API
    # -------------------------------

    def extractPrompt(self, user_prompt: str) -> ExtractResponse:
        """
        Returns:
          {"success": True, "tool_calls": [...], "llm_response": "<model content>"} on success
          or {"success": False, "message": "..."} if non-analytics, per contract
          or {"success": False, "clarify": "..."} when params missing
          or {"success": False, "error": "..."} on internal errors
        """
        if not getattr(self, "database_schema", None):
            return self._error("Service not initialized: missing database schema.")

        if is_instruction_override(user_prompt):
            return {
                "success": False,
                "clarify": "Your message appears to override system instructions. Please rephrase your analytics request.",
                "llm_response": ""
            }

        # If LLM unavailable → fail to trigger reasoning engine fallback
        if not self.use_llm:
            # Raise an exception to trigger the fallback to reasoning engine in main.py
            raise Exception("LLM service unavailable - OpenAI API key not configured or client failed to initialize")

        # LLM path
        start = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": user_prompt}
                ],
                tools=TOOL_FUNCTION_SCHEMAS,
                temperature=0.1,
                # Some SDKs do not support 'timeout' directly on create; if so, wrap in your own timeout.
            )
        except Exception as e:
            logger.exception("LLM call failed")
            return self._error(f"LLM call failed: {e}")

        latency_ms = int((time.time() - start) * 1000)
        logger.info("llm_latency_ms=%d model=%s", latency_ms, self.model)

        choice = resp.choices[0]
        msg = choice.message
        content = (msg.content or "").strip()
        tool_calls_raw = msg.tool_calls or []

        # If model returned the JSON “not analytics” message, pass through untouched
        if content.startswith("{") and '"success"' in content:
            try:
                parsed = json.loads(content)
                if parsed.get("success") is False and "message" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        # Normalize tool calls to canonical shape
        norm_calls: List[Dict[str, Any]] = []
        for tc in tool_calls_raw:
            try:
                name = tc.function.name
                args_str = tc.function.arguments or "{}"
                args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
                norm_calls.append({"name": name, "arguments": args})
            except Exception:
                norm_calls.append({"name": getattr(tc.function, "name", None), "arguments": {}})

        # Validate against TOOL_FUNCTION_SCHEMAS
        invalid = self._validate_tool_calls(norm_calls)
        if invalid:  # returns an error/clarify envelope
            return invalid

        if norm_calls:
            return {"success": True, "tool_calls": norm_calls, "llm_response": content}

        # No tool calls and no structured rejection → ask for clarification
        return {
            "success": False,
            "llm_response": content,
            "clarify": "Please ask for data records, success/fail rates, or available files (e.g., 'Show success rate for file X')."
        }

    def respond_with_chart(self, query_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the presence of chart_type and chart_data.
        """
        if not {"chart_type", "chart_data"} <= set(query_result):
            return {"success": False, "error": "Invalid query result format.", "fallback_response": "Error processing the request."}
        return query_result

# Instantiate for importers
llm_service = LLMService()