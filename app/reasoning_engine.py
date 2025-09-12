from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re
from enum import Enum

class QueryType(Enum):
    DATA_RETRIEVAL = "data_retrieval"
    SUCCESS_RATE_ANALYSIS = "success_rate_analysis"
    CHART_GENERATION = "chart_generation"
    MULTI_STEP = "multi_step"
    UNKNOWN = "unknown"

@dataclass
class Step:
    tool: str
    params: Dict[str, Any]
    expects: str
    postconditions: Optional[List[str]] = None
    fallback: Optional["Step"] = None

@dataclass
class Plan:
    execution_strategy: str
    steps: List[Step]
    expected_output: str
    notes: Optional[str] = None

class ReasoningEngine:
    """
    Advanced reasoning and planning workflow for the analytics agent.
    Analyzes user queries, decomposes complex requests, and plans multi-step execution.
    """
    
    # Priority order for query type resolution
    PRIORITY = [
        QueryType.CHART_GENERATION,
        QueryType.SUCCESS_RATE_ANALYSIS,
        QueryType.DATA_RETRIEVAL,
        QueryType.MULTI_STEP,
    ]

    def __init__(self):
        # Pre-compiled patterns for performance
        self.PAT_INTENTS = {
            QueryType.DATA_RETRIEVAL: [
                re.compile(r'\b(show|get|find|list|retrieve)\b.*\b(records?|data|entries)\b(?!.*\b(rate|percentage|ratio)\b)', re.I),
                re.compile(r'\b(records?|data|entries)\b.*\b(for|in|from)\b(?!.*\b(success|fail|failure)\b)', re.I),
                re.compile(r'\bstatus\b.*\b(records?|data|table)\b(?!.*\b(rate|percentage|ratio)\b)', re.I),
                re.compile(r'\b(list|show)\b.*\b(all|every)\b.*\b(records?|entries)\b', re.I),
            ],
            QueryType.SUCCESS_RATE_ANALYSIS: [
                re.compile(r'\b(success|failure|fail)\b.*\b(rate|percentage|ratio)\b', re.I),
                re.compile(r'\b(calculate|compute|analy[sz]e)\b.*\b(success|performance)\b', re.I),
                re.compile(r'\bhow\s+(many|much)\b.*\b(successful|failed)\b', re.I),
                re.compile(r'\b(show|get)\b.*\b(success|fail|failure)\b.*\b(rate|percentage)\b', re.I),
                # Enhanced patterns for implicit rate queries
                re.compile(r'\b(show|get|display)\b.*\b(success|fail|failure)\b.*\b(for|in|from)\b.*\bfile\b', re.I),
                re.compile(r'\b(success|fail|failure)\b.*\b(status|result)\b.*\b(for|of)\b', re.I),
                re.compile(r'\b(how|what)\b.*\b(success|fail|failure)\b', re.I),
                re.compile(r'\b(check|see|view)\b.*\b(success|fail|failure)\b', re.I),
            ],
            QueryType.CHART_GENERATION: [
                re.compile(r'\b(chart|graph|visuali[sz]e|plot|diagram)\b', re.I),
                re.compile(r'\b(show|create|generate|make)\b.*\b(chart|graph)\b', re.I),
            ],
        }
        
        # Entity extraction patterns
        self.RE_FILE = re.compile(r'(?:"|"|")?([\w\-\s./]+?\.(?:csv|json|xlsx|parquet))(?:"|"|")?', re.I)
        self.RE_FILE_FALLBACKS = [
            re.compile(r'file\s+["\']?([\w\-\s./]+(?:\.\w+)?)["\']?', re.I),
            re.compile(r'for\s+["\']?([\w\-\s./]+(?:\.\w+)?)["\']?', re.I),
        ]
        self.RE_SUCCESS = re.compile(r'\b(success|successful|succeeded)\b', re.I)
        self.RE_FAIL = re.compile(r'\b(fail|failed|failure|error)\b', re.I)
        self.RE_SEQUENCE = re.compile(r'\b(then|after that|also|and then)\b', re.I)
        self.RE_GROUPBY = re.compile(r'\bby\s+([a-z_][\w\s-]+)\b', re.I)
        self.RE_CHART_TYPE = re.compile(r'\b(bar|line|pie|scatter)\b', re.I)
        
        # Tool registry for validation and planning
        self.TOOL_REGISTRY = {
            "get_success_rate_by_file_name": {"required": ["file_name"], "expects": "analytics"},
        }
    
    def analyze_query(self, user_prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Advanced query analysis with multi-entity extraction and rationale tracking.
        """
        prompt = user_prompt
        plower = prompt.lower()

        # Intent scoring with pattern matching
        scores: Dict[QueryType, int] = {qt: 0 for qt in QueryType}
        matches: Dict[QueryType, List[str]] = {qt: [] for qt in QueryType}
        
        for qt, patterns in self.PAT_INTENTS.items():
            for pat in patterns:
                if pat.search(plower):
                    scores[qt] += 1
                    matches[qt].append(pat.pattern)

        # Choose best type deterministically using priority
        max_score = max(scores.values()) if scores else 0
        qtype = QueryType.UNKNOWN if max_score == 0 else next(
            (qt for qt in self.PRIORITY if scores.get(qt, 0) == max_score), QueryType.UNKNOWN
        )

        # Entity extraction (single pass for efficiency)
        file_name = self._extract_file_name(prompt, context)
        status = self._extract_status(plower)  # returns List[str] or None
        group_by = self._extract_group_by(plower)
        chart_type = self._extract_chart_type(plower)

        # Complexity assessment
        complexity = self._assess_complexity(plower, scores, bool(self.RE_SEQUENCE.search(plower)))

        # Confidence calculation with rationale
        confidence = min(
            0.4 * (qtype != QueryType.UNKNOWN)
            + 0.3 * bool(file_name)
            + 0.2 * bool(status)
            + 0.1 * (qtype in [QueryType.DATA_RETRIEVAL, QueryType.CHART_GENERATION]),
            1.0
        )
        
        rationale = {qt.value: matches[qt] for qt in self.PAT_INTENTS}

        return {
            "original_prompt": user_prompt,
            "query_type": qtype,
            "entities": {
                "file_name": file_name,
                "status": status,
                "group_by": group_by,
                "chart_type": chart_type,
            },
            "complexity": complexity,
            "confidence": confidence,
            "rationale": rationale,
            "requires_decomposition": complexity == "high",
            "has_sequence_markers": bool(self.RE_SEQUENCE.search(plower)),
        }
    
    def plan_execution(self, analysis: Dict[str, Any]) -> Plan:
        """
        Advanced execution planning with multi-step workflows and auto-resolution.
        """
        qtype: QueryType = analysis["query_type"]
        e = analysis["entities"]
        seq = analysis["has_sequence_markers"]
        missing_file = not e.get("file_name")

        steps: List[Step] = []

        # Auto-resolution for missing critical entities
        if missing_file:
            steps.append(Step("list_available_files", {}, "file_list"))

        # Plan execution steps based on query type
        if qtype in (QueryType.DATA_RETRIEVAL, QueryType.UNKNOWN):
            # Check if this is actually a success/fail query that should get rate data
            original_query = analysis.get("original_prompt", "")
            has_success_fail_keywords = bool(e.get("status")) or bool(self.RE_SUCCESS.search(original_query)) or bool(self.RE_FAIL.search(original_query))
            
            if has_success_fail_keywords and e.get("file_name"):
                # Treat as implicit success rate query
                show_only = "both"  # default
                if e.get("status"):
                    status_list = e.get("status")
                    if status_list == ["success"]:
                        show_only = "success"
                    elif status_list == ["fail"]:
                        show_only = "fail"
                
                steps.append(Step(
                    "get_success_rate_by_file_name",
                    {"file_name": e.get("file_name"), "show_only": show_only},
                    "analytics",
                    postconditions=["rate_in_[0,1]"]
                ))
            elif e.get("status"):
                # Explicit record retrieval
                steps.append(Step(
                    "get_records_by_status",
                    {"file_name": e.get("file_name"), "status": e.get("status")},
                    "tabular_data",
                    postconditions=["non_empty"]
                ))
                
        if qtype in (QueryType.SUCCESS_RATE_ANALYSIS, QueryType.CHART_GENERATION):
            # Determine show_only parameter based on extracted status
            show_only = "both"  # default
            if e.get("status"):
                status_list = e.get("status")
                if status_list == ["success"]:
                    show_only = "success"
                elif status_list == ["fail"]:
                    show_only = "fail"
                # If both success and fail are mentioned, keep "both"
            
            steps.append(Step(
                "get_success_rate_by_file_name",
                {"file_name": e.get("file_name"), "show_only": show_only},
                "analytics",
                postconditions=["rate_in_[0,1]"]
            ))
            
        if qtype == QueryType.CHART_GENERATION or seq:
            steps.append(Step(
                "render_chart",
                {"metric": "success_rate", "chart_type": e.get("chart_type") or "bar", "group_by": e.get("group_by")},
                "chart"
            ))

        # Fallback if no steps planned
        if not steps:
            steps = [Step("list_available_files", {}, "file_list")]

        # Determine expected output and strategy
        expected = "chart_with_data" if any(s.tool == "render_chart" for s in steps) else steps[-1].expects
        
        return Plan(
            execution_strategy="multi_tool" if len(steps) > 1 else "single_tool",
            steps=steps,
            expected_output=expected,
            notes="Auto-added file listing step due to missing filename" if missing_file else None,
        )
    
    # ---------- Helper Methods ----------
    
    def _extract_file_name(self, prompt: str, context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Enhanced file name extraction with multiple fallback patterns."""
        # Primary pattern for common file extensions
        m = self.RE_FILE.search(prompt)
        if m: 
            return m.group(1).strip()
            
        # Fallback patterns
        for pat in self.RE_FILE_FALLBACKS:
            m2 = pat.search(prompt)
            if m2: 
                return m2.group(1).strip()
                
        # Context fallback
        if context and context.get("last_file_queried"):
            return context["last_file_queried"]
            
        return None

    def _extract_status(self, plower: str) -> Optional[List[str]]:
        """Extract status filters, supporting multiple statuses."""
        hits = []
        if self.RE_SUCCESS.search(plower): 
            hits.append("success")
        if self.RE_FAIL.search(plower): 
            hits.append("fail")
        return hits or None

    def _extract_group_by(self, plower: str) -> Optional[str]:
        """Extract group-by field for aggregation."""
        m = self.RE_GROUPBY.search(plower)
        return m.group(1).strip() if m else None

    def _extract_chart_type(self, plower: str) -> Optional[str]:
        """Extract preferred chart type."""
        m = self.RE_CHART_TYPE.search(plower)
        return m.group(1).lower() if m else None

    def _assess_complexity(self, plower: str, scores: Dict[QueryType, int], has_seq: bool) -> str:
        """Enhanced complexity assessment with multiple indicators."""
        indicators = 0
        
        # Sequence markers indicate multi-step
        if has_seq: 
            indicators += 1
            
        # Multiple query types suggest complexity
        if scores.get(QueryType.DATA_RETRIEVAL, 0) and scores.get(QueryType.CHART_GENERATION, 0): 
            indicators += 1
            
        # Conjunction words suggest multiple operations
        if " and " in plower or " also " in plower: 
            indicators += 1
            
        return "high" if indicators >= 2 else ("medium" if indicators == 1 else "low")

    # ---------- Legacy Method Support (for backward compatibility) ----------
    
    # Legacy methods removed - they were not being used in the current codebase
    # The modern plan_execution() method handles all planning logic

# Initialize reasoning engine
reasoning_engine = ReasoningEngine()
