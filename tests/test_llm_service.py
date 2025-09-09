import pytest
from app.llm_service import LLMService


class DummyResponse:
    def __init__(self, content, tool_calls):
        class Choice:
            def __init__(self, content, tool_calls):
                class Message:
                    def __init__(self, content, tool_calls):
                        self.content = content
                        self.tool_calls = tool_calls
                self.message = Message(content, tool_calls)
        self.choices = [Choice(content, tool_calls)]


class DummyClient:
    def __init__(self, response):
        self.response = response

    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        return self.response


def test_mock_fallback_returns_tool_calls():
    svc = LLMService()
    svc.client = None
    svc.use_llm = False

    out = svc.extractPrompt("list files")
    assert out["success"] is True
    assert isinstance(out["tool_calls"], list)
    assert out["tool_calls"][0]["name"] == "list_available_files"


def test_normalize_tool_calls_from_dict_shape():
    svc = LLMService()
    # Simulate client returning dict-shaped tool call
    resp = DummyResponse("ok", [{"function": {"name": "get_success_rate_by_file_name", "arguments": {"file_name": "a.csv"}}}])
    svc.client = DummyClient(resp)
    svc.use_llm = True

    out = svc.extractPrompt("success rate for a.csv")
    assert out["success"] is True
    assert isinstance(out["tool_calls"], list)
    tc = out["tool_calls"][0]
    assert tc["name"] == "get_success_rate_by_file_name"
    assert tc["arguments"]["file_name"] == "a.csv"


def test_normalize_tool_calls_from_flat_shape():
    svc = LLMService()
    resp = DummyResponse("ok", [{"name": "list_available_files", "arguments": {}}])
    svc.client = DummyClient(resp)
    svc.use_llm = True

    out = svc.extractPrompt("list files")
    assert out["success"] is True
    assert out["tool_calls"][0]["name"] == "list_available_files"
