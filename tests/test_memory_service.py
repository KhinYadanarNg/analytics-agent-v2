import pytest
from datetime import datetime, timedelta

from app.memory_service import ConversationMemory


def test_create_session_and_context():
    mem = ConversationMemory()
    session_id = mem.create_session("user-123")
    assert isinstance(session_id, str)
    ctx = mem.get_session_context(session_id)
    assert ctx["user_id"] == "user-123"
    assert ctx["interaction_count"] == 0


def test_store_interaction_and_history_limit():
    mem = ConversationMemory()
    mem.max_session_history = 3
    sid = mem.create_session("u1")
    for i in range(5):
        ok = mem.store_interaction(sid, f"q{i}", "tool", {"success": True, "tool": "t", "file_name": f"f{i}", "row_count": i})
        assert ok
    hist = mem.get_conversation_history(sid)
    assert len(hist) == 3  # limited to last 3
    latest = hist[-1]
    assert latest["user_prompt"] == "q4"


def test_cleanup_old_sessions():
    mem = ConversationMemory()
    sid = mem.create_session("u2")
    # artificially set created_at to older than threshold
    mem.sessions[sid]["created_at"] = datetime.now() - timedelta(hours=48)
    removed = mem.cleanup_old_sessions(max_age_hours=24)
    assert removed == 1
