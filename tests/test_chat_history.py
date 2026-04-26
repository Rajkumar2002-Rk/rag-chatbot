"""Tests for api/rag_api.py — chat history formatting."""
from api.rag_api import _format_chat_history


def test_none_returns_none_string():
    assert _format_chat_history(None) == "None"


def test_empty_returns_none_string():
    assert _format_chat_history([]) == "None"


def test_truncates_long_assistant():
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "A" * 500},
    ]
    result = _format_chat_history(messages)
    assert "..." in result
    # Should not contain the full 500-char string
    assert "A" * 300 not in result


def test_keeps_last_six_messages():
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
        for i in range(10)
    ]
    result = _format_chat_history(messages)
    # Messages 0-3 should be excluded, 4-9 should be present
    assert "msg4" in result
    assert "msg9" in result
    assert "msg3" not in result
