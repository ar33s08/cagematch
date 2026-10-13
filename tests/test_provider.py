"""Provider layer: payload parsing, None-content handling, transport failures."""

from __future__ import annotations

import pytest

from cagematch.provider import OpenAICompatProvider, ProviderError, parse_completion


def test_parse_plain_completion():
    payload = {"choices": [{"message": {"content": "hello"}}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 3}}
    c = parse_completion(payload, 5.0)
    assert c.content == "hello"
    assert c.tool_calls == []
    assert (c.prompt_tokens, c.completion_tokens, c.latency_ms) == (10, 3, 5.0)


def test_parse_content_none_with_reasoning_present():
    # vLLM thinking stacks return content=None when reasoning ate the budget
    payload = {"choices": [{"message": {"content": None, "reasoning": "lots of thinking"}}]}
    c = parse_completion(payload, 1.0)
    assert c.content is None
    assert c.reasoning == "lots of thinking"  # scores as lost, never a crash


def test_parse_malformed_raises_provider_error():
    with pytest.raises(ProviderError):
        parse_completion({"nope": 1}, 1.0)


def test_parse_unparseable_tool_arguments_becomes_finding():
    payload = {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "t1", "function": {"name": "calc", "arguments": "{oops"}}]}}]}
    c = parse_completion(payload, 1.0)
    assert c.tool_calls[0].arguments == {"__unparseable__": "{oops"}


def test_headers_built_from_env_name_not_literal(monkeypatch):
    monkeypatch.setenv("CM_TEST_KEY", "sekrit")
    p = OpenAICompatProvider("http://example.invalid/v1", name="t",
                             api_key_env="CM_TEST_KEY")
    assert p._headers()["Authorization"] == "Bearer " + "sekrit"
    monkeypatch.delenv("CM_TEST_KEY")
    assert "Authorization" not in p._headers()


def test_transport_error_wrapped(monkeypatch):
    p = OpenAICompatProvider("http://127.0.0.1:1/v1", name="dead")
    with pytest.raises(ProviderError) as excinfo:
        p.complete([], [], "m")
    assert "transport error" in str(excinfo.value)
