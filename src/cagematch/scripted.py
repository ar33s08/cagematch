"""Deterministic scripted provider for hermetic CI runs.

The full agent loop, trace chain, and eval gate run in CI with no network and
no API keys. ScriptedProvider replays a fixed transcript of completions,
consuming one per provider call. It is a TEST DOUBLE and is labelled as such
everywhere; it must never be used to judge a real cell.
"""

from __future__ import annotations

from cagematch.provider import Completion, ToolCall


class ScriptedProvider:
    simulated = True

    def __init__(self, name: str, transcript: list):
        self.name = name
        self.transcript = list(transcript)
        self.calls: list[dict] = []
        self._index = 0

    def complete(self, messages: list, tools: list, model: str, max_tokens: int = 1024,
                 temperature: float = 0.0) -> Completion:
        self.calls.append({"n_messages": len(messages), "model": model})
        if self._index >= len(self.transcript):
            return Completion(content=None, tool_calls=[], prompt_tokens=0,
                              completion_tokens=0, latency_ms=0.0, simulated=True)
        step = self.transcript[self._index]
        self._index += 1
        tool_calls = [ToolCall(id=f"call_{self._index}_{i}", name=t["name"],
                               arguments=dict(t["arguments"]))
                      for i, t in enumerate(step.get("tool_calls", []))]
        return Completion(content=step.get("content"), tool_calls=tool_calls,
                          prompt_tokens=step.get("prompt_tokens", 200),
                          completion_tokens=step.get("completion_tokens", 50),
                          latency_ms=1.0, simulated=True)
