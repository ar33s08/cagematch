"""Minimal OpenAI-compatible chat-completions client (stdlib only).

Zero third-party runtime dependencies on purpose: an eval harness must run on
any machine that has Python 3.10+, including locked-down corporate or trading
floor workstations where pip installs are a negotiation.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


class ProviderError(RuntimeError):
    """Raised when the upstream endpoint answers non-2xx or malformed."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Completion:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    simulated: bool = False


def extract_text(message: dict) -> str | None:
    """Pull answer text out of a chat message, tolerating reasoning stacks.

    vLLM-style servers that run thinking models can return content=None when
    the token budget was consumed by reasoning. Treating None as a crash is a
    harness bug; the reasoning field is inspected as a fallback so a
    budget-starved reply scores as lost, never as an exception.
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    for key in ("reasoning", "reasoning_content"):
        alt = message.get(key)
        if isinstance(alt, str) and alt.strip():
            return None  # keep None: answer lost, but reasoning retained by caller
    return None


def parse_completion(payload: dict, latency_ms: float) -> Completion:
    try:
        choice = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"malformed completion payload: {exc!r}") from exc
    tool_calls = []
    for call in choice.get("tool_calls") or []:
        fn = call.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                args = {"__unparseable__": raw_args}
        else:
            args = raw_args
        tool_calls.append(ToolCall(id=str(call.get("id", "")), name=str(fn.get("name", "")),
                                   arguments=args))
    usage = payload.get("usage") or {}
    reasoning = choice.get("reasoning") or choice.get("reasoning_content")
    return Completion(
        content=extract_text(choice),
        tool_calls=tool_calls,
        reasoning=reasoning if isinstance(reasoning, str) else None,
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        latency_ms=latency_ms,
    )


class OpenAICompatProvider:
    """One cell in the derby: any /v1/chat/completions endpoint.

    Auth is read from an environment variable NAME (never a literal key in
    config or code). Self-hosted endpoints on localhost/tailnet may run with
    no key at all.
    """

    def __init__(self, base_url: str, name: str = "cell", api_key_env: str | None = None,
                 timeout_s: float = 120.0, extra_payload: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        # per-provider quirks that ride in the request body, e.g. vLLM thinking
        # models need {"chat_template_kwargs": {"enable_thinking": false}} or
        # every call spends the whole budget on reasoning (60s+ per call, and
        # content comes back null). Quirks belong in the cell config, not in
        # loop code.
        self.extra_payload = dict(extra_payload or {})

    @property
    def url(self) -> str:
        return self.base_url + "/chat/completions"

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env or "", "")
            if key:
                headers["Authorization"] = "Bearer " + key
        return headers

    def complete(self, messages: list, tools: list, model: str, max_tokens: int = 1024,
                 temperature: float = 0.0) -> Completion:
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        body.update(self.extra_payload)
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"),
                                     headers=self._headers(), method="POST")
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise ProviderError(f"{self.name}: HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"{self.name}: transport error: {exc!r}") from exc
        latency_ms = (time.monotonic() - started) * 1000.0
        return parse_completion(payload, latency_ms)
