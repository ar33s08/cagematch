"""The agent loop shared by every cell — with budgets that actually bite.

A harness cell = provider + model + system prompt + loop policy (parallel vs
serial tool execution, error surfacing style). The loop enforces step and cost
budgets BEFORE dispatching the next call, so a rambling cell fails with a
budget finding instead of burning the clock.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from cagematch.provider import OpenAICompatProvider, ProviderError
from cagematch.tools import TOOLS_SCHEMA, ToolError, make_tool_registry
from cagematch.trace import RunStore

# USD per 1M tokens, keyed by provider cell name; unknown cells default to
# zero-cost (self-hosted) but still record tokens.
DEFAULT_PRICING = {
    "gpt": {"prompt": 2.50, "completion": 10.00},
    "claude": {"prompt": 3.00, "completion": 15.00},
    "spark-qwen": {"prompt": 0.0, "completion": 0.0},
    "scripted": {"prompt": 0.0, "completion": 0.0},
}


@dataclass
class CellConfig:
    name: str
    provider: OpenAICompatProvider
    model: str
    system_prompt: str
    parallel_tools: bool = True
    max_steps: int = 16
    max_cost_usd: float = 0.50
    pricing: dict = field(default_factory=lambda: {"prompt": 0.0, "completion": 0.0})


@dataclass
class RunOutcome:
    run_id: str
    status: str            # finished | budget_steps | budget_cost | provider_error | no_finish | error
    answer: dict | None
    steps: int
    cost_usd: float
    wall_ms: float
    tool_errors: int
    detail: str = ""


def _tool_result_text(fn, call) -> tuple[bool, str]:
    try:
        result = fn(**call.arguments)
        return True, json.dumps(result, default=str)
    except ToolError as exc:
        return False, f"TOOL_ERROR: {exc}"
    except TypeError as exc:
        return False, f"TOOL_ERROR: bad arguments: {exc}"
    except Exception as exc:  # noqa: BLE001 - tool bugs must surface to the model, not crash the cell
        return False, f"TOOL_ERROR: internal: {exc.__class__.__name__}: {exc}"


def run_task(task: dict, cell: CellConfig, store: RunStore, data_dir, ws_dir,
             tool_registry=None) -> RunOutcome:
    run_id = f"{task['id']}__{cell.name}"
    registry = tool_registry or make_tool_registry(data_dir, ws_dir)
    store.start_run(run_id, task["id"], cell.name,
                    {"model": cell.model, "parallel_tools": cell.parallel_tools,
                     "max_steps": cell.max_steps, "max_cost_usd": cell.max_cost_usd})
    messages = [
        {"role": "system", "content": task.get("system", "") + "\n\n" + cell.system_prompt},
        {"role": "user", "content": task["prompt"]},
    ]
    store.append(run_id, {"kind": "start", "messages": messages})

    started = time.monotonic()
    spent = 0.0
    steps = 0
    tool_errors = 0
    status, answer, detail = "no_finish", None, "model never called finish()"

    while True:
        if steps >= cell.max_steps:
            status, detail = "budget_steps", f"step budget exhausted at {cell.max_steps}"
            break
        if spent >= cell.max_cost_usd:
            status, detail = "budget_cost", f"cost budget exhausted (${spent:.4f} >= ${cell.max_cost_usd})"
            break
        try:
            completion = cell.provider.complete(messages, TOOLS_SCHEMA, cell.model)
        except ProviderError as exc:
            status, detail = "provider_error", str(exc)[:300]
            store.append(run_id, {"kind": "provider_error", "detail": detail})
            break
        steps += 1
        spent += (completion.prompt_tokens / 1e6) * cell.pricing.get("prompt", 0.0)
        spent += (completion.completion_tokens / 1e6) * cell.pricing.get("completion", 0.0)
        store.append(run_id, {
            "kind": "completion", "step": steps, "cost_usd": spent,
            "content": completion.content,
            "reasoning_present": completion.reasoning is not None,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments}
                           for c in completion.tool_calls],
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "latency_ms": round(completion.latency_ms, 1),
        })

        assistant_msg = {"role": "assistant", "content": completion.content}
        if completion.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in completion.tool_calls
            ]
        messages.append(assistant_msg)

        if not completion.tool_calls:
            # A completion with no tool call and no usable answer is a dead turn.
            if completion.content:
                detail = f"model answered in prose without finish(): {completion.content[:200]}"
            break

        calls = list(completion.tool_calls)
        if not cell.parallel_tools:
            calls = calls[:1]  # serial policy: dispatch one tool per turn
        finished_here = None
        for call in calls:
            fn = registry.get(call.name)
            if fn is None:
                ok, text = False, f"TOOL_ERROR: unknown tool '{call.name}'"
            else:
                ok, text = _tool_result_text(fn, call)
            if not ok:
                tool_errors += 1
            store.append(run_id, {"kind": "tool_result", "step": steps, "call_id": call.id,
                                  "name": call.name, "ok": ok, "result": text[:4000]})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
            if call.name == "finish" and ok:
                finished_here = call.arguments.get("answer")
        if finished_here is not None:
            status, answer, detail = "finished", finished_here, ""
            break

    store.append(run_id, {"kind": "end", "status": status, "answer": answer,
                          "steps": steps, "cost_usd": spent, "tool_errors": tool_errors})
    store.finish_run(run_id)
    return RunOutcome(run_id=run_id, status=status, answer=answer, steps=steps,
                      cost_usd=spent, wall_ms=(time.monotonic() - started) * 1000.0,
                      tool_errors=tool_errors, detail=detail)
