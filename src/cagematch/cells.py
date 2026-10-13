"""Cell matrix: provider + model + system prompt + loop policy = one contender.

Config is data, not code, so adding a provider or a harness variant is a JSON
edit. API keys live ONLY in environment variables named by api_key_env.
"""

from __future__ import annotations

import json
from pathlib import Path

from cagematch.agent import DEFAULT_PRICING, CellConfig
from cagematch.provider import OpenAICompatProvider

BASE_SYSTEM = (
    "Work only with the provided tools. Use series_stats for aggregate questions "
    "instead of eyeballing rows. One tool call may be wrong: read TOOL_ERROR text "
    "and adjust. Always end by calling finish with the required answer object."
)

SERIAL_SYSTEM = (
    BASE_SYSTEM + " Issue exactly one tool call per turn."
)


def default_cells() -> list:
    """The standing derby. Live cells self-skip when their env vars are absent."""
    return [
        {
            "name": "gpt",
            "kind": "live",
            "base_url_env": "OPENAI_BASE_URL",
            "base_url_default": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "model": "gpt-4.1-mini",
            "system_prompt": BASE_SYSTEM,
            "parallel_tools": True,
        },
        {
            "name": "claude",
            "kind": "live",
            "base_url_env": "ANTHROPIC_BASE_URL",
            "base_url_default": "https://api.anthropic.com/v1",
            "api_key_env": "ANTHROPIC_API_KEY",
            "model": "claude-sonnet-4-5",
            "system_prompt": BASE_SYSTEM,
            "parallel_tools": True,
        },
        {
            "name": "spark-qwen",
            "kind": "live",
            "base_url_env": "SPARK_BASE_URL",
            "base_url_default": "http://127.0.0.1:18300/v1",
            "api_key_env": None,
            "model": "qwen3.8-flash-next",
            "system_prompt": BASE_SYSTEM,
            "parallel_tools": True,
        },
        {
            "name": "gpt-serial",
            "kind": "live",
            "base_url_env": "OPENAI_BASE_URL",
            "base_url_default": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "model": "gpt-4.1-mini",
            "system_prompt": SERIAL_SYSTEM,
            "parallel_tools": False,
        },
        {
            "name": "scripted",
            "kind": "scripted",
            "transcript": "scripted",
            "system_prompt": BASE_SYSTEM,
            "parallel_tools": True,
        },
    ]


def scripted_transcript(task_id: str) -> list:
    """A correct-but-not-optimal scripted run per task (test double, CI only)."""
    if task_id == "drawdown-spread":
        return [
            {"tool_calls": [{"name": "series_stats", "arguments":
                            {"file": "data/spy_daily.csv", "column": "close",
                             "stat": "max_drawdown"}}]},
            {"tool_calls": [{"name": "series_stats", "arguments":
                            {"file": "data/qqq_daily.csv", "column": "close",
                             "stat": "max_drawdown"}}]},
            {"content": "computing which fell deeper",
             "tool_calls": [{"name": "calc", "arguments":
                            {"expr": "0.0"}}]},
            {"tool_calls": [{"name": "finish", "arguments": {"answer": {
                "spy_drawdown": -0.253606, "qqq_drawdown": -0.356172,
                "worst_asset": "qqq"}}}]},
        ]
    if task_id == "recent-window":
        return [
            # deliberate wrong-file probe: exercises the TOOL_ERROR recovery path
            {"tool_calls": [{"name": "series_stats", "arguments":
                            {"file": "data/sp500_daily.csv", "column": "close",
                             "stat": "mean"}}]},
            {"tool_calls": [{"name": "series_stats", "arguments":
                            {"file": "data/spy_daily.csv", "column": "close",
                             "stat": "last"}}]},
            {"tool_calls": [{"name": "calc", "arguments":
                            {"expr": "777.859924 - 0.0"}}]},
            {"content": "deriving the trailing mean over the last 252 rows",
             "tool_calls": [{"name": "read_data_csv", "arguments":
                            {"file": "data/spy_daily.csv", "offset": 1003, "limit": 252}}]},
            {"tool_calls": [{"name": "finish", "arguments": {"answer": {
                "spy_252_mean": 713.999643, "fred_last": 7811.54}}}]},
        ]
    raise KeyError(f"no scripted transcript for {task_id}")


def _env(name, default):
    import os
    if name and os.environ.get(name):
        return os.environ[name]
    return default


def build_cell(spec: dict):
    """Materialize a CellConfig; returns None when a live cell's key is absent."""
    kind = spec.get("kind", "live")
    max_steps = spec.get("max_steps", 16)
    max_cost = spec.get("max_cost_usd", 0.50)
    if kind == "scripted":
        return CellConfig(
            name=spec["name"], provider=None, model="scripted",
            system_prompt=spec.get("system_prompt", BASE_SYSTEM),
            parallel_tools=spec.get("parallel_tools", True),
            max_steps=max_steps, max_cost_usd=max_cost,
            pricing=DEFAULT_PRICING.get("scripted", {}),
        )
    import os
    base_url = _env(spec.get("base_url_env"), spec.get("base_url_default", ""))
    key_env = spec.get("api_key_env")
    if key_env and not os.environ.get(key_env):
        return None  # cell not available in this environment
    if not key_env and spec.get("base_url_env") and not os.environ.get(spec["base_url_env"]):
        # self-hosted cell with no credential gate: enable it only when the
        # operator explicitly points it somewhere (keeps bare `cagematch run`
        # hermetic instead of failing on a refused localhost connection)
        return None
    provider = OpenAICompatProvider(base_url=base_url, name=spec["name"],
                                    api_key_env=key_env)
    return CellConfig(
        name=spec["name"], provider=provider, model=spec["model"],
        system_prompt=spec.get("system_prompt", BASE_SYSTEM),
        parallel_tools=spec.get("parallel_tools", True),
        max_steps=max_steps, max_cost_usd=max_cost,
        pricing=spec.get("pricing", DEFAULT_PRICING.get(spec["name"],
                        {"prompt": 0.0, "completion": 0.0})),
    )


def load_cell_matrix(path: Path | None) -> list:
    if path is None:
        return default_cells()
    return json.loads(Path(path).read_text(encoding="utf-8"))
