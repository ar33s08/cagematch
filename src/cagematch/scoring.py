"""Evidence-locked scoring: ground truth is recomputed, never trusted from the agent.

Two independent locks:
1. NUMERIC — expected values are recomputed from the raw data files at scoring
   time (the agent's own numbers are never the reference).
2. REPLAY — every tool call recorded in a run's trace is re-executed against
   the same data files and compared to the stored result. A cell cannot score
   well on numbers it did not actually derive.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from cagematch.agent import _tool_result_text
from cagematch.provider import ToolCall
from cagematch.tools import make_tool_registry


def load_csv_column(path: Path, column: str) -> list[float]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for row in rows:
        raw = row.get(column, "")
        if raw is None or str(raw).strip() in ("", "."):
            continue
        out.append(float(raw))
    return out


def max_drawdown(values: list[float]) -> float:
    peak, worst = values[0], 0.0
    for v in values:
        peak = max(peak, v)
        worst = min(worst, v / peak - 1.0)
    return worst


def ground_truth(task_id: str, data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    if task_id == "drawdown-spread":
        spy = load_csv_column(data_dir / "spy_daily.csv", "close")
        qqq = load_csv_column(data_dir / "qqq_daily.csv", "close")
        return {"spy_drawdown": max_drawdown(spy), "qqq_drawdown": max_drawdown(qqq)}
    if task_id == "recent-window":
        spy = load_csv_column(data_dir / "spy_daily.csv", "close")
        fred = load_csv_column(data_dir / "sp500_daily.csv", "SP500")
        return {"spy_252_mean": sum(spy[-252:]) / 252, "fred_last": fred[-1]}
    raise KeyError(f"no ground truth for task {task_id}")


def _close(a, b, tol) -> bool:
    if a is None or b is None:
        return False
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)
    except (TypeError, ValueError):
        return False


def score_answer(task_id: str, answer: dict | None, data_dir: Path) -> dict:
    truth = ground_truth(task_id, Path(data_dir))
    checks = {}
    if task_id == "drawdown-spread":
        checks["spy_drawdown"] = _close((answer or {}).get("spy_drawdown"), truth["spy_drawdown"], 1e-4)
        checks["qqq_drawdown"] = _close((answer or {}).get("qqq_drawdown"), truth["qqq_drawdown"], 1e-4)
        worst_expected = "qqq" if truth["qqq_drawdown"] < truth["spy_drawdown"] else "spy"
        checks["worst_asset"] = str((answer or {}).get("worst_asset", "")).lower() == worst_expected
    elif task_id == "recent-window":
        checks["spy_252_mean"] = _close((answer or {}).get("spy_252_mean"), truth["spy_252_mean"], 1e-2)
        checks["fred_last"] = _close((answer or {}).get("fred_last"), truth["fred_last"], 1e-2)
    else:
        raise KeyError(f"no scorer for task {task_id}")
    return {"checks": checks, "passed": all(checks.values()), "n_checks": len(checks),
            "n_passed": sum(checks.values())}


def replay_run(store, run_id: str, data_dir: Path, ws_dir: Path) -> dict:
    """Re-execute every recorded tool call; divergence means the score is unproven."""
    registry = make_tool_registry(Path(data_dir), Path(ws_dir))
    events = store.events(run_id)
    total, matched, diverged = 0, 0, []
    for ev in events:
        if ev.get("kind") != "completion":
            continue
        for call in ev.get("tool_calls", []):
            if call["name"] == "finish":
                continue
            total += 1
            fn = registry.get(call["name"])
            if fn is None:
                diverged.append({"step": ev.get("step"), "tool": call["name"],
                                 "why": "tool no longer exists"})
                continue
            try:
                _ok, fresh_text = _tool_result_text(fn, ToolCall(id="", name=call["name"],
                                                                arguments=call["arguments"]))
                fresh_text = fresh_text[:4000]
            except Exception as exc:  # noqa: BLE001 - a replay crash is a divergence finding
                diverged.append({"step": ev.get("step"), "tool": call["name"],
                                 "why": f"replay raised {exc.__class__.__name__}"})
                continue
            stored = next((e["result"] for e in events
                           if e.get("kind") == "tool_result" and e.get("call_id") == call["id"]),
                          None)
            if stored is None:
                diverged.append({"step": ev.get("step"), "tool": call["name"],
                                 "why": "no stored result for call"})
                continue
            try:
                stored_norm = json.dumps(json.loads(stored), sort_keys=True, default=str)[:4000]
                fresh_norm = json.dumps(json.loads(fresh_text), sort_keys=True, default=str)[:4000]
            except json.JSONDecodeError:
                stored_norm, fresh_norm = stored, fresh_text
            if fresh_norm == stored_norm:
                matched += 1
            else:
                diverged.append({"step": ev.get("step"), "tool": call["name"],
                                 "why": "result differs on replay"})
    return {"total": total, "matched": matched, "diverged": diverged,
            "proven": total > 0 and not diverged}
