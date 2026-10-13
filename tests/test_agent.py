"""Agent loop: budgets bite, errors are surfaced as findings, traces replay."""

from __future__ import annotations

import json

from conftest import make_cell

from cagematch.agent import run_task
from cagematch.scoring import replay_run, score_answer
from cagematch.scripted import ScriptedProvider
from cagematch.trace import RunStore

TASKS = json.loads((
    __import__("pathlib").Path(__file__).resolve().parents[1] / "tasks" / "tasks.json"
).read_text(encoding="utf-8"))
BY_ID = {t["id"]: t for t in TASKS}


def store_data_dir(tmp_path):
    import shutil
    from pathlib import Path
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    for f in (Path(__file__).resolve().parents[1] / "data").glob("*.csv"):
        target = d / f.name
        if not target.exists():
            shutil.copy(f, target)
    return d


def test_scripted_cells_pass_and_replay(tmp_path):
    store = RunStore(tmp_path / "r.db")
    for task_id in sorted(BY_ID):
        cell = make_cell("scripted", task_id)
        outcome = run_task(BY_ID[task_id], cell, store, store_data_dir(tmp_path),
                           tmp_path / f"ws-{task_id}")
        assert outcome.status == "finished", outcome.detail
        scored = score_answer(task_id, outcome.answer, store_data_dir(tmp_path))
        assert scored["passed"], scored
        replay = replay_run(store, outcome.run_id, store_data_dir(tmp_path),
                            tmp_path / f"ws-{task_id}")
        assert replay["proven"], replay


def test_step_budget_stops_runaway_cell(tmp_path):
    # transcript that never calls finish: every turn is a harmless calc
    loop = [{"tool_calls": [{"name": "calc", "arguments": {"expr": "1+1"}}]}] * 30
    provider = ScriptedProvider("runaway", loop)
    cell = make_cell("runaway", "drawdown-spread", max_steps=5)
    cell.provider = provider
    store = RunStore(tmp_path / "r.db")
    outcome = run_task(BY_ID["drawdown-spread"], cell, store, store_data_dir(tmp_path),
                       tmp_path / "ws")
    assert outcome.status == "budget_steps"
    assert outcome.steps == 5


def test_cost_budget_stops_expensive_cell(tmp_path):
    loop = [{"tool_calls": [{"name": "calc", "arguments": {"expr": "1+1"}}],
             "prompt_tokens": 100_000, "completion_tokens": 1000}] * 30
    provider = ScriptedProvider("rich", loop)
    cell = make_cell("rich", "drawdown-spread",
                     pricing={"prompt": 10.0, "completion": 10.0})
    cell.provider = provider
    store = RunStore(tmp_path / "r.db")
    outcome = run_task(BY_ID["drawdown-spread"], cell, store, store_data_dir(tmp_path),
                       tmp_path / "ws")
    assert outcome.status == "budget_cost"
    assert outcome.steps < 30  # stopped long before the transcript ran out


def test_tool_errors_surfaced_not_fatal(tmp_path):
    # first call references a missing column; the loop must deliver the
    # TOOL_ERROR text to the model and keep going
    loop = [
        {"tool_calls": [{"name": "series_stats", "arguments":
                        {"file": "data/spy_daily.csv", "column": "price", "stat": "mean"}}]},
        {"tool_calls": [{"name": "series_stats", "arguments":
                        {"file": "data/spy_daily.csv", "column": "close", "stat": "mean"}}]},
        {"tool_calls": [{"name": "finish", "arguments": {"answer": {"ok": True}}}]},
    ]
    cell = make_cell("recovers", "drawdown-spread")
    cell.provider = ScriptedProvider("recovers", loop)
    store = RunStore(tmp_path / "r.db")
    outcome = run_task(BY_ID["drawdown-spread"], cell, store, store_data_dir(tmp_path),
                       tmp_path / "ws")
    assert outcome.status == "finished"
    assert outcome.tool_errors == 1
    events = store.events(outcome.run_id)
    err = next(e for e in events if e.get("kind") == "tool_result" and not e["ok"])
    assert "column 'price' not in file" in err["result"]
    assert err["result"].startswith("TOOL_ERROR")


def test_no_finish_yields_no_finish_status(tmp_path):
    loop = [{"content": "I think the answer is around -25% and -35%", "tool_calls": []}]
    cell = make_cell("prose", "drawdown-spread")
    cell.provider = ScriptedProvider("prose", loop)
    store = RunStore(tmp_path / "r.db")
    outcome = run_task(BY_ID["drawdown-spread"], cell, store, store_data_dir(tmp_path),
                       tmp_path / "ws")
    assert outcome.status == "no_finish"
    assert "prose" in outcome.detail


def test_replay_catches_data_drift(tmp_path):
    cell = make_cell("scripted", "drawdown-spread")
    data = store_data_dir(tmp_path)
    store = RunStore(tmp_path / "r.db")
    outcome = run_task(BY_ID["drawdown-spread"], cell, store, data, tmp_path / "ws")
    assert replay_run(store, outcome.run_id, data, tmp_path / "ws")["proven"]
    # silently edit the price file under the run; replay must now refuse to certify
    p = data / "qqq_daily.csv"
    lines = p.read_text().splitlines()
    lines[100] = lines[100].replace(lines[100].split(",")[1], "999.000000")
    p.write_text("\n".join(lines) + "\n")
    replay = replay_run(store, outcome.run_id, data, tmp_path / "ws")
    assert not replay["proven"]
    assert replay["diverged"]
