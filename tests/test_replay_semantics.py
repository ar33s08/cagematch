"""Live-run found this: replay used to demand results for tool calls the
serial policy never dispatched. Replay certifies executed work, not proposals;
dispatched (not proposed) calls are what rides in the completion event."""

from pathlib import Path

from conftest import make_cell  # noqa: F401

from cagematch.agent import CellConfig, run_task
from cagematch.scoring import replay_run
from cagematch.scripted import ScriptedProvider
from cagematch.trace import RunStore

DATA = Path(__file__).resolve().parents[1] / "data"


def _data(tmp_path):
    import shutil
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    for f in DATA.glob("*.csv"):
        if not (d / f.name).exists():
            shutil.copy(f, d / f.name)
    return d


def test_serial_policy_records_only_dispatched_calls(tmp_path):
    loop = [
        {"tool_calls": [{"name": "series_stats", "arguments":
                        {"file": "data/spy_daily.csv", "column": "close",
                         "stat": "max_drawdown"}},
                       {"name": "series_stats", "arguments":
                        {"file": "data/qqq_daily.csv", "column": "close",
                         "stat": "max_drawdown"}}]},  # second dropped by policy
        {"tool_calls": [{"name": "finish", "arguments": {"answer": {"done": True}}}]},
    ]
    cell = CellConfig(name="serial", provider=ScriptedProvider("serial", loop),
                      model="scripted", system_prompt="one per turn",
                      parallel_tools=False, max_steps=8, max_cost_usd=1.0)
    store = RunStore(tmp_path / "r.db")
    outcome = run_task({"id": "drawdown-spread", "prompt": "go", "system": ""},
                       cell, store, _data(tmp_path), tmp_path / "ws")
    events = store.events(outcome.run_id)
    comp = next(e for e in events if e["kind"] == "completion")
    assert len(comp["tool_calls"]) == 1          # only the dispatched call
    assert comp["proposed_n"] == 2               # proposal count is still auditable
    # every dispatched call has a stored result -> replay must certify or
    # legitimately diverge on DATA, never on a phantom missing result
    replay = replay_run(store, outcome.run_id, _data(tmp_path), tmp_path / "ws")
    phantom = [d for d in replay["diverged"] if "no stored result" in d["why"]]
    assert not phantom, replay["diverged"]


def test_write_workspace_path_is_stable_across_replays(tmp_path):
    # The first live run's other divergence: the write tool echoed a tempdir
    # path that differs per run; workspace-relative output was already stable,
    # but the absolute echo must never ride into a certified result.
    from cagematch.tools import make_tool_registry
    data = _data(tmp_path)
    ws = tmp_path / "ws"
    reg = make_tool_registry(data, ws)
    out = reg["write_data_csv"](file="sub/x.csv", rows=[{"a": 1}])
    assert not Path(out["path"]).is_absolute()
    reg2 = make_tool_registry(data, ws)  # "replay" with a resolved-differently root
    out2 = reg2["write_data_csv"](file="sub/x.csv", rows=[{"a": 1}])
    assert out == out2
