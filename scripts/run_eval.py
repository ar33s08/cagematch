"""CI eval gate: run the hermetic derby, score it, certify the traces.

FAILS the build if:
  - any scripted cell fails its answer checks
  - any scripted cell's run does not replay clean
  - any trace chain is broken
  - replay coverage drops below the floor
No network, no API keys: this gate runs the loop, scoring, and chain exactly as
the live derby does, against the deterministic test double.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cagematch.agent import CellConfig, run_task
from cagematch.cells import scripted_transcript
from cagematch.scoring import replay_run, score_answer
from cagematch.scripted import ScriptedProvider
from cagematch.trace import RunStore, TraceTampered

MIN_REPLAY_COVERAGE = 6  # tool calls that must be replay-certified across the suite


def main() -> int:
    tasks = json.loads((ROOT / "tasks" / "tasks.json").read_text(encoding="utf-8"))
    store = RunStore(Path(tempfile.mkdtemp(prefix="gate-")) / "gate.db")
    rows, failures = [], []
    total_replays = 0
    for task in tasks:
        provider = ScriptedProvider("scripted", scripted_transcript(task["id"]))
        cell = CellConfig(name="scripted", provider=provider, model="scripted",
                          system_prompt="Use tools. Finish with the answer object.",
                          max_steps=16, max_cost_usd=0.50)
        ws = Path(tempfile.mkdtemp(prefix="gate-ws-"))
        outcome = run_task(task, cell, store, ROOT / "data", ws)
        scored = score_answer(task["id"], outcome.answer, ROOT / "data")
        replay = replay_run(store, outcome.run_id, ROOT / "data", ws)
        total_replays += replay["total"]
        ok = outcome.status == "finished" and scored["passed"] and replay["proven"]
        rows.append({"task": task["id"], "status": outcome.status,
                     "checks": f"{scored['n_passed']}/{scored['n_checks']}",
                     "replay": f"{replay['matched']}/{replay['total']}",
                     "verdict": "PASS" if ok else "FAIL"})
        if not ok:
            failures.append(f"{task['id']}: status={outcome.status} "
                            f"checks={scored['checks']} diverged={replay['diverged']}")

    try:
        chain = store.verify()
    except TraceTampered as exc:
        failures.append(f"trace chain: {exc}")
        chain = []

    if total_replays < MIN_REPLAY_COVERAGE:
        failures.append(f"replay coverage {total_replays} < floor {MIN_REPLAY_COVERAGE}")

    for r in rows:
        print(f"[{r['verdict']}] {r['task']}: status={r['status']} "
              f"checks={r['checks']} replay={r['replay']}")
    pass_rate = sum(r["verdict"] == "PASS" for r in rows) / max(1, len(rows))
    print(f"pass_rate={pass_rate:.2f} replay_coverage={total_replays} "
          f"chains={len(chain)} simulated_test_double=yes")
    if failures:
        print("GATE FAIL:")
        for f in failures:
            print("  -", f)
        return 1
    print("GATE PASS (hermetic; test double only, proves plumbing not intelligence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
