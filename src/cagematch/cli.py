"""cagematch CLI: derby runs, reports, and trace verification.

    cagematch run   --tasks tasks/tasks.json [--cells cells.json] [--out runs/derby.db]
    cagematch report --db runs/derby.db [--markdown]
    cagematch verify --db runs/derby.db
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from cagematch import __version__
from cagematch.agent import run_task
from cagematch.cells import build_cell, load_cell_matrix, scripted_transcript
from cagematch.scoring import replay_run, score_answer
from cagematch.scripted import ScriptedProvider
from cagematch.trace import RunStore, TraceTampered

REPO_ROOT = Path(__file__).resolve().parents[2]


def cmd_run(args) -> int:
    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    specs = load_cell_matrix(Path(args.cells) if args.cells else None)
    store = RunStore(args.out)
    data_dir = Path(args.data_dir)
    failures = 0
    for spec in specs:
        cell = build_cell(spec)
        if cell is None:
            print(f"[skip] cell '{spec['name']}': env credentials not set")
            continue
        for task in tasks:
            ws_dir = Path(tempfile.mkdtemp(prefix="cagematch-ws-"))
            if spec.get("kind") == "scripted":
                cell.provider = ScriptedProvider(spec["name"], scripted_transcript(task["id"]))
            outcome = run_task(task, cell, store, data_dir, ws_dir)
            scored = score_answer(task["id"], outcome.answer, data_dir)
            replay = replay_run(store, outcome.run_id, data_dir, ws_dir)
            store.append(outcome.run_id, {"kind": "score", "answer_checks": scored["checks"],
                                          "passed": scored["passed"], "replay": {
                                              "total": replay["total"],
                                              "matched": replay["matched"],
                                              "proven": replay["proven"],
                                              "diverged": replay["diverged"][:5]}})
            verdict = "PASS" if scored["passed"] and replay["proven"] else "FAIL"
            if verdict == "FAIL":
                failures += 1
            print(f"[{verdict}] {task['id']} x {cell.name}: status={outcome.status} "
                  f"checks={scored['n_passed']}/{scored['n_checks']} "
                  f"replay={replay['matched']}/{replay['total']} "
                  f"steps={outcome.steps} cost=${outcome.cost_usd:.4f} "
                  f"wall={outcome.wall_ms:.0f}ms tool_errors={outcome.tool_errors}")
            if outcome.detail:
                print(f"        detail: {outcome.detail[:200]}")
    store.close()
    print(f"db: {args.out}  failures: {failures}")
    return 1 if failures else 0


def cmd_report(args) -> int:
    store = RunStore(args.db)
    rows = []
    for run in store.runs():
        events = store.events(run["run_id"])
        score = next((e for e in events if e.get("kind") == "score"), None)
        end = next((e for e in events if e.get("kind") == "end"), None)
        completions = [e for e in events if e.get("kind") == "completion"]
        rows.append({
            "task": run["task_id"], "cell": run["cell"],
            "status": end["status"] if end else "?",
            "passed": bool(score["passed"]) if score else None,
            "replay_proven": bool(score["replay"]["proven"]) if score else None,
            "steps": end["steps"] if end else len(completions),
            "cost_usd": round(end["cost_usd"], 4) if end else None,
            "wall_ms": round(sum(e.get("latency_ms", 0) for e in completions)),
            "tool_errors": end["tool_errors"] if end else None,
        })
    store.close()
    if args.markdown:
        print("| task | cell | status | answer_checks | replay_proven | steps | cost_usd | model_ms | tool_errors |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['task']} | {r['cell']} | {r['status']} | "
                  f"{'PASS' if r['passed'] else 'FAIL'} | "
                  f"{'yes' if r['replay_proven'] else 'NO'} | {r['steps']} | "
                  f"${r['cost_usd']} | {r['wall_ms']} | {r['tool_errors']} |")
    else:
        print(json.dumps(rows, indent=2))
    return 0


def cmd_verify(args) -> int:
    store = RunStore(args.db)
    try:
        for line in store.verify():
            print(line)
    except TraceTampered as exc:
        print(f"TAMPERED: {exc}", file=sys.stderr)
        store.close()
        return 1
    store.close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="cagematch",
                                     description="Evidence-locked arena for agent harnesses.")
    parser.add_argument("--version", action="version", version=f"cagematch {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the derby")
    p_run.add_argument("--tasks", default=str(REPO_ROOT / "tasks" / "tasks.json"))
    p_run.add_argument("--cells", default=None, help="JSON cell matrix (default: built-in)")
    p_run.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    p_run.add_argument("--out", default=str(REPO_ROOT / "runs" / "derby.db"))
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="render the results table")
    p_report.add_argument("--db", default=str(REPO_ROOT / "runs" / "derby.db"))
    p_report.add_argument("--markdown", action="store_true")
    p_report.set_defaults(func=cmd_report)

    p_verify = sub.add_parser("verify", help="verify trace hash chains")
    p_verify.add_argument("--db", default=str(REPO_ROOT / "runs" / "derby.db"))
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
