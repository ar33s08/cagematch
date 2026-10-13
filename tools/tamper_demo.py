"""Forge one trace event (payload + its own recomputed hash) and show verify()
catching it at the chain link. This is the README's tamper demo, committed so
the transcript can never rot: `python tools/tamper_demo.py` reproduces it.

Run against a fresh demo derby; prints the FORGED and CAUGHT lines and exits
nonzero if the tamper is NOT detected (a silent check is no check).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cagematch.agent import CellConfig, run_task
from cagematch.cells import scripted_transcript
from cagematch.scripted import ScriptedProvider
from cagematch.trace import RunStore, TraceTampered


def main() -> int:
    task = json.loads((ROOT / "tasks" / "tasks.json").read_text(encoding="utf-8"))[0]
    db_path = Path(tempfile.mkdtemp(prefix="tamper-")) / "demo.db"
    store = RunStore(db_path)
    cell = CellConfig(name="scripted", provider=ScriptedProvider(
        "scripted", scripted_transcript(task["id"])), model="scripted",
        system_prompt="Use tools.", max_steps=16, max_cost_usd=0.50)
    outcome = run_task(task, cell, store, ROOT / "data",
                       Path(tempfile.mkdtemp(prefix="tamper-ws-")))
    print(f"run {outcome.run_id}: {len(store.events(outcome.run_id))} events, chain intact")
    store.close()

    # --- the attack: rewrite one tool_result's number AND its hash ---
    db = sqlite3.connect(db_path)
    run_id, idx, payload, prev_hash = db.execute(
        "SELECT e.run_id, e.idx, e.payload, p.hash FROM events e "
        "JOIN events p ON p.run_id = e.run_id AND p.idx = e.idx - 1 "
        "WHERE json_extract(e.payload, '$.kind') = 'tool_result' "
        "AND e.run_id LIKE 'drawdown%' LIMIT 1").fetchone()
    rec = json.loads(payload)
    original = rec["result"]
    rec["result"] = json.dumps({"stat": "max_drawdown", "value": -0.001})
    forged = json.dumps(rec, sort_keys=True)

    def event_hash(prev: str, blob: str) -> str:
        return hashlib.sha256((prev + "\n" + blob.encode("utf-8").hex()).encode("utf-8")).hexdigest()

    db.execute("UPDATE events SET payload = ?, hash = ? WHERE run_id = ? AND idx = ?",
               (forged, event_hash(prev_hash, forged), run_id, idx))
    db.commit()
    db.close()
    print(f"FORGED: {run_id} event {idx}: result {original[:60]}... -> {rec['result']}")

    # --- the catch ---
    store2 = RunStore(db_path)
    try:
        store2.verify()
    except TraceTampered as exc:
        print(f"CAUGHT: {exc}")
        print("(the attacker could recompute one event's hash but cannot rewrite "
              "the forward chain without touching every later row)")
        return 0
    print("NOT CAUGHT — tamper detection is broken", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
