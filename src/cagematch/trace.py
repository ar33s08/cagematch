"""SQLite run store with a hash-chained event log.

Every event appended to a run is chained: hash(n) = sha256(prev_hash + payload).
A score is only as trustworthy as the trace behind it, so the store can prove
that a recorded run was replayed faithfully — and localize the exact event
index if anyone edited the database by hand.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

GENESIS = "0" * 64


class TraceTampered(RuntimeError):
    def __init__(self, run_id: str, index: int, expected_prev: str, actual_prev: str):
        self.run_id = run_id
        self.index = index
        self.expected_prev = expected_prev
        self.actual_prev = actual_prev
        super().__init__(
            f"trace for run {run_id} diverges at event index {index}: "
            f"chain hash mismatch (expected prev {expected_prev[:12]}…, stored {actual_prev[:12]}…)"
        )


def _event_hash(prev_hash: str, payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256((prev_hash + "\n" + blob.hex()).encode("utf-8")).hexdigest()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    cell TEXT NOT NULL,
    started_utc REAL NOT NULL,
    finished_utc REAL,
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    ts_utc REAL NOT NULL,
    payload TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL,
    PRIMARY KEY (run_id, idx)
);
"""


class RunStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(_SCHEMA)
        self.db.commit()

    def start_run(self, run_id: str, task_id: str, cell: str, meta: dict | None = None) -> None:
        self.db.execute(
            "INSERT INTO runs (run_id, task_id, cell, started_utc, meta) VALUES (?,?,?,?,?)",
            (run_id, task_id, cell, time.time(), json.dumps(meta or {}, sort_keys=True)),
        )
        self.db.commit()

    def finish_run(self, run_id: str) -> None:
        self.db.execute("UPDATE runs SET finished_utc=? WHERE run_id=?", (time.time(), run_id))
        self.db.commit()

    def append(self, run_id: str, payload: dict) -> str:
        row = self.db.execute(
            "SELECT hash FROM events WHERE run_id=? ORDER BY idx DESC LIMIT 1", (run_id,)
        ).fetchone()
        prev = row[0] if row else GENESIS
        h = _event_hash(prev, payload)
        idx = self.db.execute(
            "SELECT COALESCE(MAX(idx), -1) + 1 FROM events WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        self.db.execute(
            "INSERT INTO events (run_id, idx, ts_utc, payload, prev_hash, hash) VALUES (?,?,?,?,?,?)",
            (run_id, idx, time.time(), json.dumps(payload, sort_keys=True), prev, h),
        )
        self.db.commit()
        return h

    def events(self, run_id: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT idx, ts_utc, payload FROM events WHERE run_id=? ORDER BY idx", (run_id,)
        ).fetchall()
        out = []
        for idx, ts, payload in rows:
            rec = json.loads(payload)
            rec["_idx"] = idx
            rec["_ts_utc"] = ts
            out.append(rec)
        return out

    def runs(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT run_id, task_id, cell, started_utc, finished_utc, meta FROM runs ORDER BY started_utc"
        ).fetchall()
        return [
            {"run_id": r[0], "task_id": r[1], "cell": r[2], "started_utc": r[3],
             "finished_utc": r[4], "meta": json.loads(r[5])}
            for r in rows
        ]

    def verify(self, run_id: str | None = None) -> list[str]:
        """Recompute the hash chain for one or all runs.

        Returns a list of human-readable integrity results; empty list means
        nothing was checked. Raises TraceTampered on the first divergence.
        """
        ids = [run_id] if run_id else [r["run_id"] for r in self.runs()]
        results = []
        for rid in ids:
            prev = GENESIS
            rows = self.db.execute(
                "SELECT idx, payload, prev_hash, hash FROM events WHERE run_id=? ORDER BY idx",
                (rid,),
            ).fetchall()
            for idx, payload, stored_prev, stored_hash in rows:
                recomputed = _event_hash(prev, json.loads(payload))
                if stored_prev != prev or recomputed != stored_hash:
                    raise TraceTampered(rid, idx, prev, stored_prev)
                prev = stored_hash
            results.append(f"{rid}: {len(rows)} events, chain intact")
        return results

    def close(self) -> None:
        self.db.close()
