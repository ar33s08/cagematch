"""Trace store: hash chain integrity and tamper localization."""

from __future__ import annotations

import pytest

from cagematch.trace import RunStore, TraceTampered


@pytest.fixture()
def store(tmp_path):
    s = RunStore(tmp_path / "runs.db")
    s.start_run("r1", "task-a", "cell-x")
    s.append("r1", {"kind": "completion", "n": 1})
    s.append("r1", {"kind": "completion", "n": 2})
    s.finish_run("r1")
    return s


def test_verify_clean(store):
    assert store.verify("r1") == ["r1: 2 events, chain intact"]


def test_tamper_detected_and_located(store):
    store.db.execute(
        "UPDATE events SET payload = '{\"kind\": \"completion\", \"n\": 999}' "
        "WHERE run_id='r1' AND idx=1")
    store.db.commit()
    with pytest.raises(TraceTampered) as excinfo:
        store.verify("r1")
    assert excinfo.value.index == 1
    assert excinfo.value.run_id == "r1"


def test_row_deletion_detected(store):
    store.db.execute("DELETE FROM events WHERE run_id='r1' AND idx=0")
    store.db.commit()
    with pytest.raises(TraceTampered):
        store.verify("r1")


def test_forge_payload_and_its_own_hash_still_detected(store):
    # The realistic single-event forgery: attacker rewrites event 0's payload
    # AND recomputes its hash correctly — but cannot update event 1's
    # prev_hash without rewriting the whole forward chain. The chain link
    # catches it at index 1.
    import json as _json

    from cagematch.trace import GENESIS, _event_hash
    forged = {"kind": "completion", "n": 42}
    h = _event_hash(GENESIS, forged)
    store.db.execute(
        "UPDATE events SET payload = ?, hash = ? WHERE run_id='r1' AND idx=0",
        (_json.dumps(forged, sort_keys=True), h))
    store.db.commit()
    with pytest.raises(TraceTampered) as excinfo:
        store.verify("r1")
    assert excinfo.value.index == 1


def test_events_roundtrip(store):
    events = store.events("r1")
    assert [e["n"] for e in events] == [1, 2]
    assert [e["_idx"] for e in events] == [0, 1]
