"""Offline regression contracts for concurrent creation on one RunStore."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from roi_swarm.telemetry import RunStore


def test_creation_checks_existence_while_holding_store_lock(tmp_path, monkeypatch):
    store = RunStore(tmp_path)
    original_path_for = store._path_for

    class TrackingLock:
        def __init__(self):
            self.depth = 0

        def __enter__(self):
            self.depth += 1
            return self

        def __exit__(self, *args):
            self.depth -= 1

    lock = TrackingLock()
    monkeypatch.setattr(store, "_lock", lock)

    def checked_path_for(run_id):
        assert lock.depth > 0, "Creation must lock before checking for an existing run"
        return original_path_for(run_id)

    monkeypatch.setattr(store, "_path_for", checked_path_for)
    assert store.create_run("original request", "shared") == "shared"
    with pytest.raises(ValueError, match="Run already exists"):
        store.create_run("replacement request", "shared")
    assert lock.depth == 0
    monkeypatch.setattr(store, "_path_for", original_path_for)
    assert store.get_run("shared")["request"] == "original request"


def test_concurrent_duplicate_creation_preserves_winner_history(tmp_path):
    store = RunStore(tmp_path)
    start = Barrier(8)

    def create(index):
        request = f"request-{index}"
        start.wait(timeout=5)
        try:
            store.create_run(request, "shared")
        except ValueError as exc:
            assert str(exc) == "Run already exists: shared"
            return None
        store.record_event("shared", "intake", "completed", {"owner": request})
        store.complete_run("shared", f"answer for {request}")
        return request

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(create, range(8)))

    winners = [request for request in outcomes if request is not None]
    assert len(winners) == 1
    winner = winners[0]
    snapshot = store.get_run("shared")
    assert snapshot["request"] == winner
    assert snapshot["status"] == "completed"
    assert snapshot["answer"] == f"answer for {winner}"
    assert len(snapshot["events"]) == 1
    assert snapshot["events"][0]["data"] == {"owner": winner}
