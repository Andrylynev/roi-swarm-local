"""The durable delta is authoritative even if the following snapshot write fails."""
import json

import pytest

from roi_swarm.state_store import CanonicalStateStore


@pytest.mark.parametrize("interval", [0, -1, True, False, 1.5, "2", None])
def test_invalid_interval_has_no_filesystem_side_effects(tmp_path, interval):
    root = tmp_path / "not-created" / "state"
    with pytest.raises(ValueError, match="snapshot_interval must be a positive integer"):
        CanonicalStateStore(root, snapshot_interval=interval)
    assert not root.parent.exists()


@pytest.mark.parametrize("interval", [1, 2, 5])
def test_positive_interval_preserves_cadence_and_current_state(tmp_path, interval):
    store = CanonicalStateStore(tmp_path, snapshot_interval=interval)
    for number in range(1, 7):
        store.save("session", {"goal": str(number)}, {})
    expected = [n for n in range(1, 7) if n == 1 or n % interval == 0]
    assert [p.name for p in sorted(tmp_path.glob("*/snapshot-*.json"))] == [
        f"snapshot-{n:06d}.json" for n in expected
    ]
    assert len(list(tmp_path.glob("*/delta-*.json"))) == 6
    assert store.load_latest("session") == {"goal": "6"}


@pytest.mark.parametrize("failed_sequence", [1, 2])
def test_durable_delta_recovers_after_snapshot_failure(tmp_path, monkeypatch, failed_sequence):
    store = CanonicalStateStore(tmp_path, snapshot_interval=2)
    if failed_sequence == 2:
        store.save("session", {"goal": "first"}, {})
    original_write = store._atomic_write

    def fail_snapshot(path, value):
        if path.name == f"snapshot-{failed_sequence:06d}.json":
            raise OSError("injected snapshot write failure")
        original_write(path, value)

    monkeypatch.setattr(store, "_atomic_write", fail_snapshot)
    with pytest.raises(OSError, match="injected snapshot"):
        store.save("session", {"goal": "recovered"}, {})
    reopened = CanonicalStateStore(tmp_path, snapshot_interval=2)
    assert reopened.load_latest("session") == {"goal": "recovered"}
    if failed_sequence == 1:
        assert not list(tmp_path.glob("*/snapshot-*.json"))
    path = next(tmp_path.glob(f"*/delta-{failed_sequence:06d}.json"))
    delta = json.loads(path.read_text(encoding="utf-8"))
    following = reopened.save("session", {"goal": "next"}, {})
    assert following["base_hash"] == delta["state_hash"]
