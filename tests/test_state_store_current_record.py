import json

from roi_swarm.state_store import CanonicalStateStore


def test_latest_state_survives_restart_between_periodic_snapshots(tmp_path):
    root = tmp_path / "state"
    store = CanonicalStateStore(root, snapshot_interval=5)
    store.save("session", {"goal": "first", "removed": True}, {})
    expected = {"goal": "second", "decisions": ["Use SQLite"], "nested": {"n": 2}}
    second = store.save("session", expected, {})

    assert len(list(root.rglob("snapshot-*.json"))) == 1
    reopened = CanonicalStateStore(root, snapshot_interval=5)
    assert reopened.load_latest("session") == expected
    third = reopened.save("session", {"goal": "third"}, {})
    assert third["base_hash"] == second["state_hash"]
    assert reopened.load_latest("other-session") is None


def test_latest_state_remains_current_across_snapshot_boundary(tmp_path):
    store = CanonicalStateStore(tmp_path, snapshot_interval=2)
    previous = None
    for number in range(1, 5):
        state = {"goal": str(number)}
        record = store.save("session", state, {})
        assert record["base_hash"] == (previous["state_hash"] if previous else None)
        assert store.load_latest("session") == state
        previous = record
    assert len(list(tmp_path.rglob("snapshot-*.json"))) == 3


def test_legacy_hash_only_deltas_fall_back_to_snapshot(tmp_path):
    store = CanonicalStateStore(tmp_path)
    store.save("legacy", {"goal": "known snapshot"}, {})
    session = next(tmp_path.iterdir())
    (session / "delta-000002.json").write_text(
        json.dumps({"sequence": 2, "base_hash": None, "state_hash": None, "delta": {}}),
        encoding="utf-8",
    )

    assert store.load_latest("legacy") == {"goal": "known snapshot"}
    store.save("legacy", {"goal": "new durable state"}, {})
    assert CanonicalStateStore(tmp_path).load_latest("legacy") == {"goal": "new durable state"}
