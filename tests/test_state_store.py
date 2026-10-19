from roi_swarm.state_store import CanonicalStateStore


def test_state_store_writes_deltas_and_periodic_snapshots(tmp_path) -> None:
    store = CanonicalStateStore(tmp_path / "state", snapshot_interval=2)

    first = store.save("session-a", {"goal": "one"}, {"goal": {"after": "one"}})
    second = store.save("session-a", {"goal": "two"}, {"goal": {"after": "two"}})

    assert first["base_hash"] is None
    assert second["base_hash"] == first["state_hash"]
    assert store.load_latest("session-a") == {"goal": "two"}
