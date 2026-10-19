"""Sequence gaps must not cause a later save to overwrite durable history."""

from roi_swarm.state_store import CanonicalStateStore


def test_removed_middle_delta_does_not_reuse_existing_sequence(tmp_path):
    store = CanonicalStateStore(tmp_path, snapshot_interval=5)
    for number in range(1, 4):
        latest = store.save("session", {"goal": str(number)}, {})
    session = next(tmp_path.iterdir())
    preserved = session / "delta-000003.json"
    original_bytes = preserved.read_bytes()
    (session / "delta-000002.json").unlink()

    reopened = CanonicalStateStore(tmp_path, snapshot_interval=5)
    following = reopened.save("session", {"goal": "four"}, {})

    assert following["sequence"] == 4
    assert following["base_hash"] == latest["state_hash"]
    assert preserved.read_bytes() == original_bytes
    assert (session / "delta-000004.json").is_file()
    assert reopened.load_latest("session") == {"goal": "four"}


def test_snapshot_reserves_sequence_when_its_delta_is_removed(tmp_path):
    store = CanonicalStateStore(tmp_path, snapshot_interval=2)
    store.save("session", {"goal": "one"}, {})
    latest = store.save("session", {"goal": "two"}, {})
    session = next(tmp_path.iterdir())
    snapshot = session / "snapshot-000002.json"
    original_bytes = snapshot.read_bytes()
    (session / "delta-000002.json").unlink()

    reopened = CanonicalStateStore(tmp_path, snapshot_interval=2)
    following = reopened.save("session", {"goal": "three"}, {})

    assert following["sequence"] == 3
    assert following["base_hash"] == latest["state_hash"]
    assert snapshot.read_bytes() == original_bytes
    assert reopened.load_latest("session") == {"goal": "three"}


def test_contiguous_records_keep_existing_sequence_and_snapshot_cadence(tmp_path):
    store = CanonicalStateStore(tmp_path, snapshot_interval=2)
    records = [store.save("session", {"goal": str(n)}, {}) for n in range(1, 5)]

    assert [record["sequence"] for record in records] == [1, 2, 3, 4]
    assert sorted(path.name for path in tmp_path.glob("*/snapshot-*.json")) == [
        "snapshot-000001.json",
        "snapshot-000002.json",
        "snapshot-000004.json",
    ]
