"""Sparse fixtures exercise sequence-width boundaries without a million writes."""

import json

from roi_swarm.state_store import CanonicalStateStore


def seed(store, kind, sequence, state=None):
    session = store._session_dir("session")
    session.mkdir(parents=True, exist_ok=True)
    record = {"sequence": sequence}
    if state is not None:
        record["state"] = state
    (session / f"{kind}-{sequence:06d}.json").write_text(
        json.dumps(record), encoding="utf-8"
    )


def test_newest_snapshot_is_selected_numerically(tmp_path):
    store = CanonicalStateStore(tmp_path)
    seed(store, "snapshot", 999999, {"goal": "old"})
    seed(store, "snapshot", 1000000, {"goal": "current"})

    assert CanonicalStateStore(tmp_path).load_latest("session") == {"goal": "current"}


def test_newer_delta_is_not_hidden_by_older_snapshot_boundary(tmp_path):
    store = CanonicalStateStore(tmp_path)
    seed(store, "snapshot", 999999, {"goal": "snapshot"})
    seed(store, "delta", 999999, {"goal": "snapshot"})
    current = {"goal": "new delta"}
    seed(store, "delta", 1000000, current)

    reopened = CanonicalStateStore(tmp_path)
    assert reopened.load_latest("session") == current
    following = reopened.save("session", {"goal": "next"}, {})
    assert following["sequence"] == 1000001
    assert following["base_hash"] == store._state_hash(current)
    assert reopened.load_latest("session") == {"goal": "next"}


def test_delta_only_restore_uses_newest_numeric_sequence(tmp_path):
    store = CanonicalStateStore(tmp_path)
    seed(store, "delta", 999999, {"goal": "old"})
    seed(store, "delta", 1000000, {"goal": "current"})
    # Legacy hash-only deltas still fall back to the newest full state.
    seed(store, "delta", 1000001)

    assert store.load_latest("session") == {"goal": "current"}
