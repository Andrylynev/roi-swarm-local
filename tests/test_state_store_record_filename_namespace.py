"""Restore and sequence allocation must recognize the same record filenames."""

import json

import pytest

from roi_swarm.state_store import CanonicalStateStore


@pytest.mark.parametrize("prefix", ["snapshot-", "delta-"])
@pytest.mark.parametrize("suffix", ["", "backup", "000002.bak", "-2", "+2", "٢", "²"])
def test_nonrecord_names_do_not_block_restore_or_save(tmp_path, prefix, suffix):
    store = CanonicalStateStore(tmp_path, snapshot_interval=5)
    store.save("session", {"goal": "snapshot"}, {})
    latest = store.save("session", {"goal": "latest delta"}, {})
    session = store._session_dir("session")
    extra = session / f"{prefix}{suffix}.json"
    # Nonrecords must not be parsed, even when their contents are invalid JSON.
    extra.write_text("not a record", encoding="utf-8")

    reopened = CanonicalStateStore(tmp_path, snapshot_interval=5)
    assert reopened.load_latest("session") == {"goal": "latest delta"}
    following = reopened.save("session", {"goal": "next"}, {})

    assert following["sequence"] == 3
    assert following["base_hash"] == latest["state_hash"]
    assert reopened.load_latest("session") == {"goal": "next"}
    assert extra.read_text(encoding="utf-8") == "not a record"


@pytest.mark.parametrize("prefix", ["snapshot-", "delta-"])
def test_numeric_record_corruption_is_still_reported(tmp_path, prefix):
    store = CanonicalStateStore(tmp_path)
    store.save("session", {"goal": "known"}, {})
    path = store._session_dir("session") / f"{prefix}000002.json"
    path.write_text("not JSON", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        store.load_latest("session")


def test_only_nonrecord_names_restore_as_empty_without_overwriting_them(tmp_path):
    store = CanonicalStateStore(tmp_path)
    session = store._session_dir("session")
    session.mkdir()
    extra = session / "snapshot-backup.json"
    extra.write_text("preserve me", encoding="utf-8")

    assert store.load_latest("session") is None
    first = store.save("session", {"goal": "first"}, {})
    assert first["sequence"] == 1
    assert first["base_hash"] is None
    assert extra.read_text(encoding="utf-8") == "preserve me"
