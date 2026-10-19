"""State-write failures must preserve durable records and clean owned staging data."""

import json
from pathlib import Path

import pytest

from roi_swarm.state_store import CanonicalStateStore


@pytest.mark.parametrize("phase", ["write", "replace"])
def test_failed_atomic_write_cleans_only_owned_staging(tmp_path, monkeypatch, phase):
    destination = tmp_path / "delta-000001.json"
    original = b'{"state": {"goal": "durable"}}'
    destination.write_bytes(original)
    other_staging = tmp_path / ".delta-000001.json.another-writer.tmp"
    other_staging.write_bytes(b"preserve another writer")
    original_write = Path.write_text
    original_replace = Path.replace
    staged = []

    def write(path, content, *args, **kwargs):
        if path.parent == tmp_path and path.suffix == ".tmp":
            staged.append(path)
            if phase == "write":
                path.write_bytes(b"partial")
                raise OSError("injected staging write failure")
        return original_write(path, content, *args, **kwargs)

    def replace(path, target):
        if target == destination:
            raise OSError("injected staging replacement failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "write_text", write)
    monkeypatch.setattr(Path, "replace", replace)

    with pytest.raises(OSError, match="injected staging"):
        CanonicalStateStore._atomic_write(destination, {"state": {"goal": "new"}})

    assert len(staged) == 1
    assert staged[0] != other_staging
    assert not staged[0].exists()
    assert destination.read_bytes() == original
    assert other_staging.read_bytes() == b"preserve another writer"
    assert set(tmp_path.iterdir()) == {destination, other_staging}


def test_successful_state_writes_leave_no_staging_files_and_restore(tmp_path):
    store = CanonicalStateStore(tmp_path, snapshot_interval=2)
    first = store.save("session", {"goal": "первый"}, {})
    second = store.save("session", {"goal": "second"}, {})

    assert second["base_hash"] == first["state_hash"]
    assert CanonicalStateStore(tmp_path).load_latest("session") == {"goal": "second"}
    session = store._session_dir("session")
    assert sorted(path.name for path in session.iterdir()) == [
        "delta-000001.json",
        "delta-000002.json",
        "snapshot-000001.json",
        "snapshot-000002.json",
    ]
    assert json.loads((session / "delta-000002.json").read_text(encoding="utf-8")) == second
