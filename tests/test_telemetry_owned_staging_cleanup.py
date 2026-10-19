"""Failed telemetry updates preserve durable evidence and clean owned staging."""

from pathlib import Path

import pytest

from roi_swarm.telemetry import RunStore


@pytest.mark.parametrize("phase", ["write", "replace"])
def test_failed_event_write_removes_only_owned_staging(tmp_path, monkeypatch, phase):
    store = RunStore(tmp_path)
    store.create_run("fixture", "one")
    destination = (tmp_path / "one.json").resolve()
    original_bytes = destination.read_bytes()
    unrelated = tmp_path / ".one.json.another-writer.tmp"
    unrelated.write_bytes(b"preserve another writer")
    original_write = Path.write_text
    original_replace = Path.replace
    staged = []

    def write(path, content, *args, **kwargs):
        if path.parent == destination.parent and path.suffix == ".tmp":
            staged.append(path)
            if phase == "write":
                path.write_bytes(b"partial history")
                raise OSError("fixture staging write failure")
        return original_write(path, content, *args, **kwargs)

    def replace(path, target):
        if target == destination:
            raise OSError("fixture staging replacement failure")
        return original_replace(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", write)
        patch.setattr(Path, "replace", replace)
        with pytest.raises(OSError, match="fixture staging"):
            store.record_event("one", "context", "completed", {"artifact_count": 2})

    assert len(staged) == 1
    assert staged[0] != unrelated
    assert not staged[0].exists()
    assert destination.read_bytes() == original_bytes
    assert unrelated.read_bytes() == b"preserve another writer"
    assert set(tmp_path.iterdir()) == {destination, unrelated}
    assert store.get_run("one")["events"] == []

    # The failed update must not consume an event sequence or prevent retry.
    store.record_event("one", "context", "completed", {"artifact_count": 2})
    store.complete_run("one", "answer")
    reopened = RunStore(tmp_path).get_run("one")
    assert reopened["status"] == "completed"
    assert reopened["answer"] == "answer"
    assert len(reopened["events"]) == 1
    assert reopened["events"][0]["seq"] == 1
    assert reopened["events"][0]["data"] == {"artifact_count": 2}
    assert set(tmp_path.iterdir()) == {destination, unrelated}
