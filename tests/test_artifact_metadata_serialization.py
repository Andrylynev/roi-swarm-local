"""Shared-instance artifact writes must preserve every run's provenance."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from roi_swarm.artifacts import ArtifactStore


def test_concurrent_puts_preserve_all_run_ids(tmp_path):
    store = ArtifactStore(tmp_path)
    text = "Shared evidence\r\nα\n"
    initial = store.put_text("initial", text)
    start = Barrier(8)

    def put(number):
        start.wait(timeout=5)
        return store.put_text(f"run-{number}", text)

    with ThreadPoolExecutor(max_workers=8) as pool:
        refs = list(pool.map(put, range(8)))

    assert all(ref == initial for ref in refs)
    metadata = store.meta_dir / f"{initial.id}.json"
    data = json.loads(metadata.read_text(encoding="utf-8"))
    assert set(data["run_ids"]) == {"initial", *(f"run-{i}" for i in range(8))}
    assert len(data["run_ids"]) == 9
    assert store.get_text(initial.id) == text
    store.put_text("run-0", text)
    assert json.loads(metadata.read_text(encoding="utf-8")) == data
    assert not list(tmp_path.rglob("*.tmp"))


def test_metadata_read_and_write_share_one_critical_section(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path)
    ref = store.put_text("first", "evidence")
    metadata = store.meta_dir / f"{ref.id}.json"
    original_read = type(metadata).read_text
    original_write = store._atomic_write
    operations = []

    class TrackingLock:
        depth = 0

        def __enter__(self):
            self.depth += 1
            return self

        def __exit__(self, *args):
            self.depth -= 1

    lock = TrackingLock()
    monkeypatch.setattr(store, "_lock", lock, raising=False)

    def read(path, *args, **kwargs):
        if path == metadata:
            assert lock.depth == 1
            operations.append("read")
        return original_read(path, *args, **kwargs)

    def write(path, content):
        assert lock.depth == 1
        operations.append("write")
        return original_write(path, content)

    monkeypatch.setattr(type(metadata), "read_text", read)
    monkeypatch.setattr(store, "_atomic_write", write)
    store.put_text("second", "evidence")
    assert operations == ["read", "write"]
    assert lock.depth == 0

    def failed_write(path, content):
        assert lock.depth == 1
        raise OSError("fixture write failure")

    monkeypatch.setattr(store, "_atomic_write", failed_write)
    with pytest.raises(OSError, match="fixture write failure"):
        store.put_text("third", "evidence")
    assert lock.depth == 0

    monkeypatch.setattr(store, "_atomic_write", write)
    store.put_text("third", "evidence")
    assert lock.depth == 0
    assert json.loads(original_read(metadata, encoding="utf-8"))["run_ids"] == [
        "first", "second", "third"
    ]
