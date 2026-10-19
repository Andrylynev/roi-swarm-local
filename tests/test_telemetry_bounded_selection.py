"""History selection preserves stable sorting while consuming records lazily."""

import json
from pathlib import Path

import pytest

from roi_swarm.telemetry import RunStore


@pytest.mark.parametrize("limit", [1, 2, 7, 30, 100])
def test_history_selection_matches_stable_sort_with_ties(tmp_path, monkeypatch, limit):
    store = RunStore(tmp_path)
    paths = []
    records = []
    for number in range(45):
        record = {
            "id": f"run-{number}",
            "created_at": f"2026-01-{1 + number % 9:02d}T00:00:00+00:00",
            "events": [{"seq": number, "data": "evidence"}],
        }
        path = tmp_path / f"run-{number}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        paths.append(path)
        records.append(record)
    damaged = tmp_path / "damaged.json"
    damaged.write_bytes(b"\xff")
    invalid_shape = tmp_path / "invalid-shape.json"
    invalid_shape.write_text("[]", encoding="utf-8")
    scan = [damaged, *reversed(paths), invalid_shape]
    original_glob = Path.glob

    def ordered_glob(path, pattern):
        if path == tmp_path and pattern == "*.json":
            return iter(scan)
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", ordered_glob)
    expected = sorted(reversed(records), key=lambda item: item["created_at"], reverse=True)
    before = {path.name: path.read_bytes() for path in scan}

    assert store.list_runs(limit=limit) == expected[:limit]
    assert {path.name: path.read_bytes() for path in scan} == before


def test_history_selection_does_not_retain_every_decoded_record(tmp_path, monkeypatch):
    import roi_swarm.telemetry as module

    store = RunStore(tmp_path)
    for number in range(80):
        (tmp_path / f"run-{number}.json").write_text(
            json.dumps({"id": str(number), "created_at": f"{number:04d}"}),
            encoding="utf-8",
        )
    original_loads = json.loads
    live = 0
    peak = 0

    class TrackedRecord(dict):
        def __init__(self, value):
            nonlocal live, peak
            super().__init__(value)
            live += 1
            peak = max(peak, live)

        def __del__(self):
            nonlocal live
            live -= 1

    def tracked_loads(text):
        return TrackedRecord(original_loads(text))

    monkeypatch.setattr(module.json, "loads", tracked_loads)
    result = store.list_runs(limit=3)

    assert [record["id"] for record in result] == ["79", "78", "77"]
    # Allow generator/selection temporaries, but not accumulation of all 80.
    # Immediate reclamation is an implementation detail of CPython.
    import sys

    if sys.implementation.name == "cpython":
        assert peak <= 6
