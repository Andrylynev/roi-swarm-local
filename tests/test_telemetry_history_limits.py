"""Offline contracts for bounded run-history requests."""

import json
from pathlib import Path

import pytest

from roi_swarm.telemetry import RunStore


def test_zero_history_limit_does_not_scan_or_read_files(tmp_path, monkeypatch):
    store = RunStore(tmp_path)
    store.create_run("fixture", "one")

    def unexpected_io(*args, **kwargs):
        raise AssertionError("A zero history limit must not access stored histories")

    monkeypatch.setattr(Path, "glob", unexpected_io)
    monkeypatch.setattr(Path, "read_text", unexpected_io)
    assert store.list_runs(limit=0) == []


@pytest.mark.parametrize("limit", [-1, -30, True, False, 1.5, "2", None])
def test_invalid_history_limits_fail_before_scanning(tmp_path, monkeypatch, limit):
    store = RunStore(tmp_path)

    def unexpected_scan(*args, **kwargs):
        raise AssertionError("Invalid limits must fail before scanning")

    monkeypatch.setattr(Path, "glob", unexpected_scan)
    with pytest.raises(ValueError, match="limit must be a nonnegative integer"):
        store.list_runs(limit=limit)


def test_positive_history_limits_preserve_newest_first_and_skip_invalid_json(tmp_path):
    store = RunStore(tmp_path)
    expected = []
    for number in range(1, 4):
        run_id = f"run-{number}"
        store.create_run("fixture", run_id)
        path = tmp_path / f"{run_id}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["created_at"] = f"2026-01-0{number}T00:00:00+00:00"
        path.write_text(json.dumps(record), encoding="utf-8")
        expected.insert(0, record)
    (tmp_path / "incomplete.json").write_text("{", encoding="utf-8")

    assert store.list_runs(limit=1) == expected[:1]
    assert store.list_runs(limit=2) == expected[:2]
    assert store.list_runs(limit=10) == expected
    assert store.list_runs() == expected
