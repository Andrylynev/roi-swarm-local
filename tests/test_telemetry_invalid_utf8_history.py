"""Unreadable history encodings must not hide healthy runs or alter evidence."""

import json

import pytest

from roi_swarm.telemetry import RunStore


def test_history_listing_skips_invalid_utf8_without_changing_files(tmp_path):
    store = RunStore(tmp_path)
    expected = []
    for number in (1, 2):
        run_id = f"healthy-{number}"
        store.create_run("Проверить данные", run_id)
        path = tmp_path / f"{run_id}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["created_at"] = f"2026-01-0{number}T00:00:00+00:00"
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        expected.insert(0, record)
    damaged = tmp_path / "damaged.json"
    damaged.write_bytes(b'{"request":"\xff"}')
    malformed = tmp_path / "malformed.json"
    malformed.write_bytes(b"{")
    before = {path.name: path.read_bytes() for path in tmp_path.glob("*.json")}

    assert store.list_runs(limit=1) == expected[:1]
    assert store.list_runs(limit=10) == expected
    assert {path.name: path.read_bytes() for path in tmp_path.glob("*.json")} == before

    # Direct inspection still exposes corruption rather than inventing a record.
    with pytest.raises(UnicodeDecodeError):
        store.get_run("damaged")


def test_history_with_only_invalid_utf8_lists_as_empty(tmp_path):
    store = RunStore(tmp_path)
    path = tmp_path / "damaged.json"
    raw = b"\xff\xfe\x00"
    path.write_bytes(raw)

    assert store.list_runs() == []
    assert path.read_bytes() == raw
