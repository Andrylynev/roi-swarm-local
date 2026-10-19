"""Invalid JSON record shapes must not hide otherwise readable run histories."""

import json

import pytest

from roi_swarm.telemetry import RunStore


@pytest.mark.parametrize(
    "payload",
    [None, [], [1], "text", 42, True, {}, {"created_at": None},
     {"created_at": 42}, {"created_at": []}, {"created_at": {}}],
)
def test_history_skips_unsortable_records_without_modifying_evidence(tmp_path, payload):
    store = RunStore(tmp_path)
    expected = []
    for number in (1, 2):
        run_id = f"healthy-{number}"
        store.create_run("fixture", run_id)
        path = tmp_path / f"{run_id}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["created_at"] = f"2026-01-0{number}T00:00:00+00:00"
        path.write_text(json.dumps(record), encoding="utf-8")
        expected.insert(0, record)
    damaged = tmp_path / "damaged.json"
    damaged.write_text(json.dumps(payload), encoding="utf-8")
    before = {path.name: path.read_bytes() for path in tmp_path.glob("*.json")}

    assert store.list_runs(limit=1) == expected[:1]
    assert store.list_runs(limit=30) == expected
    assert {path.name: path.read_bytes() for path in tmp_path.glob("*.json")} == before
    # Listing does not repair or replace what direct inspection returns.
    assert store.get_run("damaged") == payload


def test_history_with_only_unsortable_records_lists_as_empty(tmp_path):
    store = RunStore(tmp_path)
    for number, payload in enumerate([None, [], {}, {"created_at": False}]):
        (tmp_path / f"damaged-{number}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    assert store.list_runs() == []
