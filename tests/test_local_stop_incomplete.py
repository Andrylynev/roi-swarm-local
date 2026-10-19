"""Unconfirmed service cleanup must retain recovery evidence and fail honestly."""

from contextlib import nullcontext

import pytest

from local import manage


def state_with_records(root):
    plan = manage.build_plan(root)
    records = {
        name: {
            "pid": 701 + i,
            "created": "11",
            "image": service["command"][0],
            "command": service["command"],
        }
        for i, (name, service) in enumerate(plan["services"].items())
    }
    manage.write_state(root, {"version": 1, "root": str(root), "services": records})
    return records


def test_refused_stop_preserves_record_and_reports_incomplete(tmp_path, monkeypatch):
    records = state_with_records(tmp_path)
    monkeypatch.setattr(manage, "lifecycle_lock", lambda: nullcontext())
    monkeypatch.setattr(manage, "terminate_owned", lambda *args: False)
    monkeypatch.setattr(
        manage,
        "ownership",
        lambda *args: {
            "owned": False,
            "reason": "command mismatch; process may still be running",
        },
    )
    result = manage.stop(tmp_path)
    assert result["stopped"] is False
    assert set(result["incomplete"]) == set(records)
    retained = manage.read_state(tmp_path)["services"]
    for name, record in records.items():
        assert all(retained[name][key] == value for key, value in record.items())
        assert "command mismatch" in retained[name]["stop_error"]
    assert manage.main(["stop", "--root", str(tmp_path)]) == 1


def test_stop_attempts_other_services_after_one_error(tmp_path, monkeypatch):
    state_with_records(tmp_path)
    monkeypatch.setattr(manage, "lifecycle_lock", lambda: nullcontext())
    called = []

    def terminate(record, service):
        called.append(record["pid"])
        if record["pid"] == 702:
            raise manage.LifecycleError("access denied")
        return True

    monkeypatch.setattr(manage, "terminate_owned", terminate)
    result = manage.stop(tmp_path)
    assert called == [702, 701]
    assert result["stopped"] is False
    assert result["terminated"] == ["inference"]
    retained = manage.read_state(tmp_path)["services"]
    assert set(retained) == {"api"}
    assert "access denied" in retained["api"]["stop_error"]


def test_start_does_not_replace_an_unconfirmed_saved_process(tmp_path, monkeypatch):
    state_with_records(tmp_path)
    monkeypatch.setattr(manage, "lifecycle_lock", lambda: nullcontext())
    monkeypatch.setattr(manage, "ownership", lambda *a: {"owned": False, "reason": "unknown"})
    monkeypatch.setattr(manage, "listener_pid", lambda *a: None)
    with pytest.raises(manage.LifecycleError, match="unconfirmed"):
        manage.start(tmp_path)
    assert len(manage.read_state(tmp_path)["services"]) == 2


@pytest.mark.skipif(manage.os.name != "nt", reason="Windows native process fixtures")
def test_real_command_mismatch_preserves_live_process_and_state(tmp_path, monkeypatch):
    import subprocess
    import sys

    monkeypatch.setattr(manage, "lifecycle_lock", lambda: nullcontext())
    service = manage.build_plan(tmp_path)["services"]["api"]
    child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    try:
        identity = manage.process_identity(child.pid)
        assert identity is not None
        record = {**identity, "command": [sys.executable, "old version command"]}
        manage.write_state(
            tmp_path, {"version": 1, "root": str(tmp_path), "services": {"api": record}}
        )
        result = manage.stop(tmp_path)
        assert result["stopped"] is False
        assert child.poll() is None
        assert manage.read_state(tmp_path)["services"]["api"]["pid"] == child.pid
        assert "command" in result["incomplete"]["api"]
        assert service["command"] != record["command"]
    finally:
        child.kill()
        child.wait(timeout=10)


def test_start_rollback_retains_process_when_termination_refused(tmp_path, monkeypatch):
    plan = manage.build_plan(tmp_path)
    plan["missing"] = []
    monkeypatch.setattr(manage, "build_plan", lambda *a: plan)
    monkeypatch.setattr(manage, "lifecycle_lock", lambda: nullcontext())
    monkeypatch.setattr(manage, "listener_pid", lambda *a: None)
    record = {"pid": 123, "created": "42", "image": plan["services"]["inference"]["command"][0]}
    monkeypatch.setattr(manage, "spawn", lambda *a: record)
    monkeypatch.setattr(manage, "terminate_owned", lambda *a: False)

    def not_ready(*args):
        raise manage.LifecycleError("not ready")

    monkeypatch.setattr(manage, "wait_ready", not_ready)
    with pytest.raises(manage.LifecycleError, match="cleanup incomplete"):
        manage.start(tmp_path)
    assert manage.read_state(tmp_path)["services"]["inference"]["pid"] == 123
