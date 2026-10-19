"""Foreground supervision never adopts or stops replacement service processes."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from local import manage


def test_serve_dry_run_does_not_launch(tmp_path, capsys):
    assert manage.main(["serve", "--root", str(tmp_path), "--dry-run"]) == 0
    assert '"dry_run": true' in capsys.readouterr().out
    assert not (tmp_path / "data").exists()


def test_supervisor_refuses_to_adopt_running_services(tmp_path, monkeypatch):
    plan = manage.build_plan(tmp_path)
    record = {
        "pid": 101,
        "created": "11",
        "image": plan["services"]["api"]["command"][0],
        "command": plan["services"]["api"]["command"],
    }
    manage.write_state(tmp_path, {"version": 1, "root": str(tmp_path), "services": {"api": record}})
    monkeypatch.setattr(manage, "lifecycle_lock", lambda: nullcontext())
    monkeypatch.setattr(manage, "ownership", lambda *a, **kw: {"owned": True})
    monkeypatch.setattr(manage, "listener_pid", lambda port: None)
    with pytest.raises(manage.LifecycleError, match="supervisor.*existing|existing.*supervisor"):
        manage.serve(tmp_path)
    assert manage.read_state(tmp_path)["services"]["api"] == record


def test_supervisor_cleanup_cannot_stop_replacement_process(tmp_path, monkeypatch):
    plan = manage.build_plan(tmp_path)
    saved = {
        "pid": 101,
        "created": "11",
        "image": plan["services"]["api"]["command"][0],
        "command": plan["services"]["api"]["command"],
    }
    newer = {**saved, "pid": 102, "created": "12"}
    manage.write_state(tmp_path, {"version": 1, "root": str(tmp_path), "services": {"api": newer}})
    monkeypatch.setattr(manage, "lifecycle_lock", lambda: nullcontext())
    calls = []
    monkeypatch.setattr(manage, "terminate_owned", lambda *a: calls.append(a) or True)
    result = manage.cleanup_supervised(tmp_path, {"api": saved}, plan)
    assert calls == []
    assert result["stopped"] is False
    assert manage.read_state(tmp_path)["services"]["api"] == newer


def test_supervisor_signal_cleans_its_exact_saved_receipts(tmp_path, monkeypatch):
    import signal

    plan = manage.build_plan(tmp_path)
    record = {
        "pid": 101,
        "created": "11",
        "image": plan["services"]["api"]["command"][0],
        "command": plan["services"]["api"]["command"],
    }
    handlers = {}
    cleanup = []

    def started(*args, **kwargs):
        kwargs["_receipts"]["api"] = record
        return {"supervised": {"api": record}}

    monkeypatch.setattr(manage, "start", started)
    monkeypatch.setattr(manage, "ownership", lambda *a: {"owned": True})

    def set_signal(sig, handler):
        previous = handlers.get(sig)
        handlers[sig] = handler
        return previous

    monkeypatch.setattr(manage.signal, "signal", set_signal)
    monkeypatch.setattr(
        manage,
        "time",
        SimpleNamespace(
            **{key: value for key, value in vars(manage.time).items() if key != "sleep"},
            sleep=lambda seconds: handlers[signal.SIGTERM](signal.SIGTERM, None),
        ),
    )
    monkeypatch.setattr(
        manage,
        "cleanup_supervised",
        lambda root, records, plan: cleanup.append(records) or {"stopped": True},
    )
    assert manage.serve(tmp_path)["stopped"] is True
    assert cleanup == [{"api": record}]
    assert handlers == {signal.SIGTERM: None, signal.SIGINT: None}


def test_supervisor_signal_during_startup_cannot_interrupt_rollback(tmp_path, monkeypatch):
    import signal

    plan = manage.build_plan(tmp_path)
    plan["missing"] = []
    monkeypatch.setattr(manage, "build_plan", lambda *args: plan)
    monkeypatch.setattr(manage, "lifecycle_lock", lambda: nullcontext())
    monkeypatch.setattr(manage, "listener_pid", lambda port: None)
    record = {
        "pid": 101,
        "created": "11",
        "image": plan["services"]["inference"]["command"][0],
        "command": plan["services"]["inference"]["command"],
    }
    handlers = {}
    stopped = []
    monkeypatch.setattr(manage, "spawn", lambda *a: record)
    monkeypatch.setattr(
        manage.signal, "signal", lambda sig, handler: handlers.setdefault(sig, handler)
    )
    monkeypatch.setattr(
        manage, "wait_ready", lambda *a, **kw: handlers[signal.SIGTERM](signal.SIGTERM, None)
    )

    def terminate(*args):
        stopped.append(args[0])
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        return True

    monkeypatch.setattr(manage, "terminate_owned", terminate)
    assert manage.serve(tmp_path)["stopped"] is True
    assert stopped == [record]
    assert manage.read_state(tmp_path)["services"] == {}
