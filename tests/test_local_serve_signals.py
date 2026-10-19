"""Signals may arrive at any startup bytecode, not just in readiness waits."""

import inspect
import signal
import sys
from contextlib import nullcontext

import pytest

from local import manage


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize(
    "statement",
    [
        "started.append(name)",
        'state["services"][name] = record',
        "_receipts[name] = record",
    ],
)
def test_signal_between_spawn_and_registration_cleans_exact_receipt(
    tmp_path, monkeypatch, statement, signum
):
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
    spawned, stopped, handlers = [], [], {}

    def spawn(*args):
        spawned.append(record)
        return record

    def set_signal(sig, handler):
        previous = handlers.get(sig)
        handlers[sig] = handler
        return previous

    monkeypatch.setattr(manage, "spawn", spawn)
    monkeypatch.setattr(manage.signal, "signal", set_signal)
    monkeypatch.setattr(
        manage, "terminate_owned", lambda receipt, service: stopped.append(receipt) or True
    )
    # On a correct implementation the pending signal is checked before readiness.
    monkeypatch.setattr(
        manage, "wait_ready", lambda *a, **kw: pytest.fail("readiness after shutdown request")
    )
    source, first = inspect.getsourcelines(manage.start)
    line = next(first + index for index, text in enumerate(source) if text.strip() == statement)
    fired = []

    def inject(frame, event, arg):
        if (
            frame.f_code is manage.start.__code__
            and event == "line"
            and frame.f_lineno == line
            and not fired
        ):
            fired.append(True)
            handlers[signum](signum, frame)
        return inject

    previous_trace = sys.gettrace()
    try:
        sys.settrace(inject)
        result = manage.serve(tmp_path)
    finally:
        sys.settrace(previous_trace)
    assert fired == [True]
    assert result["stopped"] is True
    assert spawned == stopped == [record]
    assert manage.read_state(tmp_path)["services"] == {}
    assert handlers == {signal.SIGTERM: None, signal.SIGINT: None}


def test_signal_during_real_readiness_cleans_registered_native_service(tmp_path, monkeypatch):
    plan = manage.build_plan(tmp_path)
    plan["missing"] = []
    for service in plan["services"].values():
        service["command"] = [sys.executable, "-c", "import time; time.sleep(90)"]
    monkeypatch.setattr(manage, "build_plan", lambda *args: plan)
    monkeypatch.setattr(manage, "lifecycle_lock", lambda: nullcontext())
    monkeypatch.setattr(manage, "listener_pid", lambda port: None)
    handlers, checked, stopped = {}, [], []

    def set_signal(sig, handler):
        previous = handlers.get(sig)
        handlers[sig] = handler
        return previous

    def health(url):
        saved = manage.read_state(tmp_path)["services"]
        assert set(saved) == {"inference"}
        checked.append(saved["inference"])
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        return {"ready": False, "status": 503}

    terminate = manage.terminate_owned

    def stop(record, service):
        handlers[signal.SIGINT](signal.SIGINT, None)
        stopped.append(record)
        return terminate(record, service)

    monkeypatch.setattr(manage.signal, "signal", set_signal)
    monkeypatch.setattr(manage, "health", health)
    monkeypatch.setattr(manage, "terminate_owned", stop)
    assert manage.serve(tmp_path, ready_timeout=30)["stopped"]
    assert len(checked) == 1 and stopped == checked
    assert manage.process_identity(checked[0]["pid"]) is None
    assert manage.read_state(tmp_path)["services"] == {}
