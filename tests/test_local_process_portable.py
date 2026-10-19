"""Ownership uses kernel evidence, never a listener PID as a kill target."""

import os
import socket
import sys
import time

import psutil
import pytest

from local import manage


def test_group_cleanup_signals_only_observed_process_objects(tmp_path, monkeypatch):
    from types import SimpleNamespace

    events = []
    service = {"command": ["owned-executable"]}
    record = {
        "pid": 101,
        "created": "42",
        "image": "owned-executable",
        "command": service["command"],
        "pgid": 101,
        "sid": 101,
    }

    class Process:
        def __init__(self, pid):
            self.pid = pid
            self.alive = True

        def create_time(self):
            return 42

        def status(self):
            return psutil.STATUS_RUNNING

        def suspend(self):
            events.append((self.pid, "stop"))

        def resume(self):
            events.append((self.pid, "continue"))

        def terminate(self):
            events.append((self.pid, "term"))
            self.alive = False

        def is_running(self):
            return self.alive

    leader, child = Process(101), Process(102)
    fake_os = SimpleNamespace(**{**vars(os), "getpgid": lambda pid: 101, "getsid": lambda pid: 101})
    monkeypatch.setattr(manage, "os", fake_os)
    monkeypatch.setattr(manage, "ownership", lambda *args: {"owned": True})
    monkeypatch.setattr(
        manage,
        "identity_from_process",
        lambda process: {k: record[k] for k in ("pid", "created", "image")},
    )
    monkeypatch.setattr(manage.psutil, "Process", lambda pid: leader)
    monkeypatch.setattr(
        manage.psutil, "process_iter", lambda: [p for p in (leader, child) if p.alive]
    )
    monkeypatch.setattr(manage.psutil, "wait_procs", lambda processes, timeout: (processes, []))
    assert manage.terminate_process_group(record, service) is True
    assert events == [
        (101, "stop"),
        (102, "stop"),
        (101, "term"),
        (102, "term"),
        (101, "continue"),
        (102, "continue"),
    ]
    events.clear()
    assert manage.terminate_process_group({**record, "sid": 999}, service) is False
    assert not events


def test_portable_identity_reads_real_process_without_win32():
    process = psutil.Process(os.getpid())
    identity = manage.identity_from_process(process)
    assert identity["pid"] == os.getpid()
    assert int(identity["created"]) > 0
    assert manage.same_path(identity["image"], process.exe())
    assert manage.identity_from_process(None) is None


def test_portable_listener_snapshot_reports_real_loopback_owner():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        assert manage.portable_listener_pid(server.getsockname()[1]) == os.getpid()


@pytest.mark.skipif(os.name == "nt", reason="POSIX session/process-group lifecycle")
def test_posix_spawn_owns_session_and_stops_grandchild(tmp_path):
    child_pid = tmp_path / "grandchild.pid"
    script = (
        "import subprocess,sys,time,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']); "
        f"pathlib.Path({str(child_pid)!r}).write_text(str(p.pid)); time.sleep(120)"
    )
    service = {"command": [sys.executable, "-c", script], "log": str(tmp_path / "child.log")}
    record = manage.spawn(service, tmp_path, manage.child_environment(tmp_path))
    try:
        deadline = time.monotonic() + 10
        while not child_pid.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert child_pid.exists()
        grandchild = psutil.Process(int(child_pid.read_text()))
        assert record["pgid"] == record["pid"] == os.getpgid(record["pid"])
        assert record["sid"] == record["pid"] == os.getsid(record["pid"])
        assert manage.terminate_owned({**record, "created": "0"}, service) is False
        assert grandchild.is_running()
        assert manage.terminate_owned(record, service) is True
        assert manage.process_identity(record["pid"]) is None
        assert not grandchild.is_running() or grandchild.status() == psutil.STATUS_ZOMBIE
    finally:
        if manage.process_identity(record["pid"]):
            manage.terminate_owned(record, service)


@pytest.mark.skipif(os.name == "nt", reason="POSIX isolated group admission")
def test_posix_refuses_foreign_or_missing_process_group_evidence(tmp_path):
    import subprocess

    process = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(120)"])
    service = {"command": [sys.executable, "-c", "import time;time.sleep(120)"]}
    try:
        record = {**manage.process_identity(process.pid), "command": service["command"]}
        assert manage.terminate_owned(record, service) is False
        assert process.poll() is None
    finally:
        process.kill()
        process.wait(timeout=10)
