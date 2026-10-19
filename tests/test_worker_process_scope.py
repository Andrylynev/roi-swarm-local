"""Real process tests: only processes created by these tests are terminated."""

import ctypes
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

if os.name == "nt":
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _kernel32.WaitForSingleObject.restype = wintypes.DWORD
    _kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.TerminateProcess.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL


class _ProcessProbe:
    """Keep an OS handle to our child, not a reusable Windows PID."""

    def __init__(self, pid):
        self.pid = pid
        self.handle = None
        self.fd = None
        if os.name == "nt":
            self.handle = _kernel32.OpenProcess(0x00100001, False, pid)
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())
        elif hasattr(os, "pidfd_open"):
            self.fd = os.pidfd_open(pid)

    def alive(self):
        if os.name == "nt":
            state = _kernel32.WaitForSingleObject(self.handle, 0)
            if state == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
            return state == 0x102
        try:
            os.kill(self.pid, 0)
            stat = Path(f"/proc/{self.pid}/stat")
            return not (stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] == "Z")
        except ProcessLookupError:
            return False

    def wait_dead(self):
        # Job accounting reaches zero just before process handles are signaled.
        deadline = time.monotonic() + 5
        while self.alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not self.alive(), f"Test-owned child {self.pid} is still running"

    def close(self):
        # Independent emergency cleanup also runs when the implementation fails.
        if self.alive():
            if os.name == "nt":
                if not _kernel32.TerminateProcess(self.handle, 1):
                    error = ctypes.get_last_error()
                    if _kernel32.WaitForSingleObject(self.handle, 5000) != 0:
                        raise ctypes.WinError(error)
                assert _kernel32.WaitForSingleObject(self.handle, 5000) == 0
            elif self.fd is not None and hasattr(signal, "pidfd_send_signal"):
                signal.pidfd_send_signal(self.fd, signal.SIGKILL)
            else:
                os.kill(self.pid, signal.SIGKILL)
        if self.handle is not None:
            assert _kernel32.CloseHandle(self.handle)
        if self.fd is not None:
            os.close(self.fd)


def _events(path, marker):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if len(records) == 2:
            assert {record["kind"] for record in records} == {"host", "child"}
            assert all(record["marker"] == marker for record in records)
            return {record["kind"]: record for record in records}
        time.sleep(0.01)
    pytest.fail(f"No host/child readiness records: {path.read_text()!r}")


@contextmanager
def _tree(tmp_path, *, exit_host):
    from roi_swarm.worker_process_scope import WorkerProcessScope

    marker = uuid.uuid4().hex
    child_script = tmp_path / "child.py"
    child_script.write_text(
        "import json, os, sys, time\n"
        "print(json.dumps({'kind': 'child', 'pid': os.getpid(), "
        "'marker': sys.argv[1]}), flush=True)\n"
        "time.sleep(60)\n"
    )
    host_script = tmp_path / "host.py"
    host_script.write_text(
        "import json, os, subprocess, sys\n"
        "assert sys.stdin.readline().strip() == 'spawn'\n"
        "child = subprocess.Popen([sys._base_executable, '-u', sys.argv[1], sys.argv[2]], "
        "stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr)\n"
        "print(json.dumps({'kind': 'host', 'pid': os.getpid(), "
        "'child_pid': child.pid, 'marker': sys.argv[2]}), flush=True)\n"
        "assert sys.stdin.readline().strip() == 'exit'\n"
    )
    scope = WorkerProcessScope()
    host = probe = None
    log = tmp_path / "stdout.jsonl"
    try:
        expected = {"creationflags": 0x00000004} if os.name == "nt" else {"start_new_session": True}
        assert scope.launch_kwargs == expected
        with log.open("w") as stdout, (tmp_path / "stderr.txt").open("w") as stderr:
            host = subprocess.Popen(
                [sys.executable, "-u", str(host_script), str(child_script), marker],
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                text=True,
                **scope.launch_kwargs,
            )
        scope.attach(host.pid)
        host.stdin.write("spawn\n")
        host.stdin.flush()
        records = _events(log, marker)
        actual_host_pid = records["host"]["pid"]
        if os.name != "nt":
            assert actual_host_pid == host.pid
        assert actual_host_pid not in (os.getpid(), records["child"]["pid"])
        child_pid = records["child"]["pid"]
        assert records["host"]["child_pid"] == child_pid
        assert child_pid not in (host.pid, os.getpid())
        probe = _ProcessProbe(child_pid)
        assert probe.alive()
        if exit_host:
            host.stdin.write("exit\n")
            host.stdin.flush()
            assert host.wait(timeout=10) == 0
            assert probe.alive(), "The host must leave a real living orphan"
        yield scope, host, probe
    finally:
        try:
            scope.close()
        finally:
            if probe is not None:
                probe.close()
            if host is not None:
                if host.poll() is None:
                    host.kill()
                host.wait(timeout=10)
                host.stdin.close()


def test_close_kills_orphan_after_host_exits_normally(tmp_path):
    with _tree(tmp_path, exit_host=True) as (scope, host, child):
        assert host.returncode == 0
        scope.close()
        assert not child.alive()
        child.wait_dead()
        scope.close()  # Successful close is idempotent.


def test_close_kills_running_host_and_descendant(tmp_path):
    with _tree(tmp_path, exit_host=False) as (scope, host, child):
        assert host.poll() is None
        scope.close()
        assert host.wait(timeout=5) != 0
        assert not child.alive()


def test_attach_accepts_already_exited_host():
    from roi_swarm.worker_process_scope import WorkerProcessScope

    host = subprocess.Popen(
        [sys._base_executable, "-c", "pass"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert host.wait(timeout=10) == 0
    scope = WorkerProcessScope()
    try:
        scope.attach(host.pid)
    finally:
        scope.close()
    scope.close()


def test_empty_scope_close_is_idempotent():
    from roi_swarm.worker_process_scope import WorkerProcessScope

    scope = WorkerProcessScope()
    scope.close()
    scope.close()
    with pytest.raises(RuntimeError, match="closed"):
        scope.attach(os.getpid())


@pytest.mark.parametrize("pid", [0, -1, True, "123", os.getpid()])
def test_attach_rejects_unsafe_pid_without_touching_processes(pid):
    from roi_swarm.worker_process_scope import WorkerProcessScope

    scope = WorkerProcessScope()
    try:
        with pytest.raises(ValueError, match="PID"):
            scope.attach(pid)
    finally:
        scope.close()


def test_second_attach_rejected_without_losing_original_subtree(tmp_path):
    with _tree(tmp_path, exit_host=False) as (scope, host, child):
        with pytest.raises(RuntimeError, match="already"):
            scope.attach(host.pid)
        scope.close()
        assert not child.alive()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job accounting")
def test_close_failure_retains_job_for_retry(tmp_path, monkeypatch):
    from roi_swarm import worker_process_scope as module

    with _tree(tmp_path, exit_host=True) as (scope, _host, child):
        with monkeypatch.context() as fault:

            def fail_query(_handle):
                raise OSError("Job accounting query failed")

            fault.setattr(module, "_active_processes", fail_query)
            with pytest.raises(OSError, match="accounting query failed"):
                scope.close()
        scope.close()
        assert not child.alive()
