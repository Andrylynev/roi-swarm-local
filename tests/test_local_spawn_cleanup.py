"""Failed launch verification owns the whole tree, including native venv shims."""

import gc
import inspect
import json
import os
import signal
import sys
import time
from contextlib import nullcontext, suppress
from pathlib import Path

import psutil
import pytest

from local import manage


@pytest.fixture
def native_tree(tmp_path):
    script = tmp_path / "tree.py"
    script.write_text(
        "import pathlib, subprocess, sys, time\n"
        "stage = int(sys.argv[1])\n"
        "if stage < 2:\n"
        "    subprocess.Popen([sys.executable, __file__, str(stage + 1)])\n"
        "else:\n"
        "    pathlib.Path('leaf.ready').touch()\n"
        "time.sleep(90)\n",
        encoding="utf-8",
    )
    service = {
        "command": [sys.executable, str(script), "0"],
        "log": str(tmp_path / "launch.log"),
    }
    observed = []

    def unverified(pid):
        deadline = time.monotonic() + 10
        while not (tmp_path / "leaf.ready").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (tmp_path / "leaf.ready").exists(), Path(service["log"]).read_text()
        root = psutil.Process(pid)
        observed.extend([root, *root.children(recursive=True)])
        for process in observed:
            process.create_time()  # All fixture cleanup uses reuse-checked objects.
        return None

    yield service, observed, unverified
    for process in reversed(observed):
        with suppress(psutil.NoSuchProcess):
            process.kill()
    psutil.wait_procs(observed, timeout=5)


def alive(process):
    return process.is_running() and process.status() != psutil.STATUS_ZOMBIE


def test_failed_verification_kills_owned_grandchildren(tmp_path, monkeypatch, native_tree):
    service, observed, unverified = native_tree
    monkeypatch.setattr(manage, "process_identity", unverified)
    with pytest.raises(manage.LifecycleError, match="cannot be verified"):
        manage.spawn(service, tmp_path, manage.child_environment(tmp_path))
    assert len(observed) >= 3
    _, survivors = psutil.wait_procs(observed, timeout=3)
    assert not [process.pid for process in survivors if alive(process)]


@pytest.mark.skipif(os.name != "nt", reason="Native Windows cleanup failure journal")
def test_unconfirmed_spawn_cleanup_blocks_duplicate_start(tmp_path, monkeypatch, native_tree):
    service, observed, unverified = native_tree
    monkeypatch.setattr(manage, "process_identity", unverified)

    def unconfirmed(scope):
        # Exercise a real cleanup, then fault only its exit confirmation.
        scope.scope.close()
        raise manage.LifecycleError("injected exit confirmation failure")

    monkeypatch.setattr(manage._WindowsLaunchScope, "close", unconfirmed)
    with pytest.raises(manage.LifecycleError, match="cleanup incomplete.*recovery"):
        manage.spawn(service, tmp_path, manage.child_environment(tmp_path))
    recovery = tmp_path / "data" / "local-spawn-recovery.json"
    receipt = json.loads(recovery.read_text(encoding="utf-8"))
    assert receipt["pid"] == observed[0].pid
    assert receipt["created"] and receipt["image"]
    assert receipt["command"] == service["command"]
    assert "exit confirmation" in receipt["stop_error"]
    assert {item["pid"] for item in receipt["members"]} == {p.pid for p in observed}
    assert all(item["created"] and item["image"] for item in receipt["members"])
    monkeypatch.setattr(manage, "lifecycle_lock", lambda: nullcontext())
    monkeypatch.setattr(manage, "listener_pid", lambda port: None)
    with pytest.raises(manage.LifecycleError, match="spawn.*recovery|recovery.*spawn"):
        manage.start(tmp_path)
    stopped = manage.stop(tmp_path)
    assert stopped["stopped"] is False
    assert "spawn" in stopped["incomplete"]
    assert manage.status(tmp_path)["running"] is False
    assert json.loads(recovery.read_text(encoding="utf-8")) == receipt


@pytest.mark.skipif(os.name != "nt", reason="Native Windows orphaned Job membership")
def test_failed_verification_kills_descendants_after_root_exits(tmp_path, monkeypatch, native_tree):
    service, observed, unverified = native_tree
    original_popen = manage.subprocess.Popen
    launched = []

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        launched.append(process)
        return process

    def root_exited(pid):
        unverified(pid)
        launched[0].kill()
        launched[0].wait(timeout=5)
        return None

    monkeypatch.setattr(manage.subprocess, "Popen", popen)
    monkeypatch.setattr(manage, "process_identity", root_exited)
    with pytest.raises(manage.LifecycleError, match="cannot be verified"):
        manage.spawn(service, tmp_path, manage.child_environment(tmp_path))
    _, survivors = psutil.wait_procs(observed, timeout=3)
    assert not [process.pid for process in survivors if alive(process)]


@pytest.mark.skipif(os.name != "nt", reason="Native detached launch handoff")
def test_successful_service_survives_launcher_exit(tmp_path):
    import subprocess

    receipt_path = tmp_path / "receipt.json"
    program = (
        "import json, pathlib, sys; from local import manage; "
        f"root=pathlib.Path({str(tmp_path)!r}); "
        "service={'command':[sys.executable,'-c','import time; time.sleep(90)'],"
        "'log':str(root/'survivor.log')}; "
        "record=manage.spawn(service,root,manage.child_environment(root)); "
        f"pathlib.Path({str(receipt_path)!r}).write_text(json.dumps(record))"
    )
    record = None
    try:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(manage.ROOT / "src") + os.pathsep + str(manage.ROOT)
        result = subprocess.run(
            [sys.executable, "-c", program],
            cwd=manage.ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        record = json.loads(receipt_path.read_text())
        assert manage.process_identity(record["pid"]) is not None
    finally:
        if record is not None:
            assert manage.terminate_owned(record, {"command": record["command"]})


@pytest.mark.skipif(os.name != "nt", reason="Explicit native handle handoff")
def test_successful_spawn_releases_its_creation_handle(tmp_path, monkeypatch):
    original_popen = manage.subprocess.Popen
    launched = []

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr(manage.subprocess, "Popen", popen)
    service = {
        "command": [sys.executable, "-c", "import time; time.sleep(90)"],
        "log": str(tmp_path / "released.log"),
    }
    record = manage.spawn(service, tmp_path, manage.child_environment(tmp_path))
    try:
        assert launched[0]._handle.closed
        assert not launched[0]._child_created
    finally:
        assert manage.terminate_owned(record, service)
        # Reap a pre-fix Popen as well, keeping the red run warning-free.
        if not launched[0]._handle.closed:
            launched[0].wait(timeout=5)


@pytest.mark.skipif(os.name != "nt", reason="Real Windows SIGINT during detached handoff")
@pytest.mark.parametrize("statement", ["process._handle.Close()", "return record"])
def test_sigint_at_detached_handoff_contains_tree_or_preserves_recovery(
    tmp_path, monkeypatch, native_tree, statement
):
    service, observed, unverified = native_tree
    original_identity = manage.process_identity

    def verified(pid):
        unverified(pid)
        return original_identity(pid)

    monkeypatch.setattr(manage, "process_identity", verified)
    lines, first = inspect.getsourcelines(manage.spawn)
    target = next(first + i for i, line in enumerate(lines) if line.strip() == statement)
    fired = []

    def interrupt(frame, event, arg):
        if event == "line" and frame.f_code is manage.spawn.__code__ and frame.f_lineno == target:
            fired.append(target)
            signal.raise_signal(signal.SIGINT)
        return interrupt

    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        sys.settrace(interrupt)
        with pytest.raises((KeyboardInterrupt, manage.LifecycleError)):
            manage.spawn(service, tmp_path, manage.child_environment(tmp_path))
    finally:
        sys.settrace(None)
        signal.signal(signal.SIGINT, previous)
    assert fired == [target]
    assert len(observed) >= 3
    _, survivors = psutil.wait_procs(observed, timeout=3)
    survivors = [p.pid for p in survivors if alive(p)]
    recovery = manage._spawn_recovery_path(tmp_path)
    if survivors:
        assert recovery.is_file(), f"Uncontained descendants without durable recovery: {survivors}"
        receipt = json.loads(recovery.read_text(encoding="utf-8"))
        assert receipt["pid"] == observed[0].pid
        assert receipt["created"] and receipt["image"]
        assert receipt["command"] == service["command"]
        assert manage.same_path(receipt["root"], str(tmp_path))
        assert set(survivors) <= {item["pid"] for item in receipt["members"]}
        assert all(item["created"] and item["image"] for item in receipt["members"])
        monkeypatch.setattr(manage, "lifecycle_lock", lambda: nullcontext())
        with pytest.raises(manage.LifecycleError, match="spawn.*recovery|recovery.*spawn"):
            manage.start(tmp_path)


@pytest.mark.skipif(os.name != "nt", reason="Durable Windows detached receipt adoption")
def test_start_adopts_detached_receipt_before_removing_recovery(tmp_path, monkeypatch):
    service = {
        "command": [sys.executable, "-c", "import time; time.sleep(90)"],
        "log": str(tmp_path / "adopted.log"),
        "port": 18775,
    }
    plan = {"missing": [], "services": {"api": service}}
    monkeypatch.setattr(manage, "build_plan", lambda *args: plan)
    monkeypatch.setattr(manage, "lifecycle_lock", lambda: nullcontext())
    monkeypatch.setattr(manage, "listener_pid", lambda port: None)
    monkeypatch.setattr(manage, "wait_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(manage, "status", lambda root: {"running": True})
    original_write = manage.write_state
    adopted = []

    def write(root, state):
        recovery = json.loads(manage._spawn_recovery_path(root).read_text(encoding="utf-8"))
        assert recovery["pid"] == state["services"]["api"]["pid"]
        assert recovery["created"] == state["services"]["api"]["created"]
        assert recovery["command"] == service["command"]
        adopted.append(recovery)
        original_write(root, state)

    monkeypatch.setattr(manage, "write_state", write)
    try:
        assert manage.start(tmp_path) == {"running": True}
        assert len(adopted) == 1
        assert not manage._spawn_recovery_path(tmp_path).exists()
    finally:
        record = manage.read_state(tmp_path)["services"].get("api")
        if record:
            assert manage.terminate_owned(record, service)


@pytest.mark.skipif(os.name != "nt", reason="Windows native launch handle accounting")
def test_interrupted_handoff_handle_count_is_stable(tmp_path, monkeypatch):
    original_identity = manage.process_identity
    lines, first = inspect.getsourcelines(manage.spawn)
    target = next(
        first + i for i, line in enumerate(lines) if line.strip() == "process._handle.Close()"
    )
    counter = psutil.Process()
    handles = []
    launched = []

    def identity(pid):
        process = psutil.Process(pid)
        process.create_time()
        launched.append(process)
        return original_identity(pid)

    monkeypatch.setattr(manage, "process_identity", identity)
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        for cycle in range(5):
            root = tmp_path / str(cycle)
            root.mkdir()
            service = {
                "command": [sys.executable, "-c", "import time; time.sleep(90)"],
                "log": str(root / "launch.log"),
            }
            launched.clear()

            def interrupt(frame, event, arg):
                if (
                    event == "line"
                    and frame.f_code is manage.spawn.__code__
                    and frame.f_lineno == target
                ):
                    signal.raise_signal(signal.SIGINT)
                return interrupt

            try:
                sys.settrace(interrupt)
                with pytest.raises((KeyboardInterrupt, manage.LifecycleError)):
                    manage.spawn(service, root, manage.child_environment(root))
            finally:
                sys.settrace(None)
                for process in launched:
                    with suppress(psutil.NoSuchProcess):
                        process.kill()
                psutil.wait_procs(launched, timeout=5)
            gc.collect()
            handles.append(counter.num_handles())
    finally:
        signal.signal(signal.SIGINT, previous)
    assert handles[1:] == [handles[1]] * 4, handles


def test_portable_spawn_failure_requires_confirmed_group_cleanup(tmp_path, monkeypatch):
    from types import SimpleNamespace

    events = []
    service = {"command": [sys.executable], "log": str(tmp_path / "group.log")}
    identity = {"pid": 101, "created": "42", "image": sys.executable}

    class Process:
        pid = 101

        def poll(self):
            return 0

        def wait(self, **kwargs):
            events.append("reaped")

    monkeypatch.setattr(
        manage,
        "os",
        SimpleNamespace(
            **{**vars(os), "name": "posix", "getpgid": lambda pid: pid, "getsid": lambda pid: pid}
        ),
    )
    monkeypatch.setattr(manage.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(manage.psutil, "Process", lambda pid: object())
    monkeypatch.setattr(manage, "identity_from_process", lambda process: identity)
    monkeypatch.setattr(manage, "process_identity", lambda pid: None)

    def cleanup(record, service, **kwargs):
        assert record["created"] == "42"
        assert record["pgid"] == record["sid"] == 101
        events.append("group cleanup")
        return False

    monkeypatch.setattr(manage, "terminate_process_group", cleanup)
    with pytest.raises(manage.LifecycleError, match="cleanup incomplete"):
        manage.spawn(service, tmp_path, {})
    assert events == ["group cleanup", "reaped"]
    assert (tmp_path / "data" / "local-spawn-recovery.json").exists()
