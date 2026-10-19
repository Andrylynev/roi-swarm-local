"""Offline lifecycle tests; never launch or stop the installed services."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest


def launcher():
    path = Path(__file__).resolve().parents[1] / "local" / "manage.py"
    assert path.is_file(), "Independent local lifecycle launcher has not been implemented"
    spec = importlib.util.spec_from_file_location("local_manage_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Tests never contend with the installed launcher's fixed 8776 lock.
    original_lock = module.lifecycle_lock
    module.lifecycle_lock = lambda port=0: original_lock(port=port)
    return module


def test_dry_start_plan_is_local_and_does_not_write(tmp_path):
    manage = launcher()
    runtime = tmp_path / "runtime" / "prism" / "nested" / "llama-server.exe"
    runtime.parent.mkdir(parents=True)
    runtime.touch()
    pythonw = tmp_path / ".venv" / "Scripts" / "pythonw.exe"
    pythonw.parent.mkdir(parents=True)
    pythonw.touch()
    preset = tmp_path / "local" / "models.ini"
    preset.parent.mkdir()
    preset.write_text("[tiny]\nmodel = models/tiny.gguf\n", encoding="utf-8")
    before = sorted(str(path) for path in tmp_path.rglob("*"))
    plan = manage.start(tmp_path, models_max=2, dry_run=True)
    assert plan["dry_run"] is True
    assert plan["missing"] == []
    assert plan["services"]["inference"]["command"] == [
        str(runtime.resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        "8091",
        "--models-preset",
        str(preset.resolve()),
        "--models-max",
        "2",
        "--no-webui",
    ]
    assert plan["services"]["api"]["command"] == [
        str(pythonw.resolve()),
        "-m",
        "uvicorn",
        "local.api:create_local_app",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        "8775",
        "--no-access-log",
    ]
    assert plan["state_file"] == str(tmp_path / "data" / "local-services.json")
    assert sorted(str(path) for path in tmp_path.rglob("*")) == before
    assert "8765" not in json.dumps(plan)
    assert "8766" not in json.dumps(plan)


def test_child_environment_removes_cloud_and_source_configuration(tmp_path):
    manage = launcher()
    source = {
        "PATH": "C:/Windows/System32",
        "SystemRoot": "C:/Windows",
        "OPENAI_API_KEY": "do-not-inherit",
        "openai_base_url": "https://remote.invalid",
        "ROI_SWARM_API_KEY": "source-secret",
        "ROI_PROVIDER_BASE_URL": "https://remote.invalid",
        "ROI_WORKER_API_KEY": "worker-secret",
        "ROI_CONFIG": "D:/ROI/config/settings.yaml",
        "ANTHROPIC_API_KEY": "also-secret",
        "HF_TOKEN": "hub-secret",
        "HERMES_HOME": "D:/ROI/hermes",
        "PYTHONPATH": "D:/ROI/src",
        "HTTPS_PROXY": "https://proxy.invalid",
        "LLAMA_ARG_MODEL": "remote.gguf",
        "VIRTUAL_ENV": "D:/ROI/.venv",
        "CONDA_PREFIX": "D:/ROI/conda",
        "ROI": "remote-route",
        "API_KEY": "generic-secret",
    }
    env = manage.child_environment(tmp_path, source)
    assert env["ROI_PROVIDER"] == "local"
    assert env["OPENAI_API_KEY"] == "local-only"
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:8091/v1"
    assert env["HERMES_HOME"] == str(tmp_path / "hermes")
    assert env["LLAMA_CACHE"] == str(tmp_path / "cache" / "llama")
    assert env["HF_HOME"].startswith(str(tmp_path))
    assert env["UV_CACHE_DIR"].startswith(str(tmp_path))
    assert env["TIKTOKEN_CACHE_DIR"].startswith(str(tmp_path))
    assert env["PATH"].endswith(source["PATH"])
    assert env["PATH"].startswith(
        str(tmp_path / ".venv" / ("Scripts" if os.name == "nt" else "bin"))
    )
    assert env["VIRTUAL_ENV"] == str(tmp_path / ".venv")
    assert not any(
        value in env.values()
        for value in (
            "do-not-inherit",
            "source-secret",
            "worker-secret",
            "also-secret",
            "hub-secret",
            "https://remote.invalid",
            "D:/ROI/hermes",
            "D:/ROI/src",
            "https://proxy.invalid",
        )
    )
    assert "ROI_SWARM_API_KEY" not in env
    assert "ROI_CONFIG" not in env
    assert "PYTHONPATH" not in env
    assert "LLAMA_ARG_MODEL" not in env
    assert source["OPENAI_API_KEY"] == "do-not-inherit"
    assert "ROI" not in env
    assert "API_KEY" not in env
    assert "CONDA_PREFIX" not in env
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


@pytest.mark.parametrize("bad_image", [None, 7, [], {}])
def test_invalid_saved_image_cannot_be_owned(tmp_path, monkeypatch, bad_image):
    manage = launcher()
    service = manage.build_plan(tmp_path)["services"]["api"]
    record = {"pid": 101, "created": "22", "image": bad_image, "command": service["command"]}
    monkeypatch.setattr(
        manage,
        "process_identity",
        lambda pid: {
            "pid": pid,
            "created": "22",
            "image": service["command"][0],
        },
    )
    assert manage.ownership(record, service)["owned"] is False


def test_identity_and_state_are_bound_to_this_copy(tmp_path, monkeypatch):
    manage = launcher()
    plan = manage.build_plan(tmp_path)
    command = plan["services"]["api"]["command"]
    identity = {"pid": 12345, "created": "9001", "image": command[0]}
    record = {**identity, "command": command}
    monkeypatch.setattr(manage, "process_identity", lambda pid: identity)
    assert manage.ownership(record, plan["services"]["api"])["owned"] is True
    monkeypatch.setattr(manage, "process_identity", lambda pid: {**identity, "created": "9002"})
    assert manage.ownership(record, plan["services"]["api"])["owned"] is False
    monkeypatch.setattr(
        manage, "process_identity", lambda pid: {**identity, "image": "C:/other.exe"}
    )
    assert manage.ownership(record, plan["services"]["api"])["owned"] is False
    manage.write_state(tmp_path, {"version": 1, "root": str(tmp_path), "services": {}})
    assert manage.read_state(tmp_path)["services"] == {}
    (tmp_path / "data" / "local-services.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(manage.LifecycleError, match="state"):
        manage.read_state(tmp_path)


@pytest.mark.skipif(os.name != "nt", reason="Win32 process identity")
def test_win32_identity_observes_current_process():
    manage = launcher()
    identity = manage.process_identity(os.getpid())
    assert identity["pid"] == os.getpid()
    assert int(identity["created"]) > 0
    assert Path(identity["image"]).name.lower() in {"python.exe", "pythonw.exe"}
    assert manage.process_identity(4294967295) is None


def test_status_never_adopts_foreign_listener(tmp_path, monkeypatch):
    manage = launcher()
    monkeypatch.setattr(manage, "listener_pid", lambda port: 333)
    monkeypatch.setattr(manage, "health", lambda url: {"ready": True, "status": 200})
    result = manage.status(tmp_path)
    assert result["running"] is False
    assert result["services"]["api"]["owned"] is False
    assert result["services"]["api"]["listener_pid"] == 333
    assert result["services"]["api"]["health"]["ready"] is True
    assert not (tmp_path / "data").exists()


@pytest.mark.skipif(os.name != "nt", reason="Win32 listener table")
def test_win32_listener_pid_and_exclusive_lifecycle_lock():
    import socket

    manage = launcher()
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        assert manage.listener_pid(port) == os.getpid()
        with (
            manage.lifecycle_lock(port=port + 1),
            pytest.raises(manage.LifecycleError, match="another lifecycle"),
            manage.lifecycle_lock(port=port + 1),
        ):
            pass


def installation(tmp_path):
    for relative in (
        "runtime/prism/llama-server.exe",
        ".venv/Scripts/pythonw.exe",
        "local/models.ini",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_start_refuses_busy_unowned_port_before_spawn(tmp_path, monkeypatch):
    manage = launcher()
    installation(tmp_path)
    monkeypatch.setattr(manage, "listener_pid", lambda port: 412)
    spawned = []
    monkeypatch.setattr(manage, "spawn", lambda *args: spawned.append(args))
    with pytest.raises(manage.LifecycleError, match="8091.*unowned"):
        manage.start(tmp_path)
    assert spawned == []
    assert not (tmp_path / "data" / "local-services.json").exists()


def test_start_persists_identity_before_wait_and_is_idempotent(tmp_path, monkeypatch):
    manage = launcher()
    installation(tmp_path)
    processes = {}
    calls = []
    monkeypatch.setattr(
        manage,
        "listener_pid",
        lambda port: {8091: 101, 8775: 102}.get(port) if len(processes) == 2 else None,
    )
    monkeypatch.setattr(manage, "process_identity", lambda pid: processes.get(pid))
    monkeypatch.setattr(manage, "health", lambda url: {"ready": True, "status": 200})

    def spawn(service, root, env):
        pid = 101 + len(processes)
        identity = {"pid": pid, "created": str(pid), "image": service["command"][0]}
        processes[pid] = identity
        calls.append(service["command"])
        return {**identity, "command": service["command"]}

    def ready(name, service, record, timeout):
        assert manage.read_state(tmp_path)["services"][name] == record

    monkeypatch.setattr(manage, "spawn", spawn)
    monkeypatch.setattr(manage, "wait_ready", ready)
    result = manage.start(tmp_path, models_max=2)
    assert result["running"] is True
    assert len(calls) == 2
    assert manage.read_state(tmp_path)["models_max"] == 2
    assert manage.start(tmp_path, models_max=2)["running"] is True
    assert len(calls) == 2


def test_start_failure_rolls_back_only_new_owned_processes(tmp_path, monkeypatch):
    manage = launcher()
    installation(tmp_path)
    monkeypatch.setattr(manage, "listener_pid", lambda port: None)
    stopped = []
    record = {"pid": 501, "created": "55", "image": "unused"}
    monkeypatch.setattr(manage, "spawn", lambda *args: record)
    monkeypatch.setattr(manage, "terminate_owned", lambda rec, service: stopped.append(rec) or True)

    def fail(*args):
        raise manage.LifecycleError("not ready")

    monkeypatch.setattr(manage, "wait_ready", fail)
    with pytest.raises(manage.LifecycleError, match="not ready"):
        manage.start(tmp_path)
    assert stopped == [record]
    assert manage.read_state(tmp_path)["services"] == {}


def test_stop_ignores_recycled_pid_and_has_no_port_killing(tmp_path, monkeypatch):
    manage = launcher()
    plan = manage.build_plan(tmp_path)
    records = {
        name: {
            "pid": 701 + index,
            "created": "11",
            "image": item["command"][0],
            "command": item["command"],
        }
        for index, (name, item) in enumerate(plan["services"].items())
    }
    manage.write_state(tmp_path, {"version": 1, "root": str(tmp_path), "services": records})
    monkeypatch.setattr(
        manage,
        "process_identity",
        lambda pid: {"pid": pid, "created": "12", "image": records["inference"]["image"]},
    )
    monkeypatch.setattr(manage, "listener_pid", lambda port: 999)
    killed = []
    monkeypatch.setattr(manage.subprocess, "run", lambda *a, **kw: killed.append(a))
    result = manage.stop(tmp_path)
    assert result["stopped"] is False
    assert result["skipped"] == ["api", "inference"]
    assert killed == []
    retained = manage.read_state(tmp_path)["services"]
    assert set(retained) == {"api", "inference"}
    assert all("recycled PID" in record["stop_error"] for record in retained.values())


def test_ready_retries_503_but_never_trusts_foreign_health(tmp_path, monkeypatch):
    manage = launcher()
    service = manage.build_plan(tmp_path)["services"]["inference"]
    record = {"pid": 51}
    monkeypatch.setattr(manage, "ownership", lambda *args: {"owned": True})
    replies = iter(({"ready": False, "status": 503}, {"ready": True, "status": 200}))
    monkeypatch.setattr(manage, "health", lambda url: next(replies))
    monkeypatch.setattr(manage, "listener_pid", lambda port: 51)
    monkeypatch.setattr(manage.time, "sleep", lambda seconds: None)
    manage.wait_ready("inference", service, record, 2)
    monkeypatch.setattr(manage, "health", lambda url: {"ready": True, "status": 200})
    monkeypatch.setattr(manage, "listener_pid", lambda port: 999)
    with pytest.raises(manage.LifecycleError, match="unowned"):
        manage.wait_ready("inference", service, record, 2)


@pytest.mark.skipif(os.name != "nt", reason="Win32 detached process flags")
def test_spawn_redirects_logs_and_uses_detached_native_process(tmp_path, monkeypatch):
    manage = launcher()
    service = manage.build_plan(tmp_path)["services"]["api"]
    calls = []

    class Process:
        pid = 888

        class _handle:
            @staticmethod
            def Close():
                pass

    class Scope:
        def __init__(self):
            from types import SimpleNamespace

            self.scope = SimpleNamespace(_closed=False)

        def identity(self, process):
            return {"pid": process.pid, "created": "42", "image": service["command"][0]}

        def attach(self, process):
            assert process.pid == 888

        def release(self):
            assert not self.scope._closed

        def members(self):
            return [self.identity(Process())]

        def detach(self):
            self.scope._closed = True

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write(b"test-log-marker\n")
        return Process()

    monkeypatch.setattr(manage.subprocess, "Popen", popen)
    monkeypatch.setattr(manage, "_WindowsLaunchScope", Scope)
    monkeypatch.setattr(
        manage,
        "process_identity",
        lambda pid: {
            "pid": pid,
            "created": "42",
            "image": service["command"][0],
        },
    )
    record = manage.spawn(service, tmp_path, {"ROI_PROVIDER": "local"})
    assert record["pid"] == 888
    command, options = calls[0]
    assert command == service["command"]
    assert options["cwd"] == tmp_path
    assert options["stdin"] == manage.subprocess.DEVNULL
    assert options["stderr"] == manage.subprocess.STDOUT
    assert options["creationflags"] & manage.subprocess.DETACHED_PROCESS
    assert options["creationflags"] & manage.subprocess.CREATE_NO_WINDOW
    assert options["creationflags"] & 0x00000004  # Assign launch Job before resuming.
    assert b"test-log-marker" in Path(service["log"]).read_bytes()


@pytest.mark.skipif(os.name != "nt", reason="Real native hidden Python lifecycle")
def test_native_spawn_owns_child_listener_and_stops_its_tree(tmp_path):
    import socket
    import sys
    import time

    manage = launcher()
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    service = {
        "command": [
            str(Path(sys.executable).with_name("pythonw.exe")),
            "-m",
            "http.server",
            str(port),
            "--bind",
            "127.0.0.1",
        ],
        "log": str(tmp_path / "native-smoke.log"),
    }
    record = manage.spawn(service, tmp_path, manage.child_environment(tmp_path))
    try:
        deadline = time.monotonic() + 10
        listener = None
        while listener is None and time.monotonic() < deadline:
            listener = manage.listener_pid(port)
            time.sleep(0.05)
        assert listener is not None, Path(service["log"]).read_text(encoding="utf-8")
        assert manage.listener_belongs_to(listener, record) is True
        assert manage.terminate_owned(record, service) is True
        assert manage.process_identity(record["pid"]) is None
        assert manage.listener_pid(port) is None
    finally:
        if manage.process_identity(record["pid"]) is not None:
            manage.terminate_owned(record, service)


def test_cli_inspect_has_json_and_models_override(tmp_path, capsys, monkeypatch):
    manage = launcher()
    monkeypatch.setattr(manage, "listener_pid", lambda port: None)
    assert manage.main(["inspect", "--root", str(tmp_path), "--models-max", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["models_max"] == 2
    assert len(payload["missing"]) == 3
    assert not (tmp_path / "data").exists()
    assert manage.main(["start", "--root", str(tmp_path)]) == 1
    error = json.loads(capsys.readouterr().out)
    assert "Missing local prerequisites" in error["error"]


def test_health_never_follows_redirects_outside_loopback(monkeypatch):
    import urllib.request

    manage = launcher()
    handlers = []

    class Opener:
        def open(self, url, timeout):
            raise manage.urllib.error.HTTPError(url, 302, "redirect", {}, None)

    def build(*items):
        handlers.extend(items)
        return Opener()

    monkeypatch.setattr(manage.urllib.request, "build_opener", build)
    assert manage.health("http://127.0.0.1:8775/health")["ready"] is False
    assert any(isinstance(handler, urllib.request.HTTPRedirectHandler) for handler in handlers)
    redirect = next(h for h in handlers if isinstance(h, urllib.request.HTTPRedirectHandler))
    with pytest.raises(manage.urllib.error.HTTPError):
        redirect.redirect_request(None, None, 302, "redirect", {}, "https://remote.invalid/")


@pytest.mark.skipif(os.name != "nt", reason="Win32 taskkill failure")
def test_taskkill_failure_is_reported_even_if_root_exits(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    manage = launcher()
    service = manage.build_plan(tmp_path)["services"]["api"]
    record = {
        "pid": 101,
        "created": "22",
        "image": service["command"][0],
        "command": service["command"],
    }
    monkeypatch.setattr(manage, "ownership", lambda *args, **kwargs: {"owned": True})
    monkeypatch.setattr(manage, "process_handle", lambda pid: nullcontext("pinned"))
    identities = iter((record, None))
    monkeypatch.setattr(manage, "identity_from_handle", lambda *args: next(identities))
    monkeypatch.setattr(
        manage.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="child could not be terminated",
        ),
    )
    with pytest.raises(manage.LifecycleError, match="taskkill"):
        manage.terminate_owned(record, service)


def test_ready_times_out_with_log_location(tmp_path, monkeypatch):
    manage = launcher()
    service = manage.build_plan(tmp_path)["services"]["inference"]
    monkeypatch.setattr(manage, "ownership", lambda *args: {"owned": True})
    monkeypatch.setattr(manage, "health", lambda url: {"ready": False, "status": 503})
    clock = iter((0, 5))
    monkeypatch.setattr(manage.time, "monotonic", lambda: next(clock))
    with pytest.raises(manage.LifecycleError, match="inference.log"):
        manage.wait_ready("inference", service, {"pid": 1}, timeout=1)


@pytest.mark.skipif(os.name != "nt", reason="Real native lifecycle")
def test_native_full_lifecycle_on_ephemeral_ports(tmp_path, monkeypatch):
    import socket
    import sys

    manage = launcher()
    ports = []
    for _ in range(2):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            ports.append(reservation.getsockname()[1])
    plan = manage.build_plan(tmp_path)
    plan["missing"] = []
    for index, service in enumerate(plan["services"].values()):
        port = ports[index]
        service.update(
            {
                "command": [
                    str(Path(sys.executable).with_name("pythonw.exe")),
                    "-m",
                    "http.server",
                    str(port),
                    "--bind",
                    "127.0.0.1",
                ],
                "port": port,
                "health_url": f"http://127.0.0.1:{port}/",
            }
        )
    monkeypatch.setattr(manage, "build_plan", lambda root, models_max=1: plan)
    try:
        first = manage.start(tmp_path, ready_timeout=15)
        assert first["running"] is True
        assert manage.status(tmp_path)["running"] is True
        assert manage.start(tmp_path, ready_timeout=5)["running"] is True
        stopped = manage.stop(tmp_path)
        assert stopped["terminated"] == ["api", "inference"]
        assert manage.read_state(tmp_path)["services"] == {}
        assert all(manage.listener_pid(port) is None for port in ports)
    finally:
        manage.stop(tmp_path)
