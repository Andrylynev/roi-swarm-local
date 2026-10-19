"""Portable install discovery and real, disposable process ownership probes."""

import json
import os
from pathlib import Path

import pytest

from local import manage


def install_record(directory, **changes):
    root = directory
    record = {
        "version": 1,
        "root": str(root.resolve()),
        "backend": "cpu",
        "profile": "fast",
        "api_port": 18775,
        "inference_port": 18091,
        "lock_port": 18776,
        "hermes_root": str(root / "runtime/hermes-agent"),
        "hermes_python": str(
            root
            / "runtime/hermes-agent/.venv"
            / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        ),
    }
    record.update(changes)
    path = root / "data/local-install.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return record


def test_plan_discovers_portable_runtime_and_install_ports(tmp_path):
    root = tmp_path / "user owned app"
    record = install_record(root)
    for relative in ("runtime/prism/llama-server", ".venv/bin/python", "local/models.ini"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    before = {str(p): p.stat().st_mtime_ns for p in root.rglob("*")}
    plan = manage.build_plan(root)
    assert plan["missing"] == []
    assert plan["services"]["api"]["command"][0] == str(root / ".venv/bin/python")
    assert plan["services"]["inference"]["command"][0].endswith("llama-server")
    assert plan["services"]["api"]["port"] == record["api_port"]
    assert plan["services"]["inference"]["port"] == record["inference_port"]
    assert plan["lock_port"] == record["lock_port"]
    assert plan["backend"] == "cpu"
    assert plan["services"]["api"]["health_url"] == "http://127.0.0.1:18775/health"
    assert {str(p): p.stat().st_mtime_ns for p in root.rglob("*")} == before


@pytest.mark.parametrize(
    "changes",
    [
        {"version": True},
        {"backend": "vulkan"},
        {"api_port": True},
        {"lock_port": 18775},
        {"root": "/another/install"},
        {"hermes_python": "/usr/bin/python"},
        {"hermes_root": "/global/hermes"},
        {"profile": "unknown"},
    ],
)
def test_invalid_install_metadata_is_never_used(tmp_path, changes):
    install_record(tmp_path, **changes)
    with pytest.raises(manage.LifecycleError, match="install record"):
        manage.build_plan(tmp_path)


def test_invalid_optional_install_record_is_not_silently_ignored(tmp_path, monkeypatch):
    path = tmp_path / "data/local-install.json"
    path.parent.mkdir()
    path.write_bytes(b"\xff")
    with pytest.raises(manage.LifecycleError, match="install record"):
        manage.build_plan(tmp_path)
    original = Path.read_text

    def denied(target, **kwargs):
        if target == path:
            raise PermissionError("denied")
        return original(target, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(manage.LifecycleError, match="install record"):
        manage.build_plan(tmp_path)


def test_child_environment_uses_portable_paths_and_runner_ports(tmp_path):
    install_record(tmp_path)
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    env = manage.child_environment(tmp_path, {"PATH": "system", "TMPDIR": "/global/tmp"})
    assert env["PATH"].startswith(str(python.parent) + os.pathsep)
    assert env["TMPDIR"] == str(tmp_path / "cache/tmp")
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:18091/v1"
    assert env["ROI_LOCAL_ROOT"] == str(tmp_path)
    assert env["ROI_LOCAL_API_PORT"] == "18775"
    assert env["ROI_LOCAL_INFERENCE_PORT"] == "18091"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["HERMES_HOME"] == str(tmp_path / "hermes")


def test_install_interpreter_path_cannot_be_a_global_symlink_target(tmp_path, monkeypatch):
    record = install_record(tmp_path)
    pinned = Path(record["hermes_python"])
    global_python = tmp_path / "managed-python/python"
    original = Path.resolve

    def resolved(path, *args, **kwargs):
        if path == pinned:
            return original(global_python)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolved)
    assert manage.read_install(tmp_path)["hermes_python"] == str(pinned)
    install_record(tmp_path, hermes_python=str(global_python))
    with pytest.raises(manage.LifecycleError, match="isolated .venv"):
        manage.read_install(tmp_path)


def test_unreadable_state_cannot_be_mistaken_for_no_saved_processes(tmp_path, monkeypatch):
    path = tmp_path / "data/local-services.json"
    original_exists = Path.exists
    original_read = Path.read_text

    def hidden(target):
        return False if target == path else original_exists(target)

    def denied(target, **kwargs):
        if target == path:
            raise PermissionError("state denied")
        return original_read(target, **kwargs)

    monkeypatch.setattr(Path, "exists", hidden)
    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(manage.LifecycleError, match="state"):
        manage.read_state(tmp_path)


def test_null_saved_process_record_is_not_discarded_on_start(tmp_path, monkeypatch):
    from contextlib import nullcontext

    manage.write_state(tmp_path, {"version": 1, "root": str(tmp_path), "services": {"api": None}})
    monkeypatch.setattr(manage, "lifecycle_lock", lambda: nullcontext())
    monkeypatch.setattr(manage, "listener_pid", lambda port: None)
    with pytest.raises(manage.LifecycleError, match="unconfirmed"):
        manage.start(tmp_path)
    assert manage.read_state(tmp_path)["services"] == {"api": None}


def test_null_saved_process_record_is_preserved_by_stop(tmp_path, monkeypatch):
    from contextlib import nullcontext

    manage.write_state(tmp_path, {"version": 1, "root": str(tmp_path), "services": {"api": None}})
    monkeypatch.setattr(manage, "lifecycle_lock", lambda: nullcontext())
    result = manage.stop(tmp_path)
    assert result["stopped"] is False
    assert "api" in result["incomplete"]
    assert manage.read_state(tmp_path)["services"] == {"api": None}


def test_lifecycle_uses_install_specific_lock(tmp_path, monkeypatch):
    from contextlib import contextmanager

    install_record(tmp_path)
    locks = []

    @contextmanager
    def lock(port=8776):
        locks.append(port)
        yield

    monkeypatch.setattr(manage, "lifecycle_lock", lock)
    monkeypatch.setattr(manage, "listener_pid", lambda port: None)
    with pytest.raises(manage.LifecycleError, match="Missing local prerequisites"):
        manage.start(tmp_path)
    assert manage.stop(tmp_path)["stopped"] is True
    assert locks == [18776, 18776]
