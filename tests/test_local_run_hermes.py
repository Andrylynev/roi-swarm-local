"""The chat launcher may only execute the install-owned pinned Hermes."""

import os
from pathlib import Path

import pytest
from test_local_manage_portable import install_record

from local import run_hermes


def test_launch_uses_isolated_hermes_and_install_api_port(tmp_path, monkeypatch):
    record = install_record(tmp_path)
    python = Path(record["hermes_python"])
    python.parent.mkdir(parents=True)
    python.touch()
    calls = []
    monkeypatch.setenv("LOCALAPPDATA", "C:/global/hermes")
    monkeypatch.setattr(
        run_hermes.subprocess, "call", lambda cmd, **kw: calls.append((cmd, kw)) or 0
    )
    assert run_hermes.main(["--query", "offline probe"], root=tmp_path) == 0
    command, options = calls[0]
    assert command[0] == str(python)
    assert command[-2:] == ["--query", "offline probe"]
    assert options["cwd"] == tmp_path
    env = options["env"]
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:18775/v1"
    assert env["VIRTUAL_ENV"] == str(python.parent.parent)
    assert env["PATH"].startswith(str(python.parent) + os.pathsep)
    assert env["HERMES_HOME"] == str(tmp_path / "hermes")
    assert env["PYTHONPATH"] == str(tmp_path / "runtime/hermes-agent")


def test_missing_isolated_hermes_never_falls_back_to_desktop(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "global"))
    global_python = tmp_path / "global/hermes/hermes-agent/venv/Scripts/python.exe"
    global_python.parent.mkdir(parents=True)
    global_python.touch()
    with pytest.raises(SystemExit, match="Hermes interpreter is missing"):
        run_hermes.main([], root=tmp_path)
