"""Process spawning from the hidden Windows service, with no real Git mutations."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from roi_swarm.improvements import ImprovementEngine


@pytest.mark.skipif(os.name != "nt", reason="Windows process creation flags")
async def test_trusted_children_never_open_console(tmp_path, monkeypatch):
    engine = ImprovementEngine(tmp_path, tmp_path / "trees", lambda diff: None)
    observed = []

    async def create(*args, **kwargs):
        observed.append((args, kwargs))
        return SimpleNamespace(
            returncode=0,
            communicate=lambda: asyncio.sleep(0, result=(b"ok", b"")),
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    assert await engine._process(["git", "status"], tmp_path) == "ok"
    assert observed[0][1]["creationflags"] & subprocess.CREATE_NO_WINDOW
    assert observed[0][1]["stdin"] == asyncio.subprocess.DEVNULL
    assert observed[0][1]["stdout"] == asyncio.subprocess.PIPE
    assert observed[0][1]["stderr"] == asyncio.subprocess.PIPE


@pytest.mark.skipif(os.name != "nt", reason="Windows cancellation helper")
async def test_cancellation_killer_is_hidden_and_reaps_child(tmp_path, monkeypatch):
    engine = ImprovementEngine(tmp_path, tmp_path / "trees", lambda diff: None)
    started = asyncio.Event()
    calls = []
    child = SimpleNamespace(pid=1234, returncode=None)

    async def communicate():
        started.set()
        await asyncio.Event().wait()

    async def waited():
        child.returncode = -1
        return -1

    child.communicate = communicate
    child.wait = waited
    child.kill = lambda: setattr(child, "returncode", -1)

    async def create(*args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "taskkill":
            child.returncode = -1
            return SimpleNamespace(wait=lambda: asyncio.sleep(0, result=0))
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    task = asyncio.create_task(engine._process(["git", "status"], tmp_path))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert len(calls) == 2
    assert calls[1][0] == ("taskkill", "/PID", "1234", "/T", "/F")
    assert calls[1][1]["creationflags"] & subprocess.CREATE_NO_WINDOW
    assert child.returncode is not None


async def test_command_failure_names_stage_without_exposing_output(tmp_path, monkeypatch):
    engine = ImprovementEngine(tmp_path, tmp_path / "trees", lambda diff: None)

    async def create(*args, **kwargs):
        return SimpleNamespace(
            returncode=1,
            communicate=lambda: asyncio.sleep(0, result=(b"private stdout", b"private stderr")),
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    with pytest.raises(Exception, match=r"pytest failed \(exit 1\)") as failure:
        await engine._process([sys.executable, "-m", "pytest", "-q"], tmp_path)
    assert "private" not in str(failure.value)


async def test_default_checks_use_console_python_when_service_is_pythonw(tmp_path, monkeypatch):
    root = tmp_path / "venv" / "Scripts"
    root.mkdir(parents=True)
    console = root / "python.exe"
    console.write_bytes(b"fixture")
    monkeypatch.setattr(sys, "executable", str(root / "pythonw.exe"))
    observed = []
    engine = ImprovementEngine(tmp_path, tmp_path / "trees", lambda diff: None)

    async def process(args, path):
        observed.append(args)
        return ""

    monkeypatch.setattr(engine, "_process", process)
    await engine._default_checks(tmp_path)
    assert all(Path(args[0]) == console for args in observed)
    assert observed[0][1:3] == ["-m", "pytest"]
    assert observed[1][1:3] == ["-m", "ruff"]
