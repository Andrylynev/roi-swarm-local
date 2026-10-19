"""The pool must reap descendants even after the worker host exits normally."""

import asyncio
import ctypes
import json
import os
import sys

import pytest

import roi_swarm.waves as waves
from roi_swarm.schemas import PlannedTask
from roi_swarm.waves import HermesWorkerPool, InstallBroker


@pytest.mark.skipif(os.name != "nt", reason="Real Windows process-handle verification")
@pytest.mark.parametrize("terminal_event", ["final", "error"])
def test_pool_reaps_orphan_before_the_next_install(tmp_path, monkeypatch, terminal_event):
    script = tmp_path / "host.py"
    pid_file = tmp_path / "child.json"
    script.write_text(
        """
import json,subprocess,sys
from pathlib import Path
task=json.loads(sys.stdin.readline())
child=subprocess.Popen(
    [sys.executable,"-c","import time; time.sleep(600)"],
    stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
)
Path(task["context"]).write_text(json.dumps(child.pid))
print(json.dumps({"type":"gate","id":1}),flush=True)
sys.stdin.readline()
print(json.dumps({"type":task["goal"],"text":"done","error":"cleanup failed"}),flush=True)
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(waves, "HOST_SCRIPT", script)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.WaitForSingleObject.restype = ctypes.c_ulong
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = None

    def dead():
        return kernel.WaitForSingleObject(handle, 0) == 0

    async def scenario():
        nonlocal handle

        class Broker(InstallBroker):
            async def enter(self, worker):
                nonlocal handle
                pid = json.loads(pid_file.read_text())
                handle = kernel.OpenProcess(0x00100000, False, pid)
                assert handle, ctypes.get_last_error()
                await super().enter(worker)

            async def leave(self, worker):
                # Called only in pool finally: host intentionally never emits gate_done.
                assert dead(), "environment lease released while orphan was alive"
                await super().leave(worker)

        async def install(command, cwd):
            assert dead(), "next installation ran with an orphan still alive"
            return {"output": "ok", "exit_code": 0, "error": None}

        broker = Broker(run=install)
        pool = HermesWorkerPool(
            python=sys.executable,
            hermes_root=str(tmp_path),
            base_url="unused",
            api_key="",
            api_mode="chat_completions",
            model="unused",
            disabled_toolsets=[],
            max_iterations=1,
        )
        outcome = await pool.run(
            PlannedTask(id="orphan", objective="probe", context=str(pid_file)),
            goal=terminal_event,
            cwd=str(tmp_path),
            broker=broker,
            on_event=lambda e: None,
        )
        assert outcome.ok is (terminal_event == "final")
        assert dead()
        assert (await broker.install("next", "pip install x", str(tmp_path)))["exit_code"] == 0

    try:
        asyncio.run(scenario())
    finally:
        if handle:
            if not dead():
                import subprocess

                subprocess.run(
                    ["taskkill", "/PID", str(json.loads(pid_file.read_text())), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
            kernel.CloseHandle(handle)
