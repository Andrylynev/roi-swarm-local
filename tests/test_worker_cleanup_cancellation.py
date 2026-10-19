"""Cancellation must not interrupt any phase of worker finalization."""

import asyncio

import pytest

import roi_swarm.waves as waves
from roi_swarm.schemas import PlannedTask
from roi_swarm.waves import HermesWorkerPool, InstallBroker


@pytest.mark.parametrize("phase", ["pipes", "lease", "gate_done", "install_settlement"])
def test_repeated_cancel_settles_entire_worker_cleanup(phase, tmp_path, monkeypatch):
    async def scenario():
        reached, release, install_started = (asyncio.Event() for _ in range(3))
        install_settled = False
        stdin_closed = False
        cleaned = False

        async def boundary(name):
            if name == phase:
                reached.set()
                await release.wait()

        class Stdin:
            def write(self, line):
                pass

            async def drain(self):
                pass

            def close(self):
                nonlocal stdin_closed
                stdin_closed = True

        async def stdout():
            yield b'{"type":"gate","id":1}\n'
            yield b'{"type":"install","id":2,"command":"worker"}\n'
            await install_started.wait()
            if phase == "gate_done":
                yield b'{"type":"gate_done"}\n'
            yield b'{"type":"final","text":"done"}\n'

        class Process:
            pid = 12345
            returncode = 0
            stdin = Stdin()

            def __init__(self):
                self.stdout = stdout()

            async def communicate(self):
                await boundary("pipes")
                return b"", b""

        process = Process()

        async def spawn(*args, **kwargs):
            return process

        class Scope:
            launch_kwargs = {}

            def attach(self, pid):
                pass

            def close(self):
                nonlocal cleaned
                cleaned = True

        class Broker(InstallBroker):
            async def leave(self, worker):
                await boundary("gate_done" if phase == "gate_done" else "lease")
                await super().leave(worker)

            async def install(self, worker, command, cwd, **kwargs):
                nonlocal install_settled
                if command != "worker":
                    return await super().install(worker, command, cwd, **kwargs)
                install_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await boundary("install_settlement")
                    install_settled = True
                    raise

        async def run_install(command, cwd):
            return {"exit_code": 0, "output": "ok", "error": None}

        monkeypatch.setattr(waves.asyncio, "create_subprocess_exec", spawn)
        monkeypatch.setattr(waves, "WorkerProcessScope", Scope)
        broker = Broker(run=run_install)
        pool = HermesWorkerPool(
            python="python",
            hermes_root=str(tmp_path),
            base_url="u",
            api_key="k",
            api_mode="chat_completions",
            model="m",
            disabled_toolsets=[],
            max_iterations=1,
        )
        running = asyncio.create_task(
            pool.run(
                PlannedTask(id="cancel", objective="x"),
                goal="",
                cwd=str(tmp_path),
                broker=broker,
                on_event=lambda event: None,
            )
        )
        await asyncio.wait_for(reached.wait(), 2)
        try:
            running.cancel()
            await asyncio.sleep(0)
            running.cancel()
            await asyncio.sleep(0)
            assert not running.done(), "cancellation escaped before cleanup settled"
        finally:
            release.set()
            result = (await asyncio.gather(running, return_exceptions=True))[0]
        assert isinstance(result, asyncio.CancelledError)
        assert cleaned and stdin_closed and install_settled
        assert broker._holders == set()
        assert broker._environment_error == []
        outcome = await asyncio.wait_for(broker.install("next", "next", str(tmp_path)), 2)
        assert outcome["exit_code"] == 0

    asyncio.run(scenario())
