"""Local worker admission limits do not restrict logical tasks or remote defaults."""

import asyncio
import subprocess
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

import roi_swarm.runner as runner_module
from roi_swarm.config import AppSettings, HermesWorkersConfig, ModelRoleConfig, ProviderConfig
from roi_swarm.schemas import PlannedTask
from roi_swarm.telemetry import RunStore
from roi_swarm.waves import WaveOrchestrator, WavePlan, WaveReview, WorkerOutcome
from roi_swarm.worktrees import WaveWorkspace


def test_hermes_worker_concurrency_is_optional_and_accepts_positive_limits() -> None:
    assert HermesWorkersConfig().maximum_concurrency is None
    assert HermesWorkersConfig(maximum_concurrency=None).maximum_concurrency is None
    assert HermesWorkersConfig(maximum_concurrency=2).maximum_concurrency == 2


@pytest.mark.parametrize("limit", [0, -1])
def test_hermes_worker_concurrency_rejects_nonpositive_limits(limit: int) -> None:
    with pytest.raises(ValidationError):
        HermesWorkersConfig(maximum_concurrency=limit)


async def _unused(*args, **kwargs):
    raise AssertionError("A wave admission test must not call a model")


class WorkerProbe:
    """Keep admitted workers blocked so queued/running states can be inspected."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.release = asyncio.Event()
        self.events: list[tuple[str, str, dict]] = []
        self.folders: dict[str, str] = {}

    def record(self, node: str, status: str, data: dict) -> None:
        self.events.append((node, status, data))

    async def run(self, task, *, goal, cwd, on_event):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.folders[task.id] = cwd
        self.started.put_nowait(task.id)
        try:
            await self.release.wait()
            return WorkerOutcome(task.id, task.objective, True, "done")
        finally:
            self.active -= 1


def _tasks(prefix: str = "t") -> list[PlannedTask]:
    return [PlannedTask(id=f"{prefix}{i}", objective=f"write {prefix}{i}.txt") for i in range(4)]


async def _stop(task: asyncio.Task) -> None:
    if not task.done():
        task.cancel()
    with suppress(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("isolated", [False, True])
async def test_four_tasks_admit_only_two_active_workers(tmp_path, isolated: bool) -> None:
    root = tmp_path / "project"
    root.mkdir()
    workspace = None
    if isolated:
        subprocess.run(["git", "init", "-q"], cwd=root, check=True, capture_output=True)
        (root / "seed.txt").write_text("seed", encoding="utf-8")
        workspace = WaveWorkspace.open(str(root), root=tmp_path / "worktrees")
        assert workspace is not None

    probe = WorkerProbe()
    orchestrator = WaveOrchestrator(
        structured=_unused,
        finalize=_unused,
        run_worker=probe.run,
        maximum_waves=4,
        maximum_concurrency=2,
        on_event=probe.record,
        open_workspace=(lambda _: workspace) if isolated else None,
    )
    pending = asyncio.create_task(orchestrator._run_wave(_tasks(), 1, "write files", str(root)))
    try:
        started = [await asyncio.wait_for(probe.started.get(), 10) for _ in range(2)]
        # Give every task a scheduling turn while the admitted workers remain blocked.
        for _ in range(8):
            await asyncio.sleep(0)
        assert probe.active == probe.peak == 2
        assert probe.started.empty()
        latest = {node: status for node, status, _ in probe.events if node.startswith("worker:")}
        assert {node for node, status in latest.items() if status == "running"} == {
            f"worker:{task_id}" for task_id in started
        }
        assert sum(status == "queued" for status in latest.values()) == 2
        if isolated:
            assert len(set(probe.folders.values())) == 2
            assert all(
                Path(folder).resolve() != root.resolve() for folder in probe.folders.values()
            )
        else:
            assert set(probe.folders.values()) == {str(root)}
        probe.release.set()
        outcomes = await asyncio.wait_for(pending, 15)
        assert [outcome.task_id for outcome in outcomes] == [task.id for task in _tasks()]
        assert all(outcome.ok for outcome in outcomes)
        assert probe.peak == 2 and probe.active == 0
        for task in _tasks():
            statuses = [status for node, status, _ in probe.events if node == f"worker:{task.id}"]
            assert statuses == ["queued", "running", "completed"]
    finally:
        await _stop(pending)
        if workspace is not None:
            workspace.close()


@pytest.mark.parametrize("limit", [None, 2])
async def test_worker_gate_is_shared_across_waves_and_reusable(tmp_path, limit) -> None:
    probe = WorkerProbe()
    options = {} if limit is None else {"maximum_concurrency": limit}
    orchestrator = WaveOrchestrator(
        structured=_unused,
        finalize=_unused,
        run_worker=probe.run,
        maximum_waves=4,
        on_event=probe.record,
        **options,
    )
    # Simultaneous wave calls must share the orchestrator's one admission gate.
    pending = asyncio.gather(
        orchestrator._run_wave(_tasks("a"), 1, "goal", str(tmp_path)),
        orchestrator._run_wave(_tasks("b"), 2, "goal", str(tmp_path)),
    )
    try:
        for _ in range(8 if limit is None else limit):
            await asyncio.wait_for(probe.started.get(), 5)
        for _ in range(8):
            await asyncio.sleep(0)
        assert probe.active == probe.peak == (8 if limit is None else limit)
        probe.release.set()
        outcomes = await asyncio.wait_for(pending, 5)
        assert all(outcome.ok for wave in outcomes for outcome in wave)
        assert sum(len(wave) for wave in outcomes) == 8
    finally:
        await _stop(pending)

    while not probe.started.empty():
        probe.started.get_nowait()
    probe.release.clear()
    probe.peak = 0
    pending = asyncio.create_task(orchestrator._run_wave(_tasks("c"), 3, "goal", str(tmp_path)))
    try:
        for _ in range(4 if limit is None else limit):
            await asyncio.wait_for(probe.started.get(), 5)
        for _ in range(8):
            await asyncio.sleep(0)
        assert probe.active == probe.peak == (4 if limit is None else limit)
        probe.release.set()
        outcomes = await asyncio.wait_for(pending, 5)
        assert [outcome.task_id for outcome in outcomes] == [task.id for task in _tasks("c")]
        assert all(outcome.ok for outcome in outcomes)
        assert probe.active == 0
    finally:
        await _stop(pending)


async def test_cancelled_workers_do_not_leak_admission_slots(tmp_path) -> None:
    probe = WorkerProbe()
    orchestrator = WaveOrchestrator(
        structured=_unused,
        finalize=_unused,
        run_worker=probe.run,
        maximum_waves=4,
        maximum_concurrency=1,
        on_event=probe.record,
    )
    first = asyncio.create_task(orchestrator._run_one(_tasks()[0], 1, "goal", str(tmp_path)))
    queued = None
    try:
        assert await asyncio.wait_for(probe.started.get(), 5) == "t0"
        queued = asyncio.create_task(orchestrator._run_one(_tasks()[1], 1, "goal", str(tmp_path)))
        for _ in range(8):
            await asyncio.sleep(0)
        await _stop(queued)
        assert set(probe.folders) == {"t0"}
        assert probe.active == 1
    finally:
        await _stop(first)
        if queued is not None:
            await _stop(queued)
    assert probe.active == 0
    probe.release.set()
    outcome = await asyncio.wait_for(
        orchestrator._run_one(_tasks()[2], 2, "goal", str(tmp_path)), 5
    )
    assert outcome.ok and outcome.task_id == "t2"
    assert probe.peak == 1


@pytest.mark.parametrize("ending", ["cancel", "error"])
async def test_shared_gate_releases_to_another_request_after_failure(tmp_path, ending):
    slots = asyncio.Semaphore(1)
    first_started, release = asyncio.Event(), asyncio.Event()
    second_started = asyncio.Event()
    active = peak = 0

    async def worker(task, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            if task.id == "t0":
                first_started.set()
                await release.wait()
                raise RuntimeError("failed worker")
            second_started.set()
            return WorkerOutcome(task.id, task.objective, True, "done")
        finally:
            active -= 1

    def request():
        return WaveOrchestrator(
            structured=_unused,
            finalize=_unused,
            run_worker=worker,
            maximum_waves=4,
            maximum_concurrency=1,
            worker_slots=slots,
            on_event=lambda *_: None,
        )

    first = asyncio.create_task(request()._run_one(_tasks()[0], 1, "goal", str(tmp_path)))
    await asyncio.wait_for(first_started.wait(), 2)
    second = asyncio.create_task(request()._run_one(_tasks()[1], 1, "goal", str(tmp_path)))
    try:
        await asyncio.sleep(0)
        assert not second_started.is_set()
        if ending == "cancel":
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        else:
            release.set()
            assert not (await first).ok
        assert (await asyncio.wait_for(second, 2)).ok
        assert peak == 1 and active == 0
    finally:
        release.set()
        await _stop(first)
        await _stop(second)


async def test_worker_failure_releases_admission_for_remaining_tasks(tmp_path) -> None:
    started: list[str] = []

    async def worker(task, **kwargs):
        started.append(task.id)
        if task.id == "t0":
            raise RuntimeError("offline worker failure")
        return WorkerOutcome(task.id, task.objective, True, "done")

    orchestrator = WaveOrchestrator(
        structured=_unused,
        finalize=_unused,
        run_worker=worker,
        maximum_waves=4,
        maximum_concurrency=1,
        on_event=lambda *_: None,
    )
    outcomes = await asyncio.wait_for(orchestrator._run_wave(_tasks(), 1, "goal", str(tmp_path)), 5)
    assert started == [task.id for task in _tasks()]
    assert not outcomes[0].ok and "offline worker failure" in outcomes[0].report
    assert all(outcome.ok for outcome in outcomes[1:])


@pytest.mark.parametrize("limit", [None, 2])
@pytest.mark.parametrize("runs", [1, 2])
async def test_runner_passes_worker_limit_to_wave_orchestrator(
    tmp_path, monkeypatch, limit, runs
) -> None:
    probe = WorkerProbe()
    runner = runner_module.SwarmRunner.__new__(runner_module.SwarmRunner)
    runner.settings = AppSettings(
        provider=ProviderConfig(base_url="http://127.0.0.1:1/v1", api_key="offline"),
        models={
            role: ModelRoleConfig(model="offline")
            for role in ("router", "planner", "worker", "critic", "synthesizer")
        },
        hermes_workers=HermesWorkersConfig(
            maximum_concurrency=limit, isolate_worktrees=False, default_cwd=str(tmp_path)
        ),
    )
    runner.store = RunStore(tmp_path / "runs")
    runner._active_operations = 0
    runner._install_broker = None
    runner._worker_slots = asyncio.Semaphore(limit) if limit is not None else None
    monkeypatch.setattr(runner, "_pause_sleep", AsyncMock())
    monkeypatch.setattr(runner, "_finish_sleep", lambda _: None)
    monkeypatch.setattr(runner, "_sleep_completion", lambda *args: None)
    monkeypatch.setattr(runner, "_stream_text", AsyncMock(return_value="done"))
    pool_options: list[dict] = []

    def pool(**kwargs):
        pool_options.append(kwargs)
        return SimpleNamespace(run=probe.run)

    async def structured(**kwargs):
        if kwargs["schema"] is WavePlan:
            return WavePlan(summary="four files", tasks=_tasks())
        assert kwargs["schema"] is WaveReview
        return WaveReview(done=True, accepted=[task.id for task in _tasks()])

    runner.gateway = SimpleNamespace(generate_structured=structured)
    monkeypatch.setattr(runner_module, "HermesWorkerPool", pool)
    options: list[dict] = []

    def orchestrator(**kwargs):
        options.append(kwargs)
        return WaveOrchestrator(**kwargs)

    monkeypatch.setattr(runner_module, "WaveOrchestrator", orchestrator)
    pending = asyncio.gather(
        *(
            runner.run_waves([{"role": "user", "content": "write four files"}], f"local-{i}")
            for i in range(runs)
        )
    )
    try:
        for _ in range(4 * runs if limit is None else limit):
            await asyncio.wait_for(probe.started.get(), 5)
        for _ in range(20):
            await asyncio.sleep(0)
        assert options[0]["maximum_concurrency"] is limit
        assert probe.active == probe.peak == (4 * runs if limit is None else limit)
        assert all(item["worker_slots"] is runner._worker_slots for item in options)
        probe.release.set()
        assert await asyncio.wait_for(pending, 5) == ["done"] * runs
        assert set(probe.folders) == {task.id for task in _tasks()}
        assert all(runner.store.get_run(f"local-{i}")["status"] == "completed" for i in range(runs))
        assert runner._active_operations == 0
        assert pool_options[0]["model"] == "offline"
        assert runner.settings.provider.timeout_seconds is None
    finally:
        await _stop(pending)
