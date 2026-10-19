import asyncio
import json

from fastapi.testclient import TestClient

import roi_swarm.api as api_module
from roi_swarm.api import create_app
from roi_swarm.telemetry import RunStore

API_HEADERS = {"Authorization": "Bearer local-roi-swarm"}

REQUEST = {
    "model": "roi-swarm",
    "stream": True,
    "messages": [
        {"role": "system", "content": "You are Hermes"},
        {"role": "user", "content": "сделай 8 модулей"},
    ],
    "tools": [{"type": "function", "function": {"name": "terminal", "parameters": {}}}],
}


class WaveRunner:
    def __init__(self, store: RunStore) -> None:
        self.store = store
        self.tool_rounds = 0

    async def run(self, request, run_id=None, messages=None):  # pragma: no cover - unused
        raise AssertionError("plain swarm must not run")

    async def respond_with_tools(self, messages, tools, run_id):
        self.tool_rounds += 1
        raise AssertionError("waves must not hand tool calls back to Hermes")

    async def should_use_waves(self, messages, tools):
        return True

    async def run_waves(self, messages, run_id, *, on_delta=None):
        self.store.create_run("x", run_id)
        self.store.record_event(run_id, "wave", "running", {"wave": 1, "task_count": 2})
        self.store.record_event(run_id, "worker:t1", "tool", {"tool": "write_file", "args": "a"})
        await asyncio.sleep(0.5)
        self.store.record_event(run_id, "verifier", "failed", {"wave": 1, "accepted": 1})
        self.store.record_event(run_id, "wave", "running", {"wave": 2, "task_count": 1})
        await asyncio.sleep(0.3)
        for piece in ("Итог: ", "всё готово"):
            on_delta(piece)
        return "Итог: всё готово"

    async def close(self) -> None:
        return None


def test_whole_task_runs_in_one_stream_with_waves(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "PROGRESS_POLL_SECONDS", 0.1)
    runner = WaveRunner(RunStore(tmp_path))
    client = TestClient(create_app(runner=runner))

    with client.stream("POST", "/v1/chat/completions", headers=API_HEADERS, json=REQUEST) as r:
        body = "".join(r.iter_text())

    chunks = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
    deltas = [c["choices"][0]["delta"] for c in chunks]
    reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
    content = "".join(d.get("content") or "" for d in deltas)
    assert "Волна 1: 2 задач" in reasoning and "Волна 2" in reasoning
    assert "t1 → write_file" in reasoning
    assert content == "Итог: всё готово"
    assert not any(d.get("tool_calls") for d in deltas)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert body.rstrip().endswith("data: [DONE]")
    assert runner.tool_rounds == 0


def test_wave_failure_is_reported_and_stream_closes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "PROGRESS_POLL_SECONDS", 0.1)

    class Broken(WaveRunner):
        async def run_waves(self, messages, run_id, *, on_delta=None):
            raise RuntimeError("boom")

    client = TestClient(create_app(runner=Broken(RunStore(tmp_path))))
    with client.stream("POST", "/v1/chat/completions", headers=API_HEADERS, json=REQUEST) as r:
        body = "".join(r.iter_text())

    assert "ROI Swarm error: RuntimeError" in body
    assert body.rstrip().endswith("data: [DONE]")


def test_failure_after_streamed_summary_is_appended_not_mixed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "PROGRESS_POLL_SECONDS", 0.1)

    class LateFailure(WaveRunner):
        async def run_waves(self, messages, run_id, *, on_delta=None):
            on_delta("Итог: часть")
            await asyncio.sleep(0.05)
            raise RuntimeError("late")

    client = TestClient(create_app(runner=LateFailure(RunStore(tmp_path))))
    with client.stream("POST", "/v1/chat/completions", headers=API_HEADERS, json=REQUEST) as r:
        body = "".join(r.iter_text())

    chunks = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
    content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert content.startswith("Итог: часть\n\nROI Swarm error: RuntimeError")


def test_closing_the_stream_early_cancels_the_running_waves() -> None:
    cancelled = asyncio.Event()

    class Slow:
        store = None

        async def run_waves(self, messages, run_id, *, on_delta=None):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

    async def scenario() -> None:
        stream = api_module._stream_wave_turn(Slow(), "id", [])
        await stream.__anext__()  # role chunk, waves task now running
        await stream.__anext__()  # first progress tick
        await stream.aclose()  # client disconnected: GeneratorExit at a yield
        await asyncio.wait_for(cancelled.wait(), 5)

    original = api_module.PROGRESS_POLL_SECONDS
    api_module.PROGRESS_POLL_SECONDS = 0.05
    try:
        asyncio.run(scenario())
    finally:
        api_module.PROGRESS_POLL_SECONDS = original


def test_follow_up_tool_rounds_stay_on_the_coordinator_path(tmp_path, monkeypatch) -> None:
    from roi_swarm.runner import SwarmRunner

    runner = SwarmRunner.__new__(SwarmRunner)
    runner.settings = type("S", (), {})()
    runner.settings.hermes_workers = type("H", (), {"enabled": True})()

    async def verdict(request):
        return True

    monkeypatch.setattr(runner, "_assess_for_swarm", verdict, raising=False)
    monkeypatch.setattr(runner, "_hermes_workers_available", lambda: True, raising=False)
    tools = REQUEST["tools"]
    fresh = REQUEST["messages"]
    tool_round = [
        *fresh,
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c"}]},
        {"role": "tool", "tool_call_id": "c", "content": "ok"},
    ]
    title_job = [{"type": "function", "function": {"name": "web_search"}}]

    assert asyncio.run(runner.should_use_waves(fresh, tools)) is True
    assert asyncio.run(runner.should_use_waves(tool_round, tools)) is False
    assert asyncio.run(runner.should_use_waves(fresh, title_job)) is False
    runner.settings.hermes_workers.enabled = False
    assert asyncio.run(runner.should_use_waves(fresh, tools)) is False
