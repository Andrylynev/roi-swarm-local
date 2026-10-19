from types import SimpleNamespace

import pytest

from roi_swarm.config import SleepConfig

pytest_plugins = ("test_project_isolation",)


def test_sleep_defaults_are_enabled_and_unlimited_without_auto_improve():
    assert SleepConfig().enabled is True
    assert SleepConfig().auto_improve is False
    assert SleepConfig().daily_call_limit is None
    assert SleepConfig().max_steps_per_cycle is None


async def test_default_enabled_runner_preserves_saved_opt_out_on_restart(scoped_runner):
    from roi_swarm.runner import SwarmRunner

    await scoped_runner.start()
    assert scoped_runner.sleep_status()["enabled"] is True
    assert scoped_runner.sleep_status()["daily_calls"] == 0
    await scoped_runner.set_sleep_enabled(False)
    await scoped_runner.close()
    restored = SwarmRunner(scoped_runner.config_path)
    try:
        await restored.start()
        assert restored.settings.sleep.enabled is True
        assert restored.sleep_status()["enabled"] is False
        assert restored.sleep_status()["daily_calls"] == 0
    finally:
        await restored.close()


@pytest.mark.parametrize("method", ["run", "respond_with_tools"])
async def test_foreground_preempts_and_only_final_turn_enqueues(scoped_runner, monkeypatch, method):
    r = scoped_runner
    events = []

    class Sleep:
        async def foreground_started(self):
            events.append("pause")

        def foreground_finished(self, completed_run=None):
            events.append(completed_run)

        async def close(self):
            pass

    r.sleep = Sleep()

    async def work(*args):
        events.append("work")
        if method == "run":
            return {"final_answer": "ok", "worker_results": []}
        return SimpleNamespace(
            content="", finish_reason="tool_calls", tool_calls=[{"id": "pending"}]
        )

    monkeypatch.setattr(r, "_run" if method == "run" else "_respond_with_tools", work)
    if method == "run":
        await r.run("task", run_id="sleep-run")
        assert events[2]["run_id"] == "sleep-run"
    else:
        await r.respond_with_tools([], [], "sleep-tools")
        # One extra pause retains the hold while external tools execute.
        assert events[2] == "pause"
        assert events[3] is None
    assert events[:2] == ["pause", "work"]


async def test_reflection_has_no_raw_request_or_cross_project_sources(scoped_runner):
    r = scoped_runner
    captured = []

    async def generate(**kwargs):
        captured.append(kwargs)
        return kwargs["schema"](observations=["reviewed"], done=True)

    r.gateway.generate_structured = generate
    result = await r._reflect_sleep(
        {
            "scope": "alpha",
            "source_run_id": "x",
            "metrics": {},
            "observations": [],
            "hypotheses": [],
            "next_step": "check",
        }
    )
    assert result["done"] is True
    assert "workflow" in captured[0]["system"].lower()
    assert captured[0]["model"] == r.settings.model_for("critic").model
