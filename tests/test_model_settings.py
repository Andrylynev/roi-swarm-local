import asyncio
from types import SimpleNamespace

import pytest

from roi_swarm.config import AppSettings
from roi_swarm.runner import SwarmRunner

ROLES = ("router", "planner", "worker", "critic", "synthesizer")


@pytest.fixture
def runner(tmp_path, monkeypatch):
    import roi_swarm.runner as module

    settings = AppSettings(
        provider={"base_url": "http://example.test/v1", "api_key": "test-only"},
        models={role: {"model": "old-model"} for role in ROLES},
        rag={"enabled": False},
    )
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "load_settings", lambda _: settings.model_copy(deep=True))
    instance = SwarmRunner(tmp_path / "config" / "settings.yaml")
    yield instance
    asyncio.run(instance.close())


def test_model_selection_is_persisted_and_loaded_without_provider_secrets(runner):
    roles = {role: f"new-{role}" for role in ROLES}
    original_tokens = runner.settings.model_for("planner").max_output_tokens
    runner._swarm_verdicts["old"] = True

    result = runner.update_model_settings(roles)

    assert result["roles"] == roles
    assert runner.settings.model_for("planner").max_output_tokens == original_tokens
    assert runner._swarm_verdicts == {}
    raw = runner.model_settings_path.read_text(encoding="utf-8")
    assert "test-only" not in raw and "base_url" not in raw
    restored = SwarmRunner(runner.config_path)
    try:
        assert restored.model_settings()["roles"] == roles
    finally:
        asyncio.run(restored.close())


def test_model_selection_rejects_busy_runner(runner):
    runner._active_operations = 1
    with pytest.raises(RuntimeError, match="active"):
        runner.update_model_settings({role: "new" for role in ROLES})
    assert runner.settings.model_for("worker").model == "old-model"
    assert not runner.model_settings_path.exists()


def test_failed_persistence_keeps_runtime_unchanged(runner, monkeypatch):
    from pathlib import Path

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError):
        runner.update_model_settings({role: "new" for role in ROLES})
    assert runner.settings.model_for("worker").model == "old-model"


@pytest.mark.parametrize("roles", [{"worker": "new"}, {role: " " for role in ROLES}])
def test_model_selection_rejects_invalid_roles(runner, roles):
    with pytest.raises(ValueError):
        runner.update_model_settings(roles)


@pytest.mark.parametrize("method", ["run", "respond_with_tools"])
def test_active_request_blocks_switch_until_cancellation(runner, method, monkeypatch):
    async def scenario():
        entered = asyncio.Event()
        blocked = asyncio.Event()

        async def waiting(*args):
            entered.set()
            await blocked.wait()

        internal = "_run" if method == "run" else "_respond_with_tools"
        monkeypatch.setattr(runner, internal, waiting)
        args = ("request",) if method == "run" else ([], [], "r")
        task = asyncio.create_task(getattr(runner, method)(*args))
        await entered.wait()
        assert runner.model_settings()["active_operations"] == 1
        with pytest.raises(RuntimeError, match="active"):
            runner.update_model_settings({role: "new" for role in ROLES})
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert runner.model_settings()["active_operations"] == 0
        runner.update_model_settings({role: "new" for role in ROLES})

    asyncio.run(scenario())


def test_new_graph_uses_selected_models(runner):
    from roi_swarm.sizing import SwarmAssessment

    seen = []

    async def structured(*, model, **kwargs):
        seen.append(model)
        return SwarmAssessment(
            task_type="simple",
            complexity=0,
            parallelizability=0,
            uncertainty=0,
            stakes=0,
            swarm_required=False,
        )

    async def text(*, model, **kwargs):
        seen.append(model)
        return "selected model answered"

    runner.gateway.generate_structured = structured
    runner.gateway.generate_text = text
    runner.update_model_settings({role: f"selected-{role}" for role in ROLES})
    result = asyncio.run(runner.graph.ainvoke({"request": "hello"}))
    assert result["final_answer"] == "selected model answered"
    assert seen == ["selected-router", "selected-synthesizer"]


def test_router_cache_does_not_reuse_verdict_for_matching_request_suffix(runner):
    from roi_swarm.sizing import SwarmAssessment

    prompts = []

    async def assess(*, prompt, **kwargs):
        prompts.append(prompt)
        return SwarmAssessment(
            task_type="test",
            complexity=0.5,
            parallelizability=0.5,
            uncertainty=0,
            stakes=0,
            swarm_required=prompt.startswith("large"),
        )

    runner.gateway.generate_structured = assess

    async def scenario():
        suffix = "same material " * 400
        assert await runner._assess_for_swarm("small " + suffix) is False
        assert await runner._assess_for_swarm("large " + suffix) is True
        assert len(prompts) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("separator", ["\u00a0", "\u2003", "\u0085", "\u200b"])
def test_model_ids_reject_invisible_unicode_separators(runner, separator):
    roles = {role: f"some{separator}model" for role in ROLES}
    with pytest.raises(ValueError):
        runner.update_model_settings(roles)


def test_model_catalog_uses_provider_without_exposing_secrets(runner):
    async def listed():
        return SimpleNamespace(data=[SimpleNamespace(id="z"), SimpleNamespace(id="a")])

    runner.client.models.list = listed
    result = asyncio.run(runner.model_options())
    assert result["models"] == ["a", "old-model", "z"]
    assert result["source"] == "provider"


def test_model_catalog_failure_keeps_configured_choices(runner):
    async def failed():
        raise RuntimeError("sensitive provider response")

    runner.client.models.list = failed
    result = asyncio.run(runner.model_options())
    assert result["models"] == ["old-model"]
    assert result["source"] == "configured"
    assert "sensitive" not in str(result)
