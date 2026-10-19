import asyncio
from types import SimpleNamespace

import pytest

from roi_swarm.config import AppSettings, ContextConfig, ExperienceConfig, RagConfig
from roi_swarm.runner import SwarmRunner


@pytest.fixture
def scoped_runner(tmp_path, monkeypatch):
    import roi_swarm.runner as module

    settings = AppSettings(
        provider={"base_url": "http://example.test/v1", "api_key": "test-only"},
        models={
            r: {"model": "test"} for r in ("router", "planner", "worker", "critic", "synthesizer")
        },
        rag={
            "enabled": True,
            "project_collections": {"alpha": ["alpha-docs"], "beta": ["beta-docs"]},
        },
    )
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "load_settings", lambda _: settings)
    runner = SwarmRunner(tmp_path / "config" / "settings.yaml")
    yield runner
    asyncio.run(runner.close())


def test_scope_config_is_explicit():
    assert ContextConfig().planner_input_limit_chars is None
    assert ContextConfig().worker_context_tokens == 800_000
    assert ContextConfig().main_context_tokens == 1_000_000
    assert RagConfig().project_collections == {}
    assert ExperienceConfig().database_path == "data/experience/experience.sqlite"
    assert ExperienceConfig().enabled is True


def test_no_project_never_searches_default_collection(scoped_runner):
    r = scoped_runner
    r.rag.upsert(collection="alpha-docs", source="alpha.py", text="booking alpha secret context")
    r.rag.upsert(collection="beta-docs", source="beta.py", text="booking beta foreign context")
    captured = []

    async def stream(state, **kwargs):
        captured.append(state["prepared_context"].rag_results)
        yield {"direct": {"final_answer": "ok", "answered_directly": True}}

    r.graph = SimpleNamespace(astream=stream)
    result = asyncio.run(r.run("booking", run_id="unscoped"))
    assert captured[-1] == []
    assert result["project_id"] is None
    asyncio.run(r.run("booking", run_id="alpha", project_id="alpha"))
    assert {i["collection"] for i in captured[-1]} == {"alpha-docs"}
    asyncio.run(r.run("booking", run_id="unknown", project_id="unknown"))
    assert captured[-1] == []


def test_tool_delegation_preserves_gathered_materials_even_if_model_omits_them(
    scoped_runner, monkeypatch
):
    import roi_swarm.runner as module
    from roi_swarm.agent_loop import AgentTurn

    r = scoped_runner
    seen = []
    source = "EARLY\n" + "source " * 2000 + "\nMIDDLE\nLATE"

    class Coordinator:
        def __init__(self, client, *, run_swarm, **kwargs):
            self.delegate = run_swarm

        async def respond(self, messages, tools, *, run_id):
            await self.delegate("Audit four areas", run_id + "-s1")
            return AgentTurn("done")

    async def stream(state, **kwargs):
        seen.append(r.context_pipeline.pack_for_planner(state["prepared_context"]))
        yield {"direct": {"final_answer": "ok"}}

    monkeypatch.setattr(module, "HermesToolAgent", Coordinator)
    r.graph = SimpleNamespace(astream=stream)
    asyncio.run(
        r.respond_with_tools(
            [
                {"role": "user", "content": "Audit the project"},
                {"role": "tool", "content": source, "tool_call_id": "read-source"},
            ],
            [],
            "coordinator",
            project_id="alpha",
        )
    )
    assert source in seen[0]
    assert "read-source" in seen[0]


def test_empty_collection_filter_never_becomes_unrestricted_search(scoped_runner):
    scoped_runner.rag.upsert(collection="alpha-docs", source="a.py", text="booking")
    assert scoped_runner.rag.search("booking", collections=[]) == []


def test_preparation_failure_is_recorded_not_left_running(scoped_runner, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("Planner input exceeds budget; split task")

    monkeypatch.setattr(scoped_runner.context_pipeline, "prepare", broken)
    with pytest.raises(ValueError):
        asyncio.run(scoped_runner.run("huge request", run_id="prep-error"))
    assert scoped_runner.store.get_run("prep-error")["status"] == "failed"
    assert scoped_runner._active_operations == 0
