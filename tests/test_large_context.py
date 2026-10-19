import asyncio

import pytest
from test_graph_e2e import FakeGateway, _settings

from roi_swarm.config import ContextConfig
from roi_swarm.context import ContextPipeline
from roi_swarm.graph import build_graph
from roi_swarm.schemas import PlannedTask, PlanResult


def test_context_defaults_are_token_budgets_without_old_character_caps():
    config = ContextConfig()
    assert config.worker_context_tokens == 800_000
    assert config.main_context_tokens == 1_000_000
    assert config.maximum_pack_chars is None
    assert config.planner_input_limit_chars is None


def test_large_worker_context_and_originals_are_not_clipped(tmp_path):
    pipeline = ContextPipeline(tmp_path)
    source = "FIRST_MARKER\n" + "data sample\n" * 55_000 + "MIDDLE_MARKER\nLAST_MARKER"
    prepared = pipeline.prepare(
        [{"role": "user", "content": "Inspect the assigned file.\n\nMATERIALS:\n" + source}],
        run_id="large-context",
    )
    planner = pipeline.pack_for_planner(prepared)
    worker = pipeline.pack_for_worker(
        prepared, PlannedTask(id="file", objective="Inspect assigned file", context=source)
    )
    assert source in planner
    assert source in worker
    assert pipeline.worker_context_tokens == 800_000
    assert pipeline.main_context_tokens == 1_000_000


@pytest.mark.parametrize("use_reference", [False, True])
def test_worker_does_not_duplicate_its_assigned_material_near_token_limit(tmp_path, use_reference):
    from roi_swarm.token_budget import count_tokens

    pipeline = ContextPipeline(tmp_path)
    source = " z" * 770_000
    prepared = pipeline.prepare(
        [{"role": "user", "content": "Inspect assigned file.\n\nMATERIALS:\n" + source}],
        run_id="near-worker-limit",
    )
    pack = pipeline.pack_for_worker(
        prepared, PlannedTask(
            id="one", objective="Inspect assigned file",
            context="[[message:0]]" if use_reference else source,
        )
    )
    assert pack.count(source) == 1
    assert count_tokens(pack) < 800_000


def test_worker_receives_only_assigned_part_of_materials_larger_than_worker_window(tmp_path):
    from roi_swarm.token_budget import count_tokens

    pipeline = ContextPipeline(tmp_path)
    assigned = "ASSIGNED_FILE\n" + " z" * 770_000 + "\nASSIGNED_END"
    unrelated = "UNASSIGNED_FILE\n" + " y" * 100_000 + "\nUNASSIGNED_END"
    materials = assigned + "\n\n" + unrelated
    assert count_tokens(materials) > 800_000
    prepared = pipeline.prepare(
        [{"role": "user", "content": "Audit assigned file.\n\nMATERIALS:\n" + materials}],
        run_id="assigned-part-only",
    )
    pack = pipeline.pack_for_worker(
        prepared, PlannedTask(id="one", objective="Audit assigned file", context=assigned)
    )
    assert pack.count(assigned) == 1
    assert unrelated not in pack
    assert "UNASSIGNED_FILE" not in pack
    assert count_tokens(pack) < 800_000


def test_worker_does_not_duplicate_assigned_hard_constraints_near_limit(tmp_path):
    from roi_swarm.token_budget import count_tokens

    pipeline = ContextPipeline(tmp_path)
    constraint = "Never discard required evidence:" + " z" * 770_000
    prepared = pipeline.prepare(
        [{"role": "system", "content": constraint}, {"role": "user", "content": "Audit"}],
        run_id="assigned-constraint",
    )
    pack = pipeline.pack_for_worker(
        prepared, PlannedTask(id="one", objective="Audit", context="[[message:0]]")
    )
    assert pack.count(constraint) == 1
    assert "HARD CONSTRAINTS:" in pack
    assert "TASK CONTEXT" in pack
    assert count_tokens(pack) < 800_000


def test_expanded_worker_evidence_obeys_worker_not_main_token_limit(tmp_path):
    pipeline = ContextPipeline(tmp_path, worker_context_tokens=300, main_context_tokens=2_000)
    source = " evidence" * 400
    prepared = pipeline.prepare(
        [{"role": "tool", "content": source}, {"role": "user", "content": "Audit"}],
        run_id="expanded-overflow",
    )
    with pytest.raises(ValueError, match="Context exceeds 300 tokens"):
        pipeline.pack_for_worker(
            prepared, PlannedTask(id="one", objective="Audit", context="[[message:0]]")
        )


def test_worker_fits_required_assigned_evidence_without_optional_goal_blocking(tmp_path):
    from roi_swarm.token_budget import count_tokens

    pipeline = ContextPipeline(tmp_path, worker_context_tokens=300, main_context_tokens=2_000)
    goal = "Audit independently. " + " background" * 500
    source = " evidence" * 180
    prepared = pipeline.prepare(
        [{"role": "tool", "content": source}, {"role": "user", "content": goal}],
        run_id="optional-goal",
    )
    pack = pipeline.pack_for_worker(
        prepared,
        PlannedTask(id="one", objective="Audit assigned evidence", context="[[message:0]]"),
    )
    assert source in pack
    assert goal not in pack
    assert count_tokens(pack) <= 300


def test_planner_near_main_limit_keeps_large_goal_constraints_once(tmp_path):
    from roi_swarm.token_budget import count_tokens

    system = "Never discard this instruction:" + " z" * 470_000
    goal = "Audit these independent requirements:" + " y" * 470_000
    pipeline = ContextPipeline(tmp_path)
    prepared = pipeline.prepare(
        [{"role": "system", "content": system}, {"role": "user", "content": goal}],
        run_id="main-canonical-dedup",
    )
    pack = pipeline.pack_for_planner(prepared)
    assert pack.count(system) == 1
    assert pack.count(goal) == 1
    assert "HARD CONSTRAINTS:" in pack
    assert "[[message:0]]" in pack
    assert "[[message:1]]" in pack
    assert prepared.metrics.packed_tokens_estimate == count_tokens(pack) < 1_000_000


def test_system_context_above_12000_still_uses_swarm(tmp_path):
    pipeline = ContextPipeline(tmp_path)
    instructions = "Never invent findings.\n" + "Required reference " * 1600
    prepared = pipeline.prepare(
        [
            {"role": "system", "content": instructions},
            {"role": "user", "content": "Audit four separate files."},
        ],
        run_id="swarm-large",
    )
    seen = []

    class Gateway(FakeGateway):
        async def generate_structured(self, *, schema, **kwargs):
            seen.append((schema.__name__, kwargs.get("context_window_tokens")))
            if schema is PlanResult:
                return PlanResult(
                    summary="four parts",
                    tasks=[
                        PlannedTask(id=str(i), objective=f"Inspect file {i}", context="source")
                        for i in range(4)
                    ],
                )
            return await super().generate_structured(schema=schema, **kwargs)

        async def generate_text(self, **kwargs):
            seen.append(("final", kwargs.get("context_window_tokens")))
            return await super().generate_text(**kwargs)

    settings = _settings(10)
    result = asyncio.run(
        build_graph(settings, Gateway(), pipeline).ainvoke(
            {
                "request": prepared.current_request,
                "prepared_context": prepared,
            }
        )
    )
    assert not result.get("answered_directly")
    assert len(result["worker_results"]) == 4
    assert ("SwarmAssessment", 800_000) in seen
    assert ("PlanResult", 1_000_000) in seen
    assert ("WorkerResult", 800_000) in seen
    assert ("final", 1_000_000) in seen


def test_runner_exposes_applied_budgets_and_keeps_full_tool_router_request(tmp_path, monkeypatch):
    from roi_swarm.runner import SwarmRunner
    from roi_swarm.sizing import SwarmAssessment

    settings = _settings(10)
    settings.rag.enabled = False
    import roi_swarm.runner as runner_module

    monkeypatch.setattr(runner_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner_module, "load_settings", lambda _: settings)
    runner = SwarmRunner(tmp_path / "config" / "settings.yaml")
    captured = []

    async def assess(**kwargs):
        captured.append(kwargs)
        return SwarmAssessment(
            task_type="test", complexity=0, parallelizability=0,
            uncertainty=0, stakes=0, swarm_required=False,
        )

    runner.gateway.generate_structured = assess
    try:
        assert runner.model_settings()["context_limits"] == {
            "worker_context_tokens": 800_000, "main_context_tokens": 1_000_000,
            "token_encoding": "o200k_base",
            "maximum_pack_chars": None, "planner_input_limit_chars": None,
        }
        request = "task " * 5000 + "LAST_REQUIREMENT"
        asyncio.run(runner._assess_for_swarm(request))
        assert captured[0]["prompt"] == request
        assert captured[0]["context_window_tokens"] == 800_000
    finally:
        asyncio.run(runner.close())


def test_gateway_checks_tokens_not_characters_and_reserves_output():
    from test_gateway import FakeClient

    from roi_swarm.gateway import ModelGateway
    from roi_swarm.token_budget import count_tokens

    prompt = " z" * 800_000
    assert count_tokens(prompt) == 800_000
    client = FakeClient("ok")
    gateway = ModelGateway(client)
    with pytest.raises(ValueError, match="(?i)context.*800000.*tokens"):
        asyncio.run(
            gateway.generate_text(
                model="worker",
                system="system",
                prompt=prompt,
                max_output_tokens=5000,
                context_window_tokens=800_000,
            )
        )
    assert client.responses.calls == []
    answer = asyncio.run(
        gateway.generate_text(
            model="main",
            system="system",
            prompt=prompt,
            max_output_tokens=5000,
            context_window_tokens=1_000_000,
        )
    )
    assert answer == "ok"
    assert len(client.responses.calls) == 1
