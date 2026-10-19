import asyncio

import pytest
from test_graph_e2e import FakeGateway, _settings

from roi_swarm.context import ContextPipeline
from roi_swarm.graph import build_graph
from roi_swarm.schemas import PlannedTask, PlanResult


@pytest.mark.parametrize("prepared_available", [True, False])
def test_unknown_message_reference_fails_worker_explicitly_without_model_call(
    tmp_path, prepared_available
):
    class ReferenceGateway(FakeGateway):
        async def generate_structured(self, *, schema, **kwargs):
            if schema is PlanResult:
                return PlanResult(
                    summary="four tasks",
                    tasks=[
                        PlannedTask(id=f"p{i}", objective=f"Check {i}", context="[[message:999]]")
                        for i in range(4)
                    ],
                )
            return await super().generate_structured(schema=schema, **kwargs)

    settings = _settings(10)
    settings.swarm.maximum_rounds = 1
    pipeline = ContextPipeline(tmp_path)
    prepared = pipeline.prepare([{"role": "user", "content": "Audit four files"}], run_id="unknown")
    gateway = ReferenceGateway()
    state = {"request": prepared.current_request}
    if prepared_available:
        state["prepared_context"] = prepared
    result = asyncio.run(
        build_graph(settings, gateway, pipeline if prepared_available else None).ainvoke(state)
    )
    assert result["verification"]["passed"] is False
    assert len(result["worker_results"]) == 4
    assert all(item["confidence"] == 0 for item in result["worker_results"])
    assert all("[[message:999]]" in item["conclusion"] for item in result["worker_results"])
    assert gateway.worker_prompts == []


def test_oversized_worker_evidence_is_reported_as_failed_not_graph_crash(tmp_path):
    class LargeContextGateway(FakeGateway):
        async def generate_structured(self, *, schema, **kwargs):
            if schema is PlanResult:
                return PlanResult(
                    summary="four tasks",
                    tasks=[
                        PlannedTask(id=f"p{i}", objective=f"Check {i}", context="source " * 500)
                        for i in range(4)
                    ],
                )
            return await super().generate_structured(schema=schema, **kwargs)

    settings = _settings(10)
    settings.swarm.maximum_rounds = 1
    pipeline = ContextPipeline(tmp_path, maximum_pack_chars=1000)
    prepared = pipeline.prepare([{"role": "user", "content": "Audit four files"}], run_id="test")
    result = asyncio.run(
        build_graph(settings, LargeContextGateway(), pipeline).ainvoke(
            {"request": prepared.current_request, "prepared_context": prepared}
        )
    )

    assert result["verification"]["passed"] is False
    assert len(result["worker_results"]) == 4
    assert all(item["confidence"] == 0 for item in result["worker_results"])
    assert all("ValueError" in item["conclusion"] for item in result["worker_results"])
