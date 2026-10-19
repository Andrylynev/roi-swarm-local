import asyncio

from roi_swarm.config import AppSettings
from roi_swarm.graph import build_graph
from roi_swarm.schemas import (
    FinalResult,
    GroupSummary,
    PlannedTask,
    PlanResult,
    WorkerResult,
)
from roi_swarm.sizing import SwarmAssessment


class FakeGateway:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self.synth_prompt = ""
        self.worker_prompts: list[str] = []

    async def generate_structured(self, *, model, system, prompt, schema, **_):
        self.calls[schema.__name__] = self.calls.get(schema.__name__, 0) + 1
        if schema is SwarmAssessment:
            return SwarmAssessment(
                task_type="research",
                complexity=1.0,
                parallelizability=1.0,
                uncertainty=1.0,
                stakes=1.0,
                swarm_required=True,
            )
        if schema is PlanResult:
            return PlanResult(
                summary="plan",
                tasks=[
                    PlannedTask(id=f"part-{i}", objective=f"Atomic part {i}: check item {i}")
                    for i in range(1, 121)
                ],
            )
        if schema is WorkerResult:
            self.worker_prompts.append(prompt)
            return WorkerResult(task_id="x", conclusion="finding", confidence=0.9)
        if schema is GroupSummary:
            return GroupSummary(summary="group", key_findings=["f"], confidence=0.9)
        if schema is FinalResult:
            self.synth_prompt = prompt
            return FinalResult(answer="done")
        raise AssertionError(schema)

    async def generate_text(self, *, model, system, prompt, **_):
        self.calls["text"] = self.calls.get("text", 0) + 1
        self.synth_prompt = prompt
        return "done"


def _settings(maximum: int) -> AppSettings:
    role = {"model": "m"}
    return AppSettings(
        provider={"base_url": "http://x/v1", "api_key": "k"},
        models={r: role for r in ("router", "planner", "worker", "critic", "synthesizer")},
        swarm={"minimum_workers": 10, "maximum_workers": maximum, "aggregation_group_size": 15},
    )


def test_large_swarm_is_hierarchically_aggregated_before_terra() -> None:
    gateway = FakeGateway()
    graph = build_graph(_settings(120), gateway)

    result = asyncio.run(graph.ainvoke({"request": "big research"}, config={"max_concurrency": 32}))

    assert result["final_answer"] == "done"
    assert result["worker_target"] == 120
    assert gateway.calls["WorkerResult"] == 120
    assert gateway.calls["GroupSummary"] == 8
    assert "group-8" in gateway.synth_prompt
    assert gateway.synth_prompt.count('"conclusion"') == 0
    # Every worker received its own atomic subtask, not a copy of the whole request.
    objectives = {p.split("Objective: ")[1].splitlines()[0] for p in gateway.worker_prompts}
    assert len(objectives) == 120


def test_planner_with_few_atomic_parts_does_not_spawn_padding_workers() -> None:
    class SmallPlan(FakeGateway):
        async def generate_structured(self, *, model, system, prompt, schema, **kw):
            if schema is PlanResult:
                self.calls["PlanResult"] = 1
                return PlanResult(
                    summary="plan",
                    tasks=[PlannedTask(id=f"p{i}", objective=f"part {i}") for i in range(3)],
                )
            return await super().generate_structured(
                model=model, system=system, prompt=prompt, schema=schema, **kw
            )

    gateway = SmallPlan()
    result = asyncio.run(build_graph(_settings(120), gateway).ainvoke({"request": "small"}))

    assert result["final_answer"] == "done"
    # <= 3 atomic parts: Opus answers by itself, no Luna workers.
    assert "WorkerResult" not in gateway.calls
    assert result["answered_directly"] is True


def test_simple_request_is_answered_by_opus_without_planner_or_workers() -> None:
    class Simple(FakeGateway):
        async def generate_structured(self, *, model, system, prompt, schema, **kw):
            if schema is SwarmAssessment:
                self.calls["SwarmAssessment"] = 1
                return SwarmAssessment(
                    task_type="qa",
                    complexity=0.1,
                    parallelizability=0.1,
                    uncertainty=0.1,
                    stakes=0.1,
                    swarm_required=False,
                )
            return await super().generate_structured(
                model=model, system=system, prompt=prompt, schema=schema, **kw
            )

    gateway = Simple()
    result = asyncio.run(build_graph(_settings(120), gateway).ainvoke({"request": "why sky blue"}))

    assert result["final_answer"] == "done"
    assert result["answered_directly"] is True
    assert "PlanResult" not in gateway.calls
    assert "WorkerResult" not in gateway.calls


def test_plan_above_direct_threshold_goes_to_luna_swarm() -> None:
    class FourParts(FakeGateway):
        async def generate_structured(self, *, model, system, prompt, schema, **kw):
            if schema is PlanResult:
                return PlanResult(
                    summary="plan",
                    tasks=[PlannedTask(id=f"p{i}", objective=f"part {i}") for i in range(4)],
                )
            return await super().generate_structured(
                model=model, system=system, prompt=prompt, schema=schema, **kw
            )

    gateway = FourParts()
    result = asyncio.run(build_graph(_settings(120), gateway).ainvoke({"request": "x"}))

    assert gateway.calls["WorkerResult"] == 4
    assert not result.get("answered_directly")


def test_concurrency_is_unlimited_by_default() -> None:
    assert _settings(70).swarm.maximum_concurrency is None
    assert _settings(70).swarm.direct_answer_max_workers == 3


def test_router_schema_failure_falls_back_to_direct_opus_answer() -> None:
    class BrokenRouter(FakeGateway):
        async def generate_structured(self, *, model, system, prompt, schema, **kw):
            if schema is SwarmAssessment:
                raise ValueError("router returned a title instead of an assessment")
            return await super().generate_structured(
                model=model, system=system, prompt=prompt, schema=schema, **kw
            )

    gateway = BrokenRouter()
    result = asyncio.run(build_graph(_settings(120), gateway).ainvoke({"request": "x"}))

    assert result["final_answer"] == "done"
    assert result["answered_directly"] is True


def test_synthesizer_writes_plain_markdown_not_json() -> None:
    """A long final report inside a JSON string gets truncated (Unterminated string)."""

    class NoJsonSynth(FakeGateway):
        async def generate_structured(self, *, model, system, prompt, schema, **kw):
            if schema is FinalResult:
                raise AssertionError("synthesizer must not use JSON output")
            return await super().generate_structured(
                model=model, system=system, prompt=prompt, schema=schema, **kw
            )

    gateway = NoJsonSynth()
    result = asyncio.run(build_graph(_settings(120), gateway).ainvoke({"request": "big"}))

    assert result["final_answer"] == "done"
    assert gateway.calls["WorkerResult"] == 120
