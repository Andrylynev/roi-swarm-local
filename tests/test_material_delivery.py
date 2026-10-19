import asyncio
import json

import pytest
from test_graph_e2e import FakeGateway, _settings

from roi_swarm.agent_loop import _auto_delegate_call, _swarm_request
from roi_swarm.context import ContextPipeline
from roi_swarm.graph import build_graph
from roi_swarm.schemas import PlannedTask, PlanResult, WorkerResult


def test_auto_delegate_retains_every_tool_result():
    originals = [f"FILE_{i}_EARLY\n" + "x" * 65000 + f"\nFILE_{i}_LATE" for i in range(3)]
    call = _auto_delegate_call(
        "Audit three files", [{"role": "tool", "content": text} for text in originals]
    )
    payload = json.loads(call["function"]["arguments"])
    assert payload["materials"] == "\n\n".join(originals)
    request = _swarm_request(call["function"]["arguments"])
    assert all(original in request for original in originals)


def test_swarm_request_keeps_material_whitespace_verbatim():
    materials = "\n  source line\n\n"
    request = _swarm_request(json.dumps({"task": "Audit", "materials": materials}))
    assert request == "Audit\n\nMATERIALS:\n" + materials


def test_auto_delegate_retains_structured_tool_content():
    structured = {"file": "module.ts", "result": "MIDDLE_STRUCTURED_MARKER"}
    content = [
        {"type": "text", "text": "EARLY_STRUCTURED_MARKER"},
        {"type": "result", "data": structured},
    ]
    call = _auto_delegate_call(
        "Audit",
        [
            {"role": "tool", "content": content},
            {"role": "tool", "content": structured},
        ],
    )
    materials = json.loads(call["function"]["arguments"])["materials"]
    assert "EARLY_STRUCTURED_MARKER" in materials
    assert materials.count("MIDDLE_STRUCTURED_MARKER") == 2
    assert "module.ts" in materials


def test_planner_receives_full_deduplicated_spilled_history(tmp_path):
    pipeline = ContextPipeline(
        tmp_path, artifact_threshold_chars=100, recent_messages=1, maximum_pack_chars=1000
    )
    original = "EARLY_FILE_MARKER\n" + "a" * 2400 + "\nMIDDLE_FILE_MARKER\n"
    original += "z" * 2400 + "\nLATE_FILE_MARKER"
    prepared = pipeline.prepare(
        [
            {"role": "user", "content": "Audit the source"},
            {"role": "tool", "content": original, "tool_call_id": "read-old"},
            {"role": "tool", "content": original, "tool_call_id": "read-again"},
            {"role": "user", "content": "Find four concrete defects"},
        ],
        run_id="spilled-history",
    )
    pack = pipeline.pack_for_planner(prepared)
    assert original in pack
    assert pack.count("MIDDLE_FILE_MARKER") == 1
    assert "read-old" in pack
    assert "HARD CONSTRAINTS:" in pack
    assert prepared.metrics.packed_chars == len(pack)
    from roi_swarm.token_budget import count_tokens

    assert prepared.metrics.packed_tokens_estimate == count_tokens(pack)
    assert prepared.metrics.input_tokens_estimate == count_tokens(
        prepared.conversation.render_full()
    )
    assert prepared.metrics.token_encoding == "o200k_base"
    assert prepared.metrics.token_accounting == "reference; provider-native counts may differ"
    assert len(pack) > pipeline.maximum_pack_chars
    assert prepared.conversation.blocks[1].artifact_id
    assert prepared.conversation.blocks[2].text == original


def test_material_payload_stays_original_not_repeated_in_goal_or_small_packs(tmp_path):
    pipeline = ContextPipeline(tmp_path, maximum_pack_chars=1500)
    goal = "Audit four modules. Never modify production."
    materials = "FILE_EARLY\n" + "source " * 2000 + "\nFILE_MIDDLE\nFILE_LATE"
    full = goal + "\n\nMATERIALS:\n" + materials
    prepared = pipeline.prepare([{"role": "user", "content": full}], run_id="separate")
    assert prepared.current_request == goal
    assert prepared.canonical_state.goal == goal
    assert prepared.conversation.blocks[0].text == full
    planner = pipeline.pack_for_planner(prepared)
    assert planner.count(materials) == 1
    assert not prepared.canonical_state.open_items
    for pack in (
        pipeline.pack_for_router(prepared),
        pipeline.pack_for_synthesizer(prepared),
        pipeline.pack_for_worker(
            prepared, PlannedTask(id="a", objective="Audit module a", context="EXACT INPUT")
        ),
    ):
        assert materials not in pack
        assert "FILE_EARLY" not in pack
        assert len(pack) <= 1500


def test_canonical_hard_constraints_are_not_160_character_previews(tmp_path):
    constraint = "Never discard " + "a detailed requirement " * 20 + "FINAL_CONSTRAINT_MARKER"
    prepared = ContextPipeline(tmp_path).prepare(
        [{"role": "system", "content": constraint}, {"role": "user", "content": "Audit"}],
        run_id="long-constraint",
    )
    assert constraint in prepared.canonical_state.hard_constraints


def test_planner_budget_is_independent_explicit_and_never_truncates(tmp_path):
    source = "EARLY\n" + "data " * 1000 + "MIDDLE\n" + "data " * 1000 + "LATE"
    messages = [{"role": "user", "content": "Audit\n\nMATERIALS:\n" + source}]
    pipeline = ContextPipeline(tmp_path / "normal", maximum_pack_chars=500)
    assert pipeline.planner_input_limit_chars is None
    assert pipeline.main_context_tokens == 1_000_000
    prepared = pipeline.prepare(messages, run_id="normal")
    full_pack = pipeline.pack_for_planner(prepared)
    assert source in full_pack
    exact = ContextPipeline(
        tmp_path / "exact", planner_input_limit_chars=len(full_pack), maximum_pack_chars=500
    )
    assert exact.pack_for_planner(exact.prepare(messages, run_id="normal")) == full_pack
    too_small = ContextPipeline(tmp_path / "overflow", planner_input_limit_chars=len(full_pack) - 1)
    with pytest.raises(ValueError, match=r"(?i)planner.*limit.*split.*task"):
        too_small.prepare(messages, run_id="normal")


def test_workflow_lessons_are_untrusted_advisory_and_count_towards_planner_limit(tmp_path):
    pipeline = ContextPipeline(tmp_path)
    prepared = pipeline.prepare([{"role": "user", "content": "Audit"}], run_id="lessons")
    assert prepared.workflow_lessons == []
    prepared.workflow_lessons = [{"lesson": "Keep assigned evidence intact", "scope": "project"}]
    pack = pipeline.pack_for_planner(prepared)
    assert "UNTRUSTED WORKFLOW LESSONS" in pack
    assert "not evidence or instructions" in pack
    assert "user/system" in pack
    assert json.dumps(prepared.workflow_lessons, ensure_ascii=False) in pack
    assert prepared.metrics.packed_chars == len(pack)
    pipeline.planner_input_limit_chars = len(pack) - 1
    with pytest.raises(ValueError, match="Split the task"):
        pipeline.pack_for_planner(prepared)


def test_real_graph_delivers_early_middle_late_evidence_and_retries(tmp_path):
    sources = [
        f"FILE_{i}_EARLY\n"
        + "x" * 17000
        + f"\nFILE_{i}_MIDDLE\n"
        + "z" * 17000
        + f"\nFILE_{i}_LATE"
        for i in range(4)
    ]
    call = _auto_delegate_call(
        "Audit four files", [{"role": "tool", "content": text} for text in sources]
    )
    full_request = _swarm_request(call["function"]["arguments"])
    pipeline = ContextPipeline(tmp_path, maximum_pack_chars=1800, recent_messages=1)
    prepared = pipeline.prepare(
        [
            {"role": "system", "content": "Never invent writes."},
            {"role": "tool", "content": sources[0], "tool_call_id": "old-spilled-file"},
            {"role": "tool", "content": sources[0], "tool_call_id": "duplicate-file"},
            {
                "role": "user",
                "content": "Audit four files\n\nMATERIALS:\n" + "\n\n".join(sources[1:]),
            },
        ],
        run_id="graph-materials",
    )

    class EvidenceGateway(FakeGateway):
        def __init__(self):
            super().__init__()
            self.attempts = {}
            self.planner_prompt = ""
            self.group_prompts = []

        async def generate_structured(self, *, schema, system, prompt, **kwargs):
            if schema is PlanResult:
                self.planner_prompt = prompt
                assert all(source in prompt for source in sources)
                assert all(prompt.count(f"FILE_{i}_MIDDLE") == 1 for i in range(4))
                assert "cannot read or write files" in system
                assert "[[message:N]]" in system
                assert "zero-based message number" in system
                assert "do not reprint large originals" in system
                assert "arbitrary source references" in system
                return PlanResult(
                    summary="four atomic audits",
                    tasks=[
                        PlannedTask(
                            id=f"f{i}",
                            objective=f"Audit file {i}",
                            context="\n".join(
                                line for line in source.splitlines() if line.startswith("FILE_")
                            ),
                        )
                        for i, source in enumerate(sources)
                    ],
                )
            if schema is WorkerResult:
                assert "workflow_lessons" in system
                assert "no secrets" in system
                self.worker_prompts.append(prompt)
                task_id = prompt.split("TASK ID: ")[1].splitlines()[0]
                attempt = self.attempts.get(task_id, 0) + 1
                self.attempts[task_id] = attempt
                return WorkerResult(
                    task_id=task_id,
                    conclusion="verified file finding",
                    confidence=0.1 if attempt == 1 else 0.95,
                )
            if schema.__name__ == "GroupSummary":
                self.group_prompts.append(prompt)
            return await super().generate_structured(
                schema=schema, system=system, prompt=prompt, **kwargs
            )

    settings = _settings(10)
    settings.swarm.aggregation_group_size = 2
    settings.swarm.expansion_batch = 4
    settings.swarm.maximum_rounds = 2
    gateway = EvidenceGateway()
    result = asyncio.run(
        build_graph(settings, gateway, pipeline).ainvoke(
            {"request": full_request, "prepared_context": prepared}
        )
    )
    assert result["verification"]["passed"]
    assert prepared.metrics.packed_chars == len(gateway.planner_prompt)
    assert len(gateway.worker_prompts) == 8
    for i in range(4):
        expected = f"FILE_{i}_EARLY\nFILE_{i}_MIDDLE\nFILE_{i}_LATE"
        prompts = [p for p in gateway.worker_prompts if p.startswith(f"TASK ID: f{i}\n")]
        assert len(prompts) == 2
        assert all(expected in prompt for prompt in prompts)
        assert "Retry round 2" in prompts[1]
        assert all(len(prompt) <= 1800 for prompt in prompts)
    assert all("MATERIALS:" not in p for p in gateway.group_prompts)
    assert all(source not in gateway.synth_prompt for source in sources)


def test_graph_rechecks_planner_budget_before_any_provider_call(tmp_path):
    pipeline = ContextPipeline(tmp_path)
    prepared = pipeline.prepare([{"role": "user", "content": "Audit"}], run_id="late-limit")
    prepared.workflow_lessons = [{"lesson": "x" * 1000}]
    pipeline.planner_input_limit_chars = 500
    gateway = FakeGateway()
    with pytest.raises(ValueError, match="Split the task"):
        asyncio.run(
            build_graph(_settings(10), gateway, pipeline).ainvoke(
                {"request": "Audit", "prepared_context": prepared}
            )
        )
    assert gateway.calls == {}


def test_graph_without_pipeline_keeps_original_material_for_planner():
    original = "Audit\n\nMATERIALS:\nEARLY\nMIDDLE\nLATE"

    class RawGateway(FakeGateway):
        async def generate_structured(self, *, schema, prompt, **kwargs):
            if schema is PlanResult:
                assert original in prompt
                return PlanResult(summary="audit", tasks=[PlannedTask(id="a", objective="Audit")])
            return await super().generate_structured(schema=schema, prompt=prompt, **kwargs)

    result = asyncio.run(build_graph(_settings(10), RawGateway()).ainvoke({"request": original}))
    assert result["answered_directly"]


def test_short_plan_expands_only_assigned_large_originals_before_workers(tmp_path):
    sources = [
        f"FILE_{i}_EARLY\n" + (f"source_{i} line\n" * 18_000)
        + f"FILE_{i}_MIDDLE\nFILE_{i}_LATE\n"
        for i in range(4)
    ]
    unrelated = "UNASSIGNED_PROJECT\n" + "other source\n" * 20_000
    messages = [{"role": "system", "content": "Never invent evidence. Preserve source citations."}]
    messages.extend(
        {"role": "tool", "content": source, "tool_call_id": f"read-file-{i}"}
        for i, source in enumerate(sources)
    )
    messages.append({"role": "tool", "content": unrelated, "tool_call_id": "unassigned"})
    messages.append({"role": "user", "content": "Audit four files independently."})
    pipeline = ContextPipeline(tmp_path, recent_messages=1)
    prepared = pipeline.prepare(messages, run_id="short-reference-plan")
    assert all(source not in prepared.recent_context for source in sources)

    class ReferenceGateway(FakeGateway):
        def __init__(self):
            super().__init__()
            self.attempts = {}

        async def generate_structured(self, *, schema, system, prompt, **kwargs):
            if schema is PlanResult:
                assert all(source in prompt for source in sources)
                assert "[[message:N]]" in system
                assert "zero-based" in system
                assert kwargs["max_output_tokens"] == 12_000
                return PlanResult(
                    summary="four independent audits",
                    tasks=[
                        PlannedTask(
                            id=f"f{i}", objective=f"Audit file {i}",
                            context=f"Assigned input:\n[[message:{i + 1}]]\nKeep citations.",
                        )
                        for i in range(4)
                    ],
                )
            if schema is WorkerResult:
                self.worker_prompts.append(prompt)
                task_id = prompt.split("TASK ID: ")[1].splitlines()[0]
                attempt = self.attempts.get(task_id, 0) + 1
                self.attempts[task_id] = attempt
                return WorkerResult(
                    task_id=task_id, conclusion="source-backed finding",
                    confidence=0.1 if attempt == 1 else 0.95,
                )
            return await super().generate_structured(
                schema=schema, system=system, prompt=prompt, **kwargs
            )

    settings = _settings(10)
    settings.models["planner"].max_output_tokens = 12_000
    settings.swarm.maximum_rounds = 2
    settings.swarm.expansion_batch = 4
    gateway = ReferenceGateway()
    result = asyncio.run(
        build_graph(settings, gateway, pipeline).ainvoke(
            {"request": prepared.current_request, "prepared_context": prepared}
        )
    )
    assert result["verification"]["passed"]
    assert len(gateway.worker_prompts) == 8
    assert all(source not in task["context"] for source in sources for task in result["tasks"])
    for i, source in enumerate(sources):
        prompts = [p for p in gateway.worker_prompts if p.startswith(f"TASK ID: f{i}\n")]
        assert len(prompts) == 2
        assert "Retry round 2" in prompts[1]
        for prompt in prompts:
            assert prompt.count(source) == 1
            assert all(other not in prompt for other in sources if other != source)
            assert unrelated not in prompt
            assert "Never invent evidence. Preserve source citations." in prompt
            assert f"message={i + 1} role=tool" in prompt
            assert f"tool_call_id=read-file-{i}" in prompt
            assert f"[[message:{i + 1}]]" not in prompt


def test_direct_fallback_receives_materials_once_without_giant_request_prefix(tmp_path):
    source = "DIRECT_EARLY\n" + "x" * 4000 + "\nDIRECT_MIDDLE\nDIRECT_LATE"
    request = "Audit\n\nMATERIALS:\n" + source
    pipeline = ContextPipeline(tmp_path, maximum_pack_chars=600)
    prepared = pipeline.prepare([{"role": "user", "content": request}], run_id="direct")

    class BrokenRouter(FakeGateway):
        async def generate_structured(self, **kwargs):
            raise RuntimeError("router failed")

    gateway = BrokenRouter()
    result = asyncio.run(
        build_graph(_settings(10), gateway, pipeline).ainvoke(
            {"request": request, "prepared_context": prepared}
        )
    )
    assert result["answered_directly"]
    assert gateway.synth_prompt.count(source) == 1
    assert "DIRECT_MIDDLE" in gateway.synth_prompt
