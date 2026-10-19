import pytest

from roi_swarm.context import ContextPipeline, NormalizedConversation
from roi_swarm.schemas import PlannedTask


@pytest.mark.parametrize(
    "marker", ["[[message:01]]", "[[message:-1]]", "[[message:bad]]", "[[message:1]"]
)
def test_worker_rejects_malformed_message_binding_instead_of_passing_it_through(tmp_path, marker):
    pipeline = ContextPipeline(tmp_path)
    prepared = pipeline.prepare([{"role": "user", "content": "Audit"}], run_id="malformed")
    with pytest.raises(ValueError, match="Invalid task context reference"):
        pipeline.pack_for_worker(
            prepared, PlannedTask(id="one", objective="Audit", context=marker)
        )


def test_repeated_and_duplicate_message_bindings_preserve_sources_without_duplicate_evidence(
    tmp_path,
):
    pipeline = ContextPipeline(tmp_path, artifact_threshold_chars=100)
    source = "\n  EXACT_SOURCE\n" + "payload " * 1000 + "\nEND_SOURCE\n\n"
    prepared = pipeline.prepare(
        [
            {"role": "tool", "content": source, "tool_call_id": "first-read"},
            {"role": "tool", "content": source, "tool_call_id": "second-read"},
            {"role": "user", "content": "Audit source"},
        ],
        run_id="duplicate-bindings",
    )
    pack = pipeline.pack_for_worker(
        prepared,
        PlannedTask(
            id="one", objective="Audit source",
            context="[[message:1]]\n[[message:0]]\n[[message:1]]",
        ),
    )
    assert pack.count(source) == 1
    assert "message=0 role=tool tool_call_id=first-read" in pack
    assert "message=1 role=tool tool_call_id=second-read" in pack
    assert "evidence already included" in pack


def test_message_expansion_preserves_nested_literal_marker_without_recursing(tmp_path):
    pipeline = ContextPipeline(tmp_path)
    source = "\n  source contains literal [[message:999]]\n\n"
    prepared = pipeline.prepare(
        [{"role": "tool", "content": source}, {"role": "user", "content": "Audit"}],
        run_id="nonrecursive",
    )
    pack = pipeline.pack_for_worker(
        prepared, PlannedTask(id="one", objective="Audit", context="Before\n[[message:0]]\nAfter")
    )
    assert source in pack
    assert "Before\n[message=0 role=tool]" in pack
    assert "After" in pack


def test_full_direct_task_context_and_paths_remain_verbatim_without_guessed_expansion(tmp_path):
    pipeline = ContextPipeline(tmp_path)
    unassigned = "PRIVATE_UNASSIGNED_SOURCE"
    prepared = pipeline.prepare(
        [
            {"role": "tool", "content": unassigned},
            {"role": "user", "content": "Audit source"},
        ],
        run_id="direct-context",
    )
    context = "\n  // assigned.ts\nconst x = 1;\nsource: assigned.ts\n\n"
    pack = pipeline.pack_for_worker(
        prepared, PlannedTask(id="one", objective="Audit source", context=context)
    )
    assert context in pack
    assert unassigned not in pack
    assert "artifact_id=" not in pack


def test_context_pipeline_preserves_full_conversation_and_constraints(tmp_path) -> None:
    pipeline = ContextPipeline(root=tmp_path, artifact_threshold_chars=200, recent_messages=2)
    messages = [
        {"role": "system", "content": "Never use Docker."},
        {"role": "user", "content": "Build an API."},
        {"role": "assistant", "content": "I will use FastAPI. " * 40},
        {"role": "user", "content": "Keep all files on D:."},
    ]

    prepared = pipeline.prepare(messages, run_id="run-context")

    assert prepared.current_request == "Keep all files on D:."
    assert prepared.conversation.message_count == 4
    assert "Never use Docker." in prepared.canonical_state.hard_constraints
    assert "Keep all files on D:." in prepared.canonical_state.hard_constraints
    planner_pack = pipeline.pack_for_planner(prepared)
    assert prepared.metrics.packed_chars == len(planner_pack)
    assert messages[2]["content"] in planner_pack


def test_exact_dedup_externalizes_large_duplicate_content(tmp_path) -> None:
    pipeline = ContextPipeline(root=tmp_path, artifact_threshold_chars=50)
    repeated = "large result " * 20
    messages = [
        {"role": "user", "content": repeated},
        {"role": "assistant", "content": repeated},
        {"role": "user", "content": "Summarize the result"},
    ]

    prepared = pipeline.prepare(messages, run_id="run-dedup")

    assert prepared.metrics.duplicate_blocks == 1
    assert prepared.metrics.artifact_count >= 1
    assert any(block.artifact_id for block in prepared.conversation.blocks)


def test_role_specific_worker_pack_is_smaller_than_full_context(tmp_path) -> None:
    pipeline = ContextPipeline(
        root=tmp_path, artifact_threshold_chars=200, recent_messages=2, maximum_pack_chars=12000
    )
    prepared = pipeline.prepare(
        [
            {"role": "system", "content": "Use Python and preserve citations."},
            {"role": "user", "content": "Research database options."},
            {"role": "assistant", "content": "We discussed SQLite and Postgres. " * 40},
            {"role": "user", "content": "Now compare SQLite security."},
        ],
        run_id="run-pack",
    )
    task = PlannedTask(id="security", role="critic", objective="Check SQLite security")

    pack = pipeline.pack_for_worker(prepared, task)

    assert "Check SQLite security" in pack
    assert len(pack) < len(prepared.conversation.render_full())


def test_delta_snapshot_records_only_changed_fields(tmp_path) -> None:
    pipeline = ContextPipeline(root=tmp_path)
    first = pipeline.prepare(
        [{"role": "user", "content": "Build the service on D:."}],
        run_id="run-delta",
    )
    second = pipeline.prepare(
        [
            {"role": "user", "content": "Build the service on D:."},
            {"role": "user", "content": "Use SQLite."},
        ],
        run_id="run-delta",
        previous_state=first.canonical_state,
    )

    assert second.delta.changed is True
    assert "goal" in second.delta.fields or "decisions" in second.delta.fields


def test_normalized_conversation_accepts_tool_messages_and_null_content() -> None:
    conversation = NormalizedConversation.from_openai_messages(
        [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
            {"role": "tool", "content": "42", "tool_call_id": "call-1"},
            {"role": "user", "content": "Use that result"},
        ]
    )

    assert conversation.message_count == 3
    assert conversation.blocks[0].role == "assistant"
    assert "call-1" in conversation.blocks[0].text


def test_worker_receives_assigned_file_and_retry_context_with_large_goal(tmp_path) -> None:
    pipeline = ContextPipeline(root=tmp_path, maximum_pack_chars=3000)
    prepared = pipeline.prepare(
        [{"role": "user", "content": "Audit application.\n" + "background material " * 5000}],
        run_id="large-audit",
    )
    for name in ("security", "performance", "seo", "dependencies"):
        source = f"// {name}.ts\nexport const {name} = 'SOURCE_{name}';\n" * 12
        context = f"Assigned source: {name}.ts\n{source}\nRETRY: verify SOURCE_{name} again."
        task = PlannedTask(id=name, objective=f"Audit {name}", context=context)

        pack = pipeline.pack_for_worker(prepared, task)

        assert context in pack
        assert len(pack) <= pipeline.maximum_pack_chars


def test_worker_preserves_hard_constraints_when_background_fills_budget(tmp_path) -> None:
    pipeline = ContextPipeline(root=tmp_path, maximum_pack_chars=1000)
    prepared = pipeline.prepare(
        [
            {"role": "system", "content": "Never use Docker. Never modify production files."},
            {"role": "user", "content": "Audit application. " + "background " * 2000},
        ],
        run_id="constraints",
    )
    task = PlannedTask(id="security", objective="Audit security", context="source " * 50)

    pack = pipeline.pack_for_worker(prepared, task)

    assert task.context in pack
    assert "Never use Docker. Never modify production files." in pack
    assert len(pack) <= pipeline.maximum_pack_chars


def test_worker_rejects_oversized_task_context_instead_of_silently_truncating(tmp_path) -> None:
    import pytest

    pipeline = ContextPipeline(root=tmp_path, maximum_pack_chars=1000)
    prepared = pipeline.prepare([{"role": "user", "content": "Audit"}], run_id="oversized")
    task = PlannedTask(id="security", objective="Audit security", context="evidence " * 300)

    with pytest.raises(ValueError, match="(?i)task context.*budget"):
        pipeline.pack_for_worker(prepared, task)
