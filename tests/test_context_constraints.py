import pytest

from roi_swarm.context import ContextPipeline


@pytest.mark.parametrize("budget", [0, 1, 2, 8, 32, 33, 34, 35, 100])
def test_limit_pack_counts_truncation_marker_inside_budget(tmp_path, budget) -> None:
    pipeline = ContextPipeline(tmp_path, maximum_pack_chars=budget)

    pack = pipeline._limit_pack("start " + "optional " * 100 + " end")

    assert len(pack) <= budget
    if budget >= len("\n[...context pack truncated...]\n"):
        assert "[...context pack truncated...]" in pack
    if budget == 100:
        assert pack.startswith("start ")
        assert pack.endswith(" end")


@pytest.mark.parametrize("role", ["synthesizer", "router"])
def test_role_pack_preserves_required_middle_constraints_before_large_optional_context(
    tmp_path, role
) -> None:
    pipeline = ContextPipeline(tmp_path, maximum_pack_chars=1800)
    constraints = [
        f"Never discard rule {index}: " + "обязательное условие " * 5 + "END" for index in range(10)
    ]
    goal = "Проверить все ограничения без потери исходного текста."
    prepared = pipeline.prepare(
        [
            {"role": "system", "content": "\n".join(constraints)},
            {"role": "user", "content": goal},
        ],
        run_id="required-first",
    )
    prepared.canonical_state.decisions = ["optional decision " * 2000]
    prepared.canonical_state.artifact_ids = ["optional-artifact-" * 2000]
    prepared.rag_results = [{"source": "optional", "text": "optional evidence " * 2000}]
    prepared.recent_context = "optional recent context " * 2000
    heading = "GOAL: " if role == "synthesizer" else "CURRENT REQUEST:\n"
    required = (
        heading
        + goal
        + "\n\nHARD CONSTRAINTS:\n"
        + "\n".join(f"- {constraint}" for constraint in constraints)
    )
    assert len(required) < pipeline.maximum_pack_chars

    pack = getattr(pipeline, f"pack_for_{role}")(prepared)

    assert pack.startswith(required)
    assert all(constraint in pack for constraint in constraints)
    assert len(pack) <= pipeline.maximum_pack_chars


@pytest.mark.parametrize("role", ["synthesizer", "router"])
@pytest.mark.parametrize("oversized_field", ["request", "constraints"])
def test_role_pack_rejects_oversized_required_text(tmp_path, role, oversized_field) -> None:
    pipeline = ContextPipeline(tmp_path, maximum_pack_chars=300)
    prepared = pipeline.prepare([{"role": "user", "content": "Audit"}], run_id="oversized")
    if oversized_field == "request":
        prepared.current_request = "Required request " * 100
        prepared.canonical_state.goal = prepared.current_request
    else:
        prepared.canonical_state.hard_constraints = ["Never discard this rule " * 100]

    with pytest.raises(ValueError, match=rf"(?i){role}.*required.*budget"):
        getattr(pipeline, f"pack_for_{role}")(prepared)


@pytest.mark.parametrize("role", ["synthesizer", "router"])
@pytest.mark.parametrize("spare_chars", [-1, 0, 1, 2, 3, 100])
def test_role_pack_accounts_for_separator_at_required_budget_boundary(
    tmp_path, role, spare_chars
) -> None:
    pipeline = ContextPipeline(tmp_path)
    prepared = pipeline.prepare([{"role": "user", "content": "Audit"}], run_id="boundary")
    prepared.canonical_state.goal = "Synthesize results"
    required = (
        "GOAL: Synthesize results" if role == "synthesizer" else "CURRENT REQUEST:\nAudit"
    ) + "\n\nHARD CONSTRAINTS:\n- none"
    pipeline.maximum_pack_chars = len(required) + spare_chars
    pack_for_role = getattr(pipeline, f"pack_for_{role}")

    if spare_chars < 0:
        with pytest.raises(ValueError, match=rf"(?i){role}.*required.*budget"):
            pack_for_role(prepared)
    else:
        pack = pack_for_role(prepared)
        assert pack.startswith(required)
        assert len(pack) <= pipeline.maximum_pack_chars
        if spare_chars <= 2:
            assert pack == required
        else:
            assert pack.startswith(required + "\n\n")
