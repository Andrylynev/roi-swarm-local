from roi_swarm.graph import (
    _resize_tasks,
    build_retry_tasks,
    merge_tasks_for_retry,
    should_expand,
)
from roi_swarm.schemas import PlannedTask, VerificationReport


def test_resize_tasks_always_generates_globally_unique_ids() -> None:
    tasks = [
        PlannedTask(id="task-2", objective="one"),
        PlannedTask(id="task-2", objective="two"),
    ]

    resized = _resize_tasks(tasks, 2, "request")

    assert len({task.id for task in resized}) == 2


def test_failed_verification_expands_until_round_limit() -> None:
    state = {
        "round_number": 1,
        "launched_workers": 10,
        "verification": VerificationReport(
            passed=False,
            missing_task_ids=["task-1"],
        ).model_dump(),
    }

    assert should_expand(state, maximum_rounds=3, maximum_workers=70) is True
    assert should_expand({**state, "round_number": 3}, 3, 70) is False


def test_retry_tasks_include_failures_and_respect_worker_budget() -> None:
    tasks = [
        PlannedTask(id="task-1", objective="one"),
        PlannedTask(id="task-2", objective="two"),
    ]
    report = VerificationReport(
        passed=False,
        missing_task_ids=["task-1"],
        low_confidence_task_ids=["task-2"],
    )

    retries = build_retry_tasks(
        tasks,
        report,
        expansion_batch=10,
        remaining_worker_budget=1,
        round_number=2,
    )

    assert len(retries) == 1
    assert retries[0].id in {"task-1", "task-2"}


def test_retry_verification_keeps_all_original_tasks_in_scope() -> None:
    original_tasks = [
        PlannedTask(id="task-1", objective="one"),
        PlannedTask(id="task-2", objective="two"),
    ]
    retry_tasks = [PlannedTask(id="task-1", objective="one retry")]

    merged = merge_tasks_for_retry(original_tasks, retry_tasks)

    assert [task.id for task in merged] == ["task-1", "task-2"]
    assert merged[0].objective == "one retry"


def test_latest_results_keep_newest_retry_per_task() -> None:
    from roi_swarm.graph import latest_results

    results = [
        {"task_id": "a", "conclusion": "old", "confidence": 0.0},
        {"task_id": "b", "conclusion": "b", "confidence": 0.9},
        {"task_id": "a", "conclusion": "new", "confidence": 0.8},
    ]

    latest = latest_results(results)

    assert [item["conclusion"] for item in latest] == ["new", "b"]
