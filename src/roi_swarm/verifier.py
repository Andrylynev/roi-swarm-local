from __future__ import annotations

from collections import Counter

from roi_swarm.schemas import PlannedTask, VerificationReport, WorkerResult


def verify_results(
    tasks: list[PlannedTask],
    results: list[WorkerResult],
    *,
    minimum_confidence: float = 0.35,
    allow_retries: bool = False,
) -> VerificationReport:
    task_counts = Counter(task.id for task in tasks)
    result_counts = Counter(result.task_id for result in results)
    duplicate_tasks = sorted(task_id for task_id, count in task_counts.items() if count > 1)
    duplicate_results = sorted(
        task_id for task_id, count in result_counts.items() if count > 1 and not allow_retries
    )

    expected = set(task_counts)
    by_task: dict[str, WorkerResult] = {}
    for result in results:
        by_task[result.task_id] = result

    missing = sorted(expected - by_task.keys())
    empty = sorted(
        task_id
        for task_id, result in by_task.items()
        if task_id in expected and not result.conclusion.strip()
    )
    low_confidence = sorted(
        task_id
        for task_id, result in by_task.items()
        if task_id in expected and result.confidence < minimum_confidence
    )
    passed = not (
        missing
        or empty
        or low_confidence
        or duplicate_tasks
        or duplicate_results
    )
    return VerificationReport(
        passed=passed,
        missing_task_ids=missing,
        empty_task_ids=empty,
        low_confidence_task_ids=low_confidence,
        duplicate_task_ids=duplicate_tasks,
        duplicate_result_ids=duplicate_results,
    )
