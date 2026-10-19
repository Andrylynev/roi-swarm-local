from roi_swarm.schemas import PlannedTask, WorkerResult
from roi_swarm.verifier import verify_results


def test_verifier_accepts_complete_nonempty_results() -> None:
    tasks = [
        PlannedTask(id="a", objective="one", role="researcher"),
        PlannedTask(id="b", objective="two", role="critic"),
    ]
    results = [
        WorkerResult(task_id="a", conclusion="finding one", confidence=0.8),
        WorkerResult(task_id="b", conclusion="finding two", confidence=0.7),
    ]

    report = verify_results(tasks, results)

    assert report.passed is True
    assert report.missing_task_ids == []


def test_verifier_reports_missing_tasks() -> None:
    tasks = [
        PlannedTask(id="a", objective="one", role="researcher"),
        PlannedTask(id="b", objective="two", role="critic"),
    ]
    results = [WorkerResult(task_id="a", conclusion="finding", confidence=0.8)]

    report = verify_results(tasks, results)

    assert report.passed is False
    assert report.missing_task_ids == ["b"]


def test_verifier_rejects_duplicate_task_ids() -> None:
    tasks = [
        PlannedTask(id="a", objective="one"),
        PlannedTask(id="a", objective="two"),
    ]
    results = [WorkerResult(task_id="a", conclusion="finding", confidence=0.8)]

    report = verify_results(tasks, results)

    assert report.passed is False
    assert report.duplicate_task_ids == ["a"]


def test_verifier_uses_latest_retry_result_and_rejects_duplicate_success_results() -> None:
    tasks = [PlannedTask(id="a", objective="one")]
    retry_results = [
        WorkerResult(task_id="a", conclusion="failed", confidence=0.0),
        WorkerResult(task_id="a", conclusion="fixed", confidence=0.8),
    ]

    report = verify_results(tasks, retry_results, allow_retries=True)

    assert report.passed is True
