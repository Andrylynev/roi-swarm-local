from roi_swarm.sizing import SwarmAssessment, choose_worker_target


def test_non_swarm_task_uses_single_worker() -> None:
    assessment = SwarmAssessment(
        task_type="simple",
        complexity=0.1,
        parallelizability=0.1,
        uncertainty=0.1,
        stakes=0.1,
        swarm_required=False,
    )

    assert choose_worker_target(assessment) == 1


def test_swarm_task_never_uses_fewer_than_ten_workers() -> None:
    assessment = SwarmAssessment(
        task_type="research",
        complexity=0.1,
        parallelizability=0.8,
        uncertainty=0.1,
        stakes=0.1,
        swarm_required=True,
    )

    assert choose_worker_target(assessment) == 10


def test_high_complexity_parallel_task_can_use_seventy_workers() -> None:
    assessment = SwarmAssessment(
        task_type="research",
        complexity=1.0,
        parallelizability=1.0,
        uncertainty=1.0,
        stakes=1.0,
        swarm_required=True,
    )

    assert choose_worker_target(assessment) == 70


def test_complex_but_sequential_task_does_not_over_scale() -> None:
    assessment = SwarmAssessment(
        task_type="proof",
        complexity=1.0,
        parallelizability=0.1,
        uncertainty=0.8,
        stakes=0.8,
        swarm_required=True,
    )

    assert choose_worker_target(assessment) <= 20


def test_sizing_scales_to_two_hundred_worker_ceiling() -> None:
    assessment = SwarmAssessment(
        task_type="research",
        complexity=1.0,
        parallelizability=1.0,
        uncertainty=1.0,
        stakes=1.0,
        swarm_required=True,
    )

    assert choose_worker_target(assessment, minimum=10, maximum=200) == 200
