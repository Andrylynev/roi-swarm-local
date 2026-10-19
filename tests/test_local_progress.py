"""Local progress must not falsely label Bonsai inference as cloud Opus."""

from roi_swarm.api import _describe_event


def test_progress_is_model_neutral():
    for node, status in [
        ("coordinator", "running"),
        ("coordinator", "delegated"),
        ("planner", "completed"),
        ("direct", "completed"),
    ]:
        text = _describe_event({"node": node, "status": status, "data": {}})
        assert "Opus" not in text


def test_wave_size_is_not_falsely_claimed_as_simultaneous_inference():
    text = _describe_event(
        {
            "node": "wave",
            "status": "running",
            "data": {
                "wave": 1,
                "task_count": 50,
            },
        }
    )
    assert "50" in text
    assert "параллельно" not in text
