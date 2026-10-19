"""Switch only the local model role and queue settings while idle."""

import pytest
import yaml

from local.profiles import profile_values


def test_fast_profile_keeps_9b_lead_and_uses_2b_workers():
    data = yaml.safe_load("models: {}\nswarm: {}\nhermes_workers: {}\n")
    result = profile_values(data, "fast")
    assert result["models"]["worker"]["model"] == "bonsai-2b"
    assert result["models"]["planner"]["model"] == "bonsai-9b"
    assert result["hermes_workers"]["maximum_concurrency"] == 2
    assert data["models"] == {}


def test_specialist_profile_is_not_a_mass_worker_pool():
    result = profile_values({"models": {}, "swarm": {}, "hermes_workers": {}}, "security")
    assert result["models"]["worker"]["model"] == "bonsai-4b"
    assert result["models"]["critic"]["model"] == "whiterabbit-7b"
    assert result["hermes_workers"]["maximum_concurrency"] == 1
    assert result["swarm"]["maximum_concurrency"] == 1


def test_cpu_profile_serializes_shared_64k_workers():
    result = profile_values({}, "fast", backend="cpu")
    assert result["swarm"]["maximum_concurrency"] == 1
    assert result["hermes_workers"]["maximum_concurrency"] == 1
    assert result["models"]["worker"]["model"] == "bonsai-2b"
    with pytest.raises(ValueError, match="backend"):
        profile_values({}, "fast", backend="metal")


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="profile"):
        profile_values({}, "unknown")


def test_balanced_profile_uses_4b_workers_and_9b_text_roles():
    result = profile_values({}, "balanced")
    assert result["models"]["worker"]["model"] == "bonsai-4b"
    assert result["models"]["router"]["model"] == "bonsai-2b"
    for role in ("planner", "critic", "synthesizer"):
        assert result["models"][role]["model"] == "bonsai-9b"


@pytest.mark.parametrize(
    "source",
    [
        None,
        [],
        "profile",
        {"models": []},
        {"models": {"worker": None}},
        {"swarm": []},
        {"hermes_workers": None},
    ],
)
def test_malformed_profile_structure_is_rejected(source):
    with pytest.raises(ValueError, match="mapping"):
        profile_values(source, "fast")
