"""Small local workers must finish reading with their atomic scope, not the full request."""

from roi_swarm.hermes_worker_host import build_prompt


def test_atomic_objective_is_repeated_after_the_background_goal():
    task = {
        "objective": "Write only b.txt",
        "goal": "Write a.txt b.txt c.txt d.txt",
        "context": "Own b.txt exclusively",
        "cwd": "D:/ROI local/project",
    }
    prompt = build_prompt(task)
    assert prompt.rfind("Write only b.txt") > prompt.find(task["goal"])
    assert prompt.endswith("Do not implement any other part of the overall goal.")
    assert task["cwd"] in prompt
    assert "Quote paths containing spaces" in prompt
    assert "line-number prefixes are NOT file contents" in prompt
    assert "verified:true" in prompt
    assert "stop and report success" in prompt


def test_success_requires_content_matching_the_assigned_deliverable():
    prompt = build_prompt({"objective": "Create b.txt containing exactly BETA"})
    assert "one read confirming the requested contents" in prompt
    assert "verify that the path and contents match your objective" in prompt
    assert "one successful read" not in prompt


def test_redundant_check_suppression_does_not_replace_code_verification():
    prompt = build_prompt({"objective": "Fix parser and run its regression tests"})
    assert "Only for such verified simple file creation" in prompt
    assert "For code changes and complex tasks, run the required tests" in prompt
