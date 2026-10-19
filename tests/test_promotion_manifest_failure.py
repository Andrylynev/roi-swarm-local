"""Real temp Git merge: post-merge metadata failure must require a restart."""

from test_improvements import approve, checks, git, proposal

from roi_swarm.improvements import ImprovementEngine

pytest_plugins = ("test_improvements", "test_project_isolation")


async def test_manifest_write_failure_keeps_engine_and_runner_restart_required(
    repo, scoped_runner, monkeypatch
):
    root, worktrees = repo
    runner = scoped_runner
    runner.settings.sleep.auto_improve = True
    engine = ImprovementEngine(root, worktrees, approve, checks)
    runner.improvements = engine
    original_save = engine._save_manifest

    def fail_only_after_promotion(record):
        if record["state"] == "promoted":
            raise OSError("synthetic manifest disk failure")
        original_save(record)

    async def propose(snapshot):
        return proposal()

    monkeypatch.setattr(engine, "_save_manifest", fail_only_after_promotion)
    monkeypatch.setattr(runner, "_propose_improvement", propose)
    base = git(root, "rev-parse", "HEAD")

    await runner._save_sleep_lessons({"source_run_id": "manifest-failure"}, {"lessons": []})

    assert git(root, "rev-parse", "HEAD") != base
    assert (root / "src/roi_swarm/example.py").read_text() == "VALUE = 2\n"
    assert git(root, "status", "--porcelain") == ""
    assert runner.sleep_status()["auto_improve_state"] == "failed"
    assert runner.sleep_status()["restart_required"] is True
    retained = await engine.wait_for_promotion()
    assert retained["state"] == "promoted"
    assert retained["restart_required"] is True
    assert runner._improvement_status["restart_required"] is True
    assert "synthetic manifest disk failure" not in str(runner._improvement_status)
