"""Real Git + trusted checks driven by idle sleep; model replies are offline fixtures."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

from test_improvements import git, proposal

from roi_swarm.config import AppSettings
from roi_swarm.reflection import ImprovementProposal, ImprovementReview, ReflectionStep
from roi_swarm.runner import SwarmRunner
from roi_swarm.sleep import SleepManager

pytest_plugins = ("test_improvements",)


async def test_idle_sleep_reviews_commits_and_promotes_only_checked_candidate(repo, monkeypatch):
    import roi_swarm.runner as runner_module

    root, worktrees = repo
    (root / ".gitignore").write_text("data/\nruns/\n__pycache__/\n.pytest_cache/\n.ruff_cache/\n")
    git(root, "add", ".gitignore")
    git(root, "commit", "-m", "ignore runtime")
    settings = AppSettings(
        provider={"base_url": "http://example.invalid/v1", "api_key": "fixture"},
        models={r: {"model": r} for r in ("router", "planner", "worker", "critic", "synthesizer")},
        rag={"enabled": False},
        experience={"enabled": False},
        sleep={"enabled": True, "auto_improve": True},
    )
    monkeypatch.setattr(runner_module, "PROJECT_ROOT", root)
    monkeypatch.setattr(runner_module, "load_settings", lambda _: settings)
    runner = SwarmRunner(root / "config/settings.yaml")
    calls = []

    async def generate(*, schema, model, **kwargs):
        calls.append((schema.__name__, model))
        if schema is ReflectionStep:
            return ReflectionStep(observations=["fixture"], done=True)
        if schema is ImprovementProposal:
            return ImprovementProposal.model_validate(proposal())
        assert schema is ImprovementReview
        assert model == "critic"
        return ImprovementReview(approved=True)

    runner.gateway = SimpleNamespace(generate_structured=generate)
    runner.improvements.worktrees = Path(worktrees)
    manager = SleepManager(
        root / "data/sleep.sqlite",
        runner._reflect_sleep,
        runner._save_sleep_lessons,
        enabled=True,
        idle_seconds=0,
        step_delay_seconds=0.01,
        max_steps_per_cycle=1,
    )
    runner.sleep = manager
    before = git(root, "rev-parse", "HEAD")
    try:
        await manager.start()
        await manager.foreground_started()
        manager.foreground_finished({"run_id": "finished", "metrics": {"task_count": 1}})
        async with asyncio.timeout(45):
            while not runner.sleep_status()["restart_required"]:
                await asyncio.sleep(0.02)
        after = git(root, "rev-parse", "HEAD")
        assert after != before
        assert git(root, "rev-list", "--count", f"{before}..{after}") == "1"
        assert git(root, "status", "--porcelain") == ""
        assert (root / "src/roi_swarm/example.py").read_text() == "VALUE = 2\n"
        assert runner.sleep_status()["auto_improve_state"] == "promoted"
        assert calls == [
            ("ReflectionStep", "critic"),
            ("ImprovementProposal", "planner"),
            ("ImprovementReview", "critic"),
        ]
    finally:
        await runner.close()
