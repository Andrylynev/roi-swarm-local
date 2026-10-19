"""Sleep mode merges verified work into the target branch and leaves no worktrees."""

import asyncio

import pytest
from test_improvements import approve, checks, git, proposal

pytest_plugins = ("test_improvements",)


def candidates(root):
    branches = git(root, "branch", "--list", "improvement/*")
    trees = [
        line
        for line in git(root, "worktree", "list", "--porcelain").splitlines()
        if line.startswith("worktree ")
    ]
    return branches, len(trees)


@pytest.mark.asyncio
async def test_promote_merges_into_target_branch_and_removes_candidate(repo):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    target = git(root, "branch", "--show-current")
    engine = ImprovementEngine(root, worktrees, approve, checks, target_branch=target)
    prepared = await engine.prepare(proposal())
    assert prepared["state"] == "tested", prepared
    result = await engine.promote(prepared["commit_id"])
    assert result["state"] == "promoted", result
    assert git(root, "rev-parse", target) == prepared["commit_id"]
    assert git(root, "status", "--porcelain") == ""
    assert candidates(root) == ("", 1)
    assert not (worktrees / prepared["branch"].split("/")[1]).exists()


@pytest.mark.asyncio
async def test_rejected_review_leaves_no_worktree_or_branch(repo):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo

    async def reject(diff):
        return False

    result = await ImprovementEngine(root, worktrees, reject, checks).prepare(proposal())
    assert result["state"] == "rejected_review", result
    assert candidates(root) == ("", 1)
    assert not any(p.is_dir() for p in worktrees.iterdir())


@pytest.mark.asyncio
async def test_failed_checks_leave_no_worktree_or_branch(repo):
    from roi_swarm.improvements import ImprovementEngine, _Blocked

    root, worktrees = repo

    async def failing(path):
        raise _Blocked("failed", "pytest failed (exit 1)")

    result = await ImprovementEngine(root, worktrees, approve, failing).prepare(proposal())
    assert result["state"] == "failed", result
    assert candidates(root) == ("", 1)


@pytest.mark.asyncio
async def test_other_branch_is_never_merged(repo):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    target = git(root, "branch", "--show-current")
    git(root, "switch", "-c", "feature")
    engine = ImprovementEngine(root, worktrees, approve, checks, target_branch=target)
    result = await engine.preflight()
    assert result["state"] == "blocked_branch", result
    assert (await engine.prepare(proposal()))["state"] == "blocked_branch"
    assert candidates(root) == ("", 1)


@pytest.mark.asyncio
async def test_stale_candidate_is_removed_when_head_moved(repo):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    engine = ImprovementEngine(root, worktrees, approve, checks)
    prepared = await engine.prepare(proposal())
    (root / "other.txt").write_text("x\n")
    git(root, "add", "other.txt")
    git(root, "commit", "-m", "moved")
    result = await engine.promote(prepared["commit_id"])
    assert result["state"] == "blocked_head", result
    assert candidates(root) == ("", 1)


@pytest.mark.asyncio
async def test_sweep_removes_orphans_but_not_foreign_branches(repo):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    engine = ImprovementEngine(root, worktrees, approve, checks)
    prepared = await engine.prepare(proposal())
    orphan = "improvement/" + "a" * 32
    git(root, "branch", orphan)
    git(root, "branch", "improvement/keep-me")
    git(root, "branch", "feature")
    assert await engine.sweep() == 2
    assert not (worktrees / prepared["branch"].split("/")[1]).exists()
    branches = git(root, "branch", "--list")
    assert "improvement/keep-me" in branches and "feature" in branches
    assert orphan not in branches and prepared["branch"] not in branches


@pytest.mark.asyncio
async def test_runner_restarts_service_only_when_idle(monkeypatch):
    from types import SimpleNamespace

    from roi_swarm import server
    from roi_swarm.runner import SwarmRunner

    requested = []
    monkeypatch.setattr(server, "request_restart", lambda: requested.append(1) or True)
    runner = SwarmRunner.__new__(SwarmRunner)
    runner.settings = SimpleNamespace(sleep=SimpleNamespace(auto_restart=True))
    runner._restart_task = None
    runner._active_operations = 1
    runner._tool_leases = {}
    runner._lease_recovery_hold = False
    runner._improvement_status = {"state": "promoted"}
    runner._schedule_restart()
    await asyncio.sleep(0.05)
    assert requested == []
    runner._active_operations = 0
    await asyncio.wait_for(runner._restart_task, 5)
    assert requested == [1]


@pytest.mark.asyncio
async def test_runner_does_not_restart_when_disabled(monkeypatch):
    from types import SimpleNamespace

    from roi_swarm import server
    from roi_swarm.runner import SwarmRunner

    monkeypatch.setattr(server, "request_restart", lambda: pytest.fail("restart"))
    runner = SwarmRunner.__new__(SwarmRunner)
    runner.settings = SimpleNamespace(sleep=SimpleNamespace(auto_restart=False))
    runner._restart_task = None
    runner._schedule_restart()
    assert runner._restart_task is None


def test_server_restart_request_without_server_is_refused():
    from roi_swarm import server

    assert server.request_restart() is False
