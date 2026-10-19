"""Real Git fixtures: never operate on the developer repository."""

import subprocess
from pathlib import Path

import pytest


def git(root, *args):
    return subprocess.run(
        ["git", "-c", "core.hooksPath=NUL", "-c", "core.autocrlf=false", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "config", "user.name", "Fixture")
    (root / "src/roi_swarm").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src/roi_swarm/example.py").write_text("VALUE = 1\n")
    (root / "tests/test_original.py").write_text("def test_original():\n    assert True\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "fixture")
    return root, tmp_path / "worktrees"


def proposal():
    return {
        "title": "Improve value",
        "changes": [
            {"path": "src/roi_swarm/example.py", "content": "VALUE = 2\n"},
            {"path": "tests/test_new.py", "content": "def test_new():\n    assert True\n"},
        ],
    }


async def approve(diff):
    assert "VALUE = 2" in diff
    assert "test_new" in diff
    return True


async def checks(path):
    assert (path / "src/roi_swarm/example.py").read_text() == "VALUE = 2\n"


@pytest.mark.asyncio
async def test_prepare_review_persist_and_fast_forward(repo):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    base = git(root, "rev-parse", "HEAD")
    engine = ImprovementEngine(root, worktrees, approve, checks)
    result = await engine.prepare(proposal())
    assert result["state"] == "tested", result
    assert result["base_commit"] == base
    assert git(root, "rev-parse", "HEAD") == base
    assert (root / "src/roi_swarm/example.py").read_text() == "VALUE = 1\n"
    assert Path(result["worktree"]).exists()
    # Persistence must work with a fresh engine, not only in-memory state.
    promoted = await ImprovementEngine(root, worktrees, approve, checks).promote(
        result["commit_id"]
    )
    assert promoted["state"] == "promoted", promoted
    assert git(root, "rev-parse", "HEAD") == result["commit_id"]
    assert git(root, "rev-list", "--count", f"{base}..HEAD") == "1"
    assert git(root, "status", "--porcelain") == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("dirty", ["tracked", "untracked"])
async def test_dirty_never_creates_worktrees(repo, dirty):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    name = "src/roi_swarm/example.py" if dirty == "tracked" else "untracked.txt"
    (root / name).write_text("dirty\n")
    base = git(root, "rev-parse", "HEAD")
    result = await ImprovementEngine(root, worktrees, approve, checks).prepare(proposal())
    assert result["state"] == "blocked_dirty"
    assert not worktrees.exists()
    assert git(root, "rev-parse", "HEAD") == base


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    [
        "../escape.py",
        "tests/../escape.py",
        "tests\\escape.py",
        "C:/evil.py",
        "/tmp/evil.py",
        "src/roi_swarm/security.py",
        "src/roi_swarm/sleep.py",
        "src/roi_swarm/improvements.py",
        "src/roi_swarm/runner.py",
        "tests/test_original.py",
        "tests/conftest.py",
        ".git/config",
        "config/test_new.py",
        "tests/CON.py",
        "tests/test_new.py:stream",
        "src/roi_swarm/auth/security.py",
        "src/roi_swarm/__init__.py",
        "src/roi_swarm/graph.py",
        "src/roi_swarm/gateway.py",
        "src/roi_swarm/verifier.py",
        "src/roi_swarm/model_settings.py",
        "src/roi_swarm/reflection.py",
        "src/roi_swarm/startup.py",
        "src/roi_swarm/waves.py",
        "src/roi_swarm/worktrees.py",
        "src/roi_swarm/hermes_worker_host.py",
        "src/roi_swarm/worker_install_policy.py",
        "src/roi_swarm/install_targets.py",
        "src/roi_swarm/environment_guard.py",
        "src/roi_swarm/worker_process_scope.py",
        "src/roi_swarm/orchestration/new.py",
    ],
)
async def test_pathguards_reject_before_writes(repo, name):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    request = proposal()
    request["changes"][0]["path"] = name
    result = await ImprovementEngine(root, worktrees, approve, checks).prepare(request)
    assert result["state"] == "rejected", result
    assert not worktrees.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["checks", "review", "mutate", "race"])
async def test_failure_never_commits_or_changes_main(repo, mode):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    base = git(root, "rev-parse", "HEAD")
    reviewed = []

    async def check(path):
        if mode == "checks":
            raise RuntimeError("secret should not leak")
        if mode == "mutate":
            (path / "tests/test_original.py").write_text("# weakened\n")
        if mode == "race":
            (root / "untracked.txt").write_text("race\n")

    async def review(diff):
        reviewed.append(diff)
        return mode != "review"

    result = await ImprovementEngine(root, worktrees, review, check).prepare(proposal())
    assert result["state"] != "tested", result
    assert "secret should not leak" not in str(result)
    if mode in {"checks", "mutate"}:
        assert not reviewed
    assert git(root, "rev-parse", "HEAD") == base
    for branch in git(root, "branch", "--format=%(refname:short)").splitlines():
        assert git(root, "rev-parse", branch) == base
    assert (root / "src/roi_swarm/example.py").read_text() == "VALUE = 1\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["unknown", "dirty", "head", "tamper", "paths"])
async def test_promote_rejects_unknown_dirty_or_stale(repo, mode):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    engine = ImprovementEngine(root, worktrees, approve, checks)
    result = await engine.prepare(proposal())
    assert result["state"] == "tested", result
    commit = result["commit_id"]
    if mode == "unknown":
        commit = "0" * 40
    elif mode == "dirty":
        (root / "untracked").write_text("dirty")
    elif mode == "head":
        git(root, "commit", "--allow-empty", "-m", "concurrent")
    else:
        import json

        manifest = worktrees / f"{commit}.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if mode == "paths":
            data["paths"] = []
        else:
            data["diff_digest"] = "0" * 64
        manifest.write_text(json.dumps(data), encoding="utf-8")
    base = git(root, "rev-parse", "HEAD")
    rejected = await engine.promote(commit)
    assert rejected["state"] != "promoted", rejected
    assert git(root, "rev-parse", "HEAD") == base


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["secret", "file_limit", "total_limit", "count", "no_test"])
async def test_content_bounds(repo, mode):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    request = proposal()
    if mode == "secret":
        request["changes"][0]["content"] = 'API_KEY = "«redacted:sk-…»"'
    elif mode == "file_limit":
        request["changes"][0]["content"] = "x" * 100001
    elif mode == "total_limit":
        request["changes"] = [
            {"path": f"tests/test_{i}.py", "content": "x" * 100000} for i in range(4)
        ]
    elif mode == "count":
        request["changes"] = [{"path": f"tests/test_{i}.py", "content": ""} for i in range(21)]
    else:
        request["changes"].pop()
    result = await ImprovementEngine(root, worktrees, approve, checks).prepare(request)
    assert result["state"] == "rejected", result
    assert not worktrees.exists()


@pytest.mark.asyncio
async def test_default_checks_use_async_plugin_and_worktree_env(repo, monkeypatch):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    engine = ImprovementEngine(root, worktrees, approve)
    observed = []

    async def capture(args, path):
        observed.append((args, path))
        return ""

    monkeypatch.setenv("VENDOR_API_KEY", "never-forward")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "999")
    monkeypatch.setattr(engine, "_process", capture)
    await engine._default_checks(root)
    assert observed[0][0][1:5] == ["-m", "pytest", "-q", "-p"]
    assert "pytest_asyncio.plugin" in observed[0][0]
    assert observed[1][0][1:] == ["-m", "ruff", "check", "src", "tests"]
    env = engine._env(root)
    assert "VENDOR_API_KEY" not in env
    assert "GIT_CONFIG_COUNT" not in env
    assert env["PYTHONPATH"] == str(root / "src")
    assert env["GIT_OPTIONAL_LOCKS"] == "0"


@pytest.mark.asyncio
async def test_real_default_checks(repo):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    (root / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n.ruff_cache/\n")
    git(root, "add", ".gitignore")
    git(root, "commit", "-m", "ignore caches")
    result = await ImprovementEngine(root, worktrees, approve).prepare(proposal())
    assert result["state"] == "tested", result


@pytest.mark.asyncio
async def test_configured_hooks_are_never_executed(repo):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    hooks = root.parent / "hostile_hooks"
    hooks.mkdir()
    marker = root.parent / "hook_executed"
    for name in ["post-checkout", "pre-commit", "commit-msg", "post-commit", "post-merge"]:
        hook = hooks / name
        hook.write_text(f'#!/bin/sh\ntouch "{marker.as_posix()}"\nexit 1\n')
        hook.chmod(0o755)
    git(root, "config", "core.hooksPath", str(hooks))
    engine = ImprovementEngine(root, worktrees, approve, checks)
    result = await engine.prepare(proposal())
    assert result["state"] == "tested", result
    assert (await engine.promote(result["commit_id"]))["state"] == "promoted"
    assert not marker.exists()


@pytest.mark.asyncio
async def test_symlink_parent_rejected(repo):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    outside = root.parent / "outside"
    outside.mkdir()
    link = root / "src/roi_swarm/linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Windows account does not have symlink privilege")
    git(root, "add", ".")
    git(root, "commit", "-m", "link fixture")
    request = proposal()
    request["changes"][0]["path"] = "src/roi_swarm/linked/escape.py"
    result = await ImprovementEngine(root, worktrees, approve, checks).prepare(request)
    assert result["state"] == "rejected", result
    assert not worktrees.exists()
    assert not (outside / "escape.py").exists()


@pytest.mark.asyncio
async def test_process_cancellation_is_bounded(repo):
    import asyncio
    import sys

    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    engine = ImprovementEngine(root, worktrees, approve, checks)
    task = asyncio.create_task(
        engine._process(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            root,
        )
    )
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 15)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "API_KEY=abcdefghijklmnopqrstuvwxyz012345",
        'password = "correct-horse-battery-staple"',
        "-----BEGIN " + "PRIVATE KEY-----",
    ],
)
async def test_secret_variants_rejected(repo, content):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    request = proposal()
    request["changes"][0]["content"] = content
    result = await ImprovementEngine(root, worktrees, approve, checks).prepare(request)
    assert result["state"] == "rejected", result
    assert not worktrees.exists()


@pytest.mark.asyncio
async def test_preflight_is_read_only_and_blocks_dirty_before_review(repo):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    base = git(root, "rev-parse", "HEAD")
    engine = ImprovementEngine(root, worktrees, approve, checks)
    assert await engine.preflight() == {"state": "ready", "base_commit": base}
    (root / "untracked.txt").write_text("operator change\n")
    assert (await engine.preflight())["state"] == "blocked_dirty"
    assert not worktrees.exists()
    assert git(root, "rev-parse", "HEAD") == base
    assert (root / "untracked.txt").read_text() == "operator change\n"


@pytest.mark.asyncio
async def test_default_checks_disable_ambient_dotenv_loading(repo):
    import sys

    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    engine = ImprovementEngine(root, worktrees, approve)
    # A synthetic file, not the user's .env: even explicit load_dotenv must be off.
    (root.parent / ".env").write_text("IMPROVEMENT_FIXTURE_SECRET=fixture-only\n")
    output = await engine._process(
        [
            sys.executable,
            "-c",
            "import os; from dotenv import load_dotenv; "
            "load_dotenv('../.env'); "
            "assert 'IMPROVEMENT_FIXTURE_SECRET' not in os.environ; print('disabled')",
        ],
        root,
    )
    assert output.strip() == "disabled"


@pytest.mark.asyncio
@pytest.mark.parametrize("race", ["dirty", "head", "content"])
async def test_promotion_reads_back_exact_target_after_merge(repo, monkeypatch, race):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    engine = ImprovementEngine(root, worktrees, approve, checks)
    result = await engine.prepare(proposal())
    assert result["state"] == "tested", result
    original_git = engine._git

    async def raced_git(path, *args):
        output = await original_git(path, *args)
        if path == root and args[0] == "merge":
            if race == "dirty":
                (root / "untracked.txt").write_text("concurrent\n")
            elif race == "head":
                git(root, "commit", "--allow-empty", "-m", "concurrent")
            else:
                (root / "src/roi_swarm/example.py").write_text("VALUE = 999\n")
        return output

    monkeypatch.setattr(engine, "_git", raced_git)
    promoted = await engine.promote(result["commit_id"])
    assert promoted["state"] != "promoted", promoted
    # Never undo the operator's concurrent work to manufacture a clean result.
    if race == "content":
        assert (root / "src/roi_swarm/example.py").read_text() == "VALUE = 999\n"


@pytest.mark.asyncio
async def test_prepare_does_not_certify_commit_changed_after_review(repo, monkeypatch):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    engine = ImprovementEngine(root, worktrees, approve, checks)
    original_git = engine._git

    async def mutate_commit(path, *args):
        output = await original_git(path, *args)
        if path != root and args[0] == "commit":
            (path / "src/roi_swarm/example.py").write_text("VALUE = 999\n")
            git(path, "add", ".")
            git(path, "commit", "--amend", "--no-edit")
        return output

    monkeypatch.setattr(engine, "_git", mutate_commit)
    result = await engine.prepare(proposal())
    assert result["state"] != "tested", result
    assert not list(worktrees.glob("*.json"))
    assert (root / "src/roi_swarm/example.py").read_text() == "VALUE = 1\n"


@pytest.mark.asyncio
async def test_promotion_guard_is_rechecked_at_final_merge_boundary(repo, monkeypatch):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    allowed = True
    engine = ImprovementEngine(
        root,
        worktrees,
        approve,
        checks,
        promotion_allowed=lambda: allowed,
    )
    candidate = await engine.prepare(proposal())
    assert candidate["state"] == "tested", candidate
    original_main = engine._main
    calls = 0

    async def concurrent_foreground(base=None):
        nonlocal allowed, calls
        head = await original_main(base)
        calls += 1
        # Introduce foreground activity at the very last await before merge.
        if calls == 4:
            allowed = False
        return head

    monkeypatch.setattr(engine, "_main", concurrent_foreground)
    promoted = await engine.promote(candidate["commit_id"])
    assert promoted["state"] == "paused", promoted
    assert git(root, "rev-parse", "HEAD") == candidate["base_commit"]
    assert (root / "src/roi_swarm/example.py").read_text() == "VALUE = 1\n"
    assert await engine.wait_for_promotion() is None


@pytest.mark.asyncio
async def test_per_promotion_epoch_guard_rejects_finished_foreground_race(repo, monkeypatch):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    epoch = 0
    engine = ImprovementEngine(root, worktrees, approve, checks, promotion_allowed=lambda: True)
    candidate = await engine.prepare(proposal())
    assert candidate["state"] == "tested", candidate
    original_main = engine._main

    async def foreground_finished(base=None):
        nonlocal epoch
        head = await original_main(base)
        epoch += 1
        return head

    monkeypatch.setattr(engine, "_main", foreground_finished)
    expected_epoch = epoch
    result = await engine.promote(
        candidate["commit_id"],
        allowed=lambda: epoch == expected_epoch,
    )
    assert result["state"] == "paused", result
    assert git(root, "rev-parse", "HEAD") == candidate["base_commit"]


@pytest.mark.asyncio
async def test_prepare_cancellation_kills_descendants_without_commit(repo):
    import asyncio
    import csv
    import os
    import sys

    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    ready = root.parent / "child.pid"
    base = git(root, "rev-parse", "HEAD")

    async def child_check(path):
        child = "import time; time.sleep(120)"
        parent = (
            "import pathlib, subprocess, sys, time; "
            f"p = subprocess.Popen([sys.executable, '-c', {child!r}]); "
            f"pathlib.Path({str(ready)!r}).write_text(str(p.pid)); time.sleep(120)"
        )
        await engine._process([sys.executable, "-c", parent], path)

    engine = ImprovementEngine(root, worktrees, approve, child_check)
    task = asyncio.create_task(engine.prepare(proposal()))
    try:
        async with asyncio.timeout(15):
            while not ready.exists():
                await asyncio.sleep(0.02)
        pid = int(ready.read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 15)
        if os.name == "nt":
            output = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            ).stdout
            assert not any(
                len(row) > 1 and row[1] == str(pid) for row in csv.reader(output.splitlines())
            )
        else:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        assert git(root, "rev-parse", "HEAD") == base
        assert not list(worktrees.glob("*.json"))
        for branch in git(root, "branch", "--format=%(refname:short)").splitlines():
            assert git(root, "rev-parse", branch) == base
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancellation_waits_for_final_merge_and_readback(repo, monkeypatch):
    import asyncio

    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    engine = ImprovementEngine(root, worktrees, approve, checks)
    candidate = await engine.prepare(proposal())
    assert candidate["state"] == "tested", candidate
    merging, release = asyncio.Event(), asyncio.Event()
    original_git = engine._git

    async def delayed_merge(path, *args):
        if path == root and args[0] == "merge":
            merging.set()
            await release.wait()
        return await original_git(path, *args)

    monkeypatch.setattr(engine, "_git", delayed_merge)
    promote = asyncio.create_task(engine.promote(candidate["commit_id"]))
    try:
        await asyncio.wait_for(merging.wait(), 10)
        promote.cancel()
        await asyncio.sleep(0.05)
        assert not promote.done(), "Cancellation must not interrupt a final Git merge"
        waiter = asyncio.create_task(engine.wait_for_promotion())
        await asyncio.sleep(0.05)
        assert not waiter.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(promote, 15)
        settled = await asyncio.wait_for(waiter, 15)
        assert settled == {
            "state": "promoted",
            "commit_id": candidate["commit_id"],
            "base_commit": candidate["base_commit"],
            "restart_required": True,
        }
        assert await engine.wait_for_promotion() == settled
        import json

        manifest = json.loads((worktrees / f"{candidate['commit_id']}.json").read_text())
        assert manifest["state"] == "promoted"
        assert manifest["restart_required"] is True
        assert git(root, "rev-parse", "HEAD") == candidate["commit_id"]
        assert (root / "src/roi_swarm/example.py").read_text() == "VALUE = 2\n"
        assert git(root, "status", "--porcelain") == ""
    finally:
        release.set()
        await asyncio.gather(promote, return_exceptions=True)


@pytest.mark.asyncio
async def test_tracked_dotenv_never_copied_to_worktree(repo):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    # Synthetic fixture; no main-repository secret file is accessed.
    (root / ".env").write_text("FIXTURE_ONLY=value\n")
    git(root, "add", ".env")
    git(root, "commit", "-m", "unsafe fixture")
    engine = ImprovementEngine(root, worktrees, approve, checks)
    assert (await engine.prepare(proposal()))["state"] == "rejected"
    assert not worktrees.exists()


@pytest.mark.asyncio
async def test_real_default_checks_reject_failing_new_test_without_review(repo):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    request = proposal()
    request["changes"][1]["content"] = "def test_new():\n    assert False\n"
    reviewed = []

    async def review(diff):
        reviewed.append(diff)
        return True

    base = git(root, "rev-parse", "HEAD")
    result = await ImprovementEngine(root, worktrees, review).prepare(request)
    assert result["state"] == "failed", result
    assert not reviewed
    assert git(root, "rev-parse", "HEAD") == base
    assert not list(worktrees.glob("*.json"))


@pytest.mark.asyncio
async def test_commit_identity_does_not_depend_on_operator_global_config(repo):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    git(root, "config", "--unset", "user.name")
    git(root, "config", "--unset", "user.email")
    result = await ImprovementEngine(root, worktrees, approve, checks).prepare(proposal())
    assert result["state"] == "tested", result
    assert git(root, "show", "-s", "--format=%an <%ae>", result["commit_id"]) == (
        "ROI Swarm <roi-swarm@localhost>"
    )


@pytest.mark.asyncio
async def test_documented_dotenv_template_does_not_block_preflight(repo):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    (root / ".env.example").write_text("API_KEY=\n")
    git(root, "add", ".env.example")
    git(root, "commit", "-m", "document required variable")
    result = await ImprovementEngine(root, worktrees, approve, checks).preflight()
    assert result == {"state": "ready", "base_commit": git(root, "rev-parse", "HEAD")}
    assert not worktrees.exists()


@pytest.mark.asyncio
async def test_git_symlink_mode_rejected_even_without_os_symlink_support(repo):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    git(root, "config", "core.symlinks", "false")
    target = root / "src/roi_swarm/example.py"
    target.write_text("../../../outside.py")
    blob = git(root, "hash-object", "-w", "src/roi_swarm/example.py")
    git(root, "update-index", "--cacheinfo", f"120000,{blob},src/roi_swarm/example.py")
    git(root, "commit", "-m", "symlink-mode fixture")
    assert git(root, "status", "--porcelain") == ""
    result = await ImprovementEngine(root, worktrees, approve, checks).prepare(proposal())
    assert result["state"] == "rejected", result
    assert not worktrees.exists()


@pytest.mark.asyncio
async def test_successful_merge_with_failed_readback_retains_restart_required(repo, monkeypatch):
    from roi_swarm.improvements import ImprovementEngine

    root, worktrees = repo
    engine = ImprovementEngine(root, worktrees, approve, checks)
    candidate = await engine.prepare(proposal())
    assert candidate["state"] == "tested", candidate
    original_main = engine._main

    async def failed_readback(base=None):
        if base == candidate["commit_id"]:
            raise OSError("synthetic readback failure")
        return await original_main(base)

    monkeypatch.setattr(engine, "_main", failed_readback)
    result = await engine.promote(candidate["commit_id"])
    assert result["state"] == "failed"
    assert result["restart_required"] is True
    assert git(root, "rev-parse", "HEAD") == candidate["commit_id"]
    retained = await engine.wait_for_promotion()
    assert retained["restart_required"] is True
