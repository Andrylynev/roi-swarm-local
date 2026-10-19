"""Real hidden Windows host; Git writes occur only in a temporary fixture repo."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_improvements import git

pytest_plugins = ("test_improvements",)


@pytest.mark.skipif(os.name != "nt", reason="Requires Windows pythonw")
async def test_pythonw_host_runs_real_checks_and_commits_fixture(repo, tmp_path):
    root, worktrees = repo
    gui = Path(sys.executable).with_name("pythonw.exe")
    assert gui.is_file()
    (root / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n.ruff_cache/\n")
    git(root, "add", ".gitignore")
    git(root, "commit", "-m", "ignore test caches")
    result_file = tmp_path / "result.json"
    source = str(Path(__file__).parents[1] / "src")
    code = """import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[4])
from roi_swarm.improvements import ImprovementEngine
async def approve(diff):
    return True
async def run():
    engine = ImprovementEngine(Path(sys.argv[1]), Path(sys.argv[2]), approve)
    proposal = {"title": "fixture", "changes": [
        {"path": "src/roi_swarm/example.py", "content": "VALUE = 2\\n"},
        {"path": "tests/test_hidden.py", "content": "def test_hidden():\\n    assert True\\n"},
    ]}
    prepared = await engine.prepare(proposal)
    promoted = None
    if prepared["state"] == "tested":
        promoted = await engine.promote(prepared["commit_id"])
    Path(sys.argv[3]).write_text(json.dumps({"prepared": prepared, "promoted": promoted}))
asyncio.run(run())
"""
    process = await asyncio.create_subprocess_exec(
        str(gui),
        "-c",
        code,
        str(root),
        str(worktrees),
        str(result_file),
        source,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        await asyncio.wait_for(process.communicate(), 60)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == 0
    result = json.loads(result_file.read_text())
    assert result["prepared"]["state"] == "tested", result
    assert result["promoted"]["state"] == "promoted", result
    assert git(root, "rev-parse", "HEAD") == result["promoted"]["commit_id"]
    assert git(root, "status", "--porcelain") == ""
