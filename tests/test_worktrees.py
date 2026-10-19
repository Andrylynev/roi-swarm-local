import subprocess
from pathlib import Path

import pytest

from roi_swarm.schemas import PlannedTask
from roi_swarm.waves import WaveOrchestrator, WavePlan, WaveReview, WorkerOutcome
from roi_swarm.worktrees import WaveWorkspace


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "t")
    _git(root, "config", "user.email", "t@t")
    (root / ".gitignore").write_text(".venv/\n")
    (root / "shared.txt").write_text("one\ntwo\nthree\n")
    (root / ".venv").mkdir()
    (root / ".venv" / "marker").write_text("env")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "init")
    return root


def test_worktree_sees_uncommitted_state_and_merges_back(repo, tmp_path) -> None:
    (repo / "draft.txt").write_text("untracked draft\n")
    (repo / "shared.txt").write_text("one\nTWO\nthree\n")
    space = WaveWorkspace.open(str(repo), root=tmp_path / "trees")
    assert space is not None
    space.snapshot()
    folder = space.prepare("a")
    assert folder != repo
    assert (folder / "draft.txt").read_text() == "untracked draft\n"
    assert (folder / "shared.txt").read_text() == "one\nTWO\nthree\n"
    assert (folder / ".venv" / "marker").read_text() == "env"
    (folder / "new.py").write_text("x = 1\n")
    (folder / "draft.txt").write_text("edited\n")

    result = space.merge("a", keep=True)

    assert result.status == "applied"
    assert sorted(result.files) == ["draft.txt", "new.py"]
    assert (repo / "new.py").read_text().replace("\r\n", "\n") == "x = 1\n"
    assert (repo / "draft.txt").read_text().replace("\r\n", "\n") == "edited\n"
    assert (repo / ".venv" / "marker").read_text() == "env"  # linked dir survives cleanup
    assert not folder.exists()
    assert _git(repo, "log", "--oneline").count("\n") == 0  # nothing committed in the repo
    space.close()
    assert "trees" not in _git(repo, "worktree", "list")


def test_conflicting_workers_second_is_rejected(repo, tmp_path) -> None:
    space = WaveWorkspace.open(str(repo), root=tmp_path / "trees")
    assert space is not None
    space.snapshot()
    first, second = space.prepare("a"), space.prepare("b")
    (first / "shared.txt").write_text("one\nA\nthree\n")
    (second / "shared.txt").write_text("one\nB\nthree\n")

    assert space.merge("a", keep=True).status == "applied"
    clash = space.merge("b", keep=True)

    assert clash.status == "conflict" and clash.files == ["shared.txt"]
    assert (repo / "shared.txt").read_text().replace("\r\n", "\n") == "one\nA\nthree\n"
    space.close()


def test_failed_worker_changes_are_discarded(repo, tmp_path) -> None:
    space = WaveWorkspace.open(str(repo), root=tmp_path / "trees")
    assert space is not None
    folder = space.prepare("a")
    (folder / "junk.txt").write_text("half done")
    assert space.merge("a", keep=False).status == "discarded"
    assert not (repo / "junk.txt").exists()
    space.close()


def test_non_git_folder_has_no_workspace(tmp_path) -> None:
    assert WaveWorkspace.open(str(tmp_path)) is None
    assert WaveWorkspace.open("") is None


def test_paths_are_mapped_between_shared_folder_and_worktree(repo, tmp_path) -> None:
    space = WaveWorkspace.open(str(repo), root=tmp_path / "trees")
    assert space is not None
    space.prepare("a")
    text = f"edit {repo / 'shared.txt'} and {repo.as_posix()}/new.py"
    private = space.to_worker(text, "a")
    assert str(repo) not in private and repo.as_posix() not in private
    assert space.to_shared(private, "a") == text
    space.merge("a", keep=False)
    space.close()


async def test_wave_runs_each_worker_in_own_worktree_and_reports_conflicts(repo, tmp_path) -> None:
    plan = WavePlan(
        summary="two",
        tasks=[
            PlannedTask(id="a", objective="edit shared.txt", context=f"file {repo}/shared.txt"),
            PlannedTask(id="b", objective="also edit shared.txt"),
        ],
    )
    reviews: list[str] = []

    async def structured(system, prompt, schema):
        if schema is WavePlan:
            return plan
        reviews.append(prompt)
        return WaveReview(done=True, accepted=["a"])

    async def finalize(system, prompt, sink):
        return "ok"

    seen: dict[str, tuple[str, str]] = {}

    async def worker(task, *, goal, cwd, on_event):
        seen[task.id] = (cwd, task.context)
        Path(cwd, "shared.txt").write_text(f"one\n{task.id}\nthree\n")
        return WorkerOutcome(task.id, task.objective, True, f"wrote {cwd}", 1, ["write_file"])

    events: list[tuple[str, str, dict]] = []
    orchestrator = WaveOrchestrator(
        structured=structured,
        finalize=finalize,
        run_worker=worker,
        maximum_waves=2,
        on_event=lambda node, status, data: events.append((node, status, data)),
        open_workspace=lambda folder: WaveWorkspace.open(folder, root=tmp_path / "trees"),
    )

    await orchestrator.run([], request="edit", cwd=str(repo))

    cwd_a, context_a = seen["a"]
    assert cwd_a != seen["b"][0] and Path(cwd_a).resolve() != repo.resolve()
    assert str(repo) not in context_a.split("You work in")[0]
    assert (repo / "shared.txt").read_text().replace("\r\n", "\n") == "one\na\nthree\n"
    assert ("merge:a", "merged") in [(n, s) for n, s, _ in events]
    assert ("merge:b", "conflict") in [(n, s) for n, s, _ in events]
    statuses = {n: s for n, s, _ in events if n.startswith("worker:")}
    assert statuses == {"worker:a": "completed", "worker:b": "failed"}
    assert "## b (FAILED" in reviews[0] and "MERGE CONFLICT" in reviews[0]
    assert "OBJECTIVE: edit shared.txt" in reviews[0]
    assert str(tmp_path / "trees") not in reviews[0]
    assert "trees" not in _git(repo, "worktree", "list")


async def test_single_task_wave_uses_shared_folder(repo, tmp_path) -> None:
    async def structured(system, prompt, schema):
        if schema is WavePlan:
            return WavePlan(summary="one", tasks=[PlannedTask(id="a", objective="x")])
        return WaveReview(done=True, accepted=["a"])

    async def finalize(system, prompt, sink):
        return "ok"

    seen: list[str] = []

    async def worker(task, *, goal, cwd, on_event):
        seen.append(cwd)
        return WorkerOutcome(task.id, task.objective, True, "done")

    await WaveOrchestrator(
        structured=structured,
        finalize=finalize,
        run_worker=worker,
        maximum_waves=1,
        on_event=lambda *_: None,
        open_workspace=lambda folder: WaveWorkspace.open(folder, root=tmp_path / "trees"),
    ).run([], request="x", cwd=str(repo))

    assert seen == [str(repo)]


def test_new_ignored_outputs_are_copied_back_but_caches_are_not(repo, tmp_path) -> None:
    (repo / ".gitignore").write_text(".venv/\nout/\n*.log\n__pycache__/\n")
    space = WaveWorkspace.open(str(repo), root=tmp_path / "trees")
    assert space is not None
    folder = space.prepare("a")
    (folder / "out").mkdir()
    (folder / "out" / "report.xlsx").write_bytes(b"\x00xlsx")
    (folder / "run.log").write_text("log")
    (folder / "__pycache__").mkdir()
    (folder / "__pycache__" / "m.pyc").write_bytes(b"pyc")

    result = space.merge("a", keep=True)

    assert result.status == "applied"
    assert sorted(result.files) == ["out/report.xlsx", "run.log"]
    assert (repo / "out" / "report.xlsx").read_bytes() == b"\x00xlsx"
    assert not (repo / "__pycache__").exists()
    assert (repo / ".venv" / "marker").read_text() == "env"
    space.close()


def test_existing_ignored_user_files_are_never_overwritten(repo, tmp_path) -> None:
    (repo / ".gitignore").write_text(".venv/\n.env\ndata/\n")
    (repo / ".env").write_text("SECRET=real")
    (repo / "data").mkdir()
    (repo / "data" / "state.db").write_text("REAL USER DATA")
    space = WaveWorkspace.open(str(repo), root=tmp_path / "trees")
    assert space is not None
    folder = space.prepare("a")
    (folder / ".env").write_text("SECRET=placeholder")
    (folder / "data").mkdir()
    (folder / "data" / "state.db").write_text("fresh")
    (folder / "data" / "new.txt").write_text("new")

    result = space.merge("a", keep=True)

    assert result.status == "applied"
    assert result.files == ["data/new.txt"]
    assert ".env" in result.detail and "data/state.db" in result.detail
    assert (repo / ".env").read_text() == "SECRET=real"
    assert (repo / "data" / "state.db").read_text() == "REAL USER DATA"
    assert (repo / "data" / "new.txt").read_text() == "new"
    space.close()


def test_two_workers_writing_different_ignored_file_conflict(repo, tmp_path) -> None:
    (repo / ".gitignore").write_text(".venv/\nout/\n")
    space = WaveWorkspace.open(str(repo), root=tmp_path / "trees")
    assert space is not None
    first, second = space.prepare("a"), space.prepare("b")
    for folder, body in ((first, "A"), (second, "B")):
        (folder / "out").mkdir()
        (folder / "out" / "r.txt").write_text(body)
    assert space.merge("a", keep=True).status == "applied"
    assert space.merge("b", keep=True).status == "conflict"
    assert (repo / "out" / "r.txt").read_text() == "A"
    space.close()


def test_path_rewrite_respects_path_boundaries(repo, tmp_path) -> None:
    space = WaveWorkspace.open(str(repo), root=tmp_path / "trees")
    assert space is not None
    tree = space.prepare("a")
    sibling = f"{repo}-backup/f and {repo}2\\x and {repo}.bak/y"
    assert space.to_worker(sibling, "a") == sibling
    assert space.to_worker(f"{repo}", "a") != f"{repo}"
    drive = repo.drive
    if len(drive) == 2:
        bash = f"/{drive[0].lower()}{repo.as_posix()[2:]}/shared.txt"
        assert repo.as_posix()[2:] not in space.to_worker(bash, "a").replace(
            tree.as_posix()[2:], ""
        )
    space.merge("a", keep=False)
    space.close()


def test_direct_writes_to_shared_folder_are_detected(repo, tmp_path) -> None:
    space = WaveWorkspace.open(str(repo), root=tmp_path / "trees")
    assert space is not None
    space.snapshot()
    assert space.changed_outside() == []
    (repo / "bypass.txt").write_text("x")
    assert space.changed_outside() == ["bypass.txt"]
    space.close()


def test_linked_dirs_reject_parent_and_nested_names(repo, tmp_path) -> None:
    space = WaveWorkspace.open(
        str(repo), root=tmp_path / "trees", linked_dirs=["..", ".", "a/b", "c:\\x", ".venv"]
    )
    assert space is not None and space.linked_dirs == (".venv",)


async def test_prepare_failure_falls_back_to_shared_folder(repo, tmp_path) -> None:
    async def structured(system, prompt, schema):
        if schema is WavePlan:
            tasks = [PlannedTask(id="a", objective="x"), PlannedTask(id="b", objective="y")]
            return WavePlan(summary="two", tasks=tasks)
        return WaveReview(done=True, accepted=["a", "b"])

    async def finalize(system, prompt, sink):
        return "ok"

    seen: list[str] = []

    async def worker(task, *, goal, cwd, on_event):
        seen.append(cwd)
        return WorkerOutcome(task.id, task.objective, True, "done")

    class Broken(WaveWorkspace):
        def prepare(self, task_id: str) -> Path:
            raise OSError("path too long")

    events: list[tuple[str, str, dict]] = []
    await WaveOrchestrator(
        structured=structured,
        finalize=finalize,
        run_worker=worker,
        maximum_waves=1,
        on_event=lambda node, status, data: events.append((node, status, data)),
        open_workspace=lambda folder: Broken(repo, repo, root=tmp_path / "trees"),
    ).run([], request="x", cwd=str(repo))

    assert seen == [str(repo), str(repo)]
    assert any(n == "wave" and s == "isolation" and "error" in d for n, s, d in events)
