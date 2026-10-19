"""Per-worker git worktrees for one wave of parallel Hermes workers.

Every worker of a wave gets its own detached worktree, checked out from a snapshot
of the shared folder (committed, uncommitted and untracked-but-not-ignored files).
After the wave each successful worker's changes are applied back to the shared
folder one by one with ``git apply``; a patch that no longer applies is a conflict
and is reported to the central agent instead of being half-merged. New gitignored
files a worker produced (reports, artifacts, logs) are copied back as well.
"""

from __future__ import annotations

import filecmp
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

DEFAULT_LINKED_DIRS = (".venv", "node_modules")
# Tool caches a worker leaves behind; never worth copying back.
_NOISE_PARTS = frozenset(
    {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".hypothesis", ".tox"}
)
_NOISE_SUFFIXES = (".pyc", ".pyo")
_IDENTITY = {
    "GIT_AUTHOR_NAME": "ROI Swarm",
    "GIT_AUTHOR_EMAIL": "roi-swarm@localhost",
    "GIT_COMMITTER_NAME": "ROI Swarm",
    "GIT_COMMITTER_EMAIL": "roi-swarm@localhost",
}


class GitError(RuntimeError):
    pass


@dataclass
class MergeResult:
    status: str  # "applied" | "unchanged" | "conflict" | "discarded"
    files: list[str] = field(default_factory=list)
    detail: str = ""


def _git(
    args: list[str],
    cwd: Path,
    *,
    env: dict[str, str] | None = None,
    stdin: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    result = subprocess.run(
        ["git", "-c", "core.quotepath=off", "-c", "core.longpaths=true", *args],
        cwd=cwd,
        input=stdin,
        capture_output=True,
        env={**os.environ, **(env or {})},
        check=False,
        **options,
    )
    if check and result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise GitError(f"git {' '.join(args[:2])} failed: {message}")
    return result


def _out(result: subprocess.CompletedProcess[bytes]) -> str:
    return result.stdout.decode("utf-8", errors="replace").strip()


def _link(source: Path, target: Path) -> None:
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(source), str(target))
    else:
        target.symlink_to(source, target_is_directory=True)


def _is_junction(path: Path) -> bool:
    """Windows directory junction check that also works on Python 3.11 (no Path.is_junction)."""
    if os.name != "nt":
        return False
    try:
        tag = getattr(os.lstat(path), "st_reparse_tag", 0)
    except OSError:
        return False
    return tag == getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)


def _unlink(path: Path) -> None:
    """Remove a junction/symlink itself, never the directory it points to."""
    if path.is_symlink():
        path.unlink()
    elif _is_junction(path):
        os.rmdir(path)


def _is_noise(relative: str) -> bool:
    parts = PurePosixPath(relative).parts
    return bool(_NOISE_PARTS.intersection(parts)) or relative.endswith(_NOISE_SUFFIXES)


def _path_forms(path: Path) -> list[str]:
    """Spellings of an absolute path a model or shell may use (native, posix, git-bash)."""
    native, posix = str(path), path.as_posix()
    forms = [native, posix]
    drive = path.drive
    if len(drive) == 2 and drive[1] == ":":
        forms.append(f"/{drive[0].lower()}{posix[2:]}")
    return list(dict.fromkeys(forms))


def _rewrite(text: str, source: Path, target: Path) -> str:
    """Replace whole-path occurrences of ``source`` (``D:/ROI`` but not ``D:/ROI-backup``)."""
    for old, new in zip(_path_forms(source), _path_forms(target), strict=False):
        pattern = re.escape(old) + r"(?![\w\-]|\.\w)"
        text = re.sub(pattern, lambda _m, value=new: value, text, flags=re.IGNORECASE)
    return text


class WaveWorkspace:
    """Isolated worktrees for the workers of one streamed request."""

    def __init__(
        self,
        top: Path,
        cwd: Path,
        *,
        root: Path | None = None,
        linked_dirs: tuple[str, ...] | list[str] = DEFAULT_LINKED_DIRS,
    ) -> None:
        self.top = top
        self.relative = cwd.relative_to(top)
        self.root = (root or Path(tempfile.gettempdir()) / "roi-waves") / uuid.uuid4().hex[:12]
        # Only plain top-level folder names; "..", "a/b" or "." are ignored.
        self.linked_dirs = tuple(
            name
            for name in linked_dirs
            if name and name not in {".", ".."} and not re.search(r"[\\/:]", name)
        )
        self.base = ""
        self._trees: dict[str, Path] = {}
        self._admin: dict[str, Path] = {}
        self._links: dict[str, list[Path]] = {}
        self._ignored_written: dict[str, str] = {}  # ignored path -> task that wrote it
        self._closed = False
        self._lock = threading.Lock()

    @classmethod
    def open(
        cls,
        cwd: str,
        *,
        root: Path | None = None,
        linked_dirs: tuple[str, ...] | list[str] = DEFAULT_LINKED_DIRS,
    ) -> WaveWorkspace | None:
        """None when ``cwd`` is not inside a git work tree (workers then share it)."""
        if not cwd or not Path(cwd).is_dir() or shutil.which("git") is None:
            return None
        folder = Path(cwd).resolve()
        probe = _git(["rev-parse", "--show-toplevel"], folder, check=False)
        if probe.returncode != 0 or not _out(probe):
            return None
        top = Path(_out(probe)).resolve()
        try:
            folder.relative_to(top)
        except ValueError:
            return None
        return cls(top, folder, root=root, linked_dirs=linked_dirs)

    def _tree_of_shared(self) -> str:
        """Tree object of the shared folder now, built in a scratch index (real index untouched)."""
        with tempfile.TemporaryDirectory(prefix="roi-index-") as scratch:
            index = Path(scratch) / "index"
            real_index = Path(_out(_git(["rev-parse", "--git-path", "index"], self.top)))
            real_index = real_index if real_index.is_absolute() else self.top / real_index
            if real_index.is_file():
                # copy2 keeps the index mtime, so git's racy-entry check still re-hashes
                # files modified in the same second the index was written.
                shutil.copy2(real_index, index)
            env = {**_IDENTITY, "GIT_INDEX_FILE": str(index)}
            _git(["add", "-A", "--", "."], self.top, env=env)
            return _out(_git(["write-tree"], self.top, env=env))

    def snapshot(self) -> str:
        """Commit object of the shared folder as it is now; worktrees start from it."""
        with self._lock:
            tree = self._tree_of_shared()
            head = _git(["rev-parse", "--verify", "-q", "HEAD"], self.top, check=False)
            parents = ["-p", _out(head)] if head.returncode == 0 else []
            commit = _git(
                ["commit-tree", tree, *parents, "-m", "roi wave snapshot"],
                self.top,
                env=_IDENTITY,
            )
            self.base = _out(commit)
        return self.base

    def changed_outside(self) -> list[str]:
        """Files changed in the shared folder since the snapshot (isolation bypassed)."""
        if not self.base:
            return []
        with self._lock:
            tree = self._tree_of_shared()
            names = _git(["diff-tree", "-r", "--name-only", self.base, tree], self.top)
        return [line for line in _out(names).splitlines() if line]

    def prepare(self, task_id: str) -> Path:
        """Create the worker's worktree; returns the directory the worker should use."""
        if not self.base:
            self.snapshot()
        tree = self.root / task_id
        with self._lock:
            if self._closed:
                raise GitError("workspace already closed")
            tree.parent.mkdir(parents=True, exist_ok=True)
            _git(["worktree", "add", "--detach", "--force", str(tree), self.base], self.top)
            self._trees[task_id] = tree
            self._links[task_id] = []
            admin = Path(_out(_git(["rev-parse", "--absolute-git-dir"], tree, check=False)))
            if admin.name and admin.parent.name == "worktrees":
                self._admin[task_id] = admin
            for name in self.linked_dirs:
                source = self.top / name
                target = tree / name
                ignored = _git(["check-ignore", "-q", name], self.top, check=False)
                if source.is_dir() and not target.exists() and ignored.returncode == 0:
                    _link(source, target)
                    self._links[task_id].append(target)
        folder = tree / self.relative
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _new_ignored(self, tree: Path) -> list[str]:
        listing = _git(["ls-files", "-z", "--others", "--ignored", "--exclude-standard"], tree)
        names = listing.stdout.decode("utf-8", errors="replace").split("\0")
        return [name for name in names if name and not _is_noise(name)]

    def merge(self, task_id: str, *, keep: bool) -> MergeResult:
        """Apply the worker's changes to the shared folder, then drop its worktree."""
        tree = self._trees.get(task_id)
        if tree is None:
            return MergeResult("discarded", detail="no worktree")
        try:
            if not keep:
                return MergeResult("discarded", detail="worker failed; its changes were dropped")
            for link in self._links.get(task_id, []):
                _unlink(link)
            self._links[task_id] = []
            _git(["add", "-A", "--", "."], tree)
            names = _out(_git(["diff", "--cached", "--name-only", self.base], tree))
            tracked = [line for line in names.splitlines() if line]
            ignored = self._new_ignored(tree)
            if not tracked and not ignored:
                return MergeResult("unchanged")
            patch = b""
            if tracked:
                patch = _git(
                    ["diff", "--cached", "--binary", "--full-index", self.base], tree
                ).stdout
            with self._lock:
                copies: list[str] = []
                kept: list[str] = []
                clashes: list[str] = []
                for name in ignored:
                    source, destination = tree / name, self.top / name
                    if source.is_symlink() or _is_junction(source) or not source.is_file():
                        kept.append(name)  # links/special files are never copied back
                    elif not destination.exists() and not destination.is_symlink():
                        copies.append(name)
                    elif destination.is_file() and filecmp.cmp(source, destination, shallow=False):
                        continue  # identical content already in place
                    elif self._ignored_written.get(name, task_id) != task_id:
                        clashes.append(name)
                    else:
                        kept.append(name)  # a user file the worker never saw: never overwrite
                if clashes:
                    detail = "another worker already wrote different content to: " + ", ".join(
                        clashes[:20]
                    )
                    return MergeResult("conflict", tracked + clashes, detail)
                apply = ["apply", "--whitespace=nowarn"]
                if patch:
                    check = _git([*apply, "--check", "-"], self.top, stdin=patch, check=False)
                    if check.returncode != 0:
                        detail = check.stderr.decode("utf-8", errors="replace").strip()
                        return MergeResult("conflict", tracked + ignored, detail[:2000])
                    _git([*apply, "-"], self.top, stdin=patch)
                copied: list[str] = []
                failed: list[str] = []
                for name in copies:
                    destination = self.top / name
                    try:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(tree / name, destination)
                    except OSError as exc:
                        failed.append(f"{name} ({type(exc).__name__})")
                        continue
                    copied.append(name)
                    self._ignored_written[name] = task_id
            notes = []
            if kept:
                notes.append(
                    "existing files in the shared folder were kept, the worker's versions "
                    "were NOT copied: " + ", ".join(kept[:20])
                )
            if failed:
                notes.append("could not copy: " + ", ".join(failed[:20]))
            return MergeResult("applied", tracked + copied, "; ".join(notes))
        finally:
            self._remove(task_id)

    def _remove(self, task_id: str) -> None:
        tree = self._trees.pop(task_id, None)
        admin = self._admin.pop(task_id, None)
        for link in self._links.pop(task_id, []):
            _unlink(link)
        if tree is None:
            return
        with self._lock:
            removed = _git(["worktree", "remove", "--force", str(tree)], self.top, check=False)
            if removed.returncode != 0:
                # Drop only this worktree's own entry; never prune the repo-wide list.
                shutil.rmtree(tree, ignore_errors=True)
                if admin is not None:
                    shutil.rmtree(admin, ignore_errors=True)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            pending = list(self._trees)
        for task_id in pending:
            self._remove(task_id)
        shutil.rmtree(self.root, ignore_errors=True)

    def to_worker(self, text: str, task_id: str) -> str:
        """Point absolute paths of the shared checkout at the worker's private worktree."""
        tree = self._trees.get(task_id)
        if tree is None or not text:
            return text
        return _rewrite(text, self.top, tree)

    def to_shared(self, text: str, task_id: str) -> str:
        """Rewrite worktree paths in a worker report to the shared folder's paths."""
        tree = self._trees.get(task_id)
        if tree is None or not text:
            return text
        return _rewrite(text, tree, self.top)
