"""Reviewed Git improvements. Worktrees are not an OS sandbox for Python tests.

Checks use a worktree cwd and filtered environment, not filesystem or network
confinement. Hostile proposals require an externally restricted OS account.
No live restart, stash, reset, push or model-supplied shell commands are supported.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path, PurePosixPath


class _Blocked(Exception):
    def __init__(self, state: str, reason: str):
        self.state, self.reason = state, reason
        super().__init__(reason)


class ImprovementEngine:
    """Prepare one commit per proposal, then explicitly fast-forward a clean root."""

    def __init__(
        self,
        root: Path,
        worktrees: Path,
        review: Callable[[str], Awaitable[bool]],
        run_checks: Callable[[Path], Awaitable[None]] | None = None,
        promotion_allowed: Callable[[], bool] | None = None,
        target_branch: str | None = None,
    ):
        self.root = Path(root).resolve()
        # Branch verified improvements are fast-forwarded into; None accepts any.
        self.target_branch = target_branch
        self.worktrees = Path(worktrees).resolve()
        self.review = review
        self.run_checks = run_checks or self._default_checks
        self.promotion_allowed = promotion_allowed or (lambda: True)
        self._lock = asyncio.Lock()
        self._promotion: asyncio.Task[dict] | None = None
        self._last_promotion: dict | None = None
        self._git_executable = shutil.which("git") or "git"

    @staticmethod
    def _env(path: Path) -> dict[str, str]:
        # Allowlist rather than guessing every vendor's credential variable name.
        allowed = {
            "PATH",
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "PATHEXT",
            "TEMP",
            "TMP",
            "HOME",
            "USERPROFILE",
            "APPDATA",
            "LOCALAPPDATA",
            "LANG",
            "LC_ALL",
            "NUMBER_OF_PROCESSORS",
            "PROCESSOR_ARCHITECTURE",
        }
        env = {k: v for k, v in os.environ.items() if k.upper() in allowed}
        env.update(
            {
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_ATTR_NOSYSTEM": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "PYTHONPATH": str(path / "src"),
                "PYTHONNOUSERSITE": "1",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "PYTHON_DOTENV_DISABLED": "1",
            }
        )
        return env

    async def _process(self, args: list[str], path: Path) -> str:
        options = (
            {"creationflags": subprocess.CREATE_NO_WINDOW}
            if os.name == "nt"
            else {"start_new_session": True}
        )
        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=path,
            env=self._env(path),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **options,
        )
        try:
            out, _ = await asyncio.wait_for(process.communicate(), timeout=600)
        except (asyncio.CancelledError, TimeoutError):
            if os.name == "nt":
                killer = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                await asyncio.wait_for(killer.wait(), 10)
            else:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            if process.returncode is None:
                process.kill()
            await asyncio.wait_for(process.wait(), 10)
            raise
        if process.returncode:
            # Identify the trusted stage, never echo arbitrary argv or output:
            # candidate source, test failures and commit titles may contain secrets.
            module = args[2] if len(args) > 2 and args[1] == "-m" else None
            label = module if module in {"pytest", "ruff"} else "Trusted command"
            if Path(args[0]).name.casefold() in {"git", "git.exe"}:
                label = "git"
            raise _Blocked("failed", f"{label} failed (exit {process.returncode})")
        return out.decode("utf-8", errors="strict")

    async def _git(self, path: Path, *args: str) -> str:
        return await self._process(
            [
                self._git_executable,
                "-c",
                f"core.hooksPath={os.devnull}",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.autocrlf=false",
                "-c",
                "core.quotePath=false",
                "-c",
                "user.name=ROI Swarm",
                "-c",
                "user.email=roi-swarm@localhost",
                *args,
            ],
            path,
        )

    async def _default_checks(self, path: Path) -> None:
        # Keep test subprocesses on console Python, independent of the GUI host:
        # piped stdio and nested test launches must not inherit pythonw semantics.
        executable = Path(sys.executable)
        if executable.name.casefold() == "pythonw.exe":
            executable = executable.with_name("python.exe")
            if not executable.is_file():
                raise _Blocked("failed", "Console Python interpreter is unavailable")
        await self._process(
            [
                str(executable),
                "-m",
                "pytest",
                "-q",
                "-p",
                "pytest_asyncio.plugin",
            ],
            path,
        )
        await self._process([str(executable), "-m", "ruff", "check", "src", "tests"], path)

    async def _main(self, base: str | None = None) -> str:
        if (await self._git(self.root, "rev-parse", "--show-toplevel")).strip() != str(
            self.root
        ).replace("\\", "/"):
            # Git returns native slash paths on Windows.
            actual = (await self._git(self.root, "rev-parse", "--show-toplevel")).strip()
            if Path(actual).resolve() != self.root:
                raise _Blocked("rejected", "Root must be the repository top level")
        if await self._git(self.root, "status", "--porcelain", "--untracked-files=all"):
            raise _Blocked("blocked_dirty", "Основной репозиторий содержит изменения")
        if self.target_branch is not None:
            try:
                branch = (
                    await self._git(self.root, "symbolic-ref", "--quiet", "--short", "HEAD")
                ).strip()
            except _Blocked:
                branch = ""
            if branch != self.target_branch:
                raise _Blocked("blocked_branch", f"Слияние разрешено только в {self.target_branch}")
        head = (await self._git(self.root, "rev-parse", "HEAD")).strip()
        if base is not None and head != base:
            raise _Blocked("blocked_head", "HEAD изменился после начала проверки")
        # Inspect tracked names only. Ignored operator secrets are never read or
        # copied; a tracked dotenv file would otherwise enter a new worktree.
        tracked = await self._git(self.root, "ls-files", "-z")
        if any(
            PurePosixPath(name).name.casefold() not in {".env.example", ".env.sample"}
            and (
                PurePosixPath(name).name.casefold() == ".env"
                or PurePosixPath(name).name.casefold().startswith(".env.")
            )
            for name in tracked.split("\x00")
            if name
        ):
            raise _Blocked("rejected", "Отслеживаемые dotenv-файлы запрещены для проверки")
        entries = await self._git(self.root, "ls-files", "--stage", "-z")
        if any(entry.startswith(("120000 ", "160000 ")) for entry in entries.split("\x00")):
            raise _Blocked("rejected", "Git-ссылки и подмодули запрещены для проверки")
        return head

    @staticmethod
    def _secret(content: str) -> bool:
        return bool(
            re.search(
                r"(?i)(-----BEGIN [A-Z ]*PRIVATE KEY-----|"
                r"\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{16,}|"
                r"AKIA[A-Z0-9]{16})\b|"
                r"(?:api[_-]?key|secret|token|password|passwd)\s*[=:]\s*"
                r"(?:[\"'][^\"'\r\n]{8,}[\"']|[^\s\"']{8,}))",
                content,
            )
        )

    def _changes(self, proposal: dict) -> tuple[str, dict[str, str]]:
        if not isinstance(proposal, dict) or set(proposal) != {"title", "changes"}:
            raise _Blocked("rejected", "Ожидаются только title и changes")
        title, changes = proposal["title"], proposal["changes"]
        if not isinstance(title, str) or not title.strip() or self._secret(title):
            raise _Blocked("rejected", "Недопустимый заголовок")
        title = re.sub(r"[\x00-\x1f\x7f]", " ", title).strip()[:120]
        if not isinstance(changes, list) or not 1 <= len(changes) <= 20:
            raise _Blocked("rejected", "Допустимо от 1 до 20 файлов")
        result: dict[str, str] = {}
        protected = {
            "auth",
            "security",
            "sleep",
            "improvements",
            "config",
            "settings",
            "runner",
            "server",
            "startup",
            "cli",
            "api",
            "agent_loop",
            "__main__",
            "__init__",
            "conftest",
            "secrets",
            "graph",
            "gateway",
            "verifier",
            "model_settings",
            "reflection",
            "orchestration",
            "safety",
            "waves",
            "worktrees",
            "hermes_worker_host",
            "worker_install_policy",
            "install_targets",
            "environment_guard",
            "worker_process_scope",
        }
        for change in changes:
            if not isinstance(change, dict) or set(change) != {"path", "content"}:
                raise _Blocked("rejected", "Некорректная замена файла")
            name, content = change["path"], change["content"]
            if not isinstance(name, str) or not isinstance(content, str):
                raise _Blocked("rejected", "Путь и содержимое должны быть строками")
            p = PurePosixPath(name)
            if (
                not name
                or "\\" in name
                or ":" in name
                or "\x00" in name
                or p.is_absolute()
                or any(x in {"", ".", ".."} for x in name.split("/"))
                or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", x) for x in p.parts)
                or any(
                    x.lower().split(".")[0]
                    in {
                        "con",
                        "prn",
                        "aux",
                        "nul",
                        *(f"com{i}" for i in range(10)),
                        *(f"lpt{i}" for i in range(10)),
                    }
                    for x in p.parts
                )
                or any(x.endswith((".", " ")) or x.startswith(".") for x in p.parts)
                or p.suffix != ".py"
                or not (name.startswith("src/roi_swarm/") or name.startswith("tests/"))
                or any(x.lower() in protected for x in p.parts[:-1])
                or p.stem.lower() in protected
                or name.casefold() in {x.casefold() for x in result}
            ):
                raise _Blocked("rejected", "Путь запрещён политикой безопасности")
            target = self.root / name
            self._safe_path(self.root, target)
            if name.startswith("tests/") and target.exists():
                raise _Blocked("rejected", "Существующие тесты изменять запрещено")
            if len(content) > 100_000 or self._secret(content):
                raise _Blocked("rejected", "Секреты или превышение лимита содержимого")
            result[name] = content
        if sum(map(len, result.values())) > 300_000:
            raise _Blocked("rejected", "Превышен общий лимит содержимого")
        if not any(
            name.startswith("tests/") and PurePosixPath(name).name.startswith("test_")
            for name in result
        ):
            raise _Blocked("rejected", "Необходим хотя бы один новый test_*.py")
        return title, result

    @staticmethod
    def _safe_path(root: Path, target: Path) -> None:
        if not target.resolve().is_relative_to(root):
            raise _Blocked("rejected", "Путь выходит за пределы worktree")
        for component in (target, *target.parents):
            if component == root:
                break
            if component.is_symlink() or (
                hasattr(component, "is_junction") and component.is_junction()
            ):
                raise _Blocked("rejected", "Символические ссылки запрещены")

    @staticmethod
    def _digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def _diff(self, path: Path) -> str:
        return await self._git(
            path,
            "diff",
            "--cached",
            "--binary",
            "--no-ext-diff",
            "--no-textconv",
        )

    async def _unchanged(self, path: Path, base: str, digest: str) -> None:
        if (
            (await self._git(path, "rev-parse", "HEAD")).strip() != base
            or await self._git(path, "diff", "--name-only", "--no-ext-diff")
            or await self._git(path, "ls-files", "--others", "--exclude-standard")
            or self._digest(await self._diff(path)) != digest
        ):
            raise _Blocked("failed", "Проверки или рецензия изменили подготовленные файлы")

    async def preflight(self) -> dict:
        """Read-only gate before a caller spends time or credits proposing changes."""
        async with self._lock:
            try:
                return {"state": "ready", "base_commit": await self._main()}
            except _Blocked as exc:
                return {"state": exc.state, "reason": exc.reason}
            except Exception as exc:
                return {"state": "failed", "reason": f"Ошибка проверки: {type(exc).__name__}"}

    async def prepare(self, proposal: dict) -> dict:
        async with self._lock:
            try:
                return await self._prepare(proposal)
            except _Blocked as exc:
                return {"state": exc.state, "reason": exc.reason}
            except Exception as exc:
                return {"state": "failed", "reason": f"Ошибка подготовки: {type(exc).__name__}"}

    async def _prepare(self, proposal: dict) -> dict:
        base = await self._main()  # Must precede validation, mkdir and all other writes.
        title, changes = self._changes(proposal)
        if self.worktrees.is_relative_to(self.root):
            raise _Blocked("rejected", "Каталог worktree должен быть вне основного репозитория")
        self.worktrees.mkdir(parents=True, exist_ok=True)
        branch = f"improvement/{uuid.uuid4().hex}"
        path = self.worktrees / branch.split("/")[1]
        await self._git(self.root, "worktree", "add", "-b", branch, str(path), base)
        try:
            return await self._prepare_in(path, branch, base, title, changes)
        except BaseException:
            # A rejected/failed/cancelled candidate is never promoted: drop it now
            # instead of leaving an orphan worktree and branch behind.
            with suppress(Exception):
                await asyncio.shield(self._discard(branch, path))
            raise

    async def _prepare_in(
        self, path: Path, branch: str, base: str, title: str, changes: dict[str, str]
    ) -> dict:
        for name, content in changes.items():
            target = path / name
            self._safe_path(path, target)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            temporary.write_text(content, encoding="utf-8", newline="")
            temporary.replace(target)
        await self._git(path, "add", "--", *changes)
        diff = await self._diff(path)
        digest = self._digest(diff)
        await self.run_checks(path)
        await self._unchanged(path, base, digest)
        if await self.review(diff) is not True:
            raise _Blocked("rejected_review", "Независимая рецензия не одобрила изменение")
        await self._unchanged(path, base, digest)
        await self._main(base)
        await self._git(path, "commit", "-m", title, "--only", "--", *changes)
        commit = (await self._git(path, "rev-parse", "HEAD")).strip()
        committed_diff = await self._git(
            path,
            "diff",
            "--binary",
            "--no-ext-diff",
            "--no-textconv",
            base,
            commit,
        )
        if (
            self._digest(committed_diff) != digest
            or (await self._git(path, "rev-list", "--parents", "-n", "1", commit)).split()
            != [commit, base]
            or await self._git(path, "status", "--porcelain", "--untracked-files=all")
        ):
            raise _Blocked("failed", "Созданный коммит не соответствует проверенной рецензии")
        await self._main(base)
        record = {
            "state": "tested",
            "commit_id": commit,
            "base_commit": base,
            "branch": branch,
            "worktree": str(path),
            "root": str(self.root),
            "diff_digest": digest,
            "tree": (await self._git(path, "rev-parse", "HEAD^{tree}")).strip(),
            "paths": sorted(changes),
        }
        self._save_manifest(record)
        return {k: record[k] for k in ("state", "commit_id", "base_commit", "branch", "worktree")}

    def _save_manifest(self, record: dict) -> None:
        manifest = self.worktrees / f"{record['commit_id']}.json"
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        temporary.replace(manifest)

    async def promote(
        self,
        commit_id: str,
        *,
        allowed: Callable[[], bool] | None = None,
    ) -> dict:
        async with self._lock:
            try:
                result = await self._promote(commit_id, allowed)
            except _Blocked as exc:
                result = {**(self._last_promotion or {}), "state": exc.state, "reason": exc.reason}
            except Exception as exc:
                result = {
                    **(self._last_promotion or {}),
                    "state": "failed",
                    "reason": f"Ошибка продвижения: {type(exc).__name__}",
                }
            # Merged or not, a candidate is single-use: its commit is either in the
            # target branch now or stale against a moved HEAD. Remove its worktree.
            with suppress(Exception):
                await self._discard_candidate(commit_id)
            return result

    async def _discard_candidate(self, commit: str) -> None:
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
            return
        manifest = self.worktrees / f"{commit}.json"
        if not manifest.is_file() or manifest.is_symlink():
            return
        record = json.loads(manifest.read_text(encoding="utf-8"))
        await self._discard(record["branch"], Path(record["worktree"]))

    async def _discard(self, branch: str, path: Path) -> bool:
        """Remove one engine-owned candidate worktree and branch; never anything else."""
        name = branch.split("/", 1)[-1]
        if (
            not re.fullmatch(r"improvement/[0-9a-f]{32}", branch)
            or Path(path).resolve() != self.worktrees / name
        ):
            return False
        removed = False
        if Path(path).exists():
            with suppress(_Blocked):
                await self._git(self.root, "worktree", "remove", "--force", str(path))
                removed = True
        with suppress(_Blocked):
            await self._git(self.root, "worktree", "prune")
        with suppress(_Blocked):
            await self._git(self.root, "branch", "-D", branch)
            removed = True
        return removed

    async def sweep(self) -> int:
        """Drop orphaned candidates (crash/restart leftovers). Returns removed count."""
        async with self._lock:
            if self._promotion is not None:
                return 0
            with suppress(_Blocked):
                await self._git(self.root, "worktree", "prune")
            listing = await self._git(self.root, "worktree", "list", "--porcelain")
            seen: set[str] = set()
            candidates: list[tuple[str, Path]] = []
            path: Path | None = None
            for line in listing.splitlines():
                if line.startswith("worktree "):
                    path = Path(line[len("worktree ") :])
                elif line.startswith("branch refs/heads/") and path is not None:
                    branch = line[len("branch refs/heads/") :]
                    if re.fullmatch(r"improvement/[0-9a-f]{32}", branch):
                        seen.add(branch)
                        candidates.append((branch, path))
            refs = await self._git(
                self.root, "for-each-ref", "--format=%(refname:short)", "refs/heads/improvement/"
            )
            for branch in refs.split():
                if branch not in seen and re.fullmatch(r"improvement/[0-9a-f]{32}", branch):
                    candidates.append((branch, self.worktrees / branch.split("/", 1)[1]))
            removed = 0
            for branch, path in candidates:
                if await self._discard(branch, path):
                    removed += 1
            return removed

    async def _promote(self, commit: str, allowed: Callable[[], bool] | None) -> dict:
        await self._main()
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
            raise _Blocked("rejected", "Недопустимый идентификатор коммита")
        manifest = self.worktrees / f"{commit}.json"
        if not manifest.is_file() or manifest.is_symlink():
            raise _Blocked("rejected", "Неизвестный проверенный коммит")
        record = json.loads(manifest.read_text(encoding="utf-8"))
        base = record["base_commit"]
        await self._main(base)
        if (
            record["commit_id"] != commit
            or record["root"] != str(self.root)
            or record["state"] != "tested"
            or not re.fullmatch(r"improvement/[0-9a-f]{32}", record["branch"])
            or Path(record["worktree"]).resolve() != self.worktrees / record["branch"].split("/")[1]
            or (await self._git(self.root, "rev-parse", record["branch"])).strip() != commit
            or (await self._git(self.root, "rev-list", "--parents", "-n", "1", commit)).split()
            != [commit, base]
            or (await self._git(self.root, "rev-parse", f"{commit}^{{tree}}")).strip()
            != record["tree"]
        ):
            raise _Blocked("rejected", "Манифест не соответствует проверенному коммиту")
        diff = await self._git(
            self.root,
            "diff",
            "--binary",
            "--no-ext-diff",
            "--no-textconv",
            base,
            commit,
        )
        if self._digest(diff) != record["diff_digest"]:
            raise _Blocked("rejected", "Содержимое коммита не соответствует рецензии")
        committed_paths = sorted(
            filter(
                None,
                (
                    await self._git(
                        self.root,
                        "diff",
                        "--name-only",
                        "-z",
                        "--no-renames",
                        base,
                        commit,
                    )
                ).split("\x00"),
            )
        )
        if committed_paths != record["paths"] or not committed_paths:
            raise _Blocked("rejected", "Пути манифеста не соответствуют коммиту")
        await self._main(base)
        # Only the final mutation + read-back are non-cancellable. Preparation,
        # checks and review remain interruptible; the caller keeps the lock here.
        pending = asyncio.create_task(self._finish_promotion(commit, base, record, allowed))
        self._promotion = pending
        cancelled = False
        try:
            while not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError:
                    cancelled = True
            if cancelled:
                # Retrieve failures before propagating cancellation; never abandon
                # a merge subprocess or release the lock while it is still active.
                with suppress(Exception):
                    pending.result()
                raise asyncio.CancelledError
            return pending.result()
        finally:
            self._promotion = None

    async def wait_for_promotion(self) -> dict | None:
        """Settle a final merge and retain its result even if promote was cancelled."""
        pending = self._promotion
        if pending is not None:
            result = (await asyncio.shield(asyncio.gather(pending, return_exceptions=True)))[0]
            if isinstance(result, dict):
                return dict(result)
        return dict(self._last_promotion) if self._last_promotion is not None else None

    async def _finish_promotion(
        self,
        commit: str,
        base: str,
        record: dict,
        allowed: Callable[[], bool] | None,
    ) -> dict:
        await self._main(base)
        if self.promotion_allowed() is not True or (allowed is not None and allowed() is not True):
            raise _Blocked("paused", "Продвижение отложено из-за активной работы")
        # Conservatively require a restart once Git mutation is admitted, even
        # if a later read-back or manifest write fails. Never hide a changed tree.
        self._last_promotion = {
            "state": "mutation_started",
            "commit_id": commit,
            "base_commit": base,
            "restart_required": True,
        }
        await self._git(self.root, "merge", "--ff-only", "--no-edit", commit)
        # Read back the exact target, not merely the merge process exit code.
        await self._main(commit)
        if (await self._git(self.root, "rev-parse", "HEAD^{tree}")).strip() != record["tree"]:
            raise _Blocked("failed", "Дерево после продвижения не соответствует рецензии")
        for name in record["paths"]:
            target = self.root / name
            self._safe_path(self.root, target)
            expected = await self._git(self.root, "show", f"{commit}:{name}")
            if target.read_bytes() != expected.encode("utf-8"):
                raise _Blocked("failed", "Файл после продвижения не соответствует коммиту")
        await self._main(commit)
        result = {
            "state": "promoted",
            "commit_id": commit,
            "base_commit": base,
            "restart_required": True,
        }
        # Retain success before persistence as well: cancellation or an I/O failure
        # after the merge must not hide that the running service needs a restart.
        self._last_promotion = result
        record.update(state="promoted", restart_required=True)
        self._save_manifest(record)
        return result
