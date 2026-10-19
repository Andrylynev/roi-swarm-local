"""Streaming wave orchestration with full Hermes agents as workers.

One user request = one open stream. The central agent (Opus) plans atomic tasks, a wave
of Hermes workers executes them in parallel (each worker runs its own tool loop inside
its subprocess and never creates new tasks), the central agent reviews every atomic
result and either launches another wave for the rejected/missing parts or, once the
task is done, streams the final summary and the stream closes.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
from collections.abc import Awaitable, Callable
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from roi_swarm.environment_guard import EnvironmentGuard
from roi_swarm.install_targets import target_error
from roi_swarm.schemas import PlannedTask
from roi_swarm.worker_install_policy import install_error
from roi_swarm.worker_process_scope import WorkerProcessScope
from roi_swarm.worktrees import MergeResult, WaveWorkspace

HOST_SCRIPT = Path(__file__).with_name("hermes_worker_host.py")
STREAM_LIMIT = 64 * 1024 * 1024
# A worker host that already reported its answer gets this long to exit before it is killed.
EXIT_GRACE_SECONDS = 30
# ROI's own credentials never reach worker shells (the worker key is passed explicitly).
SECRET_ENV_PREFIXES = ("ROI_SWARM_", "ROI_PROVIDER_", "ROI_WORKER_")
_CWD_PATTERN = re.compile(r"(?:Current working directory|Root):\s*(\S[^\r\n]*)")

EventSink = Callable[[str, str, dict[str, Any]], None]
DeltaSink = Callable[[str], None]


class WavePlan(BaseModel):
    summary: str
    # Absolute project folder the request is about; empty = the Hermes working directory.
    workdir: str = ""
    tasks: list[PlannedTask] = Field(default_factory=list)


class WaveReview(BaseModel):
    done: bool
    accepted: list[str] = Field(default_factory=list)
    feedback: str = ""
    next_tasks: list[PlannedTask] = Field(default_factory=list)


@dataclass
class WorkerOutcome:
    task_id: str
    objective: str
    ok: bool
    report: str
    tool_calls: int = 0
    tools: list[str] = field(default_factory=list)


PLAN_SYSTEM = (
    "You are the central agent of the ROI swarm. Workers are full Hermes agents with file, "
    "terminal, web and code tools that run IN PARALLEL. Split the user's CURRENT request "
    "into ATOMIC tasks: one concrete deliverable each, self-contained (put every fact, path "
    "and constraint the worker needs into 'context'), non-overlapping, jointly covering the "
    "request. Every worker runs in its OWN git worktree (a private copy of the folder); "
    "after the wave their changes are merged back, and two workers editing the same lines "
    "is a merge conflict that rejects the later one. So give each task exclusive ownership "
    "of the files it may change and name them in 'context'. Never pad the plan with generic "
    "tasks; a task that needs another task's output belongs in a later wave, not this one. "
    "Prefer MANY small parallel tasks over one big task: every independent file, module, "
    "document or check is its own task. Set 'workdir' to the absolute project folder the "
    "request is about when the user names one (it may differ from the Hermes working "
    "directory); otherwise leave it empty."
)

SPLIT_SYSTEM = (
    "You are the central agent of the ROI swarm. The work below remains after a review. "
    "Split it into ATOMIC tasks that independent Hermes workers can run IN PARALLEL in this "
    "wave: one concrete deliverable each, self-contained context (paths, facts, what to "
    "fix), exclusive file ownership, non-overlapping, jointly covering ALL remaining work. "
    "Do not merge unrelated fixes into one task. Parts that depend on each other's output "
    "must NOT be split into the same wave: then return only the parts that can start now, "
    "or an empty task list when the work truly cannot be divided. Leave 'workdir' empty."
)

REVIEW_SYSTEM = (
    "You are the central agent of the ROI swarm reviewing one wave of atomic worker "
    "results against the user's request. Accept a task id only when its report shows the "
    "deliverable was actually produced and verified. For rejected or still-missing parts "
    "create next_tasks: SEPARATE atomic tasks that run in parallel, one per independent "
    "deliverable, each with full context and what to fix. Never bundle several remaining "
    "parts into one task. Set done=true only when the whole request is satisfied and "
    "next_tasks is empty."
)

FINAL_SYSTEM = (
    "You are the central agent of the ROI swarm. The work is finished. Write the final "
    "answer to the user in the user's language: what was done, files changed, how it was "
    "verified, and what remains open. Be concise and only report facts from the reports."
)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "") if isinstance(part, dict) else str(part) for part in content
        )
    return "" if content is None else json.dumps(content, ensure_ascii=False)


def render_transcript(messages: list[dict[str, Any]]) -> str:
    """Hermes conversation as plain text (tool rows would need matching tool schemas)."""
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        body = _text(message.get("content"))
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            body += f"\n[tool call {function.get('name')}: {function.get('arguments')}]"
        lines.append(f"### {role.upper()}\n{body.strip()}")
    return "\n\n".join(lines)


_DRIVE_PATH = re.compile(r"(?<![\w])[A-Za-z]:[\\/]|(?:^|(?<=[\s\"'`(]))[\\/](?![\\/])")
_PATH_TAIL = ".,;:!?)]}»\"'`"
# Longest path prefix tried per match; bounds filesystem probes on long pasted lines.
_PATH_SCAN_CHARS = 400


def _system_folders() -> list[Path]:
    names = ("WINDIR", "SystemRoot", "ProgramFiles", "ProgramFiles(x86)", "ProgramData")
    return [Path(os.environ[name]).resolve() for name in names if os.environ.get(name)]


_LINE_REF = re.compile(r"(?::\d+){1,2}$")


def _spellings(candidate: str) -> list[str]:
    """The text as typed, without trailing punctuation, and without a ``:line`` suffix."""
    raw = candidate.strip()
    bare = raw.rstrip(_PATH_TAIL).strip()
    options = [raw, bare, _LINE_REF.sub("", bare)]
    return [o for i, o in enumerate(options) if o and o not in options[:i]]


def _project_folder(candidate: str, *, files: bool = True) -> str | None:
    """Absolute existing folder for a named dir (or file); never a drive root or OS folder."""
    for text in _spellings(candidate):
        folder = _existing_folder(text, files=files)
        if folder:
            return folder
    return None


def _existing_folder(text: str, *, files: bool) -> str | None:
    # Accept Windows-style separators in pasted POSIX paths as well.
    path = Path(text if os.name == "nt" else text.replace("\\", "/"))
    if not path.is_absolute():
        return None
    try:
        if files and path.is_file():
            path = path.parent
        if not path.is_dir():
            return None
        resolved = path.resolve()
    except OSError:
        return None
    if resolved.parent == resolved:
        return None
    for system in _system_folders():
        if resolved == system or resolved.is_relative_to(system):
            return None
    return str(resolved)


def _named_folder(text: str, *, files: bool = True) -> str | None:
    for match in _DRIVE_PATH.finditer(text):
        line = text[match.start() : match.start() + _PATH_SCAN_CHARS].splitlines()[0]
        # Paths may contain spaces: try the longest whitespace-delimited prefix first.
        cuts = [m.start() for m in re.finditer(r"\s+", line)] + [len(line)]
        for end in reversed(cuts):
            folder = _project_folder(line[:end], files=files)
            if folder:
                return folder
    return None


def project_directory(planned: str, texts: str | list[str], default: str) -> str:
    """Folder the workers operate in: planner choice, else a folder the user named.

    ``texts`` is the current request first, then earlier user turns newest first, so a
    follow-up ("continue") keeps the project named before. Earlier turns count only when
    they name a folder: a file mentioned in passing (a log, ``~/.ssh/known_hosts``) must
    not redirect every later wave. Hermes' own working directory
    is only the fallback: users often run Hermes from one project and ask for work in
    another, and isolating the wrong repository makes every worker report "project files
    are missing".
    """
    folder = _project_folder(planned)
    if folder:
        return folder
    for index, text in enumerate([texts] if isinstance(texts, str) else texts):
        folder = _named_folder(text, files=index == 0)
        if folder:
            return folder
    return default


def working_directory(messages: list[dict[str, Any]], default: str = "") -> str:
    for message in messages:
        if message.get("role") in {"system", "developer"}:
            match = _CWD_PATTERN.search(_text(message.get("content")))
            if match and Path(match.group(1).strip()).is_dir():
                return match.group(1).strip()
    return default


# Install output handed back to the worker: the tail is what matters (errors, summary).
INSTALL_OUTPUT_CHARS = 100_000
InstallRunner = Callable[[str, str], Awaitable[dict[str, Any]]]


def _worker_env() -> dict[str, str]:
    return {
        name: value
        for name, value in os.environ.items()
        if not name.upper().startswith(SECRET_ENV_PREFIXES)
    }


def _install_env(cwd: str) -> dict[str, str]:
    env = _worker_env()
    for name in (
        "PYTHONHOME",
        "PIP_TARGET",
        "PIP_PREFIX",
        "PIP_USER",
        "UV_PROJECT_ENVIRONMENT",
        "UV_PYTHON",
        "UV_SYSTEM_PYTHON",
        "CONDA_PREFIX",
    ):
        env.pop(name, None)
    inherited = env.pop("VIRTUAL_ENV", None)
    entries = env.get("PATH", "").split(os.pathsep)
    if inherited:
        old = Path(inherited).resolve()
        entries = [
            item
            for item in entries
            if item and Path(item).resolve() not in {old / "Scripts", old / "bin"}
        ]
    root = Path(cwd or os.getcwd()).resolve()
    for parent in [root, *root.parents]:
        venv = parent / ".venv"
        scripts = venv / ("Scripts" if os.name == "nt" else "bin")
        if (venv / "pyvenv.cfg").is_file() and (
            scripts / ("python.exe" if os.name == "nt" else "python")
        ).is_file():
            entries.insert(0, str(scripts))
            env["VIRTUAL_ENV"] = str(venv)
            # uv sync must not silently select a different interpreter from a
            # .python-version file and replace the live shared environment.
            env["UV_PYTHON"] = str(scripts / ("python.exe" if os.name == "nt" else "python"))
            env["UV_PYTHON_DOWNLOADS"] = "never"
            break
    env["PATH"] = os.pathsep.join(entries)
    # Environment config outranks .npmrc. Do not inherit a global npm destination.
    for name in list(env):
        if name.lower().startswith("npm_config_"):
            del env[name]
    env.update(
        {
            "npm_config_global": "false",
            "npm_config_location": "project",
            "npm_config_prefix": str(root),
        }
    )
    env["PYTHONIOENCODING"] = "utf-8"
    return env


async def run_install_command(command: str, cwd: str) -> dict[str, Any]:
    """Run one install command in the SHARED project with the shell Hermes workers use."""
    bash = os.environ.get("HERMES_GIT_BASH_PATH") or shutil.which("bash")
    if bash and Path(bash).is_file():
        argv = [bash, "-c", command]
    elif os.name == "nt":
        argv = ["cmd", "/c", command]
    else:
        argv = ["/bin/sh", "-c", command]
    options: dict[str, Any] = (
        {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    )
    env = _install_env(cwd)
    # This function also runs harmless smoke-test commands; policy is enforced only
    # for recognized mutations, while the broker validates all worker requests.
    from roi_swarm.worker_install_policy import is_install

    if is_install(command):
        error = install_error(command) or target_error(
            command, env.get("VIRTUAL_ENV"), env["PATH"], cwd=cwd
        )
        if error:
            return {"output": "", "exit_code": -1, "error": error}
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd or None,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
        **options,
    )
    try:
        raw, _ = await process.communicate()
    except BaseException:
        await _kill(process)
        raise
    output = raw.decode("utf-8", errors="replace")
    if len(output) > INSTALL_OUTPUT_CHARS:
        output = "…" + output[-INSTALL_OUTPUT_CHARS:]
    return {"output": output, "exit_code": process.returncode, "error": None}


class InstallBroker:
    """The central agent owns package installs; tools of other workers pause meanwhile.

    Each worker holds the gate while its tool calls run. Nested install requests are
    refused; standalone installs wait for all active tools and prevent new calls until
    their serial queue is empty. Cancellation settles any already-started mutation.
    """

    def __init__(self, run: InstallRunner | None = None, on_event: EventSink | None = None):
        self._run = run or run_install_command
        self._on_event = on_event or (lambda *_: None)
        self._holders: set[str] = set()
        self._pending_installs: dict[str, int] = {}
        self._condition = asyncio.Condition()
        self._serial = asyncio.Lock()
        self._environments: dict[Path, EnvironmentGuard] = {}
        self._environment_error: list[str] = []

    async def watch(self, folder: str) -> None:
        async with self._serial:
            root = Path(folder).resolve()
            # A named source file may have selected src/pkg as cwd; its linked
            # dependencies live at the enclosing project root, not beside the file.
            for parent in [root, *root.parents]:
                if any((parent / name).exists() for name in (".venv", "node_modules", ".git")):
                    root = parent
                    break
            if root not in self._environments:
                guard = EnvironmentGuard(root)
                await asyncio.to_thread(guard.capture)
                self._environments[root] = guard
            await self._check_environments()

    async def _check_environments(self) -> list[str]:
        problems = set(self._environment_error)
        for guard in self._environments.values():
            try:
                changed = await asyncio.to_thread(guard.check)
                broken = await asyncio.to_thread(guard.broken_editables)
                problems.update(changed + broken)
            except OSError as exc:
                problems.add(f"environment check failed: {type(exc).__name__}")
        # Threaded scans yielded the loop: process cleanup may have quarantined us
        # meanwhile. Merge under the same condition used by admission/quarantine.
        async with self._condition:
            self._environment_error = sorted(problems | set(self._environment_error))
            self._condition.notify_all()
            return list(self._environment_error)

    async def check_environment(self) -> list[str]:
        async with self._serial:
            return await self._check_environments()

    async def _install_checked(self, command: str, cwd: str) -> dict[str, Any]:
        result = await self._run(command, cwd)
        # The mutation AND its baseline update are one cancellation-shielded unit.
        for guard in self._environments.values():
            try:
                broken = await asyncio.to_thread(guard.broken_editables)
                if broken:
                    self._environment_error = sorted(set(self._environment_error + broken))
                else:
                    await asyncio.to_thread(guard.capture)
            except OSError as exc:
                self._environment_error.append(f"environment check failed: {type(exc).__name__}")
        if self._environment_error:
            return {
                **result,
                "exit_code": -1,
                "error": "Shared environment has broken editable paths; repair required",
            }
        return result

    async def quarantine(self, reason: str) -> None:
        async with self._condition:
            self._environment_error.append(reason)
            self._condition.notify_all()

    async def enter(self, worker: str) -> None:
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._environment_error or not self._pending_installs
            )
            if self._environment_error:
                raise RuntimeError(
                    "Shared environment changed outside central installation; repair required"
                )
            self._holders.add(worker)

    async def leave(self, worker: str) -> None:
        async with self._condition:
            self._holders.discard(worker)
            self._condition.notify_all()

    async def install(
        self, worker: str, command: str, cwd: str, *, on_event: EventSink | None = None
    ) -> dict[str, Any]:
        def emit(status: str, **data: Any) -> None:
            # Telemetry failure must not strand an acquired installation lock.
            with suppress(Exception):
                (on_event or self._on_event)(f"install:{worker}", status, data)

        policy_error = install_error(command)
        if policy_error:
            return {"output": "", "exit_code": -1, "error": policy_error}
        # Refuse nested requests instead of waiting for the caller's own tool. This
        # breaks the A-held-gate -> B-install -> A-install cycle without mutating the
        # environment while A's other tool threads are still using it.
        async with self._condition:
            if worker in self._holders:
                return {
                    "output": "",
                    "exit_code": -1,
                    "error": "Run package installation as a standalone foreground terminal "
                    "call, not from inside execute_code or another running tool.",
                }
            self._pending_installs[worker] = self._pending_installs.get(worker, 0) + 1
            self._condition.notify_all()
            waiting = sorted(self._holders - self._pending_installs.keys())
        emit("waiting", waiting_for=waiting, command=command[:500])
        try:
            async with self._serial:
                async with self._condition:
                    await self._condition.wait_for(
                        lambda: self._environment_error or not self._holders
                    )
                problems = await self._check_environments()
                if problems:
                    return {
                        "output": "",
                        "exit_code": -1,
                        "error": "Shared environment changed "
                        "outside central installation; repair required",
                    }
                emit("running", command=command[:500], cwd=cwd)
                job = asyncio.create_task(self._install_checked(command, cwd))
                cancelled = False
                try:
                    # Once mutation starts, settle it even if the request is cancelled.
                    # Killing pip/npm halfway leaves the shared environment inconsistent.
                    while not job.done():
                        try:
                            await asyncio.shield(job)
                        except asyncio.CancelledError:
                            if job.cancelled():
                                raise
                            cancelled = True
                    result = job.result()
                except Exception as exc:  # noqa: BLE001 - worker receives tool error
                    result = {
                        "output": "",
                        "exit_code": -1,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ok = result.get("exit_code") == 0 and not result.get("error")
                emit(
                    "completed" if ok else "failed",
                    command=command[:500],
                    exit_code=result.get("exit_code"),
                )
                if cancelled:
                    raise asyncio.CancelledError
                return result
        finally:
            async with self._condition:
                count = self._pending_installs[worker] - 1
                if count:
                    self._pending_installs[worker] = count
                else:
                    del self._pending_installs[worker]
                self._condition.notify_all()


class HermesWorkerPool:
    """Runs each atomic task in its own Hermes agent subprocess."""

    def __init__(
        self,
        *,
        python: str,
        hermes_root: str,
        base_url: str,
        api_key: str,
        api_mode: str,
        model: str,
        disabled_toolsets: list[str],
        max_iterations: int,
        max_output_tokens: int | None = None,
    ) -> None:
        self.python = python
        self.hermes_root = hermes_root
        self.base_url = base_url
        self.api_key = api_key
        self.api_mode = api_mode
        self.model = model
        self.disabled_toolsets = disabled_toolsets
        self.max_iterations = max_iterations
        self.max_output_tokens = max_output_tokens

    async def run(
        self,
        task: PlannedTask,
        *,
        goal: str,
        cwd: str,
        on_event: Callable[[dict[str, Any]], None],
        broker: InstallBroker | None = None,
        to_shared: Callable[[str], str] | None = None,
    ) -> WorkerOutcome:
        broker = broker or InstallBroker()
        shared = to_shared or (lambda text: text)
        payload = {
            "task_id": task.id,
            "objective": task.objective,
            "context": task.context,
            "goal": goal,
            "cwd": cwd,
            "hermes_root": self.hermes_root,
            "model": self.model,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "disabled_toolsets": self.disabled_toolsets,
            "max_iterations": self.max_iterations,
            "max_output_tokens": self.max_output_tokens,
        }
        env = _worker_env()
        env.update({"ROI_WORKER_API_KEY": self.api_key, "PYTHONIOENCODING": "utf-8"})
        options: dict[str, Any] = (
            {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        )
        scope = WorkerProcessScope()
        launch_options = {**options, **scope.launch_kwargs}
        if "creationflags" in options or "creationflags" in scope.launch_kwargs:
            launch_options["creationflags"] = options.get(
                "creationflags", 0
            ) | scope.launch_kwargs.get("creationflags", 0)
        try:
            process = await asyncio.create_subprocess_exec(
                self.python,
                str(HOST_SCRIPT),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=cwd or None,
                env=env,
                limit=STREAM_LIMIT,
                **launch_options,
            )
        except BaseException:
            scope.close()
            raise
        try:
            # Host waits for stdin: attach the OS process subtree before any tools start.
            scope.attach(process.pid)
        except BaseException:
            await _kill(process)
            scope.close()
            raise
        worker_key = f"{task.id}@{process.pid}"
        tools: list[str] = []
        final: dict[str, Any] | None = None
        holding = False
        installs: set[asyncio.Task[None]] = set()
        writer = asyncio.Lock()

        async def reply(number: Any, **data: Any) -> None:
            assert process.stdin is not None
            line = json.dumps({"id": number, **data}, ensure_ascii=False) + "\n"
            async with writer:
                with suppress(ConnectionError, RuntimeError):
                    process.stdin.write(line.encode("utf-8"))
                    await process.stdin.drain()

        async def install(event: dict[str, Any]) -> None:
            try:
                command = shared(str(event.get("command") or ""))
                folder = shared(str(event.get("cwd") or cwd))
                result = await broker.install(
                    worker_key,
                    command,
                    folder,
                    on_event=lambda _node, status, data: on_event(
                        {"type": "install_status", "status": status, "data": data}
                    ),
                )
                note = (
                    f"[central agent ran the command in shared project {folder}; "
                    "the environment is shared with your worktree]\n"
                )
                result = {**result, "output": note + str(result.get("output") or "")}
            except BaseException as exc:  # always answer; a lost reply strands the host
                result = {"output": "", "exit_code": -1, "error": f"{type(exc).__name__}: {exc}"}
                await reply(event.get("id"), result=json.dumps(result, ensure_ascii=False))
                if not isinstance(exc, Exception):
                    raise
                return
            await reply(event.get("id"), result=json.dumps(result, ensure_ascii=False))

        try:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            await process.stdin.drain()
            async for raw in process.stdout:
                try:
                    event = json.loads(raw.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                kind = event.get("type")
                if kind in {"final", "error"}:
                    final = event
                    break
                if kind == "gate":
                    try:
                        await broker.enter(worker_key)
                    except Exception as exc:
                        await reply(event.get("id"), error=f"{type(exc).__name__}: {exc}")
                    else:
                        holding = True
                        await reply(event.get("id"))
                    continue
                if kind == "gate_done":
                    if holding:
                        await broker.leave(worker_key)
                        holding = False
                    continue
                if kind == "install":
                    # Off the read loop: this worker's other tool threads keep reporting.
                    job = asyncio.create_task(install(event))
                    installs.add(job)
                    job.add_done_callback(installs.discard)
                    on_event(
                        {"type": "tool_start", "name": "install", "args": event.get("command")}
                    )
                    continue
                if kind == "tool_start":
                    tools.append(str(event.get("name")))
                on_event(event)
            # EOF without a final event means the host crashed: settle its actual
            # exit code before forced subtree cleanup can replace it with code 1.
            if final is None:
                try:
                    await asyncio.wait_for(process.wait(), EXIT_GRACE_SECONDS)
                except TimeoutError:
                    await _kill(process)
            # A final/error event ended tool work. Do not wait for the normal host
            # exit while background children may still hold its pipes open.
        finally:

            async def cleanup_worker() -> None:
                # One finalizer owns ALL cleanup phases. Shielding only scope.close
                # leaks the lease if cancellation lands in communicate/leave/gather.
                cleanup_error: Exception | None = None
                try:
                    try:
                        await asyncio.to_thread(scope.close)
                        # Reap pipe transports after the complete OS subtree is gone.
                        await asyncio.wait_for(process.communicate(), EXIT_GRACE_SECONDS)
                    except Exception as exc:
                        cleanup_error = exc
                        await broker.quarantine("worker process cleanup was not confirmed")
                        # Keep the lease quarantined; best-effort host transport cleanup
                        # does not claim every descendant was successfully stopped.
                        with suppress(Exception):
                            await _kill(process)
                            await asyncio.wait_for(process.communicate(), EXIT_GRACE_SECONDS)
                    if holding and cleanup_error is None:
                        try:
                            await broker.leave(worker_key)
                        except Exception as exc:
                            cleanup_error = exc
                            await broker.quarantine("worker lease release was not confirmed")
                finally:
                    try:
                        jobs = list(installs)
                        for job in jobs:
                            job.cancel()
                        if jobs:
                            await asyncio.gather(*jobs, return_exceptions=True)
                    finally:
                        with suppress(Exception):
                            process.stdin.close()
                if cleanup_error is not None:
                    raise RuntimeError(
                        "worker process cleanup failed; environment quarantined"
                    ) from cleanup_error

            cleanup = asyncio.create_task(cleanup_worker())
            cleanup_cancelled = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    if cleanup.cancelled():
                        raise
                    cleanup_cancelled = True
            cleanup.result()
            if cleanup_cancelled:
                raise asyncio.CancelledError
        if final is None:
            final = {"type": "error", "error": f"worker exited with code {process.returncode}"}
        ok = final["type"] == "final" and bool(str(final.get("text") or "").strip())
        report = str(final.get("text") or final.get("error") or "")
        return WorkerOutcome(task.id, task.objective, ok, report, len(tools), tools)


async def _kill(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
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
        with suppress(TimeoutError):
            await asyncio.wait_for(killer.wait(), 10)
    with suppress(ProcessLookupError):
        process.kill()
    with suppress(TimeoutError):
        await asyncio.wait_for(process.wait(), 10)


Planner = Callable[[str, str, type[BaseModel]], Awaitable[BaseModel]]
Finalizer = Callable[[str, str, DeltaSink | None], Awaitable[str]]
WorkerRunner = Callable[..., Awaitable[WorkerOutcome]]


class WaveOrchestrator:
    def __init__(
        self,
        *,
        structured: Planner,
        finalize: Finalizer,
        run_worker: WorkerRunner,
        maximum_waves: int,
        on_event: EventSink,
        open_workspace: Callable[[str], WaveWorkspace | None] | None = None,
        install_broker: InstallBroker | None = None,
        maximum_concurrency: int | None = None,
        worker_slots: asyncio.Semaphore | None = None,
    ) -> None:
        self._worker_slots = (
            worker_slots
            if worker_slots is not None
            else (
                asyncio.Semaphore(maximum_concurrency) if maximum_concurrency is not None else None
            )
        )
        # Real Hermes workers route installs through this broker; test doubles get none.
        self.install_broker = install_broker
        self.structured = structured
        self.finalize = finalize
        self.run_worker = run_worker
        self.maximum_waves = maximum_waves
        self.on_event = on_event
        self.open_workspace = open_workspace

    async def run(
        self,
        messages: list[dict[str, Any]],
        *,
        request: str,
        cwd: str,
        on_delta: DeltaSink | None = None,
    ) -> str:
        transcript = render_transcript(messages)
        plan = await self.structured(
            PLAN_SYSTEM, f"{transcript}\n\n### CURRENT REQUEST\n{request}", WavePlan
        )
        assert isinstance(plan, WavePlan)
        tasks = _unique(plan.tasks)
        users = [_text(m.get("content")) for m in messages if m.get("role") == "user"]
        if users and users[-1] == request:
            users.pop()  # the current request is scanned first, with files allowed
        cwd = project_directory(plan.workdir, [request, *reversed(users)], cwd)
        if len(tasks) == 1:
            # The router already judged this a multi-part request: one task is a bundle.
            tasks = await self._split(tasks, request, "PLAN SUMMARY", plan.summary, {}, [])
        self.on_event("planner", "completed", {"task_count": len(tasks), "round": 1, "cwd": cwd})
        accepted: dict[str, WorkerOutcome] = {}
        history: list[WorkerOutcome] = []
        unfinished: list[str] = []
        wave = 0
        if self.install_broker is not None:
            await self.install_broker.watch(cwd)
        while tasks and wave < self.maximum_waves:
            wave += 1
            self.on_event("wave", "running", {"wave": wave, "task_count": len(tasks)})
            outcomes = await self._run_wave(tasks, wave, request, cwd)
            history.extend(outcomes)
            if self.install_broker is not None:
                problems = await self.install_broker.check_environment()
                if problems:
                    self.on_event("environment", "failed", {"files": problems[:50], "wave": wave})
                    unfinished = [
                        "Shared environment was changed outside central installation; "
                        "automatic work stopped pending repair: " + ", ".join(problems[:20])
                    ]
                    tasks = []
                    break
            review = await self.structured(
                REVIEW_SYSTEM, _review_prompt(request, plan.summary, accepted, outcomes), WaveReview
            )
            assert isinstance(review, WaveReview)
            for outcome in outcomes:
                if outcome.task_id in review.accepted and outcome.ok:
                    accepted[outcome.task_id] = outcome
            self.on_event(
                "verifier",
                "completed" if review.done else "failed",
                {"wave": wave, "accepted": len(accepted), "feedback": review.feedback[:500]},
            )
            if review.done:
                tasks = []
                break
            tasks = _unique(review.next_tasks, taken={o.task_id for o in history})
            if len(tasks) == 1 and wave < self.maximum_waves:
                tasks = await self._split(
                    tasks, request, "REVIEW FEEDBACK", review.feedback, accepted, history
                )
            if tasks:
                self.on_event("retry-planner", "completed", {"task_count": len(tasks)})
            else:
                unfinished = [review.feedback or "the reviewer rejected the result"]
        reports = list(accepted.values()) or history
        if tasks:
            unfinished = [task.objective for task in tasks]
        open_note = (
            f"\nNOT FINISHED after {wave} waves: " + "; ".join(unfinished) if unfinished else ""
        )
        answer = await self.finalize(
            FINAL_SYSTEM, _final_prompt(request, reports) + open_note, on_delta
        )
        self.on_event("synthesizer", "completed", {"waves": wave})
        return answer

    async def _split(
        self,
        tasks: list[PlannedTask],
        request: str,
        heading: str,
        note: str,
        accepted: dict[str, WorkerOutcome],
        history: list[WorkerOutcome],
    ) -> list[PlannedTask]:
        """Reviewers tend to bundle all remaining work into one task; divide it again."""
        remaining = tasks[0]
        done = "\n".join(f"- {o.task_id}: {o.objective}" for o in accepted.values()) or "(none)"
        prompt = (
            f"USER REQUEST:\n{request}\n\n{heading}:\n{note}\n\n"
            f"ALREADY ACCEPTED (do not redo, do not touch their files):\n{done}\n\n"
            f"REMAINING WORK:\nOBJECTIVE: {remaining.objective}\nCONTEXT:\n{remaining.context}"
        )
        try:
            plan = await self.structured(SPLIT_SYSTEM, prompt, WavePlan)
        except Exception as exc:  # noqa: BLE001 - keep the reviewer's task on a bad split
            self.on_event("retry-planner", "split", {"error": f"{type(exc).__name__}: {exc}"[:300]})
            return tasks
        if not isinstance(plan, WavePlan):
            return tasks
        parts = _unique(
            plan.tasks,
            taken={o.task_id for o in history},
            done_objectives={o.objective for o in accepted.values()},
        )
        if len(parts) < 2:
            return tasks
        return parts

    async def _run_wave(
        self, tasks: list[PlannedTask], wave: int, goal: str, cwd: str
    ) -> list[WorkerOutcome]:
        """Each worker in its own git worktree; successful changes merged back in order."""
        workspace = None
        if self.open_workspace is not None and len(tasks) > 1:
            try:
                workspace = await asyncio.to_thread(self.open_workspace, cwd)
                if workspace is not None:
                    await asyncio.to_thread(workspace.snapshot)
            except Exception as exc:  # noqa: BLE001 - fall back to the shared folder
                self.on_event("wave", "isolation", {"wave": wave, "error": str(exc)[:500]})
                workspace = None
        if workspace is None:
            return list(
                await asyncio.gather(*(self._run_one(task, wave, goal, cwd) for task in tasks))
            )
        try:
            try:
                folders = [await asyncio.to_thread(workspace.prepare, task.id) for task in tasks]
            except Exception as exc:  # noqa: BLE001 - worktree failed: run in the shared folder
                self.on_event("wave", "isolation", {"wave": wave, "error": str(exc)[:500]})
                await asyncio.shield(asyncio.to_thread(workspace.close))
                workspace = None
                return list(
                    await asyncio.gather(*(self._run_one(task, wave, goal, cwd) for task in tasks))
                )
            self.on_event("wave", "isolation", {"wave": wave, "worktrees": len(folders)})
            isolated = [
                task.model_copy(
                    update={
                        "objective": workspace.to_worker(task.objective, task.id),
                        "context": workspace.to_worker(task.context, task.id)
                        + f"\n\nYou work in a private git worktree: {folder}. Change files only "
                        "there; they are merged into the shared folder after the wave.",
                    }
                )
                for task, folder in zip(tasks, folders, strict=True)
            ]
            outcomes = list(
                await asyncio.gather(
                    *(
                        self._run_one(
                            task,
                            wave,
                            workspace.to_worker(goal, task.id),
                            str(folder),
                            to_shared=lambda text, tid=task.id: workspace.to_shared(text, tid),
                        )
                        for task, folder in zip(isolated, folders, strict=True)
                    )
                )
            )
            outside: list[str] = []
            try:
                outside = await asyncio.to_thread(workspace.changed_outside)
            except Exception:  # noqa: BLE001 - detection is best effort
                outside = []
            if outside:
                self.on_event("wave", "bypass", {"wave": wave, "files": outside[:50]})
            for index, outcome in enumerate(outcomes):
                outcomes[index] = await self._merge(workspace, outcome, wave)
                outcomes[index].objective = tasks[index].objective
                if outside:
                    outcomes[index].report += (
                        "\n\nWARNING: during this wave files were changed directly in the "
                        "shared folder, bypassing the worktrees: " + ", ".join(outside[:20])
                    )
            return outcomes
        finally:
            if workspace is not None:
                await asyncio.shield(asyncio.to_thread(workspace.close))

    async def _merge(
        self, workspace: WaveWorkspace, outcome: WorkerOutcome, wave: int
    ) -> WorkerOutcome:
        report = workspace.to_shared(outcome.report, outcome.task_id)
        try:
            merged = await asyncio.to_thread(workspace.merge, outcome.task_id, keep=outcome.ok)
        except Exception as exc:  # noqa: BLE001 - an unmergeable result is a rejected result
            merged = MergeResult("conflict", detail=f"{type(exc).__name__}: {exc}")
        # Merge status lives on its own node so the worker's completed/failed count stays right.
        self.on_event(
            f"merge:{outcome.task_id}",
            "merged" if merged.status in {"applied", "unchanged"} else merged.status,
            {"wave": wave, "files": merged.files[:50], "detail": merged.detail[:500]},
        )
        if merged.status == "conflict":
            self.on_event(
                f"worker:{outcome.task_id}", "failed", {"wave": wave, "error": "merge conflict"}
            )
            note = (
                "\n\nMERGE CONFLICT: these changes were NOT applied to the shared folder because "
                "another worker of the same wave changed the same content "
                f"({', '.join(merged.files) or 'unknown files'}). {merged.detail}"
            )
            return replace(outcome, ok=False, report=report + note)
        if merged.status == "applied":
            report += "\n\nFiles merged into the shared folder: " + ", ".join(merged.files)
            if merged.detail:
                report += f"\nMERGE NOTE: {merged.detail}"
        return replace(outcome, report=report)

    async def _run_one(
        self,
        task: PlannedTask,
        wave: int,
        goal: str,
        cwd: str,
        to_shared: Callable[[str], str] | None = None,
    ) -> WorkerOutcome:
        node = f"worker:{task.id}"
        self.on_event(node, "queued", {"objective": task.objective, "wave": wave})

        def forward(event: dict[str, Any]) -> None:
            if event.get("type") == "install_status":
                self.on_event(f"install:{task.id}", str(event["status"]), event["data"])
            elif event.get("type") == "tool_start":
                self.on_event(
                    node, "tool", {"tool": event.get("name"), "args": event.get("args", "")}
                )

        async with self._worker_slots if self._worker_slots is not None else nullcontext():
            self.on_event(node, "running", {"objective": task.objective, "wave": wave})
            try:
                extra: dict[str, Any] = {}
                if self.install_broker is not None:
                    extra = {"broker": self.install_broker, "to_shared": to_shared}
                outcome = await self.run_worker(task, goal=goal, cwd=cwd, on_event=forward, **extra)
            except Exception as exc:
                outcome = WorkerOutcome(
                    task.id, task.objective, False, f"{type(exc).__name__}: {exc}"
                )
            self.on_event(
                node,
                "completed" if outcome.ok else "failed",
                {"tool_calls": outcome.tool_calls, "confidence": 1.0 if outcome.ok else 0.0},
            )
            return outcome


def _unique(
    tasks: list[PlannedTask],
    taken: set[str] | None = None,
    done_objectives: set[str] | None = None,
) -> list[PlannedTask]:
    used = set(taken or ())
    seen_objectives = {" ".join(o.split()).casefold() for o in done_objectives or ()}
    result: list[PlannedTask] = []
    for task in tasks:
        key = " ".join(task.objective.split()).casefold()
        if not key or key in seen_objectives:
            continue
        seen_objectives.add(key)
        base = re.sub(r"[^a-zA-Z0-9_-]", "", task.id) or "t"
        task_id, suffix = base, 2
        while task_id in used:
            task_id, suffix = f"{base}-{suffix}", suffix + 1
        used.add(task_id)
        result.append(task.model_copy(update={"id": task_id}))
    return result


def _review_prompt(
    request: str, summary: str, accepted: dict[str, WorkerOutcome], outcomes: list[WorkerOutcome]
) -> str:
    done = "\n".join(f"- {o.task_id}: {o.objective}" for o in accepted.values()) or "(none)"
    wave = "\n\n".join(
        f"## {o.task_id} ({'finished' if o.ok else 'FAILED'}, {o.tool_calls} tool calls)\n"
        f"OBJECTIVE: {o.objective}\nREPORT:\n{o.report}"
        for o in outcomes
    )
    return (
        f"USER REQUEST:\n{request}\n\nPLAN: {summary}\n\nALREADY ACCEPTED:\n{done}\n\n"
        f"THIS WAVE:\n{wave}"
    )


def _final_prompt(request: str, reports: list[WorkerOutcome]) -> str:
    body = "\n\n".join(f"## {o.task_id}: {o.objective}\n{o.report}" for o in reports)
    return f"USER REQUEST:\n{request}\n\nWORKER REPORTS:\n{body}"
