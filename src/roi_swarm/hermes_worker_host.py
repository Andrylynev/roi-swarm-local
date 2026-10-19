"""Run ONE atomic swarm task as a full Hermes agent (executed by Hermes' own Python).

ROI runs on a newer Python than Hermes, so this file is started as a script with the
Hermes venv interpreter and must only import the standard library and Hermes itself.

Protocol: the task arrives as the first JSON line on stdin. Events are written to stdout
as JSON lines: ``tool_start`` / ``tool_complete`` while the agent works, then exactly one
``final`` (or ``error``). Everything else Hermes prints is redirected to stderr so it
cannot corrupt the event stream. The provider key is read from ``ROI_WORKER_API_KEY``.

Package installs are the central agent's job: workers share one linked ``.venv`` /
``node_modules``, so an install run inside a private worktree would point the shared
environment at a temporary folder. Every tool call therefore passes a gate (``gate`` /
``gate_done`` requests), and install commands become an ``install`` request that the
orchestrator runs in the shared project while all other workers are paused. Replies
arrive as JSON lines ``{"id": n, ...}`` on stdin.
"""

from __future__ import annotations

import ast
import io
import itertools
import json
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

PREVIEW_CHARS = 400

WORKER_PROMPT = (
    "You are one worker of the ROI swarm, running as a full Hermes agent with Hermes' "
    "tools, skills and permissions. Solve ONLY your OBJECTIVE; the overall goal is "
    "background. Use your tools directly (read, write, run, verify). Do not create new "
    "tasks, delegate or ask the user questions: if something is impossible, say exactly "
    "what blocks it. Touch only files your objective needs. Package installs (pip, uv, "
    "npm, ...) are executed for you by the central agent in the shared project: run them "
    "with the terminal tool as a standalone foreground command and read the result. "
    "Use workdir rather than 'cd ... && install ...'. Never modify .venv/node_modules "
    "directly or install from scripts; uv run tests MUST use --no-sync. Finish with a short "
    "report: what you did, files changed, how you verified it, open problems."
)

# This script is launched directly under Hermes' Python, not imported as roi_swarm.
if __package__:
    from .native_budget import NativeBudget, guard_worker_auxiliary_clients, guard_worker_clients
    from .worker_install_policy import install_error, is_install
else:
    from native_budget import NativeBudget, guard_worker_auxiliary_clients, guard_worker_clients
    from worker_install_policy import install_error, is_install


class Channel:
    """Request/response over stdin for the gate and install requests (thread safe)."""

    def __init__(self, emit: Callable[..., None], stream: Any) -> None:
        self._emit = emit
        self._ids = itertools.count(1)
        self._pending: dict[int, list[Any]] = {}
        self._lock = threading.Lock()
        self._closed = False
        threading.Thread(target=self._read, args=(stream,), daemon=True).start()

    def _read(self, stream: Any) -> None:
        try:
            for line in stream:
                try:
                    reply = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(reply, dict) or not isinstance(reply.get("id"), int):
                    continue
                with self._lock:
                    waiter = self._pending.pop(reply["id"], None)
                if waiter is not None:
                    waiter[1] = reply
                    waiter[0].set()
        finally:  # EOF or a broken pipe: release every pending request, fail closed
            with self._lock:
                self._closed = True
                pending, self._pending = list(self._pending.values()), {}
            for waiter in pending:
                waiter[0].set()

    def request(self, kind: str, **data: Any) -> dict[str, Any]:
        number = next(self._ids)
        waiter: list[Any] = [threading.Event(), None]
        with self._lock:
            if self._closed:
                return {"error": "orchestrator closed the channel"}
            self._pending[number] = waiter
        try:
            self._emit(kind, id=number, **data)
        except Exception as exc:
            with self._lock:
                self._pending.pop(number, None)
            return {"error": f"orchestrator channel failed: {exc}"}
        waiter[0].wait()
        return waiter[1] or {"error": "orchestrator closed the channel"}

    def notify(self, kind: str, **data: Any) -> None:
        self._emit(kind, **data)


class Gate:
    """One slot per worker process: the orchestrator pauses tools while it installs."""

    def __init__(self, channel: Channel) -> None:
        self._channel = channel
        self._lock = threading.Lock()
        self._count = 0

    def __enter__(self) -> Gate:
        with self._lock:
            if self._count == 0:
                reply = self._channel.request("gate")
                if reply.get("error"):
                    raise RuntimeError(str(reply["error"]))
            self._count += 1
        return self

    def __exit__(self, *_: Any) -> None:
        with self._lock:
            self._count -= 1
            if self._count == 0:
                self._channel.notify("gate_done")


def code_installs(code: Any) -> bool:
    if not isinstance(code, str):
        return False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return is_install(code)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and is_install(node.value)
        ):
            return True
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [e.value for e in node.elts if isinstance(e, ast.Constant)]
            if (
                len(values) == len(node.elts)
                and all(isinstance(v, str) for v in values)
                and is_install(" ".join(values))
            ):
                return True
    return False


def session_cwd(task_id: str | None) -> str:
    from tools.terminal_tool import get_session_cwd

    return get_session_cwd(task_id) or os.getcwd()


def wait_background(session_id: str) -> None:
    from tools.process_registry import process_registry

    while True:
        session = process_registry.get(session_id)
        if session is None or session.exited:
            return
        time.sleep(0.2)


def guard_tools(model_tools: Any, channel: Channel) -> Callable[[], None]:
    """Route install commands to the central agent and gate every other tool call."""
    original = model_tools.handle_function_call
    gate = Gate(channel)
    background: dict[str, threading.Thread] = {}
    background_lock = threading.Lock()
    admission = threading.Condition()
    closing = False
    active = 0

    def release_background(session_id: str) -> None:
        wait_background(session_id)
        gate.__exit__()
        with background_lock:
            background.pop(session_id, None)

    def cleanup() -> None:
        nonlocal closing
        # Atomically stop admission, then drain calls that can still register a process.
        # This is shutdown grace, not a provider/tool execution limit. A stuck abandoned
        # thread is terminated by the parent's OS process scope after this error.
        with admission:
            closing = True
            if not admission.wait_for(lambda: active == 0, timeout=30):
                raise RuntimeError("worker cleanup: active tool threads did not settle")
        # Do not call Hermes ProcessRegistry.kill_process here: it closes a pipe
        # synchronously while its reader (or an inherited grandchild) can hold the
        # stream lock forever. The parent's OS Job owns every descendant and kills
        # the entire subtree before releasing the broker gate, including normal exit.
        # Leave background references held until that parent cleanup completes.

    def dispatch(function_name: str, function_args: Any, *args: Any, **kwargs: Any) -> Any:
        values = function_args if isinstance(function_args, dict) else {}
        if function_name == "terminal" and is_install(values.get("command")):
            command = str(values.get("command"))
            error = install_error(command)
            if error:
                return json.dumps({"error": error})
            if values.get("background") or values.get("pty"):
                return json.dumps(
                    {"error": "Package installation must be a foreground terminal call."}
                )
            reply = channel.request(
                "install",
                command=str(values.get("command")),
                cwd=str(
                    values.get("workdir")
                    or session_cwd(kwargs.get("task_id") or (args[0] if args else None))
                ),
            )
            return str(reply.get("result") or json.dumps({"error": reply.get("error")}))
        if function_name == "execute_code" and code_installs(values.get("code")):
            return json.dumps(
                {
                    "error": "Package installs are run by the central agent: call the "
                    "terminal tool with the install command instead of execute_code."
                }
            )
        try:
            gate.__enter__()
        except RuntimeError as exc:
            return json.dumps({"error": str(exc)})
        transferred = False
        try:
            result = original(function_name, function_args, *args, **kwargs)
            if function_name == "terminal":
                try:
                    data = json.loads(result) if isinstance(result, str) else result
                except (ValueError, TypeError):
                    data = None
                if isinstance(data, dict) and isinstance(data.get("session_id"), str):
                    sid = data["session_id"]
                    thread = threading.Thread(target=release_background, args=(sid,), daemon=True)
                    with background_lock:
                        background[sid] = thread
                    thread.start()
                    transferred = True
            return result
        finally:
            if not transferred:
                gate.__exit__()

    def guarded(function_name: str, function_args: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal active
        with admission:
            if closing:
                return json.dumps({"error": "Worker is shutting down; no new tools accepted."})
            active += 1
        try:
            return dispatch(function_name, function_args, *args, **kwargs)
        finally:
            with admission:
                active -= 1
                admission.notify_all()

    model_tools.handle_function_call = guarded
    return cleanup


def _preview(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "…"


def build_prompt(task: dict[str, Any]) -> str:
    parts = [f"OBJECTIVE:\n{task['objective']}"]
    if task.get("context"):
        parts.append(f"CONTEXT FOR YOUR OBJECTIVE:\n{task['context']}")
    if task.get("goal"):
        parts.append(f"OVERALL GOAL (background only):\n{task['goal']}")
    if task.get("cwd"):
        parts.append(f"Working directory: {task['cwd']}")
        parts.append(
            "Quote paths containing spaces exactly as given. Do not translate or invent paths. "
            "Prefer read_file/write_file with literal paths over shell directory commands."
        )
    parts.append(
        "Tool result conventions: read_file line-number prefixes are NOT file contents: "
        "'1|BETA' means the file contains BETA, not 1|BETA. "
        "write_file verified:true confirms exact bytes were saved; "
        "verify that the path and contents match your objective. For a simple file creation "
        "task, after a matching verified:true write or one read confirming the requested contents, "
        "stop and report success. Only for such verified simple file creation, avoid redundant "
        "rewrites and shell checks. For code changes and complex tasks, run the required tests "
        "and all verification specified by your objective."
    )
    parts.append(
        f"YOUR ONLY DELIVERABLE (repeat): {task['objective']}\n"
        "Do not implement any other part of the overall goal."
    )
    return "\n\n".join(parts)


def generation_options(task: dict[str, Any]) -> dict[str, Any]:
    value = task.get("max_output_tokens")
    if value is None:
        return {}
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("Worker max_output_tokens must be a positive integer")
    return {"max_tokens": value}


def main() -> int:
    events = sys.stdout
    sys.stdout = sys.stderr  # Hermes banners/prints must never reach the event stream.
    lock = threading.Lock()

    def emit(kind: str, **data: Any) -> None:
        with lock:
            events.write(json.dumps({"type": kind, **data}, ensure_ascii=False) + "\n")
            events.flush()

    try:
        task = json.loads(sys.stdin.readline())
        if task.get("cwd"):
            os.environ["TERMINAL_CWD"] = task["cwd"]
            os.chdir(task["cwd"])
        hermes_root = task["hermes_root"]
        if hermes_root not in sys.path:
            sys.path.insert(0, hermes_root)
        import model_tools
        from run_agent import AIAgent

        replies = sys.stdin
        # Do not let an interactive MCP/OAuth helper steal a broker reply in batch mode.
        sys.stdin = io.StringIO("")
        cleanup = guard_tools(model_tools, Channel(emit, replies))

        agent = AIAgent(
            **generation_options(task),
            model=task["model"],
            base_url=task["base_url"],
            api_key=os.environ.pop("ROI_WORKER_API_KEY", ""),  # keep it out of worker shells
            provider="custom",
            api_mode=task.get("api_mode") or "chat_completions",
            max_iterations=int(task.get("max_iterations") or 60),
            disabled_toolsets=list(task.get("disabled_toolsets") or []),
            quiet_mode=True,
            platform="batch",
            skip_background_review=True,
            ephemeral_system_prompt=WORKER_PROMPT,
            tool_start_callback=lambda call_id, name, args: emit(
                "tool_start", id=str(call_id), name=str(name), args=_preview(args)
            ),
            tool_complete_callback=lambda call_id, name, args, result: emit(
                "tool_complete", id=str(call_id), name=str(name), result=_preview(result)
            ),
        )
        auxiliary_guard = nullcontext()
        if os.environ.get("ROI_LOCAL_NATIVE_BUDGET") == "1":
            budget = NativeBudget(task["base_url"])
            guard_worker_clients(agent, budget)
            auxiliary_guard = guard_worker_auxiliary_clients(budget)
        try:
            with auxiliary_guard:
                result = agent.run_conversation(build_prompt(task), task_id=task.get("task_id"))
        finally:
            cleanup()
        emit("final", text=str(result.get("final_response") or ""))
        return 0
    except BaseException as exc:  # noqa: BLE001 - report every failure to the orchestrator
        traceback.print_exc(file=sys.stderr)
        emit("error", error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
