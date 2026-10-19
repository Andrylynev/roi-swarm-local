"""Only the central agent installs packages; other workers pause while it does."""

import asyncio
import json
import sys
import types

import pytest

import roi_swarm.waves as waves
from roi_swarm.hermes_worker_host import Channel, guard_tools, is_install
from roi_swarm.schemas import PlannedTask
from roi_swarm.waves import HermesWorkerPool, InstallBroker


@pytest.mark.parametrize(
    "command",
    [
        "pip install requests",
        "python -m pip install -e .",
        ".venv/Scripts/python.exe -m pip install -r requirements.txt",
        "uv sync",
        "uv sync --extra dev",
        "uv pip install -e .",
        "cd app && npm install",
        "npm ci",
        "poetry install",
        "uv add httpx",
        "C:\\venv\\Scripts\\pip.exe install x",
    ],
)
def test_install_commands_are_detected(command) -> None:
    assert is_install(command)


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q",
        "pip list",
        "pip show requests",
        "uv run --no-sync pytest",
        "npm test",
        "echo installing",
        "git status",
        "python -c 'print(1)'",
    ],
)
def test_ordinary_commands_are_not_installs(command) -> None:
    assert not is_install(command)


def test_install_waits_for_running_tools_and_pauses_new_ones() -> None:
    order: list[str] = []

    async def fake_install(command, cwd):
        order.append("install-start")
        await asyncio.sleep(0.05)
        order.append("install-end")
        return {"output": "ok", "exit_code": 0, "error": None}

    async def scenario() -> None:
        broker = InstallBroker(run=fake_install)
        await broker.enter("b")  # worker b is mid tool call
        job = asyncio.create_task(broker.install("a", "uv sync", "D:/p"))
        await asyncio.sleep(0.02)
        assert order == []  # waits for b's running call
        order.append("b-done")
        await broker.leave("b")
        await asyncio.sleep(0.01)
        late = asyncio.create_task(broker.enter("c"))  # c starts a tool during install
        await asyncio.sleep(0.01)
        assert not late.done()
        await job
        await late
        order.append("c-runs")
        await broker.leave("c")

    asyncio.run(scenario())
    assert order == ["b-done", "install-start", "install-end", "c-runs"]


def test_nested_install_is_refused_without_mutating_an_active_environment() -> None:
    async def fake_install(command, cwd):
        pytest.fail("installation started while a tool was using the environment")

    async def scenario() -> dict:
        broker = InstallBroker(run=fake_install)
        await broker.enter("a")
        return await asyncio.wait_for(broker.install("a", "pip install x", "."), 2)

    assert "standalone" in asyncio.run(scenario())["error"]


def test_nested_install_behind_another_install_cannot_deadlock() -> None:
    async def scenario() -> None:
        ran: list[str] = []

        async def run(command, cwd):
            ran.append(command)
            return {"output": "ok", "exit_code": 0, "error": None}

        broker = InstallBroker(run=run)
        await broker.enter("a")
        first = asyncio.create_task(broker.install("b", "pip install b", "."))
        await asyncio.sleep(0)
        second = asyncio.create_task(broker.install("a", "pip install a", "."))
        try:
            rejected = await asyncio.wait_for(second, 2)
            assert "standalone" in rejected["error"]
        finally:
            await broker.leave("a")
        await asyncio.wait_for(first, 2)
        assert ran == ["pip install b"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "command",
    [
        "npm --location global install x",
        "npm --global true install x",
        "uv tool --allow-insecure-host localhost install ruff",
        "uv --allow-insecure-host localhost tool install ruff",
        "npm --future-global-mode enabled install x",
        "npm --global true --loglevel info install x",
        "npm --global true --help false install x",
        "npm --version false install x",
        "npm -v false install x",
        "npm --version --no-version install x",
        "npm install x --help false",
        "npm --unknown-value info install x",
        "uv --unknown-value tree tool install x",
        "pip --unknown-value list install x",
    ],
)
def test_installer_options_never_bypass_central_dispatch(command):
    calls = []

    class Central:
        def request(self, kind, **data):
            calls.append((kind, data))
            return {"result": "central result"}

        def notify(self, *a, **k):
            pass

    tools = types.SimpleNamespace(handle_function_call=lambda *a, **k: pytest.fail("ran locally"))
    guard_tools(tools, Central())
    result = tools.handle_function_call("terminal", {"command": command, "workdir": "D:/project"})
    assert result == "central result" or "error" in json.loads(result)
    assert not calls or calls[0][0] == "install"


def test_failed_install_reaches_the_worker_and_releases_the_pause() -> None:
    async def broken(command, cwd):
        raise OSError("no bash")

    events: list[tuple[str, str]] = []

    async def scenario() -> dict:
        broker = InstallBroker(run=broken, on_event=lambda n, s, d: events.append((n, s)))
        result = await broker.install("a", "uv sync", ".")
        await asyncio.wait_for(broker.enter("b"), 1)
        return result

    result = asyncio.run(scenario())
    assert "OSError" in result["error"]
    assert events == [
        ("install:a", "waiting"),
        ("install:a", "running"),
        ("install:a", "failed"),
    ]


def test_host_guard_routes_installs_and_gates_other_tools() -> None:
    sent: list[dict] = []
    replies: list[str] = []

    class Stream:
        """stdin of the host: answers every request immediately."""

        def __iter__(self):
            while True:
                if replies:
                    yield replies.pop(0)
                else:
                    import time

                    time.sleep(0.005)

    def emit(kind, **data):
        sent.append({"type": kind, **data})
        if kind == "gate":
            replies.append(json.dumps({"id": data["id"]}) + "\n")
        if kind == "install":
            result = json.dumps({"output": "installed", "exit_code": 0})
            replies.append(json.dumps({"id": data["id"], "result": result}) + "\n")

    calls: list[str] = []
    tools = types.SimpleNamespace(
        handle_function_call=lambda name, args, *a, **k: calls.append(name) or "ran"
    )
    guard_tools(tools, Channel(emit, Stream()))

    assert tools.handle_function_call("read_file", {"path": "x"}) == "ran"
    result = tools.handle_function_call("terminal", {"command": "uv sync", "workdir": "D:/w"})
    blocked = tools.handle_function_call("execute_code", {"code": "os.system('pip install x')"})

    assert calls == ["read_file"]  # the install never ran inside the worker
    assert json.loads(result)["output"] == "installed"
    assert "central agent" in json.loads(blocked)["error"]
    kinds = [event["type"] for event in sent]
    assert kinds == ["gate", "gate_done", "install"]
    assert sent[2]["command"] == "uv sync" and sent[2]["cwd"] == "D:/w"


INSTALL_HOST = """
import json, sys
task = json.loads(sys.stdin.readline())
def ask(kind, number, **data):
    print(json.dumps({"type": kind, "id": number, **data}), flush=True)
    return json.loads(sys.stdin.readline())
ask("gate", 1)
print(json.dumps({"type": "tool_start", "name": "read_file", "args": "a"}), flush=True)
print(json.dumps({"type": "gate_done"}), flush=True)
reply = ask("install", 2, command="uv sync", cwd=task["cwd"] + "/sub")
print(json.dumps({"type": "final", "text": reply["result"]}), flush=True)
"""


def test_pool_runs_worker_installs_in_the_shared_project(tmp_path, monkeypatch) -> None:
    host = tmp_path / "host.py"
    host.write_text(INSTALL_HOST, encoding="utf-8")
    monkeypatch.setattr(waves, "HOST_SCRIPT", host)
    ran: list[tuple[str, str]] = []

    async def fake_install(command, cwd):
        ran.append((command, cwd))
        return {"output": "done", "exit_code": 0, "error": None}

    pool = HermesWorkerPool(
        python=sys.executable,
        hermes_root=str(tmp_path),
        base_url="u",
        api_key="k",
        api_mode="chat_completions",
        model="m",
        disabled_toolsets=[],
        max_iterations=1,
    )
    seen: list[dict] = []
    (tmp_path / "tree").mkdir()
    worktree = str(tmp_path / "tree")
    outcome = asyncio.run(
        pool.run(
            PlannedTask(id="t", objective="o"),
            goal="",
            cwd=worktree,
            on_event=seen.append,
            broker=InstallBroker(run=fake_install),
            to_shared=lambda text: text.replace(worktree, "D:/shared"),
        )
    )

    assert ran == [("uv sync", "D:/shared/sub")]
    assert outcome.ok and json.loads(outcome.report)["exit_code"] == 0
    assert "central agent" in json.loads(outcome.report)["output"]
    assert outcome.tools == ["read_file"]
    assert [e["type"] for e in seen if e["type"] == "tool_start"] == ["tool_start", "tool_start"]
    assert [e["status"] for e in seen if e["type"] == "install_status"] == [
        "waiting",
        "running",
        "completed",
    ]


def test_install_runner_executes_in_the_given_folder(tmp_path) -> None:
    result = asyncio.run(waves.run_install_command("echo installed-here", str(tmp_path)))

    assert result["exit_code"] == 0
    assert "installed-here" in result["output"]


def test_cancellation_waits_for_started_install_before_releasing_workers() -> None:
    async def scenario() -> None:
        started, finish = asyncio.Event(), asyncio.Event()

        async def run(command, cwd):
            started.set()
            await finish.wait()
            return {"output": "ok", "exit_code": 0, "error": None}

        broker = InstallBroker(run=run)
        job = asyncio.create_task(broker.install("a", "pip install a", "."))
        await started.wait()
        job.cancel()
        await asyncio.sleep(0)
        newcomer = asyncio.create_task(broker.enter("b"))
        await asyncio.sleep(0)
        assert not job.done() and not newcomer.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await job
        await asyncio.wait_for(newcomer, 2)

    asyncio.run(scenario())


def test_channel_ignores_nondict_reply_and_releases_pending_on_eof() -> None:
    import queue
    import threading

    lines = queue.Queue()
    sent = threading.Event()
    channel = Channel(lambda *a, **k: sent.set(), iter(lines.get, None))
    results = []
    thread = threading.Thread(target=lambda: results.append(channel.request("gate")), daemon=True)
    thread.start()
    assert sent.wait(2)
    lines.put("[]\n")
    lines.put("null\n")
    lines.put(None)
    thread.join(2)
    assert not thread.is_alive()
    assert results and "closed" in results[0]["error"]


def test_gate_failure_never_runs_a_worker_tool() -> None:
    class Closed:
        def request(self, *args, **kwargs):
            return {"error": "closed"}

        def notify(self, *args, **kwargs):
            pass

    tools = types.SimpleNamespace(handle_function_call=lambda *a, **k: pytest.fail("ran tool"))
    guard_tools(tools, Closed())
    result = tools.handle_function_call("read_file", {"path": "x"})
    assert "closed" in json.loads(result)["error"]


def test_pool_replies_when_worktree_path_mapping_fails(tmp_path, monkeypatch) -> None:
    host = tmp_path / "host.py"
    host.write_text(INSTALL_HOST, encoding="utf-8")
    monkeypatch.setattr(waves, "HOST_SCRIPT", host)

    def broken(text):
        raise ValueError("path mapping failed")

    async def scenario():
        return await asyncio.wait_for(
            waves_pool(tmp_path).run(
                PlannedTask(id="a", objective="x"),
                goal="",
                cwd=str(tmp_path),
                on_event=lambda e: None,
                to_shared=broken,
            ),
            4,
        )

    result = asyncio.run(scenario())
    assert result.ok
    assert "path mapping failed" in json.loads(result.report)["error"]


def test_host_install_uses_recorded_terminal_cwd(tmp_path, monkeypatch) -> None:
    from roi_swarm import hermes_worker_host as host

    requested = []

    class Replies:
        def request(self, kind, **values):
            requested.append((kind, values))
            return {"result": "ok"}

    monkeypatch.setattr(host, "session_cwd", lambda task_id: str(tmp_path / task_id))
    tools = types.SimpleNamespace(handle_function_call=lambda *a, **k: "ordinary")
    guard_tools(tools, Replies())
    assert tools.handle_function_call("terminal", {"command": "pip install x"}, "worker-a") == "ok"
    assert requested[0][1]["cwd"] == str(tmp_path / "worker-a")


def test_waiting_install_is_visible_and_cancellation_releases_queue() -> None:
    async def scenario() -> None:
        events = []
        broker = InstallBroker(on_event=lambda n, s, d: events.append((s, d)))
        await broker.enter("long-test")
        install = asyncio.create_task(broker.install("a", "pip install a", "."))
        await asyncio.sleep(0)
        assert events[0] == ("waiting", {"waiting_for": ["long-test"], "command": "pip install a"})
        install.cancel()
        with pytest.raises(asyncio.CancelledError):
            await install
        await asyncio.wait_for(broker.enter("new-worker"), 2)
        await broker.leave("long-test")

    asyncio.run(scenario())


def test_quoted_installer_text_in_python_is_blocked_before_execution() -> None:
    tools = types.SimpleNamespace(handle_function_call=lambda *a, **k: pytest.fail("ran code"))
    guard_tools(tools, None)
    result = tools.handle_function_call(
        "execute_code", {"code": "import subprocess; subprocess.run(['pip', 'install', 'x'])"}
    )
    assert "central agent" in json.loads(result)["error"]


def test_background_process_holds_gate_until_its_actual_exit(monkeypatch) -> None:
    import threading

    from roi_swarm import hermes_worker_host as host

    class Replies:
        def __init__(self):
            self.events = []
            self.left = threading.Event()

        def request(self, kind, **data):
            self.events.append(kind)
            return {}

        def notify(self, kind, **data):
            self.events.append(kind)
            self.left.set()

    finished = threading.Event()
    channel = Replies()
    monkeypatch.setattr(host, "wait_background", lambda sid: finished.wait(2))
    tools = types.SimpleNamespace(
        handle_function_call=lambda *a, **k: json.dumps(
            {"session_id": "proc_123", "exit_code": 0, "output": "Background process started"}
        )
    )
    guard_tools(tools, channel)
    tools.handle_function_call("terminal", {"command": "pytest", "background": True})
    assert channel.events == ["gate"]
    finished.set()
    assert channel.left.wait(2)
    assert channel.events == ["gate", "gate_done"]


def test_install_failure_has_unambiguous_progress_message() -> None:
    from roi_swarm.api import _describe_event

    message = _describe_event({"node": "install:a", "status": "failed", "data": {}})
    assert "не удалась" in message and "✓" not in message


def test_pool_keeps_consuming_events_while_a_gate_waits(tmp_path, monkeypatch) -> None:
    script = tmp_path / "host.py"
    script.write_text(
        """
import json,sys
sys.stdin.readline()
print(json.dumps({"type":"gate","id":1}),flush=True)
sys.stdin.readline()
print(json.dumps({"type":"gate_done"}),flush=True)
print(json.dumps({"type":"final","text":"ok"}),flush=True)
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(waves, "HOST_SCRIPT", script)

    async def scenario():
        broker = InstallBroker()
        await broker.enter("other")
        job = asyncio.create_task(broker.install("installer", "echo x", str(tmp_path)))
        await asyncio.sleep(0)
        worker = asyncio.create_task(
            waves_pool(tmp_path).run(
                PlannedTask(id="a", objective="x"),
                goal="",
                cwd=str(tmp_path),
                broker=broker,
                on_event=lambda e: None,
            )
        )
        await asyncio.sleep(0.02)
        await broker.leave("other")
        await job
        return await asyncio.wait_for(worker, 3)

    assert asyncio.run(scenario()).ok


def test_unbrokered_environment_change_blocks_next_install(tmp_path) -> None:
    async def scenario():
        site = tmp_path / ".venv" / "Lib" / "site-packages"
        site.mkdir(parents=True)
        marker = site / "project.pth"
        marker.write_text(str(tmp_path))

        async def run(command, cwd):
            pytest.fail("ran install after environment drift")

        broker = InstallBroker(run=run)
        await broker.watch(str(tmp_path))
        marker.write_text(str(tmp_path / "roi-waves" / "gone"))
        problems = await broker.check_environment()
        assert problems
        result = await broker.install("a", "pip install x", str(tmp_path))
        assert "environment" in result["error"].lower()

    asyncio.run(scenario())


def test_environment_check_cannot_erase_concurrent_cleanup_quarantine(tmp_path):
    import threading

    async def scenario():
        broker = InstallBroker(run=lambda *a: pytest.fail("installation must stay blocked"))
        await broker.watch(str(tmp_path))
        guard = next(iter(broker._environments.values()))
        entered, finish = threading.Event(), threading.Event()
        original = guard.check

        def paused_check():
            entered.set()
            assert finish.wait(3)
            return original()

        guard.check = paused_check
        checking = asyncio.create_task(broker.check_environment())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            await broker.quarantine("worker process cleanup was not confirmed")
        finally:
            finish.set()
        assert "worker process cleanup was not confirmed" in await checking
        with pytest.raises(RuntimeError, match="environment"):
            await broker.enter("next")
        result = await broker.install("next", "pip install x", str(tmp_path))
        assert result["exit_code"] == -1

    asyncio.run(scenario())


def test_legitimate_install_updates_environment_baseline(tmp_path) -> None:
    async def scenario():
        site = tmp_path / ".venv" / "Lib" / "site-packages"
        site.mkdir(parents=True)
        target = tmp_path / "package"
        target.mkdir()

        async def run(command, cwd):
            (site / "project.pth").write_text(str(target))
            return {"output": "ok", "exit_code": 0, "error": None}

        broker = InstallBroker(run=run)
        await broker.watch(str(tmp_path))
        result = await broker.install("a", "pip install x", str(tmp_path))
        assert result["exit_code"] == 0
        assert await broker.check_environment() == []

    asyncio.run(scenario())


def test_two_concurrent_installs_are_serial_and_share_one_pause() -> None:
    async def scenario():
        release, first_started = asyncio.Event(), asyncio.Event()
        order = []

        async def run(command, cwd):
            order.append(command)
            if command == "first":
                first_started.set()
                await release.wait()
            return {"output": "", "exit_code": 0, "error": None}

        broker = InstallBroker(run=run)
        first = asyncio.create_task(broker.install("a", "first", "."))
        await first_started.wait()
        second = asyncio.create_task(broker.install("b", "second", "."))
        await asyncio.sleep(0)
        outsider = asyncio.create_task(broker.enter("c"))
        await asyncio.sleep(0)
        assert not outsider.done()
        release.set()
        await asyncio.gather(first, second, outsider)
        assert order == ["first", "second"]

    asyncio.run(scenario())


def test_cleanup_closes_admission_and_drains_late_background_registration(monkeypatch):
    import threading

    from roi_swarm import hermes_worker_host as host

    entered, release, process_exited = (threading.Event() for _ in range(3))
    events = []

    class Replies:
        def request(self, *a, **k):
            return {}

        def notify(self, kind, **data):
            events.append(kind)

    def dispatch(*a, **k):
        entered.set()
        assert release.wait(3)
        return json.dumps({"session_id": "late-process"})

    monkeypatch.setattr(host, "wait_background", lambda sid: process_exited.wait(3))
    tools = types.SimpleNamespace(handle_function_call=dispatch)
    cleanup = guard_tools(tools, Replies())
    caller = threading.Thread(
        target=lambda: tools.handle_function_call("terminal", {}), daemon=True
    )
    caller.start()
    assert entered.wait(2)
    finished = threading.Event()
    shutdown = threading.Thread(target=lambda: (cleanup(), finished.set()), daemon=True)
    shutdown.start()
    # Cleanup must still be waiting for the tool's late background registration.
    assert not finished.wait(0.05)
    release.set()
    caller.join(3)
    shutdown.join(3)
    assert finished.is_set()
    # Hand ownership to the parent's Job without releasing the process's gate.
    assert "gate_done" not in events
    refused = json.loads(tools.handle_function_call("terminal", {}))
    assert "shutting down" in refused["error"]
    process_exited.set()


def test_cleanup_hands_all_background_processes_to_parent_without_registry_kill(monkeypatch):
    import threading

    from roi_swarm import hermes_worker_host as host

    stopped = {"one": threading.Event(), "two": threading.Event()}
    events = []

    class Replies:
        def request(self, *a, **k):
            return {}

        def notify(self, kind, **data):
            events.append(kind)

    def dispatch(name, args, *a, **k):
        return json.dumps({"session_id": args["name"]})

    monkeypatch.setattr(host, "wait_background", lambda sid: stopped[sid].wait(3))
    tools = types.SimpleNamespace(handle_function_call=dispatch)
    cleanup = guard_tools(tools, Replies())
    for name in stopped:
        tools.handle_function_call("terminal", {"name": name})
    finished = threading.Event()
    thread = threading.Thread(target=lambda: (cleanup(), finished.set()), daemon=True)
    thread.start()
    try:
        assert finished.wait(1), "host cleanup must not block on registry pipe closure"
        assert events == [], "only parent OS cleanup may release outstanding process ownership"
    finally:
        for done in stopped.values():
            done.set()


def test_parent_scope_cleanup_precedes_broker_release(tmp_path, monkeypatch):
    script = tmp_path / "host.py"
    script.write_text(
        """
import json,sys
sys.stdin.readline()
print(json.dumps({"type":"gate","id":1}),flush=True)
sys.stdin.readline()
print(json.dumps({"type":"error","error":"cleanup failed"}),flush=True)
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(waves, "HOST_SCRIPT", script)
    order = []

    class Scope:
        launch_kwargs = {}

        def attach(self, pid):
            order.append("attach")

        def close(self):
            order.append("kill descendants")

    class Broker(InstallBroker):
        async def leave(self, worker):
            order.append("release")
            await super().leave(worker)

    monkeypatch.setattr(waves, "WorkerProcessScope", Scope)
    result = asyncio.run(
        waves_pool(tmp_path).run(
            PlannedTask(id="a", objective="x"),
            goal="",
            cwd=str(tmp_path),
            broker=Broker(),
            on_event=lambda e: None,
        )
    )
    assert not result.ok
    assert order == ["attach", "kill descendants", "release"]


def test_unconfirmed_process_cleanup_quarantines_environment(tmp_path, monkeypatch):
    host = tmp_path / "host.py"
    host.write_text(INSTALL_HOST, encoding="utf-8")
    monkeypatch.setattr(waves, "HOST_SCRIPT", host)

    class Scope:
        launch_kwargs = {}

        def attach(self, pid):
            pass

        def close(self):
            raise OSError("cleanup unconfirmed")

    async def fake_install(command, cwd):
        return {"output": "ok", "exit_code": 0, "error": None}

    monkeypatch.setattr(waves, "WorkerProcessScope", Scope)

    async def scenario():
        broker = InstallBroker(run=fake_install)
        with pytest.raises(RuntimeError, match="quarantined"):
            await waves_pool(tmp_path).run(
                PlannedTask(id="a", objective="x"),
                goal="",
                cwd=str(tmp_path),
                broker=broker,
                on_event=lambda e: None,
            )
        with pytest.raises(RuntimeError, match="environment"):
            await asyncio.wait_for(broker.enter("next"), 2)
        result = await broker.install("next", "pip install x", str(tmp_path))
        assert result["exit_code"] == -1

    asyncio.run(scenario())


def waves_pool(tmp_path):
    return HermesWorkerPool(
        python=sys.executable,
        hermes_root=str(tmp_path),
        base_url="u",
        api_key="k",
        api_mode="chat_completions",
        model="m",
        disabled_toolsets=[],
        max_iterations=1,
    )
