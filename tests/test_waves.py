import asyncio
import json
import sys

import pytest

import roi_swarm.waves as waves
from roi_swarm.schemas import PlannedTask
from roi_swarm.waves import (
    HermesWorkerPool,
    WaveOrchestrator,
    WavePlan,
    WaveReview,
    WorkerOutcome,
    working_directory,
)


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def __call__(self, node: str, status: str, data: dict) -> None:
        self.events.append((node, status, data))


def _orchestrator(reviews, worker, events, finals=None, maximum_waves=4):
    prompts: list[tuple[str, str]] = []
    plan = WavePlan(
        summary="two parts",
        tasks=[
            PlannedTask(id="a", objective="write a.py"),
            PlannedTask(id="b", objective="write b.py"),
        ],
    )
    queue = list(reviews)
    planned = [plan]

    async def structured(system, prompt, schema):
        prompts.append((schema.__name__, prompt))
        if schema is WavePlan:
            # Later planner calls are split requests; an empty plan means "cannot split".
            return planned.pop(0) if planned else WavePlan(summary="", tasks=[])
        return queue.pop(0)

    async def finalize(system, prompt, sink):
        (finals if finals is not None else []).append(prompt)
        for piece in ("Готово", ": a и b"):
            if sink:
                sink(piece)
        return "Готово: a и b"

    return (
        WaveOrchestrator(
            structured=structured,
            finalize=finalize,
            run_worker=worker,
            maximum_waves=maximum_waves,
            on_event=events,
        ),
        prompts,
    )


def test_workers_of_one_wave_run_in_parallel_and_rejections_spawn_a_new_wave() -> None:
    started: list[str] = []
    both_running = asyncio.Event()

    async def worker(task, *, goal, cwd, on_event):
        started.append(task.id)
        on_event({"type": "tool_start", "name": "write_file", "args": task.id})
        if len(started) == 2:
            both_running.set()
        if task.id in {"a", "b"}:
            # Each first-wave worker waits for the other: proves real overlap.
            await asyncio.wait_for(both_running.wait(), 2)
        return WorkerOutcome(task.id, task.objective, True, f"{task.id} done", 1, ["write_file"])

    events = Recorder()
    finals: list[str] = []
    reviews = [
        WaveReview(
            done=False,
            accepted=["a"],
            feedback="b has no tests",
            next_tasks=[PlannedTask(id="b", objective="fix b.py and add tests")],
        ),
        WaveReview(done=True, accepted=["b-2"]),
    ]
    orchestrator, prompts = _orchestrator(reviews, worker, events, finals)
    deltas: list[str] = []

    answer = asyncio.run(
        orchestrator.run(
            [{"role": "user", "content": "do a and b"}],
            request="do a and b",
            cwd=".",
            on_delta=deltas.append,
        )
    )

    assert answer == "Готово: a и b"
    assert "".join(deltas) == answer
    assert started == ["a", "b", "b-2"]  # new wave got a unique id, not a duplicate
    waves_run = [data["wave"] for node, status, data in events.events if node == "wave"]
    assert waves_run == [1, 2]
    assert ("worker:a", "tool", {"tool": "write_file", "args": "a"}) in events.events
    assert [name for name, _ in prompts] == ["WavePlan", "WaveReview", "WavePlan", "WaveReview"]
    assert "fix b.py and add tests" in finals[0] and "write a.py" in finals[0]
    assert "NOT FINISHED" not in finals[0]
    assert events.events[-1][0] == "synthesizer"


def test_failed_worker_is_never_accepted_and_wave_limit_reports_open_work() -> None:
    async def worker(task, *, goal, cwd, on_event):
        raise RuntimeError("hermes crashed")

    events = Recorder()
    finals: list[str] = []
    reviews = [
        WaveReview(
            done=False,
            accepted=["a", "b"],
            next_tasks=[PlannedTask(id="a", objective="retry a")],
        )
    ]
    orchestrator, _ = _orchestrator(reviews, worker, events, finals, maximum_waves=1)

    asyncio.run(orchestrator.run([], request="x", cwd="."))

    statuses = {node: status for node, status, _ in events.events if node.startswith("worker:")}
    assert statuses == {"worker:a": "failed", "worker:b": "failed"}
    assert "NOT FINISHED after 1 waves: retry a" in finals[0]
    assert "RuntimeError: hermes crashed" in finals[0]


FAKE_HOST = """
import json, sys
task = json.loads(sys.stdin.readline())
print("banner noise")
print(json.dumps({"type": "tool_start", "id": "1", "name": "terminal", "args": "ls"}))
print(json.dumps({"type": "final", "text": "did " + task["objective"] + " in " + task["cwd"]}))
"""


def test_worker_pool_streams_tool_events_from_the_hermes_subprocess(tmp_path, monkeypatch) -> None:
    host = tmp_path / "host.py"
    host.write_text(FAKE_HOST, encoding="utf-8")
    monkeypatch.setattr(waves, "HOST_SCRIPT", host)
    pool = HermesWorkerPool(
        python=sys.executable,
        hermes_root=str(tmp_path),
        base_url="http://x/v1",
        api_key="k",
        api_mode="chat_completions",
        model="luna",
        disabled_toolsets=["delegation"],
        max_iterations=5,
    )
    seen: list[dict] = []

    outcome = asyncio.run(
        pool.run(
            PlannedTask(id="t1", objective="list files"),
            goal="g",
            cwd=str(tmp_path),
            on_event=seen.append,
        )
    )

    assert outcome.ok and outcome.report == f"did list files in {tmp_path}"
    assert outcome.tools == ["terminal"]
    assert seen == [{"type": "tool_start", "id": "1", "name": "terminal", "args": "ls"}]


def test_worker_pool_reports_crash_without_final_event(tmp_path, monkeypatch) -> None:
    host = tmp_path / "host.py"
    host.write_text("import sys; sys.exit(3)", encoding="utf-8")
    monkeypatch.setattr(waves, "HOST_SCRIPT", host)
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

    outcome = asyncio.run(
        pool.run(PlannedTask(id="t", objective="o"), goal="", cwd="", on_event=lambda e: None)
    )

    assert not outcome.ok and "code 3" in outcome.report


def _pool(tmp_path):
    return HermesWorkerPool(
        python=sys.executable,
        hermes_root=str(tmp_path),
        base_url="u",
        api_key="worker-key",
        api_mode="chat_completions",
        model="m",
        disabled_toolsets=[],
        max_iterations=1,
    )


def test_lingering_worker_host_is_killed_after_its_final_answer(tmp_path, monkeypatch) -> None:
    host = tmp_path / "host.py"
    host.write_text(
        "import json, sys, time\n"
        "sys.stdin.readline()\n"
        "print(json.dumps({'type': 'final', 'text': 'done'}), flush=True)\n"
        "time.sleep(600)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(waves, "HOST_SCRIPT", host)
    monkeypatch.setattr(waves, "EXIT_GRACE_SECONDS", 0.5)

    outcome = asyncio.run(
        asyncio.wait_for(
            _pool(tmp_path).run(
                PlannedTask(id="t", objective="o"), goal="", cwd="", on_event=lambda e: None
            ),
            30,
        )
    )

    assert outcome.ok and outcome.report == "done"


def test_worker_env_carries_only_the_worker_key_not_roi_secrets(tmp_path, monkeypatch) -> None:
    host = tmp_path / "host.py"
    host.write_text(
        "import json, os, sys\n"
        "sys.stdin.readline()\n"
        "names = sorted(n for n in os.environ if n.upper().startswith('ROI_'))\n"
        "print(json.dumps({'type': 'final', 'text': ','.join(names)}))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(waves, "HOST_SCRIPT", host)
    monkeypatch.setenv("ROI_SWARM_API_KEY", "server-secret")
    monkeypatch.setenv("ROI_PROVIDER_API_KEY", "provider-secret")

    outcome = asyncio.run(
        _pool(tmp_path).run(
            PlannedTask(id="t", objective="o"), goal="", cwd="", on_event=lambda e: None
        )
    )

    assert outcome.report == "ROI_WORKER_API_KEY"


def test_review_not_done_without_next_tasks_is_reported_as_unfinished() -> None:
    async def worker(task, *, goal, cwd, on_event):
        return WorkerOutcome(task.id, task.objective, True, "half", 0, [])

    finals: list[str] = []
    reviews = [WaveReview(done=False, accepted=["a"], feedback="b is wrong", next_tasks=[])]
    orchestrator, _ = _orchestrator(reviews, worker, Recorder(), finals)

    asyncio.run(orchestrator.run([], request="x", cwd="."))

    assert "NOT FINISHED after 1 waves: b is wrong" in finals[0]


def test_project_folder_named_in_plan_overrides_hermes_cwd(tmp_path) -> None:
    project = tmp_path / "lilac-archive"
    project.mkdir()
    seen: list[str] = []

    async def worker(task, *, goal, cwd, on_event):
        seen.append(cwd)
        return WorkerOutcome(task.id, task.objective, True, "ok", 1, [])

    plan = WavePlan(
        summary="s",
        workdir=str(project),
        tasks=[PlannedTask(id="a", objective="x"), PlannedTask(id="b", objective="y")],
    )
    opened: list[str] = []

    async def structured(system, prompt, schema):
        return plan if schema is WavePlan else WaveReview(done=True, accepted=["a", "b"])

    async def finalize(system, prompt, sink):
        return "ok"

    orchestrator = WaveOrchestrator(
        structured=structured,
        finalize=finalize,
        run_worker=worker,
        maximum_waves=2,
        on_event=Recorder(),
        open_workspace=lambda folder: opened.append(folder),
    )
    asyncio.run(orchestrator.run([], request=f"edit {project}", cwd="D:/ROI"))

    assert seen == [str(project.resolve())] * 2
    assert opened == [str(project.resolve())]


@pytest.mark.parametrize("candidate", ["/srv/my app", "\\srv\\my app"])
def test_named_folder_scans_posix_absolute_paths(monkeypatch, candidate):
    import roi_swarm.waves as module

    def existing(text, *, files):
        return "fixture-project" if text.replace("\\", "/") == "/srv/my app" else None

    monkeypatch.setattr(module, "_existing_folder", existing)
    assert module._named_folder(f"edit {candidate} then test") == "fixture-project"
    for text in (
        "see https://example.test/srv/my app",
        "see https://example.test/releases-/srv/my app",
        "see https://example.test/?path=/srv/my app",
        "edit ./srv/my app then test",
        "edit ../srv/my app then test",
    ):
        assert module._named_folder(text) is None


def test_project_folder_falls_back_to_existing_path_in_request(tmp_path) -> None:
    from roi_swarm.waves import project_directory

    project = tmp_path / "my app"
    (project / "src").mkdir(parents=True)
    (project / "src" / "x.py").write_text("")
    win = str(project).replace("/", "\\")
    want = str(project.resolve())

    assert project_directory("", f"можешь редактировать файлы {win}\n\nсделай", "D:/x") == want
    assert project_directory("", f"правь {win}\\,\tпотом тесты", "D:/x") == want
    assert project_directory("Z:/missing", "no path here", "D:/x") == "D:/x"
    assert project_directory(str(project), "anything", "D:/x") == want
    # A named file selects its folder.
    assert project_directory("", f'fix "{win}\\src\\x.py" now', "D:/x") == str(
        (project / "src").resolve()
    )
    # Relative planner folders resolve against the server process, never trusted.
    assert project_directory(".", "nothing", "D:/x") == "D:/x"


def test_project_folder_skips_drive_roots_and_system_folders(tmp_path, monkeypatch) -> None:
    from roi_swarm.waves import project_directory

    project = tmp_path / "proj"
    project.mkdir()
    system = tmp_path / "Windows"
    system.mkdir()
    monkeypatch.setenv("WINDIR", str(system))
    root = project.anchor
    text = f"see {root} and {system} then work in {project}"

    assert project_directory("", text, "D:/x") == str(project.resolve())
    assert project_directory(root, "x", "D:/x") == "D:/x"
    assert project_directory(str(system), "x", "D:/x") == "D:/x"


def test_follow_up_turn_keeps_the_project_named_earlier(tmp_path) -> None:
    from roi_swarm.waves import project_directory

    project = tmp_path / "lilac"
    project.mkdir()

    texts = ["продолжай", "ok", f"можешь редактировать файлы {project}"]
    assert project_directory("", texts, "D:/ROI") == str(project.resolve())


def test_files_from_earlier_turns_do_not_redirect_the_wave(tmp_path) -> None:
    from roi_swarm.waves import project_directory

    log = tmp_path / "logs" / "app.log"
    log.parent.mkdir()
    log.write_text("")

    assert project_directory("", ["продолжай", f"read {log} and fix"], "D:/ROI") == "D:/ROI"
    assert project_directory("", [f"read {log}"], "D:/ROI") == str(log.parent.resolve())


def test_folder_names_with_parentheses_and_line_references(tmp_path) -> None:
    from roi_swarm.waves import project_directory

    copy = tmp_path / "x (1)"
    copy.mkdir()
    source = copy / "app.py"
    source.write_text("")

    assert project_directory("", f"work in {copy}", "D:/x") == str(copy.resolve())
    assert project_directory("", f"see {source}:12:3 please", "D:/x") == str(copy.resolve())


def test_single_remaining_task_is_split_into_parallel_atomic_tasks() -> None:
    started: list[str] = []

    async def worker(task, *, goal, cwd, on_event):
        started.append(task.id)
        return WorkerOutcome(task.id, task.objective, True, "ok", 1, [])

    plans = [
        WavePlan(
            summary="s",
            tasks=[PlannedTask(id="a", objective="a"), PlannedTask(id="b", objective="b")],
        ),
        WavePlan(
            summary="split",
            tasks=[
                PlannedTask(id="cache", objective="fix cache"),
                PlannedTask(id="outbox", objective="fix outbox"),
                PlannedTask(id="docs", objective="write runbook"),
            ],
        ),
    ]
    reviews = [
        WaveReview(
            done=False,
            accepted=["a", "b"],
            next_tasks=[PlannedTask(id="rest", objective="fix cache, outbox and write runbook")],
        ),
        WaveReview(done=True, accepted=["cache", "outbox", "docs"]),
    ]
    prompts: list[tuple[str, str]] = []

    async def structured(system, prompt, schema):
        prompts.append((schema.__name__, prompt))
        return plans.pop(0) if schema is WavePlan else reviews.pop(0)

    async def finalize(system, prompt, sink):
        return "ok"

    events = Recorder()
    orchestrator = WaveOrchestrator(
        structured=structured,
        finalize=finalize,
        run_worker=worker,
        maximum_waves=4,
        on_event=events,
    )
    asyncio.run(orchestrator.run([], request="r", cwd="."))

    assert started == ["a", "b", "cache", "outbox", "docs"]
    assert [name for name, _ in prompts] == ["WavePlan", "WaveReview", "WavePlan", "WaveReview"]
    assert "fix cache, outbox and write runbook" in prompts[2][1]
    sizes = [d["task_count"] for n, s, d in events.events if n == "wave" and s == "running"]
    assert sizes == [2, 3]


def test_split_never_repeats_accepted_work() -> None:
    started: list[str] = []

    async def worker(task, *, goal, cwd, on_event):
        started.append(task.id)
        return WorkerOutcome(task.id, task.objective, True, "ok", 1, [])

    plans = [
        WavePlan(
            summary="s",
            tasks=[
                PlannedTask(id="spec", objective="write spec"),
                PlannedTask(id="cache", objective="fix cache"),
            ],
        ),
        WavePlan(
            summary="split",
            tasks=[
                PlannedTask(id="spec", objective="Write  SPEC"),
                PlannedTask(id="tests", objective="add tests"),
                PlannedTask(id="docs", objective="write runbook"),
            ],
        ),
    ]
    reviews = [
        WaveReview(
            done=False,
            accepted=["spec"],
            next_tasks=[PlannedTask(id="rest", objective="cache, tests and runbook")],
        ),
        WaveReview(done=True, accepted=["tests", "docs"]),
    ]
    prompts: list[str] = []

    async def structured(system, prompt, schema):
        prompts.append(prompt)
        return plans.pop(0) if schema is WavePlan else reviews.pop(0)

    async def finalize(system, prompt, sink):
        return "ok"

    orchestrator = WaveOrchestrator(
        structured=structured,
        finalize=finalize,
        run_worker=worker,
        maximum_waves=3,
        on_event=Recorder(),
    )
    asyncio.run(orchestrator.run([], request="r", cwd="."))

    assert started == ["spec", "cache", "tests", "docs"]
    assert "- spec: write spec" in prompts[2]


def test_failed_split_keeps_the_reviewer_task() -> None:
    from roi_swarm.waves import WaveOrchestrator as Orchestrator

    calls = {"plan": 0}
    started: list[str] = []

    async def worker(task, *, goal, cwd, on_event):
        started.append(task.id)
        return WorkerOutcome(task.id, task.objective, True, "ok", 1, [])

    reviews = [
        WaveReview(done=False, accepted=["a"], next_tasks=[PlannedTask(id="r", objective="rest")]),
        WaveReview(done=True, accepted=["r"]),
    ]

    async def structured(system, prompt, schema):
        if schema is WavePlan:
            calls["plan"] += 1
            if calls["plan"] == 1:
                return WavePlan(summary="s", tasks=[PlannedTask(id="a", objective="a")])
            raise RuntimeError("bad json")
        return reviews.pop(0)

    async def finalize(system, prompt, sink):
        return "ok"

    orchestrator = Orchestrator(
        structured=structured,
        finalize=finalize,
        run_worker=worker,
        maximum_waves=3,
        on_event=Recorder(),
    )
    asyncio.run(orchestrator.run([], request="r", cwd="."))

    assert started == ["a", "r"]


def test_working_directory_comes_from_hermes_system_prompt(tmp_path) -> None:
    messages = [{"role": "system", "content": f"Workspace:\n- Root: {tmp_path}\n- Branch: x"}]

    assert working_directory(messages, "fallback") == str(tmp_path)
    assert working_directory([{"role": "system", "content": "Root: Z:/nope"}], "fb") == "fb"


def test_worker_host_prompt_carries_only_its_objective_and_context() -> None:
    from roi_swarm.hermes_worker_host import build_prompt

    prompt = build_prompt({"objective": "write a.py", "context": "owns a.py", "goal": "app"})

    assert prompt.startswith("OBJECTIVE:\nwrite a.py")
    assert "owns a.py" in prompt and "background only" in prompt


@pytest.mark.parametrize("payload", ["not json"])
def test_worker_host_reports_bad_input_as_error_event(payload, monkeypatch, capsys) -> None:
    import io

    from roi_swarm import hermes_worker_host

    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    monkeypatch.setattr(sys, "stdout", sys.stdout)  # main() redirects it; restore afterwards
    assert hermes_worker_host.main() == 1
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    assert lines[-1]["type"] == "error"
