"""Runner sleep integration: no network, no real repository changes."""

from roi_swarm.agent_loop import AgentTurn
from roi_swarm.config import SleepConfig

pytest_plugins = ("test_project_isolation",)


def test_sleep_has_unlimited_default_budgets():
    config = SleepConfig()
    assert config.max_steps_per_cycle is None
    assert config.daily_call_limit is None
    assert config.enabled is True
    assert config.auto_improve is False


class FakeSleep:
    def __init__(self):
        self.count = 0
        self.completed = []

    async def foreground_started(self):
        self.count += 1

    def foreground_finished(self, completed_run=None):
        self.count = max(0, self.count - 1)
        if completed_run:
            self.completed.append(completed_run)

    async def close(self):
        pass

    def status(self):
        return {"foreground_count": self.count}


async def test_external_tools_keep_sleep_paused_between_http_rounds(scoped_runner, monkeypatch):
    runner = scoped_runner
    runner.sleep = FakeSleep()
    outputs = [
        AgentTurn(
            "",
            [
                {
                    "id": "read-one",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
            "tool_calls",
        ),
        AgentTurn("done", [], "stop"),
    ]

    async def respond(*args, **kwargs):
        assert runner.sleep.count >= 1
        return outputs.pop(0)

    monkeypatch.setattr(runner, "_respond_with_tools", respond)
    messages = [{"role": "user", "content": "inspect"}]
    first = await runner.respond_with_tools(messages, [], "round-one")
    assert runner.sleep.count == 1
    assert runner.sleep_status()["pending_tool_turns"] == 1
    assert runner.sleep.completed == []
    messages += [
        {"role": "assistant", "content": None, "tool_calls": first.tool_calls},
        {"role": "tool", "tool_call_id": first.tool_calls[0]["id"], "content": "file data"},
    ]
    await runner.respond_with_tools(messages, [], "round-two")
    assert runner.sleep.count == 0
    assert runner.sleep_status()["pending_tool_turns"] == 0
    assert len(runner.sleep.completed) == 1


async def test_tool_leases_survive_restart_and_can_be_explicitly_released(
    scoped_runner, monkeypatch
):
    runner = scoped_runner
    runner.sleep = FakeSleep()

    async def respond(*args, **kwargs):
        return AgentTurn("", [{"id": "pending", "function": {"name": "read_file"}}], "tool_calls")

    monkeypatch.setattr(runner, "_respond_with_tools", respond)
    await runner.respond_with_tools([{"role": "user", "content": "inspect"}], [], "lease")
    assert runner._tool_lease_path.exists()
    runner._load_tool_leases()
    assert len(runner._tool_leases) == 1
    result = await runner.release_tool_leases()
    assert result["pending_tool_turns"] == 0
    assert runner.sleep.count == 0


async def test_reflection_budget_and_failed_git_preflight_prevent_proposal_call(scoped_runner):
    runner = scoped_runner
    runner.settings.sleep.auto_improve = True
    calls = []

    class Engine:
        async def preflight(self):
            return {"state": "blocked_dirty", "reason": "dirty"}

    runner.improvements = Engine()

    async def generate(**kwargs):
        calls.append(kwargs)
        return kwargs["schema"](observations=["reviewed"], done=True)

    runner.gateway.generate_structured = generate
    snapshot = {"scope": "alpha", "source_run_id": "x", "metrics": {}}
    await runner._reflect_sleep(snapshot)
    assert calls[0]["context_window_tokens"] == 800000
    await runner._save_sleep_lessons(snapshot, {"lessons": [], "done": True})
    assert len(calls) == 1
    assert runner.sleep_status()["auto_improve_state"] == "blocked_dirty"


async def test_auto_improve_runs_propose_tests_review_then_promote(scoped_runner):
    runner = scoped_runner
    runner.settings.sleep.auto_improve = True
    events = []

    class Engine:
        async def preflight(self):
            return {"state": "ready", "base_commit": "a" * 40}

        async def prepare(self, proposal):
            events.append(("prepare", proposal))
            return {"state": "tested", "commit_id": "b" * 40}

        async def promote(self, commit):
            events.append(("promote", commit))
            return {"state": "promoted", "commit_id": commit}

    runner.improvements = Engine()

    async def propose(snapshot):
        events.append(("propose", snapshot["scope"]))
        return {
            "title": "test improvement",
            "changes": [
                {"path": "tests/test_added.py", "content": "def test_ok():\n    assert True\n"}
            ],
        }

    runner._propose_improvement = propose
    await runner._save_sleep_lessons(
        {"scope": "alpha", "source_run_id": "x"}, {"lessons": [], "done": True}
    )
    assert [event[0] for event in events] == ["propose", "prepare", "promote"]
    assert runner.sleep_status()["restart_required"] is True
    assert runner.sleep_status()["auto_improve_state"] == "promoted"


async def test_foreground_blocks_pending_promotion(scoped_runner):
    runner = scoped_runner
    runner.settings.sleep.auto_improve = True
    runner._active_operations = 1
    called = []

    async def propose(snapshot):
        called.append(snapshot)

    runner._propose_improvement = propose
    await runner._save_sleep_lessons({"scope": "a", "source_run_id": "x"}, {"lessons": []})
    assert called == []


async def test_late_background_proposal_cannot_advance_after_foreground_epoch(scoped_runner):
    runner = scoped_runner
    runner.settings.sleep.auto_improve = True
    prepared = []

    class Engine:
        async def preflight(self):
            return {"state": "ready"}

        async def prepare(self, proposal):
            prepared.append(proposal)
            return {"state": "tested", "commit_id": "a" * 40}

    runner.improvements = Engine()

    async def propose(snapshot):
        runner._foreground_epoch += 1
        return {"title": "change", "changes": [{"path": "tests/test_added.py", "content": ""}]}

    runner._propose_improvement = propose
    await runner._save_sleep_lessons({"source_run_id": "x"}, {"lessons": []})
    assert prepared == []


async def test_lease_storage_failure_never_releases_foreground_hold(scoped_runner, monkeypatch):
    runner = scoped_runner
    runner.sleep = FakeSleep()

    async def respond(*args, **kwargs):
        return AgentTurn(
            "", [{"id": "unsafe-storage", "function": {"name": "read_file"}}], "tool_calls"
        )

    def failed_persist():
        runner._lease_recovery_hold = True
        runner._sleep_error = "tool-lease-storage"
        return False

    monkeypatch.setattr(runner, "_respond_with_tools", respond)
    monkeypatch.setattr(runner, "_persist_tool_leases", failed_persist)
    import pytest

    with pytest.raises(OSError, match="Tool lease storage unavailable"):
        await runner.respond_with_tools(
            [{"role": "user", "content": "inspect"}], [], "broken-lease"
        )
    assert runner.sleep.count >= 1
    with pytest.raises(OSError):
        await runner.release_tool_leases()
    assert runner.sleep.count >= 1
    assert runner.sleep_status()["tool_lease_recovery_hold"] is True


async def test_disk_replace_failure_does_not_issue_external_tools(scoped_runner, monkeypatch):
    from pathlib import Path

    import pytest

    runner = scoped_runner
    runner.sleep = FakeSleep()
    runner._tool_lease_path.parent.mkdir(parents=True, exist_ok=True)
    runner._tool_lease_path.write_text("[]", encoding="utf-8")
    original_replace = Path.replace

    def fail_lease_replace(path, target):
        if target == runner._tool_lease_path:
            raise PermissionError("synthetic locked lease file")
        return original_replace(path, target)

    async def respond(*args, **kwargs):
        return AgentTurn("", [{"id": "external", "function": {"name": "read_file"}}], "tool_calls")

    monkeypatch.setattr(Path, "replace", fail_lease_replace)
    monkeypatch.setattr(runner, "_respond_with_tools", respond)
    with pytest.raises(OSError, match="Tool lease storage unavailable"):
        await runner.respond_with_tools([], [], "locked-lease")
    assert runner._tool_lease_path.read_text(encoding="utf-8") == "[]"
    assert runner.sleep_status()["tool_lease_recovery_hold"] is True


async def test_foreground_waits_only_for_already_started_promotion(scoped_runner):
    runner = scoped_runner
    events = []
    runner.sleep = FakeSleep()

    class Engine:
        async def wait_for_promotion(self):
            events.append("merge settled")
            return {"state": "promoted", "commit_id": "a" * 40}

    runner.improvements = Engine()
    await runner._pause_sleep()
    assert events == ["merge settled"]
    assert runner.sleep_status()["restart_required"] is True
    assert runner.sleep.count == 1
    runner._finish_sleep(None)


async def test_same_model_tool_id_cannot_release_another_dialog(scoped_runner, monkeypatch):
    runner = scoped_runner
    runner.sleep = FakeSleep()

    async def respond(messages, *args, **kwargs):
        if messages[-1].get("role") == "tool":
            return AgentTurn("done", [], "stop")
        return AgentTurn("", [{"id": "same-id", "function": {"name": "read_file"}}], "tool_calls")

    monkeypatch.setattr(runner, "_respond_with_tools", respond)
    a = [{"role": "user", "content": "dialog a"}]
    b = [{"role": "user", "content": "dialog b"}]
    first = await runner.respond_with_tools(a, [], "dialog-a")
    second = await runner.respond_with_tools(b, [], "dialog-b")
    assert first.tool_calls[0]["id"] != second.tool_calls[0]["id"]
    await runner.respond_with_tools(
        [
            *a,
            {"role": "assistant", "tool_calls": first.tool_calls},
            {"role": "tool", "tool_call_id": first.tool_calls[0]["id"], "content": "a finished"},
        ],
        [],
        "dialog-a-return",
    )
    assert runner.sleep.count == 1
    assert runner.sleep_status()["pending_tool_turns"] == 1


async def test_recovered_sleep_manager_inherits_active_requests(scoped_runner, monkeypatch):
    import roi_swarm.sleep as module

    runner = scoped_runner
    runner._active_operations = 1

    class Manager(FakeSleep):
        def __init__(self, *args, **kwargs):
            super().__init__()

        async def start(self):
            assert self.count >= 1

    monkeypatch.setattr(module, "SleepManager", Manager)
    await runner.start()
    assert runner.sleep.count == 1
    runner._active_operations = 0
    runner._finish_sleep(None)
    assert runner.sleep.count == 0


def test_promotion_guard_fences_foreground_tools_restart_and_old_epoch(scoped_runner):
    runner = scoped_runner
    assert runner.improvements.promotion_allowed() is True
    runner._active_operations = 1
    assert runner.improvements.promotion_allowed() is False
    runner._active_operations = 0
    runner._tool_leases.add("inflight")
    assert runner.improvements.promotion_allowed() is False
    runner._tool_leases.clear()
    runner._foreground_epoch += 1
    assert runner.improvements.promotion_allowed() is False
    runner._promotion_epoch = runner._foreground_epoch
    assert runner.improvements.promotion_allowed() is True
    runner._restart_required = True
    assert runner.improvements.promotion_allowed() is False


async def test_reflection_distinguishes_completed_task_from_research_steps(scoped_runner):
    runner = scoped_runner
    seen = []

    async def generate(**kwargs):
        seen.append(kwargs)
        return kwargs["schema"](done=True)

    runner.gateway.generate_structured = generate
    result = runner._sleep_completion(
        "completed",
        "alpha",
        {
            "tasks": [{"id": "one"}],
            "worker_results": [{"task_id": "one"}, {"task_id": "one"}],
            "verification": {"passed": True},
        },
    )
    assert result["metrics"]["retry_count"] == 1
    assert result["metrics"]["verification_passed"] == 1
    await runner._reflect_sleep({"source_run_id": "completed", "steps_completed": 0})
    assert "already completed" in seen[0]["system"]
    assert "research steps" in seen[0]["system"]


async def test_independent_improvement_review_fails_closed(scoped_runner):
    runner = scoped_runner
    seen = []

    async def review(**kwargs):
        seen.append(kwargs)
        return kwargs["schema"](approved=False, concerns=["not justified"])

    runner.gateway.generate_structured = review
    assert await runner._review_improvement("diff text") is False
    assert seen[0]["model"] == runner.settings.model_for("critic").model
    assert seen[0]["context_window_tokens"] == 800000
