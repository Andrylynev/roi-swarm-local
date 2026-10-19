"""One dashboard task across external Hermes tool rounds; no provider calls."""

import asyncio
import json

import pytest
from test_sleep_runtime import FakeSleep

from roi_swarm.agent_loop import AgentTurn

pytest_plugins = ("test_project_isolation",)

MODEL_MARKER = (
    "[System: The active model for this chat has changed to roi-swarm via provider roi-swarm. "
    "From this point forward, use this runtime metadata when answering questions about "
    "what model/provider is active.]"
)


@pytest.fixture
def coordinator(scoped_runner, monkeypatch):
    import roi_swarm.runner as module

    runner = scoped_runner
    runner.sleep = FakeSleep()
    seen = []
    outputs = []

    class Coordinator:
        def __init__(self, *args, **kwargs):
            self.delegate = kwargs["run_swarm"]

        async def respond(self, messages, tools, *, run_id):
            seen.append((messages, run_id, self.delegate))
            result = outputs.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result

    monkeypatch.setattr(module, "HermesToolAgent", Coordinator)
    return runner, outputs, seen


def tool_turn():
    return AgentTurn(
        "Inspecting",
        [{"id": "model-reused-id", "function": {"name": "read_file", "arguments": "{}"}}],
        "tool_calls",
    )


def with_result(messages, turn):
    return [
        *messages,
        {"role": "assistant", "content": turn.content, "tool_calls": turn.tool_calls},
        {"role": "tool", "tool_call_id": turn.tool_calls[0]["id"], "content": "file data"},
    ]


async def test_tool_rounds_share_one_running_task_until_final_answer(coordinator):
    runner, outputs, _ = coordinator
    outputs.extend([tool_turn(), tool_turn(), AgentTurn("Verified result")])
    messages = [{"role": "user", "content": "Inspect the project"}]
    first = await runner.respond_with_tools(messages, [], "http-first")
    initial = runner.store.get_run("http-first")
    assert initial["status"] == "running"
    assert runner.sleep.completed == []

    messages = with_result(messages, first)
    second = await runner.respond_with_tools(messages, [], "http-second")
    assert len(runner.store.list_runs()) == 1
    assert runner.store.get_run("http-first")["status"] == "running"
    messages = with_result(messages, second)
    await runner.respond_with_tools(messages, [], "http-final")

    records = runner.store.list_runs()
    assert len(records) == 1
    assert records[0]["id"] == "http-first"
    assert records[0]["request"] == "Inspect the project"
    assert records[0]["status"] == "completed"
    assert records[0]["answer"] == "Verified result"
    assert [item["run_id"] for item in runner.sleep.completed] == ["http-first"]
    assert runner.sleep.count == 0
    assert runner._tool_leases == set()


@pytest.mark.parametrize("as_blocks", [False, True])
async def test_model_notice_does_not_replace_task_or_split_its_tool_rounds(coordinator, as_blocks):
    from roi_swarm.agent_loop import _current_turn
    from roi_swarm.api import ChatMessage, _last_user_text

    runner, outputs, seen = coordinator
    outputs.extend([tool_turn(), AgentTurn("done")])
    notice = [{"type": "text", "text": MODEL_MARKER}] if as_blocks else MODEL_MARKER
    messages = [
        {"role": "user", "content": "Inspect the project"},
        {"role": "user", "content": notice},
    ]
    assert _last_user_text([ChatMessage(**m) for m in messages]) == "Inspect the project"
    assert _current_turn(messages)[0] == "Inspect the project"
    first = await runner.respond_with_tools(messages, [], "notice-first")
    assert runner.store.get_run("notice-first")["request"] == "Inspect the project"
    assert seen[0][0] == messages  # Do not strip runtime metadata from the provider context.
    messages = with_result(messages, first)
    messages.append({"role": "user", "content": MODEL_MARKER})
    await runner.respond_with_tools(messages, [], "notice-final")
    assert len(runner.store.list_runs()) == 1
    assert runner.store.get_run("notice-first")["status"] == "completed"


@pytest.mark.parametrize("as_blocks", [False, True])
def test_notice_prefixed_request_keeps_actual_request(tmp_path, as_blocks):
    from roi_swarm.agent_loop import _current_turn
    from roi_swarm.api import ChatMessage, _last_user_text
    from roi_swarm.context import ContextPipeline

    text = MODEL_MARKER + "\n\nInspect the project"
    content = [{"type": "text", "text": text}] if as_blocks else text
    messages = [{"role": "user", "content": content}]
    assert _last_user_text([ChatMessage(**m) for m in messages]) == "Inspect the project"
    assert _current_turn(messages)[0] == "Inspect the project"
    pipeline = ContextPipeline(tmp_path)
    prepared = pipeline.prepare(messages, run_id="prefixed")
    assert prepared.current_request == "Inspect the project"
    assert MODEL_MARKER in pipeline.pack_for_planner(prepared)


async def test_model_notice_after_tool_result_cannot_start_fresh_waves(coordinator, monkeypatch):
    runner, _, _ = coordinator
    runner.settings.hermes_workers.enabled = True
    monkeypatch.setattr(runner, "_hermes_workers_available", lambda: True)
    assessed = []

    async def assess(request):
        assessed.append(request)
        return True

    monkeypatch.setattr(runner, "_assess_for_swarm", assess)
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    fresh = [
        {"role": "user", "content": "Inspect the project"},
        {"role": "user", "content": MODEL_MARKER},
    ]
    assert await runner.should_use_waves(fresh, tools)
    assert assessed == ["Inspect the project"]
    assessed.clear()
    continued = with_result(fresh, tool_turn())
    continued.append({"role": "user", "content": MODEL_MARKER})
    assert not await runner.should_use_waves(continued, tools)
    assert assessed == []


async def test_tool_continuation_survives_runner_restart(coordinator, tmp_path, monkeypatch):
    from roi_swarm.runner import SwarmRunner

    runner, outputs, _ = coordinator
    outputs.extend([tool_turn(), AgentTurn("after restart")])
    messages = [{"role": "user", "content": "Inspect the project"}]
    first = await runner.respond_with_tools(messages, [], "before-restart")
    restarted = SwarmRunner(tmp_path / "config" / "settings.yaml")
    restarted.sleep = FakeSleep()
    restarted._load_tool_leases()
    try:
        await restarted.respond_with_tools(with_result(messages, first), [], "after-restart")
        records = restarted.store.list_runs()
        assert len(records) == 1
        assert records[0]["id"] == "before-restart"
        assert records[0]["status"] == "completed"
        assert restarted._tool_leases == set()
    finally:
        await restarted.close()


async def test_parallel_dialogs_with_identical_requests_stay_separate(coordinator):
    runner, outputs, _ = coordinator
    messages = [{"role": "user", "content": "Same request"}]
    outputs.extend([tool_turn(), tool_turn(), AgentTurn("a"), AgentTurn("b")])
    a, b = await asyncio.gather(
        runner.respond_with_tools(messages, [], "dialog-a"),
        runner.respond_with_tools(messages, [], "dialog-b"),
    )
    await asyncio.gather(
        runner.respond_with_tools(with_result(messages, a), [], "dialog-a-final"),
        runner.respond_with_tools(with_result(messages, b), [], "dialog-b-final"),
    )
    records = runner.store.list_runs()
    assert {r["id"] for r in records} == {"dialog-a", "dialog-b"}
    assert {r["answer"] for r in records} == {"a", "b"}


@pytest.mark.parametrize("failure", [RuntimeError("provider failed"), asyncio.CancelledError()])
async def test_failed_tool_continuation_finishes_original_run(coordinator, failure):
    runner, outputs, _ = coordinator
    outputs.extend([tool_turn(), failure])
    messages = [{"role": "user", "content": "Inspect the project"}]
    first = await runner.respond_with_tools(messages, [], "failed-first")
    with pytest.raises(type(failure)):
        await runner.respond_with_tools(with_result(messages, first), [], "failed-next")
    records = runner.store.list_runs()
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert runner.sleep.completed == []
    assert runner.sleep.count == 0


async def test_new_user_turn_starts_new_task_even_with_old_tool_history(coordinator):
    runner, outputs, _ = coordinator
    outputs.extend([tool_turn(), AgentTurn("new answer")])
    messages = [{"role": "user", "content": "first task"}]
    first = await runner.respond_with_tools(messages, [], "old-task")
    messages = with_result(messages, first)
    messages.append({"role": "user", "content": "second task"})
    await runner.respond_with_tools(messages, [], "new-task")
    records = {r["id"]: r for r in runner.store.list_runs()}
    assert set(records) == {"old-task", "new-task"}
    assert records["new-task"]["request"] == "second task"


async def test_unissued_tool_ids_and_different_project_cannot_join_task(coordinator):
    runner, outputs, _ = coordinator
    outputs.extend([tool_turn(), AgentTurn("other project"), AgentTurn("unknown token")])
    messages = [{"role": "user", "content": "task"}]
    first = await runner.respond_with_tools(messages, [], "scope-one", project_id="alpha")
    continued = with_result(messages, first)
    leases = runner._tool_leases.copy()
    with pytest.raises(ValueError, match="project scope"):
        await runner.respond_with_tools(continued, [], "scope-two", project_id="beta")
    assert runner._tool_leases == leases
    assert runner.sleep.count == 1
    forged = with_result(messages, first)
    forged[-1]["tool_call_id"] = "call_roi_scope-one__unissued"
    forged[-2]["tool_calls"] = [
        {**first.tool_calls[0], "id": "call_roi_scope-one__unissued"}
    ]
    await runner.respond_with_tools(forged, [], "scope-three", project_id="alpha")
    assert len(runner.store.list_runs()) == 2
    assert runner.store.get_run("scope-one")["status"] == "running"


async def test_external_tool_ids_fit_openai_limit(coordinator):
    runner, outputs, _ = coordinator
    outputs.append(tool_turn())
    first = await runner.respond_with_tools(
        [{"role": "user", "content": "Inspect"}], [], "chatcmpl-" + "a" * 18
    )
    assert len(first.tool_calls[0]["id"]) <= 64


async def test_stream_continuation_reports_original_task_progress(coordinator, monkeypatch):
    import roi_swarm.api as api

    runner, outputs, _ = coordinator
    outputs.append(tool_turn())
    messages = [{"role": "user", "content": "Inspect"}]
    first = await runner.respond_with_tools(messages, [], "progress-first")
    started = asyncio.Event()

    async def respond(messages, tools, run_id, **kwargs):
        runner.store.record_event("progress-first", "coordinator", "running")
        started.set()
        await asyncio.sleep(0.05)
        return AgentTurn("done")

    monkeypatch.setattr(runner, "respond_with_tools", respond)
    monkeypatch.setattr(api, "PROGRESS_POLL_SECONDS", 0.005)
    body = "".join([
        part async for part in api._stream_tool_turn(
            runner, "progress-next", with_result(messages, first), []
        )
    ])
    assert started.is_set()
    assert "Координатор работает с инструментами Hermes" in body


def test_follow_up_delegation_progress_uses_persisted_parent_links(tmp_path):
    from roi_swarm.api import _family_progress
    from roi_swarm.telemetry import RunStore

    store = RunStore(tmp_path)
    store.create_run("task", "logical-task")
    store.create_run("child task", "next-http-completion-s1")
    store.record_event(
        "logical-task", "coordinator", "delegated", {"run": "next-http-completion-s1"}
    )
    store.record_event(
        "next-http-completion-s1", "worker:one", "running", {"objective": "child objective"}
    )
    seen = {}
    assert any("child objective" in line for line in _family_progress(store, "logical-task", seen))
    assert _family_progress(store, "logical-task", seen) == []


@pytest.mark.parametrize("stream", [False, True])
def test_api_tool_rounds_keep_distinct_response_ids_but_one_dashboard_run(
    coordinator, stream, monkeypatch
):
    from fastapi.testclient import TestClient

    from roi_swarm.api import create_app

    runner, outputs, _ = coordinator
    outputs.extend([tool_turn(), AgentTurn("verified")])
    payload = {
        "messages": [
            {"role": "user", "content": "Inspect"},
            {"role": "user", "content": MODEL_MARKER},
        ],
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
        "stream": stream,
    }
    monkeypatch.setenv("ROI_SWARM_API_KEY", "test-only-client-token")

    def response_turn(response):
        assert response.status_code == 200
        if not stream:
            data = response.json()
            return data["id"], data["choices"][0]["message"]
        chunks = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith("data: {")]
        calls = [call for chunk in chunks for call in
                 chunk["choices"][0]["delta"].get("tool_calls", [])]
        return chunks[0]["id"], {"tool_calls": calls}

    client = TestClient(
        create_app(runner), headers={"Authorization": "Bearer test-only-client-token"}
    )
    first_id, first = response_turn(client.post("/v1/chat/completions", json=payload))
    records = client.get("/api/runs").json()
    assert len(records) == 1
    assert records[0]["status"] == "running"
    call = first["tool_calls"][0]
    payload["messages"].extend([
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": "read"},
        {"role": "user", "content": MODEL_MARKER},
    ])
    final_id, _ = response_turn(client.post("/v1/chat/completions", json=payload))
    assert final_id != first_id
    records = client.get("/api/runs").json()
    assert len(records) == 1
    assert records[0]["id"] == first_id
    assert records[0]["request"] == "Inspect"
    assert records[0]["status"] == "completed"


def test_standalone_notice_never_becomes_context_goal(tmp_path):
    from roi_swarm.context import ContextPipeline

    prepared = ContextPipeline(tmp_path).prepare([
        {"role": "user", "content": "Inspect"},
        {"role": "user", "content": MODEL_MARKER},
    ], run_id="notice-context")
    assert prepared.current_request == "Inspect"
    assert prepared.canonical_state.open_items == []


@pytest.mark.parametrize("content", [
    "", None, [{"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}],
    [{"type": "text", "text": MODEL_MARKER},
     {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}],
])
def test_non_notice_user_input_cannot_reuse_previous_task(content):
    from roi_swarm.conversation import current_task, is_runtime_notice

    messages = [
        {"role": "user", "content": "previous task"},
        {"role": "user", "content": content},
    ]
    assert current_task(messages)[0] == 1
    assert not is_runtime_notice(messages[-1])


@pytest.mark.parametrize("restart", [False, True])
async def test_incomplete_tool_batch_is_rejected_without_changing_task(
    coordinator, tmp_path, restart
):
    from roi_swarm.runner import SwarmRunner

    runner, outputs, seen = coordinator
    batch = tool_turn()
    batch.tool_calls.append({"id": "second", "function": {"name": "read_file", "arguments": "{}"}})
    outputs.extend([batch, AgentTurn("finished")])
    messages = [{"role": "user", "content": "Inspect"}]
    first = await runner.respond_with_tools(messages, [], "batch-first")
    original = runner.store.get_run("batch-first")
    continued = with_result(messages, first)
    restored = SwarmRunner(tmp_path / "config" / "settings.yaml") if restart else runner
    if restart:
        restored.sleep = FakeSleep()
    try:
        leases = restored._tool_leases.copy()
        with pytest.raises(ValueError, match="outstanding tool results"):
            await restored.respond_with_tools(continued, [], "batch-partial")
        assert len(seen) == 1  # Never call the provider with a partial batch.
        assert restored.store.get_run("batch-first") == original
        assert restored._tool_leases == leases
        assert restored._active_operations == 0
        continued.append(
            {"role": "tool", "tool_call_id": first.tool_calls[1]["id"], "content": "two"}
        )
        await restored.respond_with_tools(continued, [], "batch-complete")
        records = restored.store.list_runs()
        assert len(records) == 1
        assert records[0]["status"] == "completed"
    finally:
        if restart:
            await restored.close()


@pytest.mark.parametrize(
    "failure", [
        PermissionError("locked"), json.JSONDecodeError("bad", "{", 1),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    ]
)
async def test_unreadable_continuation_fails_closed_without_leaking_activity(coordinator, failure):
    from unittest.mock import patch

    runner, outputs, _ = coordinator
    outputs.extend([tool_turn(), AgentTurn("done")])
    messages = [{"role": "user", "content": "Inspect"}]
    first = await runner.respond_with_tools(messages, [], "storage-first")
    continued = with_result(messages, first)
    leases = runner._tool_leases.copy()
    with (
        patch.object(runner.store, "get_run", side_effect=failure),
        pytest.raises(type(failure)),
    ):
        await runner.respond_with_tools(continued, [], "storage-failed")
    assert runner._active_operations == 0
    assert runner._tool_leases == leases
    assert len(runner.store.list_runs()) == 1
    await runner.respond_with_tools(continued, [], "storage-recovered")
    assert runner.store.get_run("storage-first")["status"] == "completed"


@pytest.mark.parametrize("stream", [False, True])
def test_api_rejects_partial_batch_before_json_or_sse_response(coordinator, monkeypatch, stream):
    from fastapi.testclient import TestClient

    from roi_swarm.api import create_app

    runner, outputs, seen = coordinator
    batch = tool_turn()
    batch.tool_calls.append({"id": "two", "function": {"name": "read_file", "arguments": "{}"}})
    outputs.extend([batch, AgentTurn("complete")])
    monkeypatch.setenv("ROI_SWARM_API_KEY", "test-only-client-token")
    client = TestClient(
        create_app(runner), headers={"Authorization": "Bearer test-only-client-token"}
    )
    payload = {
        "messages": [{"role": "user", "content": "Inspect"}],
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
    }
    response = client.post("/v1/chat/completions", json=payload)
    data = response.json()
    first = AgentTurn(**{
        "content": "", "tool_calls": data["choices"][0]["message"]["tool_calls"],
        "finish_reason": "tool_calls",
    })
    payload["messages"] = with_result(payload["messages"], first)
    payload["stream"] = stream
    rejected = client.post("/v1/chat/completions", json=payload)
    assert rejected.status_code == 422
    assert len(seen) == 1
    assert runner.store.get_run(data["id"])["status"] == "running"
    payload["messages"].append(
        {"role": "tool", "tool_call_id": first.tool_calls[1]["id"], "content": "two"}
    )
    assert client.post("/v1/chat/completions", json=payload).status_code == 200
    assert len(runner.store.list_runs()) == 1
    assert runner.store.get_run(data["id"])["status"] == "completed"


async def test_missing_entire_tool_batch_cannot_create_another_task(coordinator):
    runner, outputs, seen = coordinator
    outputs.extend([tool_turn(), AgentTurn("unexpected")])
    messages = [{"role": "user", "content": "Inspect"}]
    first = await runner.respond_with_tools(messages, [], "no-results-first")
    messages.append({"role": "assistant", "tool_calls": first.tool_calls})
    with pytest.raises(ValueError, match="outstanding tool results"):
        await runner.respond_with_tools(messages, [], "no-results-next")
    assert len(seen) == 1
    assert len(runner.store.list_runs()) == 1


async def test_long_embedded_coordinator_id_is_rejected_before_starting(coordinator):
    runner, outputs, seen = coordinator
    outputs.append(tool_turn())
    with pytest.raises(ValueError, match="29 characters"):
        await runner.respond_with_tools([{"role": "user", "content": "Inspect"}], [], "a" * 128)
    assert seen == []
    assert runner._active_operations == 0
    assert runner.store.list_runs() == []


@pytest.mark.parametrize("finished", [False, True])
async def test_stale_http_retry_never_creates_task_or_calls_provider(coordinator, finished):
    runner, outputs, seen = coordinator
    outputs.extend([tool_turn(), AgentTurn("done") if finished else tool_turn(), AgentTurn("bad")])
    messages = [{"role": "user", "content": "Inspect"}]
    first = await runner.respond_with_tools(messages, [], "retry-first")
    continued = with_result(messages, first)
    await runner.respond_with_tools(continued, [], "retry-next")
    snapshot = runner.store.get_run("retry-first")
    leases = runner._tool_leases.copy()
    with pytest.raises(ValueError):
        await runner.respond_with_tools(continued, [], "retry-stale")
    assert len(seen) == 2
    assert len(runner.store.list_runs()) == 1
    assert runner.store.get_run("retry-first") == snapshot
    assert runner._tool_leases == leases


async def test_concurrent_continuation_is_rejected_before_provider_admission(
    coordinator, monkeypatch
):
    import roi_swarm.runner as module

    runner, outputs, _ = coordinator
    outputs.append(tool_turn())
    messages = [{"role": "user", "content": "Inspect"}]
    first = await runner.respond_with_tools(messages, [], "concurrent-first")
    continued = with_result(messages, first)
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    class Coordinator:
        def __init__(self, *args, **kwargs):
            pass

        async def respond(self, *args, **kwargs):
            calls.append(True)
            entered.set()
            await release.wait()
            return tool_turn()

    monkeypatch.setattr(module, "HermesToolAgent", Coordinator)
    admitted = asyncio.create_task(runner.respond_with_tools(continued, [], "concurrent-next"))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(ValueError, match="already active"):
            await asyncio.wait_for(
                runner.respond_with_tools(continued, [], "concurrent-duplicate"), 0.2
            )
    finally:
        release.set()
        await admitted
    assert len(calls) == 1
    assert len(runner.store.list_runs()) == 1
    assert runner.store.get_run("concurrent-first")["status"] == "running"
    assert len(runner._tool_leases) == 1


async def test_length_finish_is_not_completed_or_queued_for_sleep(scoped_runner):
    from test_agent_loop import ScriptedClient, _response

    runner = scoped_runner
    runner.sleep = FakeSleep()
    response = _response(content="truncated")
    response.choices[0].finish_reason = "length"
    runner.client.chat.completions.create = ScriptedClient([response])._create
    turn = await runner.respond_with_tools([{"role": "user", "content": "Inspect"}], [], "length")
    assert turn.finish_reason == "length"
    assert runner.store.get_run("length")["status"] == "failed"
    assert runner.sleep.completed == []


async def test_corrupt_utf8_ownership_file_cannot_create_duplicate(coordinator):
    runner, outputs, seen = coordinator
    outputs.extend([tool_turn(), AgentTurn("must not run")])
    messages = [{"role": "user", "content": "Inspect"}]
    first = await runner.respond_with_tools(messages, [], "utf8-original")
    path = runner.store.root / "utf8-original.json"
    original = path.read_bytes()
    leases = runner._tool_leases.copy()
    try:
        path.write_bytes(b"\xff")
        with pytest.raises(UnicodeDecodeError):
            await runner.respond_with_tools(with_result(messages, first), [], "utf8-duplicate")
        assert len(seen) == 1
        assert runner._active_operations == 0
        assert runner._tool_leases == leases
        assert not (runner.store.root / "utf8-duplicate.json").exists()
    finally:
        path.write_bytes(original)


async def test_concurrent_sse_requests_are_rejected_before_response_start(coordinator, monkeypatch):
    from httpx import ASGITransport, AsyncClient

    from roi_swarm.api import create_app

    runner, outputs, seen = coordinator
    outputs.extend([tool_turn(), AgentTurn("done")])
    monkeypatch.setenv("ROI_SWARM_API_KEY", "test-only-client-token")
    messages = [{"role": "user", "content": "Inspect"}]
    first = await runner.respond_with_tools(messages, [], "sse-original")
    payload = {
        "messages": with_result(messages, first), "stream": True,
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
    }
    started = asyncio.Event()
    release = asyncio.Event()
    app = create_app(runner)
    first_response = True

    async def gated(scope, receive, send):
        nonlocal first_response

        async def gated_send(message):
            nonlocal first_response
            if message["type"] == "http.response.start" and first_response:
                first_response = False
                started.set()
                await release.wait()
            await send(message)

        await app(scope, receive, gated_send)

    async with AsyncClient(
        transport=ASGITransport(app=gated), base_url="http://test",
        headers={"Authorization": "Bearer test-only-client-token"},
    ) as client:
        admitted = asyncio.create_task(client.post("/v1/chat/completions", json=payload))
        try:
            await asyncio.wait_for(started.wait(), 5)
            rejected = await client.post("/v1/chat/completions", json=payload)
            assert rejected.status_code == 422
            assert len(seen) == 1
        finally:
            release.set()
            accepted = await admitted
        assert accepted.status_code == 200
        assert "ROI Swarm error:" not in accepted.text
    assert len(seen) == 2
    assert len(runner.store.list_runs()) == 1
    assert runner._coordinator_active == set()
    assert runner._active_operations == 0


@pytest.mark.parametrize("after_role_chunk", [False, True])
async def test_failed_stream_send_releases_unclaimed_reservation(
    coordinator, after_role_chunk
):
    from starlette.requests import ClientDisconnect

    from roi_swarm.api import _ReservedToolResponse, _stream_tool_turn

    runner, outputs, seen = coordinator
    outputs.extend([tool_turn(), AgentTurn("done")])
    messages = [{"role": "user", "content": "Inspect"}]
    first = await runner.respond_with_tools(messages, [], "send-first")
    continued = with_result(messages, first)
    leases = runner._tool_leases.copy()
    reservation = runner.reserve_tool_turn(continued, "send-next")
    response = _ReservedToolResponse(
        _stream_tool_turn(runner, "send-next", continued, [], reservation=reservation),
        reservation=reservation,
    )

    async def send(message):
        if message["type"] == ("http.response.body" if after_role_chunk else "http.response.start"):
            raise OSError("fixture lost connection")

    async def receive():
        await asyncio.Event().wait()

    with pytest.raises(ClientDisconnect):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert reservation.closed
    assert len(seen) == 1
    assert runner._active_operations == 0
    assert runner._coordinator_active == set()
    assert runner._tool_leases == leases
    await runner.respond_with_tools(continued, [], "send-retry")
    assert runner.store.get_run("send-first")["status"] == "completed"


async def test_duplicate_run_id_cannot_change_completed_history(coordinator):
    runner, outputs, seen = coordinator
    outputs.extend([AgentTurn("original answer"), AgentTurn("must not run")])
    messages = [{"role": "user", "content": "Inspect"}]
    await runner.respond_with_tools(messages, [], "same-id")
    path = runner.store.root / "same-id.json"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Run already exists"):
        await runner.respond_with_tools(messages, [], "same-id")
    assert path.read_bytes() == original
    assert len(seen) == 1
    assert runner._active_operations == 0
    assert runner._coordinator_active == set()


@pytest.mark.parametrize("repeat_cancel", [False, True])
async def test_asgi23_disconnect_settles_async_cleanup_before_releasing_admission(
    coordinator, monkeypatch, repeat_cancel
):
    from roi_swarm.api import create_app

    runner, _, _ = coordinator
    monkeypatch.setenv("ROI_SWARM_API_KEY", "test-only-client-token")
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    cleaned = asyncio.Event()

    async def work(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            assert runner._active_operations == 1
            cleaned.set()

    monkeypatch.setattr(runner, "_respond_with_tools", work)
    payload = json.dumps({
        "messages": [{"role": "user", "content": "Inspect"}], "stream": True,
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
    }).encode()
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await entered.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "method": "POST", "scheme": "http", "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions", "query_string": b"",
        "http_version": "1.1", "server": ("test", 80), "client": ("test", 1),
        "headers": [(b"content-type", b"application/json"),
                    (b"authorization", b"Bearer test-only-client-token")],
    }
    request = asyncio.create_task(create_app(runner)(scope, receive, send))
    try:
        await asyncio.wait_for(cleaning.wait(), 5)
        if repeat_cancel:
            request.cancel()
        await asyncio.wait({request}, timeout=0.05)
        assert not request.done()
        assert runner._active_operations == 1
        assert len(runner._coordinator_active) == 1
    finally:
        release.set()
        try:
            await asyncio.wait_for(request, 5)
        except asyncio.CancelledError:
            assert repeat_cancel
    assert cleaned.is_set()
    assert runner._active_operations == 0
    assert runner._coordinator_active == set()
