import json

from fastapi.testclient import TestClient

from roi_swarm.api import create_app

API_HEADERS = {"Authorization": "Bearer local-roi-swarm"}


class FakeRunner:
    async def run(
        self,
        request: str,
        run_id: str | None = None,
        messages: list[dict] | None = None,
    ) -> dict:
        return {
            "final_answer": f"swarm:{request}",
            "run_id": run_id or "run-test",
            "worker_target": 10,
        }

    async def close(self) -> None:
        return None


def test_openai_chat_completions_endpoint() -> None:
    client = TestClient(create_app(runner=FakeRunner()))

    response = client.post(
        "/v1/chat/completions",
        headers=API_HEADERS,
        json={
            "model": "roi-swarm",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "roi-swarm"
    assert data["choices"][0]["message"]["content"] == "swarm:hello"


def test_openai_chat_completions_streaming_endpoint() -> None:
    client = TestClient(create_app(runner=FakeRunner()))

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=API_HEADERS,
        json={
            "model": "roi-swarm",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"content":"swarm:hello"' in body
    assert "data: [DONE]" in body


def test_models_endpoint_exposes_swarm_model() -> None:
    client = TestClient(create_app(runner=FakeRunner()))

    response = client.get("/v1/models", headers=API_HEADERS)

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "roi-swarm"


def test_dashboard_page_is_available() -> None:
    client = TestClient(create_app(runner=FakeRunner()))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "ROI Swarm" in response.text
    assert "swarm-graph" in response.text


def test_provider_endpoints_require_bearer_token(monkeypatch) -> None:
    monkeypatch.setenv("ROI_SWARM_API_KEY", "expected")
    client = TestClient(create_app(runner=FakeRunner()))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "roi-swarm", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 401


def test_chat_rejects_empty_user_message() -> None:
    client = TestClient(create_app(runner=FakeRunner()))

    response = client.post(
        "/v1/chat/completions",
        headers=API_HEADERS,
        json={"model": "roi-swarm", "messages": [{"role": "assistant", "content": "x"}]},
    )

    assert response.status_code == 422


def test_chat_accepts_tool_calls_and_null_assistant_content() -> None:
    client = TestClient(create_app(runner=FakeRunner()))

    response = client.post(
        "/v1/chat/completions",
        headers=API_HEADERS,
        json={
            "model": "roi-swarm",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "calc", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "content": "42", "tool_call_id": "call-1"},
                {"role": "user", "content": "Use the tool result"},
            ],
            "tools": [{"type": "function", "function": {"name": "calc", "parameters": {}}}],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "swarm:Use the tool result"


def test_run_path_traversal_is_rejected(tmp_path, monkeypatch) -> None:
    from roi_swarm.telemetry import RunStore

    monkeypatch.delenv("ROI_SWARM_API_KEY", raising=False)

    store = RunStore(tmp_path / "runs")
    runner = FakeRunner()
    runner.store = store
    client = TestClient(create_app(runner=runner))

    response = client.get("/api/runs/%2E%2E%5Cescape")

    assert response.status_code in {400, 404}


def test_run_history_requires_bearer_token(monkeypatch, tmp_path) -> None:
    from roi_swarm.telemetry import RunStore

    monkeypatch.setenv("ROI_SWARM_API_KEY", "expected")
    runner = FakeRunner()
    runner.store = RunStore(tmp_path / "runs")
    client = TestClient(create_app(runner=runner))

    assert client.get("/api/runs").status_code == 401
    assert client.get("/api/runs/run-x").status_code == 401
    ok = client.get("/api/runs", headers={"Authorization": "Bearer expected"})
    assert ok.status_code == 200


def test_streaming_reports_runner_failure_and_finishes() -> None:
    class FailingRunner(FakeRunner):
        async def run(self, request, run_id=None, messages=None):
            raise RuntimeError("boom")

    client = TestClient(create_app(runner=FailingRunner()))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=API_HEADERS,
        json={"messages": [{"role": "user", "content": "x"}], "stream": True},
    ) as response:
        body = "".join(response.iter_text())

    assert "ROI Swarm error: RuntimeError" in body
    assert body.rstrip().endswith("data: [DONE]")


def test_streaming_emits_swarm_progress_as_reasoning(tmp_path) -> None:
    import asyncio

    from roi_swarm.telemetry import RunStore

    class SlowRunner(FakeRunner):
        def __init__(self) -> None:
            self.store = RunStore(tmp_path)

        async def run(self, request, run_id=None, messages=None):
            self.store.create_run(request, run_id)
            self.store.record_event(run_id, "planner", "completed", {"task_count": 7})
            await asyncio.sleep(1.5)
            return {"final_answer": "ok", "run_id": run_id}

    import roi_swarm.api as api_module

    api_module.PROGRESS_POLL_SECONDS = 0.2
    client = TestClient(create_app(runner=SlowRunner()))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=API_HEADERS,
        json={"messages": [{"role": "user", "content": "x"}], "stream": True},
    ) as response:
        body = "".join(response.iter_text())

    assert '"reasoning_content"' in body
    assert "7" in body
    assert '"content":"ok"' in body


class ToolRunner(FakeRunner):
    """Runner that exposes the Hermes tool loop (Opus + delegate_to_swarm)."""

    def __init__(self) -> None:
        self.seen: list[tuple[list, list]] = []

    async def respond_with_tools(self, messages, tools, run_id):
        from roi_swarm.agent_loop import AgentTurn

        self.seen.append((messages, tools))
        return AgentTurn(
            content="Распаковываю архив",
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": '{"command": "unzip a.zip"}'},
                }
            ],
            finish_reason="tool_calls",
        )


HERMES_TOOL_REQUEST = {
    "model": "roi-swarm",
    "messages": [
        {"role": "system", "content": "You are Hermes"},
        {"role": "user", "content": "@file:C:/att/a.zip переделай проект"},
    ],
    "tools": [{"type": "function", "function": {"name": "terminal", "parameters": {}}}],
}


def test_chat_with_hermes_tools_returns_tool_calls() -> None:
    runner = ToolRunner()
    client = TestClient(create_app(runner=runner))

    response = client.post("/v1/chat/completions", headers=API_HEADERS, json=HERMES_TOOL_REQUEST)

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "terminal"
    assert choice["message"]["content"] == "Распаковываю архив"
    messages, tools = runner.seen[0]
    assert messages[0] == {"role": "system", "content": "You are Hermes"}
    assert tools[0]["function"]["name"] == "terminal"


def test_streamed_chat_with_hermes_tools_emits_tool_call_deltas() -> None:
    client = TestClient(create_app(runner=ToolRunner()))

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=API_HEADERS,
        json={**HERMES_TOOL_REQUEST, "stream": True},
    ) as response:
        body = "".join(response.iter_text())

    chunks = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
    tool_deltas = [
        c["choices"][0]["delta"]["tool_calls"]
        for c in chunks
        if c["choices"][0]["delta"].get("tool_calls")
    ]
    assert tool_deltas[0][0]["index"] == 0
    assert tool_deltas[0][0]["id"] == "c1"
    assert tool_deltas[0][0]["function"]["name"] == "terminal"
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert body.rstrip().endswith("data: [DONE]")
