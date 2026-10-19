from fastapi.testclient import TestClient
from test_api import HERMES_TOOL_REQUEST, FakeRunner, ToolRunner

from roi_swarm.api import create_app


def test_explicit_project_scope_reaches_run_in_stream_and_json(monkeypatch):
    monkeypatch.delenv("ROI_SWARM_API_KEY", raising=False)

    class Scoped(FakeRunner):
        async def run(self, request, run_id=None, messages=None, *, project_id=None):
            assert project_id == "booking"
            return await super().run(request, run_id, messages)

    client = TestClient(create_app(Scoped()))
    for stream in (False, True):
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "audit"}],
                "project_id": "booking",
                "stream": stream,
            },
        )
        assert response.status_code == 200
        assert "swarm:audit" in response.text
    assert (
        client.post("/api/run", json={"prompt": "audit", "project_id": "booking"}).status_code
        == 200
    )


def test_planner_budget_error_is_actionable_in_json_and_stream(monkeypatch):
    monkeypatch.delenv("ROI_SWARM_API_KEY", raising=False)

    class Overflow(FakeRunner):
        async def run(self, request, **kwargs):
            raise ValueError(
                "Planner input is 501000 characters and exceeds planner input limit. "
                "Split the task into smaller tasks."
            )

    client = TestClient(create_app(Overflow()))
    for stream in (False, True):
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "huge"}],
                "stream": stream,
            },
        )
        assert response.status_code == (200 if stream else 413)
        assert "Split the task" in response.text


def test_explicit_scope_reaches_hermes_tools(monkeypatch):
    monkeypatch.delenv("ROI_SWARM_API_KEY", raising=False)
    seen = []

    class ScopedTools(ToolRunner):
        async def respond_with_tools(self, messages, tools, run_id, *, project_id=None):
            seen.append(project_id)
            return await super().respond_with_tools(messages, tools, run_id)

    client = TestClient(create_app(ScopedTools()))
    for stream in (False, True):
        response = client.post(
            "/v1/chat/completions",
            json={
                **HERMES_TOOL_REQUEST,
                "project_id": "booking",
                "stream": stream,
            },
        )
        assert response.status_code == 200
    assert seen == ["booking", "booking"]
