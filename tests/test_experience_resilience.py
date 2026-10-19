import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from roi_swarm.api import create_app
from roi_swarm.experience import ExperienceStore, WorkflowLesson

pytest_plugins = ("test_project_isolation",)


def lesson():
    return WorkflowLesson(
        topic="coordination",
        situation="Many independent tasks",
        recommendation="Give each task a distinct owner",
        evidence="No overlap",
    )


@pytest.mark.parametrize("tools", [False, True])
def test_global_is_not_a_runnable_project(scoped_runner, tools):
    r = scoped_runner
    with pytest.raises(ValueError, match="reserved"):
        if tools:
            asyncio.run(r.respond_with_tools([], [], "reserved", project_id="global"))
        else:
            asyncio.run(r.run("task", run_id="reserved", project_id="global"))
    assert r._active_operations == 0


def test_optional_memory_startup_failure_keeps_service_usable(scoped_runner, monkeypatch):
    import roi_swarm.runner as module

    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("unavailable")

    monkeypatch.setattr(module, "ExperienceStore", broken)
    runner = module.SwarmRunner(scoped_runner.config_path)
    try:
        assert runner.experience is None
        assert runner.model_settings()["experience_status"] == "unavailable"
    finally:
        asyncio.run(runner.close())


def test_api_rejects_reserved_global_before_stream_start(scoped_runner, monkeypatch):
    monkeypatch.delenv("ROI_SWARM_API_KEY", raising=False)
    client = TestClient(create_app(scoped_runner))
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "task"}],
            "project_id": "global",
            "stream": True,
        },
    )
    assert response.status_code == 422


def test_models_cannot_write_global_notes(tmp_path):
    store = ExperienceStore(tmp_path / "experience.db")
    try:
        with pytest.raises(ValueError, match="reserved"):
            store.record(lesson(), project_id="global", run_id="model")
    finally:
        store.close()


def test_unavailable_optional_memory_does_not_fail_run(scoped_runner, monkeypatch):
    r = scoped_runner

    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("sensitive filesystem detail")

    monkeypatch.setattr(r.experience, "lessons", broken)

    async def stream(state, **kwargs):
        assert state["prepared_context"].workflow_lessons == []
        yield {"direct": {"final_answer": "useful answer"}}

    r.graph = SimpleNamespace(astream=stream)
    result = asyncio.run(r.run("task", run_id="memory-failed"))
    assert result["final_answer"] == "useful answer"
    events = r.store.get_run("memory-failed")["events"]
    assert any(e["node"] == "experience" and e["status"] == "failed" for e in events)


@pytest.mark.parametrize("method", ["GET", "POST", "PATCH"])
def test_experience_endpoints_hide_storage_errors(scoped_runner, monkeypatch, method):
    monkeypatch.delenv("ROI_SWARM_API_KEY", raising=False)

    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("sensitive path")

    for name in ("list_notes", "record", "review", "publish"):
        monkeypatch.setattr(scoped_runner.experience, name, broken, raising=False)
    client = TestClient(create_app(scoped_runner), raise_server_exceptions=False)
    if method == "GET":
        response = client.get("/api/experience?project_id=alpha")
    elif method == "POST":
        response = client.post(
            "/api/experience",
            json={"project_id": "alpha", "run_id": "r", "lesson": lesson().model_dump()},
        )
    else:
        response = client.patch("/api/experience/1", json={"status": "approved"})
    assert response.status_code == 503
    assert "sensitive" not in response.text


def test_global_publication_replaces_legacy_model_provenance_atomically(tmp_path):
    path = tmp_path / "experience.db"
    store = ExperienceStore(path)
    try:
        shared = store.record(lesson(), project_id="global", run_id="legacy", origin="operator")
        # A legacy database may contain old model-origin global entries.
        with sqlite3.connect(path) as db:
            db.execute("UPDATE experience_notes SET origin='model' WHERE id=?", (shared["id"],))
        local = store.record(lesson(), project_id="alpha", run_id="run-1")
        result = store.publish(local["id"])
        assert result["origin"] == "operator" and result["status"] == "approved"
        assert result["id"] == shared["id"]
        assert store.lessons("global")[0]["origin"] == "operator"
        assert store.list_notes("alpha")[0]["status"] == "approved"
    finally:
        store.close()


def test_metrics_distinguish_plan_direct_and_executed_attempts(scoped_runner):
    r = scoped_runner

    async def direct(state, **kwargs):
        yield {"plan": {"tasks": [{"id": "a"}, {"id": "b"}]}}
        yield {"direct": {"final_answer": "direct", "answered_directly": True}}

    r.graph = SimpleNamespace(astream=direct)
    asyncio.run(r.run("direct", run_id="metrics-direct", project_id="alpha"))
    with sqlite3.connect(r.experience.path) as db:
        metrics = json.loads(
            db.execute(
                "SELECT metrics_json FROM experience_runs WHERE run_id='metrics-direct'"
            ).fetchone()[0]
        )
    assert metrics["task_count"] == 2 and metrics["worker_count"] == 0
    assert metrics["verification_count"] == 0

    async def retries(state, **kwargs):
        yield {"plan": {"tasks": [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]}}
        for task_id in ("a", "b", "a"):
            yield {"worker": {"worker_results": [{"task_id": task_id, "confidence": 0.9}]}}
        report = {
            "passed": False,
            "missing_task_ids": ["c"],
            "empty_task_ids": ["d"],
            "low_confidence_task_ids": ["b", "d"],
        }
        yield {"verify": {"verification": report}}
        yield {"verify": {"verification": report}}
        yield {"synthesize": {"final_answer": "partial"}}

    r.graph = SimpleNamespace(astream=retries)
    asyncio.run(r.run("retry", run_id="metrics-retry", project_id="alpha"))
    with sqlite3.connect(r.experience.path) as db:
        metrics = json.loads(
            db.execute(
                "SELECT metrics_json FROM experience_runs WHERE run_id='metrics-retry'"
            ).fetchone()[0]
        )
    assert metrics["worker_count"] == 2 and metrics["retry_count"] == 1
    assert metrics["verification_count"] == 2 and metrics["failed_count"] == 3
