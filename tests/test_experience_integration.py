import asyncio
from types import SimpleNamespace

from roi_swarm.experience import WorkflowLesson
from roi_swarm.schemas import WorkerResult

pytest_plugins = ("test_project_isolation",)


def lesson():
    return WorkflowLesson(
        topic="materials",
        situation="Workers lacked assigned input",
        recommendation="Include the exact source segment in each atomic task",
        evidence="Input marker was absent from the worker prompt",
    )


def test_workers_can_propose_lessons():
    result = WorkerResult(task_id="p1", conclusion="ok", workflow_lessons=[lesson()])
    assert result.workflow_lessons == [lesson()]


def test_runner_records_candidates_and_uses_only_reviewed_exact_scope(scoped_runner):
    r = scoped_runner
    seen = []

    async def stream(state, **kwargs):
        seen.append(state["prepared_context"].workflow_lessons)
        yield {
            "worker": {
                "worker_results": [
                    WorkerResult(
                        task_id="p1",
                        conclusion="ok",
                        workflow_lessons=[lesson()],
                    ).model_dump()
                ]
            }
        }
        yield {"synthesize": {"final_answer": "ok"}}

    r.graph = SimpleNamespace(astream=stream)
    asyncio.run(r.run("audit", run_id="learn-1", project_id="alpha"))
    notes = r.experience.list_notes("alpha")
    assert len(notes) == 1 and notes[0]["status"] == "candidate"
    assert seen == [[]]
    r.experience.review(notes[0]["id"], "approved")
    asyncio.run(r.run("audit again", run_id="learn-2", project_id="alpha"))
    assert seen[-1][0]["recommendation"] == lesson().recommendation
    asyncio.run(r.run("other project", run_id="learn-3", project_id="beta"))
    assert seen[-1] == []


def test_operator_approved_global_workflow_lesson_is_reused(scoped_runner):
    r = scoped_runner
    note = r.experience.record(lesson(), project_id="global", run_id="operator", origin="operator")
    r.experience.review(note["id"], "approved")
    captured = []

    async def stream(state, **kwargs):
        captured.extend(state["prepared_context"].workflow_lessons)
        yield {"direct": {"final_answer": "ok"}}

    r.graph = SimpleNamespace(astream=stream)
    asyncio.run(r.run("a new unscoped task", run_id="global-use"))
    assert captured[0]["recommendation"] == lesson().recommendation


def test_legacy_global_model_notes_are_never_loaded(scoped_runner):
    import sqlite3

    r = scoped_runner
    note = r.experience.record(lesson(), project_id="global", run_id="legacy", origin="operator")
    r.experience.review(note["id"], "approved")
    with sqlite3.connect(r.experience.path) as db:
        db.execute("UPDATE experience_notes SET origin='model' WHERE id=?", (note["id"],))
    assert r.experience.lessons("global") == []
    assert r._experience_lessons("alpha", "unused") == []


def test_unscoped_candidates_cannot_leak_to_later_requests(scoped_runner):
    r = scoped_runner
    seen = []

    async def stream(state, **kwargs):
        seen.append(state["prepared_context"].workflow_lessons)
        yield {
            "worker": {
                "worker_results": [
                    WorkerResult(
                        task_id="p1",
                        conclusion="ok",
                        workflow_lessons=[lesson()],
                    ).model_dump()
                ]
            }
        }
        yield {"synthesize": {"final_answer": "ok"}}

    r.graph = SimpleNamespace(astream=stream)
    asyncio.run(r.run("unscoped", run_id="unscoped-1"))
    notes = r.experience.list_notes("run:unscoped-1")
    assert len(notes) == 1
    r.experience.review(notes[0]["id"], "approved")
    asyncio.run(r.run("unscoped next", run_id="unscoped-2"))
    assert seen[-1] == []
