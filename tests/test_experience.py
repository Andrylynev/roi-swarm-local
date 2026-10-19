"""Offline contracts for project-scoped, explicitly reviewed workflow memory."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError


def test_workflow_lesson_strips_whitespace_and_forbids_extra_fields():
    from roi_swarm.experience import WorkflowLesson

    lesson = WorkflowLesson(
        topic="verification",
        situation="  A deliverable has checks.  ",
        recommendation="  Run its checks before returning it.  ",
        evidence="  run-1 verified checks  ",
    )
    assert lesson.model_dump() == {
        "topic": "verification",
        "situation": "A deliverable has checks.",
        "recommendation": "Run its checks before returning it.",
        "evidence": "run-1 verified checks",
    }
    for field, invalid in [
        ("topic", "unknown"),
        ("situation", "  "),
        ("situation", "s" * 241),
        ("recommendation", "  "),
        ("recommendation", "r" * 801),
        ("evidence", "  "),
        ("evidence", "e" * 501),
        ("confidence", 0.99),
    ]:
        with pytest.raises(ValidationError):
            WorkflowLesson(**(lesson.model_dump() | {field: invalid}))
    WorkflowLesson(
        topic="decomposition", situation="s" * 240, recommendation="r" * 800, evidence="e" * 500
    )


def make_lesson(**changes):
    from roi_swarm.experience import WorkflowLesson

    return WorkflowLesson(
        **(
            {
                "topic": "verification",
                "situation": "Multiple deliverables",
                "recommendation": "Verify each deliverable independently",
                "evidence": "Checks were executed in the observed run",
            }
            | changes
        )
    )


def test_record_persists_candidate_with_explicit_scope_and_wal(tmp_path):
    from roi_swarm.experience import ExperienceStore

    path = tmp_path / "isolated" / "experience.sqlite3"
    store = ExperienceStore(path)
    note = store.record(make_lesson(), project_id="project-a", run_id="run-1")
    assert (
        note.items()
        >= {
            "project_id": "project-a",
            "run_id": "run-1",
            "status": "candidate",
            "origin": "model",
            "seen_count": 1,
            **make_lesson().model_dump(),
        }.items()
    )
    assert isinstance(note["id"], int)
    assert note["created_at"] == note["updated_at"]
    store.close()
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    reopened = ExperienceStore(path)
    assert reopened.list_notes("project-a") == [note]
    assert reopened.list_notes("project-b") == []
    reopened.close()


def test_observe_run_persists_only_safe_numeric_telemetry_and_never_promotes(tmp_path):
    from roi_swarm.experience import ExperienceStore

    path = tmp_path / "experience.db"
    store = ExperienceStore(path)
    note = store.record(make_lesson(), project_id="p", run_id="r")
    metrics = {
        "worker_count": 4,
        "duration_seconds": 1.25,
        "input_tokens": 200,
        "output_tokens": 80,
        "retry_count": 1,
        "artifact_count": 2,
        "request": "private original task",
        "source": "private source contents",
        "confidence": 0.99,
        "score": 1.0,
        "improvement": 999,
        "arbitrary_count": 99,
        "completed_count": -1,
        "failed_count": True,
        "tool_call_count": "4",
        "elapsed_seconds": float("nan"),
        "total_tokens": float("inf"),
        "secret": "Bearer fake-secret",
    }
    assert store.observe_run(run_id="r", project_id="p", metrics=metrics) is None
    assert store.lessons("p") == []
    assert store.list_notes("p")[0] == note
    store.observe_run(run_id="r", project_id="other", metrics={"worker_count": 2})
    store.close()
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT project_id, run_id, metrics_json FROM experience_runs ORDER BY project_id"
        ).fetchall()
    by_scope = {scope: json.loads(payload) for scope, _, payload in rows}
    assert by_scope == {
        "p": {
            "worker_count": 4,
            "duration_seconds": 1.25,
            "input_tokens": 200,
            "output_tokens": 80,
            "retry_count": 1,
            "artifact_count": 2,
        },
        "other": {"worker_count": 2},
    }
    reopened = ExperienceStore(path)
    reopened.observe_run(run_id="r", project_id="p", metrics={"worker_count": 5})
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM experience_runs").fetchone()[0] == 2
        assert json.loads(
            connection.execute(
                "SELECT metrics_json FROM experience_runs WHERE project_id = ?", ("p",)
            ).fetchone()[0]
        ) == {"worker_count": 5}
    for changes in [{"project_id": ""}, {"run_id": ""}, {"metrics": "raw text"}]:
        with pytest.raises(ValueError):
            reopened.observe_run(**({"run_id": "r", "project_id": "p", "metrics": {}} | changes))
    reopened.close()


def test_injection_like_text_and_scope_are_bound_as_values(tmp_path):
    from roi_swarm.experience import ExperienceStore

    store = ExperienceStore(tmp_path / "experience.db")
    payload = "x'); DROP TABLE experience_notes; --"
    scope = "p' OR 1=1 --"
    lesson = make_lesson(situation=payload, recommendation=payload, evidence=payload)
    note = store.record(lesson, project_id=scope, run_id=payload)
    assert store.list_notes(scope)[0]["situation"] == payload
    assert store.list_notes("p") == []
    store.review(note["id"], "approved")
    assert len(store.lessons(scope)) == 1
    assert store.lessons("p") == []
    store.observe_run(run_id=payload, project_id=scope, metrics={"worker_count": 1})
    assert store.record(make_lesson(), project_id="normal", run_id="r")["id"] != note["id"]
    store.close()


@pytest.mark.parametrize("field", ["situation", "recommendation", "evidence"])
@pytest.mark.parametrize(
    "secret",
    [
        "sk-" + "synthetic-fixture-" * 3,
        "Bearer fake-opaque-credential",
        "api_key = 'fake-secret'",
        '"password": "fake-secret"',
        "API-KEY=fake-secret",
    ],
)
def test_secrets_are_rejected_without_echoing_credentials(tmp_path, field, secret):
    from roi_swarm.experience import ExperienceStore

    store = ExperienceStore(tmp_path / "experience.db")
    with pytest.raises(ValueError) as caught:
        store.record(make_lesson(**{field: secret}), project_id="p", run_id="r")
    assert secret not in str(caught.value)
    assert "fake-secret" not in repr(caught.value)
    assert store.list_notes("p") == []
    store.close()


def test_model_notes_do_not_persist_source_specific_absolute_paths(tmp_path):
    from roi_swarm.experience import ExperienceStore

    store = ExperienceStore(tmp_path / "experience.db")
    for source in [r"C:\private\source.py", "/home/user/private/source.py", "file:///private/a"]:
        with pytest.raises(ValueError):
            store.record(make_lesson(evidence=source), project_id="p", run_id="r")
    note = store.record(
        make_lesson(evidence=r"C:\project\verified.py"),
        project_id="p",
        run_id="r",
        origin="operator",
    )
    assert note["origin"] == "operator"
    assert note["status"] == "candidate"
    store.close()


def test_identifiers_origin_and_forged_lesson_are_validated(tmp_path):
    from roi_swarm.experience import ExperienceStore, WorkflowLesson

    store = ExperienceStore(tmp_path / "experience.db")
    for changes in [
        {"project_id": ""},
        {"project_id": "  "},
        {"project_id": None},
        {"run_id": ""},
        {"run_id": None},
        {"origin": "approved"},
    ]:
        with pytest.raises(ValueError):
            store.record(make_lesson(), **({"project_id": "p", "run_id": "r"} | changes))
    for project in ["", "   ", None]:
        with pytest.raises(ValueError):
            store.list_notes(project)
        with pytest.raises(ValueError):
            store.lessons(project)
    forged = WorkflowLesson.model_construct(
        topic="verification", situation="", recommendation="Valid", evidence="Valid"
    )
    with pytest.raises(ValueError):
        store.record(forged, project_id="p", run_id="r")
    store.close()


def test_dedup_is_normalized_project_scoped_and_counts_distinct_runs(tmp_path):
    from roi_swarm.experience import ExperienceStore

    store = ExperienceStore(tmp_path / "experience.db")
    original = store.record(make_lesson(), project_id="p", run_id="r1")
    repeat = store.record(
        make_lesson(situation="MULTIPLE   deliverables", evidence="Second observation"),
        project_id="p",
        run_id="r1",
    )
    assert repeat["id"] == original["id"]
    assert repeat["seen_count"] == 1
    another = store.record(make_lesson(evidence="New run evidence"), project_id="p", run_id="r2")
    assert another["id"] == original["id"]
    assert another["seen_count"] == 2
    assert another["status"] == "candidate"
    assert another["created_at"] == original["created_at"]
    assert another["updated_at"] >= original["updated_at"]
    foreign = store.record(make_lesson(), project_id="P", run_id="r1")
    assert foreign["id"] != original["id"]
    assert len(store.list_notes("p")) == len(store.list_notes("P")) == 1
    store.close()
    with sqlite3.connect(tmp_path / "experience.db") as connection:
        rows = connection.execute(
            "SELECT run_id, evidence FROM experience_observations WHERE note_id = ?",
            (original["id"],),
        ).fetchall()
    assert set(rows) == {
        ("r1", make_lesson().evidence),
        ("r1", "Second observation"),
        ("r2", "New run evidence"),
    }


def test_concurrent_records_have_atomic_dedup_and_distinct_run_counts(tmp_path):
    from roi_swarm.experience import ExperienceStore

    store = ExperienceStore(tmp_path / "experience.db")

    def record(index):
        return store.record(make_lesson(), project_id="p", run_id=f"r{index % 10}")

    with ThreadPoolExecutor(max_workers=12) as pool:
        records = list(pool.map(record, range(100)))
    assert len({record["id"] for record in records}) == 1
    assert store.list_notes("p")[0]["seen_count"] == 10
    assert store.list_notes("p")[0]["status"] == "candidate"
    store.close()


def test_only_explicitly_approved_exact_project_lessons_are_retrieved(tmp_path):
    from roi_swarm.experience import ExperienceStore

    store = ExperienceStore(tmp_path / "experience.db")
    candidate = store.record(make_lesson(), project_id="p", run_id="r1")
    operator = store.record(
        make_lesson(situation="Operator observation"),
        project_id="p",
        run_id="r1",
        origin="operator",
    )
    assert operator["status"] == "candidate"
    assert store.lessons("p") == []
    approved = store.review(candidate["id"], "approved")
    assert approved["status"] == "approved"
    assert approved["updated_at"] >= candidate["updated_at"]
    assert store.lessons("p") == [approved]
    assert store.lessons("other") == []
    assert store.list_notes("p", status="approved") == [approved]
    assert store.list_notes("p", status="candidate") == [operator]
    store.record(make_lesson(), project_id="p", run_id="r2")
    assert store.lessons("p")[0]["status"] == "approved"
    rejected = store.review(candidate["id"], "rejected")
    assert rejected["status"] == "rejected"
    store.record(make_lesson(), project_id="p", run_id="r3")
    assert store.list_notes("p", "rejected")[0]["status"] == "rejected"
    assert store.lessons("p") == []
    with pytest.raises(KeyError):
        store.review(999999, "approved")
    for invalid in ["candidate", "APPROVED", "", None]:
        with pytest.raises(ValueError):
            store.review(operator["id"], invalid)
    assert store.list_notes("p", limit=1) == [operator]
    assert store.list_notes("p", limit=0) == []
    assert store.lessons("p", limit=0) == []
    for invalid_limit in [-1, 1.5, True, "10"]:
        with pytest.raises(ValueError):
            store.list_notes("p", limit=invalid_limit)
        with pytest.raises(ValueError):
            store.lessons("p", limit=invalid_limit)
    with pytest.raises(ValueError):
        store.list_notes("p", status="not-a-status")
    store.close()
