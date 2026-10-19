from fastapi.testclient import TestClient

from roi_swarm.api import create_app

pytest_plugins = ("test_project_isolation",)

LESSON = {
    "topic": "coordination",
    "situation": "Parallel writes collided",
    "recommendation": "Assign a single writer per file",
    "evidence": "Two attempted writes had the same target",
}


def test_experience_candidate_review_and_global_publish(scoped_runner, monkeypatch):
    monkeypatch.delenv("ROI_SWARM_API_KEY", raising=False)
    client = TestClient(create_app(scoped_runner))
    created = client.post(
        "/api/experience",
        json={
            "project_id": "alpha",
            "run_id": "manual-1",
            "lesson": LESSON,
        },
    )
    assert created.status_code == 200
    note = created.json()
    assert note["status"] == "candidate"
    assert scoped_runner.experience.lessons("alpha") == []
    assert len(client.get("/api/experience?project_id=alpha").json()) == 1
    reviewed = client.patch(
        f"/api/experience/{note['id']}",
        json={
            "status": "approved",
            "publish_global": True,
        },
    )
    assert reviewed.status_code == 200
    assert scoped_runner.experience.lessons("alpha")[0]["status"] == "approved"
    shared = scoped_runner.experience.lessons("global")
    assert len(shared) == 1 and shared[0]["origin"] == "operator"
    assert client.patch("/api/experience/99999", json={"status": "approved"}).status_code == 404


def test_experience_api_requires_auth(scoped_runner, monkeypatch):
    monkeypatch.setenv("ROI_SWARM_API_KEY", "expected")
    client = TestClient(create_app(scoped_runner))
    assert client.get("/api/experience?project_id=alpha").status_code == 401
    assert client.post("/api/experience", json={}).status_code == 401
    assert client.patch("/api/experience/1", json={"status": "approved"}).status_code == 401
