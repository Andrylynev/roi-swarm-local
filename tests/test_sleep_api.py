from fastapi.testclient import TestClient
from test_api import FakeRunner

from roi_swarm.api import create_app


class Sleeping(FakeRunner):
    def sleep_status(self):
        return {"enabled": False, "state": "idle"}

    def sleep_snapshot(self):
        return {"current": None, "queue": []}

    async def set_sleep_enabled(self, enabled):
        return {"enabled": enabled, "state": "waiting" if enabled else "disabled"}


def test_sleep_lifecycle_starts_before_requests(monkeypatch):
    monkeypatch.delenv("ROI_SWARM_API_KEY", raising=False)
    events = []

    class Lifecycle(Sleeping):
        async def start(self):
            events.append("started")

        async def close(self):
            events.append("closed")

    with TestClient(create_app(Lifecycle())) as client:
        assert events == ["started"]
        assert client.get("/health").status_code == 200
    assert events == ["started", "closed"]


def test_sleep_storage_errors_are_safe_503(monkeypatch):
    monkeypatch.delenv("ROI_SWARM_API_KEY", raising=False)

    class Broken(Sleeping):
        async def set_sleep_enabled(self, enabled):
            raise OSError("private path")

    response = TestClient(create_app(Broken())).put("/api/sleep", json={"enabled": True})
    assert response.status_code == 503
    assert "private" not in response.text


def test_sleep_routes_and_auth(monkeypatch):
    monkeypatch.delenv("ROI_SWARM_API_KEY", raising=False)
    client = TestClient(create_app(Sleeping()))
    assert client.get("/api/sleep").json()["enabled"] is False
    assert client.get("/api/sleep/snapshot").json()["queue"] == []
    assert client.put("/api/sleep", json={"enabled": True}).json()["enabled"] is True
    monkeypatch.setenv("ROI_SWARM_API_KEY", "test")
    assert client.get("/api/sleep").status_code == 401
    assert client.put("/api/sleep", json={"enabled": False}).status_code == 401
