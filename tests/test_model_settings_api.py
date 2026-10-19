import pytest
from fastapi.testclient import TestClient
from test_api import FakeRunner

from roi_swarm.api import create_app

ROLES = ("router", "planner", "worker", "critic", "synthesizer")


class ModelRunner(FakeRunner):
    def __init__(self):
        self.roles = {role: "old" for role in ROLES}
        self.busy = False

    def model_settings(self):
        return {"roles": self.roles, "active_operations": int(self.busy)}

    def update_model_settings(self, roles):
        if self.busy:
            raise RuntimeError("active")
        self.roles = roles
        return self.model_settings()

    async def model_options(self):
        return {"models": ["old", "new"], "source": "provider"}


def test_dashboard_model_endpoints_save_and_read_back(monkeypatch):
    monkeypatch.delenv("ROI_SWARM_API_KEY", raising=False)
    runner = ModelRunner()
    client = TestClient(create_app(runner))
    assert client.get("/api/model-options").json()["models"] == ["old", "new"]
    assert client.get("/api/model-settings").json()["roles"] == runner.roles
    roles = {role: "new" for role in ROLES}
    response = client.put("/api/model-settings", json={"roles": roles})
    assert response.status_code == 200
    assert client.get("/api/model-settings").json()["roles"] == roles


@pytest.mark.parametrize("roles", [{"worker": "new"}, {role: " " for role in ROLES}])
def test_model_api_rejects_invalid_selection(monkeypatch, roles):
    monkeypatch.delenv("ROI_SWARM_API_KEY", raising=False)
    client = TestClient(create_app(ModelRunner()))
    assert client.put("/api/model-settings", json={"roles": roles}).status_code == 422


def test_model_api_busy_and_disk_errors_are_not_success(monkeypatch):
    monkeypatch.delenv("ROI_SWARM_API_KEY", raising=False)
    runner = ModelRunner()
    runner.busy = True
    client = TestClient(create_app(runner))
    roles = {role: "new" for role in ROLES}
    assert client.put("/api/model-settings", json={"roles": roles}).status_code == 409

    def failed(roles):
        raise OSError("sensitive disk path")

    runner.update_model_settings = failed
    response = client.put("/api/model-settings", json={"roles": roles})
    assert response.status_code == 503
    assert "sensitive" not in response.text


def test_model_endpoints_preserve_auth(monkeypatch):
    monkeypatch.setenv("ROI_SWARM_API_KEY", "test-access")
    client = TestClient(create_app(ModelRunner()))
    assert client.get("/api/model-settings").status_code == 401
    assert client.get("/api/model-options").status_code == 401
    assert client.put("/api/model-settings", json={"roles": {}}).status_code == 401
