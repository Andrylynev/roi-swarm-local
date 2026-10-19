from fastapi.testclient import TestClient
from test_api import FakeRunner

from roi_swarm.api import create_app


def test_token_budget_overflow_is_clear_json_and_sse(monkeypatch):
    monkeypatch.delenv("ROI_SWARM_API_KEY", raising=False)
    error = (
        "Context exceeds 1000000 tokens: input=1000001, output reserve=0 "
        "(reference tokenizer o200k_base). Split the task; no input was silently truncated."
    )

    class Overflow(FakeRunner):
        async def run(self, *args, **kwargs):
            raise ValueError(error)

    client = TestClient(create_app(Overflow()))
    payload = {"messages": [{"role": "user", "content": "large"}]}
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 413
    assert response.json()["detail"] == error
    response = client.post("/v1/chat/completions", json={**payload, "stream": True})
    assert "Context exceeds 1000000 tokens" in response.text
    assert "[DONE]" in response.text
