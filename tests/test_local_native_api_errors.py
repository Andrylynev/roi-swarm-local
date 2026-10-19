"""Native tokenizer overflow must be actionable in both JSON and streaming APIs."""

import pytest
from fastapi.testclient import TestClient
from test_api import FakeRunner

from roi_swarm.api import create_app
from roi_swarm.native_budget import NativeBudget


@pytest.mark.parametrize("stream", [False, True])
def test_real_native_budget_error_reaches_client_without_disclosing_input(monkeypatch, stream):
    monkeypatch.delenv("ROI_SWARM_API_KEY", raising=False)
    generated = []

    def post(path, payload):
        if path.startswith("/props"):
            return {"default_generation_settings": {"n_ctx": 65536}}
        if path.startswith("/apply-template"):
            return {"prompt": "PRIVATE_SOURCE_MUST_NOT_LEAK"}
        return {"tokens": [0] * 66066}

    budget = NativeBudget("http://127.0.0.1:8091/v1", post=post)

    class Overflow(FakeRunner):
        async def run(self, *args, **kwargs):
            budget.check({"model": "test", "messages": [], "max_tokens": 4096})
            generated.append(True)
            raise AssertionError("Overflow reached generation")

    client = TestClient(create_app(Overflow()))
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "large"}], "stream": stream},
    )
    if stream:
        assert response.status_code == 200
        assert "[DONE]" in response.text
    else:
        assert response.status_code == 413
    for number in ("66066", "4096", "65536"):
        assert number in response.text
    assert "PRIVATE_SOURCE_MUST_NOT_LEAK" not in response.text
    assert generated == []
