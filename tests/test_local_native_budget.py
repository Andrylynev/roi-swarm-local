"""Native template/token counts, not reference tokenizer estimates, gate generation."""

import asyncio
import types

import pytest

from local.messages import LocalCompletionClient
from roi_swarm.native_budget import NativeBudget, guard_sync_client


@pytest.mark.parametrize("stream", [False, True])
def test_native_count_rejects_reference_undercount_before_transport(stream):
    seen = []

    def post(path, payload):
        seen.append((path, payload))
        if path.startswith("/props"):
            return {"default_generation_settings": {"n_ctx": 65536}}
        if path.startswith("/apply-template"):
            assert payload["tools"][0]["function"]["name"] == "tool"
            assert payload["chat_template_kwargs"]["enable_thinking"] is False
            return {"prompt": "native exact template"}
        assert payload["content"] == "native exact template"
        assert payload["parse_special"] is True
        return {"tokens": [0] * 66066}

    budget = NativeBudget("http://127.0.0.1:8091/v1", post=post)

    async def create(**kwargs):
        pytest.fail("oversized request reached generation")

    sdk = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )
    client = LocalCompletionClient(sdk, budget=budget)
    with pytest.raises(ValueError, match="66066.*4096.*65536"):
        asyncio.run(
            client.chat.completions.create(
                model="bonsai-9b",
                messages=[{"role": "user", "content": "reference count 60831"}],
                tools=[{"type": "function", "function": {"name": "tool"}}],
                max_tokens=4096,
                stream=stream,
            )
        )
    assert len(seen) == 3


def test_boundary_accepts_exact_fit_and_uses_actual_slot_limit():
    def post(path, payload):
        if path.startswith("/props"):
            return {"default_generation_settings": {"n_ctx": 8192}}
        if path.startswith("/apply-template"):
            return {"prompt": "x"}
        return {"tokens": [0] * 8000}

    budget = NativeBudget("http://127.0.0.1:8091/v1", post=post)
    assert budget.check({"model": "bonsai-2b", "messages": [], "max_tokens": 192}) == 8000
    with pytest.raises(ValueError, match="8192"):
        budget.check({"model": "bonsai-2b", "messages": [], "max_tokens": 193})


@pytest.mark.parametrize("reserve", [None, -1, 0, True, "4096"])
def test_missing_or_unbounded_output_reserve_fails_closed(reserve):
    budget = NativeBudget("http://127.0.0.1:8091/v1", post=lambda *a: pytest.fail("queried"))
    with pytest.raises(ValueError, match="output reserve"):
        budget.check({"model": "bonsai-9b", "messages": [], "max_tokens": reserve})


def test_worker_sync_guard_checks_before_returning_stream():
    order = []

    def create(**kwargs):
        order.append("generate")
        return "stream object"

    sdk = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )
    counter = types.SimpleNamespace(check=lambda value: order.append("count"))
    guard_sync_client(sdk, counter)
    guard_sync_client(sdk, counter)  # idempotent: do not count twice
    assert sdk.chat.completions.create(model="bonsai-4b", stream=True) == "stream object"
    assert order == ["count", "generate"]


def test_native_template_omits_sdk_transport_objects_not_model_payload():
    import json

    import httpx

    seen = []

    def post(path, payload):
        json.dumps(payload)  # Metadata requests must be a genuine JSON body.
        if path.startswith("/props"):
            return {"default_generation_settings": {"n_ctx": 65536}}
        if path.startswith("/apply-template"):
            seen.append(payload)
            return {"prompt": "x"}
        return {"tokens": [0] * 10}

    budget = NativeBudget("http://127.0.0.1:8091/v1", post=post)
    request = {
        "model": "bonsai-4b",
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {"name": "tool"}}],
        "max_tokens": 4096,
        "timeout": httpx.Timeout(1800),
        "extra_headers": {"X-Local": "transport-only"},
        "extra_query": {"trace": "1"},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    assert budget.check(request) == 10
    assert "timeout" not in seen[0] and "extra_headers" not in seen[0]
    assert "extra_query" not in seen[0] and "extra_body" not in seen[0]
    assert seen[0]["tools"] == request["tools"]
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert isinstance(request["timeout"], httpx.Timeout)


def test_unavailable_native_counter_never_falls_back_to_reference_tokenizer():
    def post(*args):
        raise OSError("native tokenizer unavailable")

    budget = NativeBudget("http://127.0.0.1:8091/v1", post=post)
    with pytest.raises(OSError, match="unavailable"):
        budget.check({"model": "bonsai-9b", "messages": [], "max_tokens": 32})
