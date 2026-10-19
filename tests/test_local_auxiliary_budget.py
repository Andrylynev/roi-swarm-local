"""Offline regression against the frozen Hermes compression and auxiliary wire path."""

import importlib
import io
import json
import os
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from roi_swarm import hermes_worker_host as host
from roi_swarm.native_budget import NativeBudget, guard_worker_auxiliary_clients

HERMES = Path(
    os.environ.get(
        "ROI_TEST_HERMES_ROOT",
        str(Path(__file__).resolve().parents[1] / "runtime/hermes-agent"),
    )
)
BASE = "http://127.0.0.1:8091/v1"


@pytest.fixture
def hermes(monkeypatch, tmp_path):
    if not (HERMES / "agent/auxiliary_client.py").is_file():
        pytest.skip("Frozen Hermes source is required for this offline integration regression")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.syspath_prepend(str(HERMES))
    monkeypatch.setattr(
        socket.socket,
        "connect",
        lambda *args: pytest.fail("Offline regression attempted a network connection"),
    )
    aux = importlib.import_module("agent.auxiliary_client")
    compressor = importlib.import_module("agent.context_compressor")
    monkeypatch.setattr(aux, "_client_cache", {})
    monkeypatch.setattr(aux, "_aux_unhealthy_until", {})
    monkeypatch.setattr(aux, "_peek_pool_entry", lambda *args: None)
    return aux, compressor


def run_worker(monkeypatch, conversation):
    class Agent:
        def __init__(self, **kwargs):
            self.client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: None))
            )

        def _create_request_openai_client(self, **kwargs):
            return self.client

        def _ensure_primary_openai_client(self, **kwargs):
            return self.client

        def run_conversation(self, *args, **kwargs):
            return {"final_response": conversation()}

    monkeypatch.setitem(sys.modules, "run_agent", SimpleNamespace(AIAgent=Agent))
    monkeypatch.setitem(sys.modules, "model_tools", SimpleNamespace())
    monkeypatch.setattr(host, "guard_tools", lambda *args: lambda: None)
    task = {
        "objective": "offline regression",
        "hermes_root": str(HERMES),
        "model": "bonsai-9b",
        "base_url": BASE,
        "max_output_tokens": 4096,
    }
    events = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(task) + "\n"))
    monkeypatch.setattr(sys, "stdout", events)
    monkeypatch.setenv("ROI_LOCAL_NATIVE_BUDGET", "1")
    status = host.main()
    return status, [json.loads(line) for line in events.getvalue().splitlines()]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("kind", ["new", "cached", "rebuild"])
@pytest.mark.parametrize("oversized", [False, True])
def test_real_compression_auxiliary_wire_is_guarded(
    hermes, monkeypatch, stream, kind, oversized
):
    aux, compressor = hermes
    calls = []
    outgoing = []
    built = []
    templates = []
    timeout = httpx.Timeout(1800)
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    monkeypatch.setattr(
        aux,
        "_get_auxiliary_task_config",
        lambda task: {
            "provider": "custom",
            "model": "bonsai-2b",
            "base_url": BASE,
            "api_key": "offline-only",
            "extra_body": extra,
        },
    )
    monkeypatch.setattr(aux, "_aux_progress_active", lambda: stream)
    monkeypatch.setattr(compressor, "_pinned_summary_call_kwargs", lambda: {"timeout": timeout})

    def transport(**kwargs):
        calls.append("TRANSPORT")
        outgoing.append(kwargs)
        message = SimpleNamespace(content="validated summary", tool_calls=None)
        if kwargs.get("stream"):
            return iter(
                [
                    SimpleNamespace(
                        id="offline",
                        model=kwargs["model"],
                        choices=[SimpleNamespace(delta=message, finish_reason="stop")],
                    )
                ]
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")]
        )

    def resolve(*args, **kwargs):
        client = SimpleNamespace(
            base_url=BASE,
            chat=SimpleNamespace(completions=SimpleNamespace(create=transport)),
        )
        built.append(client)
        return client, "bonsai-2b"

    monkeypatch.setattr(aux, "resolve_provider_client", resolve)

    def post(path, payload):
        json.dumps(payload)  # SDK timeout objects must never enter metadata JSON.
        calls.append(path.split("?", 1)[0])
        assert path.endswith("?model=bonsai-2b")
        if path.startswith("/props"):
            return {"default_generation_settings": {"n_ctx": 65536}}
        if path.startswith("/apply-template"):
            templates.append(payload)
            assert "timeout" not in payload
            assert payload["chat_template_kwargs"] == extra["chat_template_kwargs"]
            return {"prompt": "native rendered auxiliary prompt"}
        assert payload["content"] == "native rendered auxiliary prompt"
        assert payload["add_special"] is False and payload["parse_special"] is True
        return {"tokens": [0] * (66066 if oversized else 32)}

    budget = NativeBudget(BASE, post=post)
    monkeypatch.setattr(host, "NativeBudget", lambda url: budget)
    cached = None
    if kind != "new":
        cached, _ = aux._get_cached_client(
            "custom", "bonsai-2b", base_url=BASE, api_key="offline-only", task="compression"
        )
    summary = SimpleNamespace(
        model="bonsai-9b",
        provider="custom",
        base_url=BASE,
        api_key="offline-only",
        api_mode="chat_completions",
        summary_model="",
        context_length=65536,
        _record_aux_compression_call=lambda **kwargs: None,
        _compression_cancelled=lambda: False,
    )
    originals = (aux._get_cached_client, aux._create_with_progress_once)

    def conversation():
        if kind == "rebuild":
            assert aux._evict_cached_client_instance(cached)
        return compressor.ContextCompressor._call_summary_llm(
            summary, "retain this exact compression evidence", time.monotonic()
        )

    status, events = run_worker(monkeypatch, conversation)
    if oversized:
        assert status == 1, "Oversized actual auxiliary compression bypassed native accounting"
        assert events[-1]["type"] == "error"
        assert "66066" in events[-1]["error"] and "4096" in events[-1]["error"]
        assert outgoing == []
    else:
        assert status == 0, events
        assert events[-1] == {"type": "final", "text": "validated summary"}
        assert outgoing[0].get("max_tokens") == 4096, "Auxiliary wire has no output reserve"
        assert outgoing[0]["model"] == "bonsai-2b"  # Do not redirect to the main 9B route.
        assert outgoing[0]["timeout"] is timeout
        assert outgoing[0].get("stream", False) is stream
        assert outgoing[0]["extra_body"] == extra
        assert outgoing[0]["messages"] == [
            {"role": "user", "content": "retain this exact compression evidence"}
        ]
        assert calls == ["/props", "/apply-template", "/tokenize", "TRANSPORT"]
    assert templates and templates[0]["max_tokens"] == 4096
    assert len(built) == (2 if kind == "rebuild" else 1)
    # A worker-scoped hook must not leak into another invocation/profile in this process.
    assert (aux._get_cached_client, aux._create_with_progress_once) == originals
    assert all(client.chat.completions.create is transport for client in built)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_local_worker_rejects_nonlocal_auxiliary_routes_before_transport(
    hermes, monkeypatch, stream, cached
):
    aux, _ = hermes
    outgoing = []
    client = SimpleNamespace(
        base_url="https://cloud.invalid/v1",
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: outgoing.append(kwargs))
        ),
    )
    monkeypatch.setattr(aux, "resolve_provider_client", lambda *args, **kwargs: (client, "remote"))
    monkeypatch.setattr(aux, "_aux_progress_active", lambda: stream)
    if cached:
        aux._get_cached_client("custom", "remote", base_url=client.base_url)
    with guard_worker_auxiliary_clients(NativeBudget(BASE)):
        selected, _ = aux._get_cached_client("custom", "remote", base_url=client.base_url)
        with pytest.raises(ValueError, match="loopback HTTP endpoint"):
            aux._create_with_progress_once(
                selected,
                {"model": "remote", "messages": [{"role": "user", "content": "private"}]},
                "compression",
            )
    assert outgoing == []


@pytest.mark.parametrize("stream", [False, True])
def test_auxiliary_fallback_dispatch_preserves_request_and_sdk_timeout(
    hermes, monkeypatch, stream
):
    aux, _ = hermes
    seen = []
    timeout = httpx.Timeout(1800)
    monkeypatch.setattr(aux, "_aux_progress_active", lambda: stream)

    def post(path, payload):
        json.dumps(payload)
        if path.startswith("/props"):
            return {"default_generation_settings": {"n_ctx": 65536}}
        if path.startswith("/apply-template"):
            assert payload["max_completion_tokens"] == 8192
            assert payload["tools"][0]["function"]["name"] == "offline"
            assert payload["chat_template_kwargs"] == {"enable_thinking": False}
            assert not {"timeout", "extra_headers", "extra_query", "extra_body"} & payload.keys()
            return {"prompt": "actual fallback template"}
        return {"tokens": [0] * 20}

    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    def transport(**kwargs):
        seen.append(kwargs)
        return response

    client = SimpleNamespace(
        base_url=BASE,
        chat=SimpleNamespace(completions=SimpleNamespace(create=transport)),
    )
    request = {
        "model": "bonsai-4b",
        "messages": [{"role": "user", "content": "exact fallback input"}],
        "tools": [{"type": "function", "function": {"name": "offline"}}],
        "max_completion_tokens": 8192,
        "timeout": timeout,
        "extra_headers": {"X-Local": "preserved"},
        "extra_query": {"trace": "1"},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    snapshot = dict(request)
    with guard_worker_auxiliary_clients(NativeBudget(BASE, post=post)):
        # A fallback candidate need not pass through the normal cache checkout.
        assert aux._create_with_progress_once(client, request, "compression") is response
    assert request == snapshot
    assert seen[0]["timeout"] is timeout
    assert seen[0]["max_completion_tokens"] == 8192
    assert "max_tokens" not in seen[0]
    assert all(seen[0][key] is value for key, value in snapshot.items())
    assert seen[0].get("stream", False) is stream
    assert client.chat.completions.create is transport
