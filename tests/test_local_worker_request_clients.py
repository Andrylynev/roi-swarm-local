"""Exercise installed Hermes factory + dispatch, not only a wrapped primary client."""

import ast
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from roi_swarm.native_budget import NativeBudget, guard_worker_clients

HERMES = Path(os.environ.get(
    "ROI_TEST_HERMES_ROOT",
    str(Path(__file__).resolve().parents[1] / "runtime/hermes-agent"),
))


def installed_function(path, name, namespace):
    if not path.is_file():
        pytest.skip("Installed Hermes source is required for this integration regression")
    source = ast.parse(path.read_text(encoding="utf-8"))
    matches = [
        node for node in ast.walk(source) if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"Installed Hermes changed its {name} seam"
    module = ast.Module(body=[matches[0]], type_ignores=[])
    exec(
        compile(
            module, str(path), "exec", flags=__import__("__future__").annotations.compiler_flag
        ),
        namespace,
    )
    return namespace[name]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("kind", ["new", "cached", "replacement"])
@pytest.mark.parametrize("oversized", [False, True])
def test_installed_request_local_clients_enforce_native_budget(stream, kind, oversized):
    calls = []
    sdk_timeout = httpx.Timeout(1800)

    def transport(**kwargs):
        assert kwargs["timeout"] is sdk_timeout
        assert kwargs["extra_headers"] == {"X-Local": "preserved"}
        calls.append("TRANSPORT")
        return "real transport result"

    def client():
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=transport)))

    def post(path, payload):
        json.dumps(payload)  # Exercise accounting's actual serialization requirement.
        assert "timeout" not in payload and "extra_headers" not in payload
        calls.append("COUNT:" + path.split("?", 1)[0])
        if path.startswith("/props"):
            return {"default_generation_settings": {"n_ctx": 65536}}
        if path.startswith("/apply-template"):
            return {"prompt": "native prompt including tools"}
        return {"tokens": [0] * (66066 if oversized else 32)}

    budget = NativeBudget("http://127.0.0.1:8091/v1", post=post)
    cache = {"client": client() if kind == "cached" else None}
    factory = installed_function(
        HERMES / "agent/client_lifecycle.py",
        "_create_request_openai_client",
        {
            "Any": object,
            "Optional": __import__("typing").Optional,
            "_OPENAI_SLOT": "request-slot",
            "base_url_host_matches": lambda *a: False,
        },
    )
    agent = SimpleNamespace(
        api_mode="chat_completions",
        provider="custom",
        base_url="http://127.0.0.1:8091/v1",
        client=client(),
        _client_kwargs={"base_url": "http://127.0.0.1:8091/v1"},
        _ensure_primary_openai_client=lambda **kw: agent.client,
        _openai_client_lock=lambda: threading.RLock(),
        _checkout_request_slot=lambda *a: (cache["client"], None),
        _close_openai_client=lambda *a, **kw: None,
        _create_openai_client=lambda *a, **kw: client(),
        _store_request_slot=lambda attr, value, key: cache.update(client=value),
        _touch_activity=lambda *a: None,
    )
    agent._create_request_openai_client = lambda **kw: factory(agent, **kw)
    guard_worker_clients(agent, budget)
    guard_worker_clients(agent, budget)  # registration must be idempotent
    if kind == "replacement":
        # A later retry/rotation rebuilds both the primary and request-local client.
        agent.client = client()
        cache["client"] = None
    kwargs = {
        "model": "bonsai-4b",
        "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 4096,
        "stream": stream,
        "timeout": sdk_timeout,
        "extra_headers": {"X-Local": "preserved"},
    }
    if stream:
        opening = installed_function(
            HERMES / "agent/chat_completion_helpers.py",
            "_open_chat_stream",
            {"Any": object, "time": time, "is_native_gemini_base_url": lambda url: False},
        )
        session = SimpleNamespace(
            agent=agent, clients=SimpleNamespace(set_client=lambda c: c), last_chunk_time={}
        )

        def dispatch():
            return opening(session, kwargs)
    else:
        opening = installed_function(
            HERMES / "agent/chat_completion_helpers.py",
            "_dispatch_nonstreaming_api_request",
            {},
        )

        def dispatch():
            return opening(
                agent,
                kwargs,
                make_client=lambda reason: agent._create_request_openai_client(reason=reason),
            )

    if oversized:
        with pytest.raises(ValueError, match="66066.*4096.*65536"):
            dispatch()
        assert "TRANSPORT" not in calls
    else:
        assert dispatch() == "real transport result"
        assert calls[-1] == "TRANSPORT"
    assert calls.count("COUNT:/props") == 1


@pytest.mark.parametrize("kind", ["cached", "replaced", "closed"])
@pytest.mark.parametrize("oversized", [False, True])
def test_installed_iteration_summary_guards_rebuilt_primary(kind, oversized):
    calls = []
    sdk_timeout = httpx.Timeout(1800)

    def transport(**kwargs):
        assert kwargs["timeout"] is sdk_timeout
        calls.append("TRANSPORT")
        return "summary response"

    def client():
        return SimpleNamespace(
            closed=False,
            chat=SimpleNamespace(completions=SimpleNamespace(create=transport)),
        )

    def post(path, payload):
        json.dumps(payload)
        calls.append("COUNT:" + path.split("?", 1)[0])
        if path.startswith("/props"):
            return {"default_generation_settings": {"n_ctx": 65536}}
        if path.startswith("/apply-template"):
            return {"prompt": "native summary"}
        return {"tokens": [0] * (66066 if oversized else 32)}

    logger = SimpleNamespace(warning=lambda *a, **kw: None)
    lifecycle = HERMES / "agent/client_lifecycle.py"
    ensure = installed_function(lifecycle, "_ensure_primary_openai_client", {"logger": logger})
    replace = installed_function(lifecycle, "_replace_primary_openai_client", {"logger": logger})
    summary = installed_function(
        HERMES / "agent/chat_completion_helpers.py",
        "_chat_summary_attempt",
        {
            "sanitize_outbound_kwargs": lambda *a: None,
            "_managed_summary_call": lambda agent, rid, request, callback, **kw: callback(request),
            "_summary_text": lambda agent, response: response,
        },
    )
    agent = SimpleNamespace(
        provider="custom",
        api_mode="chat_completions",
        client=client(),
        _client_kwargs={"base_url": "http://127.0.0.1:8091/v1"},
        _openai_client_lock=lambda: threading.RLock(),
        _is_openai_client_closed=lambda value: value.closed,
        _create_openai_client=lambda *a, **kw: client(),
        _retire_shared_openai_client=lambda *a, **kw: None,
        _close_openai_client=lambda *a, **kw: None,
        _client_log_context=lambda: "local fixture",
        _build_api_kwargs=lambda messages: {
            "model": "bonsai-4b",
            "messages": messages,
            "max_tokens": 4096,
            "timeout": sdk_timeout,
        },
        _create_request_openai_client=lambda **kw: client(),
    )
    agent._ensure_primary_openai_client = lambda **kw: ensure(agent, **kw)
    guard_worker_clients(agent, NativeBudget("http://127.0.0.1:8091/v1", post=post))
    guard_worker_clients(agent, NativeBudget("http://127.0.0.1:8091/v1", post=post))
    if kind == "replaced":
        assert replace(agent, reason="recovery")
    elif kind == "closed":
        agent.client.closed = True
    attempt = summary(agent, [{"role": "user", "content": "summarize"}], "summary-test")
    if oversized:
        with pytest.raises(ValueError, match="66066.*4096.*65536"):
            attempt(0)
        assert "TRANSPORT" not in calls
    else:
        assert attempt(0) == "summary response"
        assert calls[-1] == "TRANSPORT"
    assert calls.count("COUNT:/props") == 1
