"""Native llama.cpp template/token accounting shared by local lead and Hermes workers.

Only standard library dependencies: this module also runs under Hermes' interpreter.
No reference-tokenizer fallback is safe when the native accounting service is unavailable.
"""

from __future__ import annotations

import json
import threading
import urllib.parse
import urllib.request
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

if __package__:
    from .native_budget_errors import NativeContextBudgetError
else:
    from native_budget_errors import NativeContextBudgetError


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Native token accounting must not redirect to another endpoint")


def guard_sync_client(client, budget) -> None:
    """Wrap a worker's real SDK call, including streaming, without editing Hermes."""
    completions = client.chat.completions
    original = completions.create
    if getattr(original, "_roi_native_budget", False):
        return

    def checked_create(*args, **kwargs):
        if args:
            raise ValueError("Native accounting requires keyword chat arguments")
        budget.check(kwargs)
        return original(**kwargs)

    checked_create._roi_native_budget = True
    completions.create = checked_create


def guard_worker_clients(agent, budget) -> None:
    """Guard the actual Hermes per-request client, including cached/rebuilt clients.

    AIAgent.client is not the normal wire client: Hermes checks out a separate
    client for both streaming and non-streaming calls. Keep its ownership and
    cleanup lifecycle intact and guard every object returned by that seam.
    """
    factory = getattr(agent, "_create_request_openai_client", None)
    ensure_primary = getattr(agent, "_ensure_primary_openai_client", None)
    if not callable(factory) or not callable(ensure_primary):
        raise RuntimeError("Installed Hermes has no supported guarded client lifecycle seams")
    if getattr(factory, "_roi_native_budget", False):
        return
    guard_sync_client(agent.client, budget)

    def guarded_primary_client(*args, **kwargs):
        # Iteration summaries use the primary directly. Re-check after every
        # ensure, since recovery/rotation can have replaced it since registration.
        client = ensure_primary(*args, **kwargs)
        guard_sync_client(client, budget)
        return client

    def guarded_request_client(*args, **kwargs):
        client = factory(*args, **kwargs)
        guard_sync_client(client, budget)
        return client

    guarded_primary_client._roi_native_budget = True
    guarded_request_client._roi_native_budget = True
    agent._ensure_primary_openai_client = guarded_primary_client
    agent._create_request_openai_client = guarded_request_client


@contextmanager
def guard_worker_auxiliary_clients(budget):
    """Guard frozen Hermes auxiliary clients for this local worker's lifetime only.

    Compression owns a separate cached client and deliberately omits max_tokens.
    Guard the cache checkout and the actual progress-dispatch seam (also used by
    fallback candidates), without changing Hermes files or its chosen local route.
    """
    from agent import auxiliary_client as auxiliary

    factory = auxiliary._get_cached_client
    dispatch = auxiliary._create_with_progress_once
    guarded = []
    lock = threading.RLock()
    budgets = {budget.base: budget}

    def guard(client):
        if client is None:
            return
        with lock:
            completions = client.chat.completions
            if any(resource is completions for resource, _, _ in guarded):
                return
            original = completions.create

            def checked_create(*args, **kwargs):
                if args:
                    raise ValueError("Native accounting requires keyword chat arguments")
                # Inspect the actual client, not requested routing metadata: cached
                # or recovered clients may differ. Never send local-worker text out.
                url = str(getattr(client, "base_url", "") or "")
                native = NativeBudget(url)
                with lock:
                    counter = budgets.setdefault(native.base, native)
                request = dict(kwargs)
                body = request.get("extra_body") or {}
                if not any(
                    key in request or key in body
                    for key in ("max_tokens", "max_completion_tokens")
                ):
                    request["max_tokens"] = 4096
                counter.check(request)
                return original(**request)

            completions.create = checked_create
            guarded.append((completions, original, checked_create))

    def guarded_factory(*args, **kwargs):
        client, model = factory(*args, **kwargs)
        guard(client)
        return client, model

    def guarded_dispatch(client, *args, **kwargs):
        guard(client)
        return dispatch(client, *args, **kwargs)

    auxiliary._get_cached_client = guarded_factory
    auxiliary._create_with_progress_once = guarded_dispatch
    try:
        yield
    finally:
        auxiliary._get_cached_client = factory
        auxiliary._create_with_progress_once = dispatch
        with lock:
            for completions, original, checked in guarded:
                if completions.create is checked:
                    completions.create = original


class NativeBudget:
    def __init__(
        self,
        base_url: str,
        *,
        post: Callable[[str, dict], dict] | None = None,
        context_limit: int = 65536,
    ) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Native model accounting requires a loopback HTTP endpoint")
        self.base = base_url.rstrip("/").removesuffix("/v1")
        self.limit = context_limit
        self._post = post or self._request

    def _request(self, path: str, payload: dict) -> dict:
        data = None if path.startswith("/props?") else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        # Metadata/template requests may load the model. This is not a generation timeout.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=120) as response:
            return json.load(response)

    def check(self, request: dict[str, Any]) -> int:
        transport_keys = {"timeout", "extra_headers", "extra_query", "extra_body"}
        payload = {key: value for key, value in request.items() if key not in transport_keys}
        payload.update(request.get("extra_body") or {})
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("Native accounting requires an explicit model")
        reserves = [
            payload[key] for key in ("max_tokens", "max_completion_tokens") if key in payload
        ]
        if not reserves or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in reserves
        ):
            raise ValueError("A positive explicit output reserve is required for local inference")
        reserve = max(reserves)
        query = "?model=" + urllib.parse.quote(model, safe="")
        props = self._post("/props" + query, {})
        active = props.get("default_generation_settings", {}).get("n_ctx")
        if isinstance(active, bool) or not isinstance(active, int) or active <= 0:
            raise ValueError("Native endpoint did not report its active per-slot context")
        limit = min(active, self.limit)
        # Forward the actual normalized messages, tools and template options, not a
        # stringified approximation. /apply-template uses this model's native Jinja.
        template = self._post("/apply-template" + query, payload)
        prompt = template.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("Native endpoint returned no rendered chat template")
        tokens = self._post(
            "/tokenize" + query,
            {
                "model": model,
                "content": prompt,
                "add_special": False,
                "parse_special": True,
            },
        ).get("tokens")
        if not isinstance(tokens, list):
            raise ValueError("Native endpoint returned no token IDs")
        count = len(tokens)
        if count + reserve > limit:
            raise NativeContextBudgetError(count, reserve, limit)
        return count
