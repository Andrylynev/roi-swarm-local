"""Stable transport adaptation for native Qwen chat templates (local deployment only)."""

import asyncio
import copy
import types

from roi_swarm.native_budget import NativeBudget


def _system_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(part, dict) and part.get("type") in {"text", "input_text"} for part in content
    ):
        return "\n".join(part.get("text", "") for part in content)
    raise ValueError("System messages must be text for the local Qwen template")


def fold_system_messages(messages):
    prefix = []
    tail = []
    seen_dialogue = False
    for message in messages:
        if message.get("role") in {"system", "developer"}:
            if seen_dialogue:
                raise ValueError("Local Qwen cannot represent mid-conversation system instructions")
            prefix.append(_system_text(message.get("content")))
        else:
            seen_dialogue = True
            tail.append(copy.deepcopy(message))
    return ([{"role": "system", "content": "\n\n".join(prefix)}] if prefix else []) + tail


class LocalCompletionClient:
    """Wrap request assembly, leaving the actual SDK streaming/transport untouched."""

    def __init__(self, client, *, budget=None):
        self._client = client
        self._budget = budget or NativeBudget(str(client.base_url))
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def __getattr__(self, name):
        return getattr(self._client, name)

    async def _create(self, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = fold_system_messages(kwargs["messages"])
        extra = dict(kwargs.get("extra_body") or {})
        extra.setdefault("reasoning_format", "auto")
        extra.setdefault("chat_template_kwargs", {"enable_thinking": False})
        kwargs["extra_body"] = extra
        await asyncio.to_thread(self._budget.check, kwargs)
        return await self._client.chat.completions.create(**kwargs)
