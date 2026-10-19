"""Qwen's native template accepts only one initial system message."""

import asyncio
import copy
import types

import pytest

from local.messages import LocalCompletionClient, fold_system_messages


def test_merge_only_initial_instruction_messages_preserves_inputs():
    messages = [
        {"role": "system", "content": "coordinator"},
        {"role": "system", "content": "Hermes system"},
        {"role": "developer", "content": [{"type": "text", "text": "constraints"}]},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call"}]},
        {"role": "tool", "tool_call_id": "call", "content": "evidence"},
    ]
    original = copy.deepcopy(messages)
    result = fold_system_messages(messages)
    assert result[0] == {"role": "system", "content": "coordinator\n\nHermes system\n\nconstraints"}
    assert result[1:] == messages[3:]
    assert messages == original


def test_midconversation_system_is_rejected_not_silently_reordered():
    with pytest.raises(ValueError, match="mid-conversation"):
        fold_system_messages(
            [{"role": "user", "content": "x"}, {"role": "system", "content": "later instruction"}]
        )


def test_transport_applies_normalization_and_leaves_schema_intact():
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return "response"

    original = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)),
        models="catalog",
    )
    client = LocalCompletionClient(original, budget=types.SimpleNamespace(check=lambda value: 0))
    result = asyncio.run(
        client.chat.completions.create(
            messages=[{"role": "system", "content": "a"}, {"role": "system", "content": "b"}],
            extra_body={"json_schema": {"type": "object"}},
        )
    )
    assert result == "response"
    assert client.models == "catalog"
    assert len(calls[0]["messages"]) == 1
    assert calls[0]["extra_body"]["json_schema"] == {"type": "object"}
    assert calls[0]["extra_body"]["reasoning_format"] == "auto"
