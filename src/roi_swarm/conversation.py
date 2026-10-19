"""Resolve a task without mistaking Hermes runtime notices for user requests.

These helpers only select task metadata. The original history stays untouched.
"""

from __future__ import annotations

from typing import Any

_MODEL_NOTICE_PREFIX = "[System: The active model for this chat has changed to "
_MODEL_NOTICE_SUFFIX = (
    ". From this point forward, use this runtime metadata when answering questions "
    "about what model/provider is active.]"
)


def task_text(content: Any) -> str:
    if isinstance(content, list):
        content = "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "input_text"}
        )
    if not isinstance(content, str):
        return ""
    text = content.strip()
    while text.startswith(_MODEL_NOTICE_PREFIX):
        end = text.find(_MODEL_NOTICE_SUFFIX, len(_MODEL_NOTICE_PREFIX))
        if end < 0:
            break  # Not a complete known runtime notice: retain the user's text.
        text = text[end + len(_MODEL_NOTICE_SUFFIX) :].strip()
    return text


def current_task(messages: list[dict[str, Any]]) -> tuple[int, str]:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "user" and not is_runtime_notice(message):
            return index, task_text(message.get("content"))
    return -1, ""


def is_runtime_notice(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if message.get("role") != "user":
        return False
    if isinstance(content, list):
        if not content or any(
            not isinstance(part, dict) or part.get("type") not in {"text", "input_text"}
            for part in content
        ):
            return False
        content = "\n".join(str(part.get("text") or "") for part in content)
    return (
        isinstance(content, str)
        and content.strip().startswith(_MODEL_NOTICE_PREFIX)
        and not task_text(content)
    )
