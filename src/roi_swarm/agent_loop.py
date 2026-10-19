"""Hermes tool loop: the swarm coordinator uses Hermes' own tools.

Hermes sends its full system prompt, history and tool list with every request. The
coordinator model (Opus) sees exactly that, so it has the same permissions, tools and
information as Hermes. Hermes tool calls (files, terminal, web, ...) are returned to
Hermes for execution on the user's machine. One extra tool, ``delegate_to_swarm``,
is executed here: it runs the Luna swarm on materials the coordinator has gathered.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from roi_swarm.conversation import current_task
from roi_swarm.token_budget import check_context

DELEGATE_TOOL_NAME = "delegate_to_swarm"

DELEGATE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": DELEGATE_TOOL_NAME,
        "description": (
            "Run the ROI swarm: a planner splits the task into atomic subtasks, many parallel "
            "workers solve them, results are verified and synthesized into one report. Use it "
            "for large analysis/design work with more than 3 independent parts. Workers have "
            "NO file or tool access: first read the needed files with your tools and put the "
            "relevant contents into 'materials'. Apply the returned report yourself with your "
            "tools (write files, run commands). The full report is also saved to a .md file "
            "whose path is returned; copy it instead of retyping long reports."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Complete, self-contained task statement for the swarm.",
                },
                "materials": {
                    "type": "string",
                    "description": (
                        "File contents, code, listings, facts and constraints the workers need."
                    ),
                },
            },
            "required": ["task"],
        },
    },
}

COORDINATOR_ADDENDUM = (
    "You are running as the ROI swarm coordinator inside Hermes. You have every Hermes tool "
    "listed in this request, with Hermes' permissions: use them for all file work (read, "
    "unpack archives, edit, write, run commands). Attached files mentioned in the conversation "
    "(e.g. @file: paths, including ones Hermes could not inline) can be opened with those "
    f"tools. For big multi-part tasks call {DELEGATE_TOOL_NAME} with gathered materials, then "
    "carry out the result with your tools. Small tasks: just do them yourself."
)

MUST_DELEGATE_ADDENDUM = (
    "The ROI router classified the current user task as a multi-part task for the swarm. "
    f"You MUST call {DELEGATE_TOOL_NAME} before creating or changing any files: first read "
    "what you need (read files, list or unpack archives), put the relevant contents into "
    "'materials', delegate, and only then implement the swarm's report with your tools. "
    "Several independent large parts may be delegated as several parallel calls."
)

# Tools that change the user's files; blocked until the swarm has produced its report.
WRITE_TOOLS = frozenset({"write_file", "patch", "excess_write_file"})
# Toolsets without any of these are Hermes auxiliary jobs (skill review, titles): never route.
WORK_TOOLS = frozenset({"read_file", "write_file", "patch", "terminal", "execute_code"})
SWARM_MARKER = "Отчёт роя"
MAX_GATHER_ROUNDS = 8

SwarmCallable = Callable[[str, str], Awaitable[str]]
AssessCallable = Callable[[str], Awaitable[bool]]


@dataclass
class AgentTurn:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"


def _call_dict(call: Any) -> dict[str, Any]:
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.function.name, "arguments": call.function.arguments or "{}"},
    }


def _swarm_request(arguments: str) -> str:
    try:
        payload = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        payload = {"task": arguments}
    if not isinstance(payload, dict):
        payload = {"task": str(payload)}
    task = str(payload.get("task") or "").strip()
    materials = str(payload.get("materials") or "")
    return f"{task}\n\nMATERIALS:\n{materials}" if materials else task


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "")
            if isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"}
            else json.dumps(part, ensure_ascii=False)
            for part in content
        )
    return json.dumps(content, ensure_ascii=False)


def _current_turn(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Last user request and every message Hermes appended after it (tool rounds)."""
    index, request = current_task(messages)
    return request, messages[index + 1 :] if index >= 0 else []


def _auto_delegate_call(request: str, turn_messages: list[dict[str, Any]]) -> dict[str, Any]:
    gathered = "\n\n".join(
        _text(m.get("content")) for m in turn_messages if m.get("role") == "tool"
    )
    return {
        "id": "auto_delegate",
        "type": "function",
        "function": {
            "name": DELEGATE_TOOL_NAME,
            "arguments": json.dumps({"task": request, "materials": gathered}, ensure_ascii=False),
        },
    }


class HermesToolAgent:
    def __init__(
        self,
        client: Any,
        *,
        model: str,
        run_swarm: SwarmCallable,
        max_output_tokens: int = 16000,
        temperature: float | None = None,
        max_internal_rounds: int = 6,
        reports_dir: str | Path | None = None,
        assess: AssessCallable | None = None,
        context_window_tokens: int = 1000000,
        token_encoding: str = "o200k_base",
    ) -> None:
        self.client = client
        self.model = model
        self.run_swarm = run_swarm
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.max_internal_rounds = max_internal_rounds
        self.reports_dir = Path(reports_dir) if reports_dir is not None else None
        self.assess = assess
        self.context_window_tokens = context_window_tokens
        self.token_encoding = token_encoding

    async def respond(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        run_id: str,
    ) -> AgentTurn:
        must_delegate = await self._must_delegate(messages, tools)
        request, turn_messages = _current_turn(messages)
        gather_rounds = sum(
            1 for m in turn_messages if m.get("role") == "assistant" and m.get("tool_calls")
        )
        system = [{"role": "system", "content": COORDINATOR_ADDENDUM}]
        if must_delegate:
            system.append({"role": "system", "content": MUST_DELEGATE_ADDENDUM})
        conversation: list[dict[str, Any]] = [*system, *messages]
        all_tools = [*(tools or []), DELEGATE_TOOL]
        reports: list[str] = []
        swarm_count = 0
        # Forced tool_choice is rejected by thinking models, so the swarm is invoked by us.
        force = must_delegate and gather_rounds >= MAX_GATHER_ROUNDS
        for _ in range(self.max_internal_rounds):
            if force:
                content = ""
                delegated = [_auto_delegate_call(request, turn_messages)]
                external = []
                calls = delegated
            else:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": conversation,
                    "tools": all_tools,
                    "max_tokens": self.max_output_tokens,
                }
                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature
                check_context(
                    {"messages": conversation, "tools": all_tools},
                    limit=self.context_window_tokens,
                    output_tokens=self.max_output_tokens,
                    encoding=self.token_encoding,
                )
                response = await self.client.chat.completions.create(**kwargs)
                finish_reason = response.choices[0].finish_reason
                message = response.choices[0].message
                content = (message.content or "").strip()
                if finish_reason not in {"stop", "tool_calls"}:
                    return AgentTurn(content, [], finish_reason or "length")
                calls = [_call_dict(call) for call in (message.tool_calls or [])]
                delegated = [c for c in calls if c["function"]["name"] == DELEGATE_TOOL_NAME]
                external = [c for c in calls if c["function"]["name"] != DELEGATE_TOOL_NAME]
            if not delegated and must_delegate:
                allowed = [c for c in external if c["function"]["name"] not in WRITE_TOOLS]
                if allowed:
                    # Gathering materials (reads, unpacking) is fine before delegation.
                    return AgentTurn(content, allowed, "tool_calls")
                # The model tried to finish or write alone: hand the work to the swarm.
                calls = [_auto_delegate_call(request, turn_messages)]
                delegated, content = calls, ""
            if delegated:
                must_delegate, force = False, False
            if not delegated:
                if external:
                    text = "\n\n".join([*reports, content]).strip()
                    return AgentTurn(text, external, "tool_calls")
                return AgentTurn(content, [], "stop")

            run_ids = [f"{run_id}-s{swarm_count + i + 1}" for i in range(len(delegated))]
            swarm_count += len(delegated)
            results = await asyncio.gather(
                *(
                    self.run_swarm(_swarm_request(call["function"]["arguments"]), rid)
                    for call, rid in zip(delegated, run_ids, strict=True)
                ),
                return_exceptions=True,
            )
            outputs: list[str] = []
            for result, rid in zip(results, run_ids, strict=True):
                if isinstance(result, BaseException):
                    outputs.append(f"Swarm failed: {type(result).__name__}: {result}")
                    continue
                saved = self._save_report(rid, result)
                reports.append(f"Отчёт роя{f' ({saved})' if saved else ''}:\n{result}")
                outputs.append(
                    f"Swarm report saved to: {saved}\n\n{result}" if saved else str(result)
                )
            # Calls issued in the same message as the delegation were written before the
            # report existed (e.g. placeholder file writes). Never forward them; the model
            # re-issues what it still needs now that it sees the report.
            conversation.append(
                {"role": "assistant", "content": content or None, "tool_calls": calls}
            )
            for call in calls:
                if call["function"]["name"] == DELEGATE_TOOL_NAME:
                    out = outputs[delegated.index(call)]
                else:
                    out = (
                        "Not executed: it was issued before the swarm report was available. "
                        "Re-issue it now if still needed, using the report."
                    )
                conversation.append({"role": "tool", "tool_call_id": call["id"], "content": out})
        return AgentTurn("\n\n".join(reports) or "Swarm round limit reached.", [], "stop")

    async def _must_delegate(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> bool:
        if self.assess is None:
            return False
        names = {tool.get("function", {}).get("name") for tool in tools or []}
        if not names & WORK_TOOLS:
            return False
        request, turn_messages = _current_turn(messages)
        if not request:
            return False
        if any(
            m.get("role") == "assistant" and SWARM_MARKER in _text(m.get("content"))
            for m in turn_messages
        ):
            return False
        try:
            return bool(await self.assess(request))
        except Exception:
            return False

    def _save_report(self, swarm_run_id: str, report: str) -> str | None:
        if self.reports_dir is None:
            return None
        try:
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            path = self.reports_dir / f"{swarm_run_id}.md"
            path.write_text(report, encoding="utf-8")
        except OSError:
            return None
        return str(path)
