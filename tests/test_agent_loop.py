import asyncio
import json
from types import SimpleNamespace

from roi_swarm.agent_loop import DELEGATE_TOOL_NAME, HermesToolAgent

HERMES_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
]


def _call(call_id: str, name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _response(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    reason = "tool_calls" if tool_calls else "stop"
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=reason)])


class ScriptedClient:
    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class SwarmRecorder:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []

    async def __call__(self, request: str, run_id: str) -> str:
        self.requests.append((request, run_id))
        return "SWARM REPORT: 8 parts analysed"


def _agent(client: ScriptedClient, swarm: SwarmRecorder) -> HermesToolAgent:
    return HermesToolAgent(client, model="claude-opus-5-5", run_swarm=swarm)


def test_hermes_tool_calls_are_returned_to_hermes_for_execution() -> None:
    client = ScriptedClient(
        [_response(tool_calls=[_call("c1", "read_file", {"path": "C:/a.zip"})])]
    )
    swarm = SwarmRecorder()

    turn = asyncio.run(
        _agent(client, swarm).respond(
            [{"role": "user", "content": "unpack C:/a.zip"}], HERMES_TOOLS, run_id="r1"
        )
    )

    assert turn.finish_reason == "tool_calls"
    assert turn.tool_calls == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "C:/a.zip"}'},
        }
    ]
    sent_names = [tool["function"]["name"] for tool in client.calls[0]["tools"]]
    assert sent_names == ["read_file", DELEGATE_TOOL_NAME]
    # Hermes' own system prompt and history are forwarded unchanged after our addendum.
    assert client.calls[0]["messages"][-1] == {"role": "user", "content": "unpack C:/a.zip"}
    assert swarm.requests == []


def test_delegate_runs_swarm_internally_and_feeds_result_back() -> None:
    client = ScriptedClient(
        [
            _response(
                tool_calls=[
                    _call("d1", DELEGATE_TOOL_NAME, {"task": "design cache", "materials": "src"})
                ]
            ),
            _response(content="Final answer based on swarm"),
        ]
    )
    swarm = SwarmRecorder()

    turn = asyncio.run(
        _agent(client, swarm).respond(
            [{"role": "user", "content": "rework project"}], HERMES_TOOLS, run_id="r2"
        )
    )

    assert turn.finish_reason == "stop"
    assert turn.tool_calls == []
    assert turn.content == "Final answer based on swarm"
    assert len(swarm.requests) == 1
    request, swarm_run_id = swarm.requests[0]
    assert "design cache" in request and "src" in request
    assert swarm_run_id.startswith("r2")
    tool_message = client.calls[1]["messages"][-1]
    assert tool_message == {
        "role": "tool",
        "tool_call_id": "d1",
        "content": "SWARM REPORT: 8 parts analysed",
    }


def test_swarm_report_survives_into_hermes_history_when_tools_follow() -> None:
    client = ScriptedClient(
        [
            _response(tool_calls=[_call("d1", DELEGATE_TOOL_NAME, {"task": "plan"})]),
            _response(
                content="Applying the plan",
                tool_calls=[_call("w1", "read_file", {"path": "x"})],
            ),
        ]
    )

    turn = asyncio.run(
        _agent(client, SwarmRecorder()).respond(
            [{"role": "user", "content": "go"}], HERMES_TOOLS, run_id="r3"
        )
    )

    assert turn.finish_reason == "tool_calls"
    assert [call["id"] for call in turn.tool_calls] == ["w1"]
    # Hermes only stores what we return, so the internal swarm result must be in content.
    assert "SWARM REPORT" in turn.content
    assert "Applying the plan" in turn.content


def test_swarm_failure_is_reported_to_the_model_not_raised() -> None:
    async def broken_swarm(request: str, run_id: str) -> str:
        raise RuntimeError("boom")

    client = ScriptedClient(
        [
            _response(tool_calls=[_call("d1", DELEGATE_TOOL_NAME, {"task": "t"})]),
            _response(content="done anyway"),
        ]
    )

    turn = asyncio.run(
        HermesToolAgent(client, model="m", run_swarm=broken_swarm).respond(
            [{"role": "user", "content": "go"}], HERMES_TOOLS, run_id="r4"
        )
    )

    assert turn.content == "done anyway"
    assert "RuntimeError" in client.calls[1]["messages"][-1]["content"]


def test_swarm_report_is_saved_to_file_and_path_given_to_model(tmp_path) -> None:
    client = ScriptedClient(
        [
            _response(tool_calls=[_call("d1", DELEGATE_TOOL_NAME, {"task": "review"})]),
            _response(content="ok"),
        ]
    )
    agent = HermesToolAgent(client, model="m", run_swarm=SwarmRecorder(), reports_dir=tmp_path)

    asyncio.run(agent.respond([{"role": "user", "content": "go"}], HERMES_TOOLS, run_id="r5"))

    saved = tmp_path / "r5-s1.md"
    assert saved.read_text(encoding="utf-8") == "SWARM REPORT: 8 parts analysed"
    tool_content = client.calls[1]["messages"][-1]["content"]
    assert str(saved) in tool_content
    assert "SWARM REPORT" in tool_content


def test_calls_issued_together_with_delegate_are_deferred_until_report_is_known() -> None:
    client = ScriptedClient(
        [
            _response(
                tool_calls=[
                    _call("d1", DELEGATE_TOOL_NAME, {"task": "plan"}),
                    _call("w1", "write_file", {"path": "x", "content": "placeholder"}),
                ]
            ),
            _response(tool_calls=[_call("w2", "write_file", {"path": "x", "content": "real"})]),
        ]
    )

    turn = asyncio.run(
        _agent(client, SwarmRecorder()).respond(
            [{"role": "user", "content": "go"}], HERMES_TOOLS, run_id="r6"
        )
    )

    assert [call["id"] for call in turn.tool_calls] == ["w2"]
    deferred = [m for m in client.calls[1]["messages"] if m.get("tool_call_id") == "w1"]
    assert deferred and "Not executed" in deferred[0]["content"]


FILE_TOOLS = [
    *HERMES_TOOLS,
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    },
]


class Assessor:
    def __init__(self, verdict: bool) -> None:
        self.verdict = verdict
        self.prompts: list[str] = []

    async def __call__(self, prompt: str) -> bool:
        self.prompts.append(prompt)
        return self.verdict


def test_big_task_forces_delegation_when_model_tries_to_finish_alone() -> None:
    client = ScriptedClient(
        [
            _response(content="I did everything myself"),
            _response(content="Done with swarm"),
        ]
    )
    swarm = SwarmRecorder()
    assessor = Assessor(True)
    agent = HermesToolAgent(client, model="m", run_swarm=swarm, assess=assessor)

    turn = asyncio.run(
        agent.respond([{"role": "user", "content": "rework 8 modules"}], FILE_TOOLS, run_id="r")
    )

    assert assessor.prompts == ["rework 8 modules"]
    assert len(swarm.requests) == 1
    assert "rework 8 modules" in swarm.requests[0]
    assert turn.content == "Done with swarm"
    # Thinking models reject forced tool_choice (400), so it must never be sent.
    assert all("tool_choice" not in call for call in client.calls)
    assert "MUST" in client.calls[0]["messages"][1]["content"]


def test_big_task_blocks_writes_before_delegation_but_allows_reads() -> None:
    client = ScriptedClient(
        [
            _response(
                tool_calls=[
                    _call("r1", "read_file", {"path": "a"}),
                    _call("w1", "write_file", {"path": "b"}),
                ]
            ),
        ]
    )
    agent = HermesToolAgent(client, model="m", run_swarm=SwarmRecorder(), assess=Assessor(True))
    blocked_client_round = ScriptedClient(
        [_response(tool_calls=[_call("r2", "read_file", {"path": "a"})])]
    )
    del blocked_client_round

    turn = asyncio.run(agent.respond([{"role": "user", "content": "big"}], FILE_TOOLS, run_id="r"))

    # Writes before the swarm ran are not forwarded; reads (gathering materials) are.
    assert [call["function"]["name"] for call in turn.tool_calls] == ["read_file"]


def test_no_forcing_after_swarm_already_reported_in_this_turn() -> None:
    client = ScriptedClient([_response(tool_calls=[_call("w1", "write_file", {"path": "b"})])])
    assessor = Assessor(True)
    agent = HermesToolAgent(client, model="m", run_swarm=SwarmRecorder(), assess=assessor)
    history = [
        {"role": "user", "content": "big"},
        {"role": "assistant", "content": "Отчёт роя:\nreport", "tool_calls": []},
        {"role": "tool", "tool_call_id": "x", "content": "ok"},
    ]

    turn = asyncio.run(agent.respond(history, FILE_TOOLS, run_id="r"))

    assert [call["function"]["name"] for call in turn.tool_calls] == ["write_file"]
    assert assessor.prompts == []
    assert "tool_choice" not in client.calls[0]


def test_small_task_and_non_file_toolsets_are_not_forced() -> None:
    client = ScriptedClient([_response(content="ok"), _response(content="ok")])
    assessor = Assessor(False)
    agent = HermesToolAgent(client, model="m", run_swarm=SwarmRecorder(), assess=assessor)
    asyncio.run(agent.respond([{"role": "user", "content": "hi"}], FILE_TOOLS, run_id="r"))
    always = Assessor(True)
    agent2 = HermesToolAgent(client, model="m", run_swarm=SwarmRecorder(), assess=always)
    # Hermes auxiliary jobs (skill review) only carry skill tools: never route them.
    asyncio.run(agent2.respond([{"role": "user", "content": "review"}], [], run_id="r"))

    assert assessor.prompts == ["hi"]
    assert always.prompts == []
    assert all("tool_choice" not in call for call in client.calls)
