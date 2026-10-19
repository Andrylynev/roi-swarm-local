"""Provider-free boundary tests for schemas, repair prompts and tool definitions."""

import json

import pytest
from test_agent_loop import ScriptedClient, SwarmRecorder, _response
from test_gateway import FakeClient

from roi_swarm.agent_loop import COORDINATOR_ADDENDUM, DELEGATE_TOOL, HermesToolAgent
from roi_swarm.gateway import ModelGateway
from roi_swarm.schemas import WorkerResult
from roi_swarm.token_budget import check_context


async def test_schema_repair_is_rechecked_before_second_api_call():
    prompt = " z" * 100
    schema_json = json.dumps(WorkerResult.model_json_schema(), ensure_ascii=False)
    full_system = (
        "system\nReturn only JSON matching this JSON Schema. "
        f"Do not add markdown. Schema: {schema_json}"
    )
    window = (
        check_context(
            [{"role": "system", "content": full_system}, {"role": "user", "content": prompt}],
            limit=100000,
        )
        + 100
    )
    client = FakeClient('{"wrong":true}')
    gateway = ModelGateway(client)
    with pytest.raises(ValueError, match="Context exceeds"):
        await gateway.generate_structured(
            model="worker",
            system="system",
            prompt=prompt,
            schema=WorkerResult,
            max_output_tokens=100,
            context_window_tokens=window,
        )
    assert len(client.responses.calls) == 1


async def test_coordinator_budgets_tools_and_output_together():
    messages = [{"role": "user", "content": "hello"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": " z" * 1000,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    value = {
        "messages": [{"role": "system", "content": COORDINATOR_ADDENDUM}, *messages],
        "tools": [*tools, DELEGATE_TOOL],
    }
    used = check_context(value, limit=100000)
    client = ScriptedClient([_response(content="ok")])
    agent = HermesToolAgent(
        client,
        model="main",
        run_swarm=SwarmRecorder(),
        max_output_tokens=100,
        context_window_tokens=used + 99,
    )
    with pytest.raises(ValueError, match="Context exceeds"):
        await agent.respond(messages, tools, run_id="too-small")
    assert client.calls == []
    agent.context_window_tokens = used + 100
    assert (await agent.respond(messages, tools, run_id="exact-fit")).content == "ok"
    assert len(client.calls) == 1
