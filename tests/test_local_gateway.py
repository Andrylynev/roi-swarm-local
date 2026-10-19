"""Constrain local planner JSON without modifying the original remote gateway."""

import asyncio
import json
import types

from pydantic import BaseModel

from local.gateway import LocalModelGateway


class Answer(BaseModel):
    value: int


def test_local_structured_request_sends_json_schema():
    calls = []

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=json.dumps({"value": 42}))
                    )
                ]
            )

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Completions()))
    gateway = LocalModelGateway(client)
    answer = asyncio.run(
        gateway.generate_structured(
            model="bonsai-9b",
            system="Return result",
            prompt="value 42",
            schema=Answer,
        )
    )
    assert answer.value == 42
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["schema"] == Answer.model_json_schema()
    assert calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert calls[0]["extra_body"]["reasoning_format"] == "auto"
