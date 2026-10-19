import pytest

from roi_swarm.gateway import ModelGateway
from roi_swarm.schemas import WorkerResult


class FakeResponses:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"output_text": self.text})()


class FakeCompletions:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        message = type("Message", (), {"content": self.text})()
        choice = type("Choice", (), {"message": message})()
        return type("Completion", (), {"choices": [choice]})()


class FakeClient:
    def __init__(self, text: str) -> None:
        self.responses = FakeResponses(text)
        completions = FakeCompletions(text)
        self.chat = type("Chat", (), {"completions": completions})()


@pytest.mark.asyncio
async def test_gateway_validates_structured_output_and_passes_temperature() -> None:
    client = FakeClient(
        '{"task_id":"t1","conclusion":"done","evidence":[],"confidence":0.8,"risks":[]}'
    )
    gateway = ModelGateway(client=client)

    result = await gateway.generate_structured(
        model="worker-model",
        system="work",
        prompt="task",
        schema=WorkerResult,
        temperature=0.25,
    )

    assert result.task_id == "t1"
    assert result.confidence == 0.8
    assert client.responses.calls[0]["model"] == "worker-model"
    assert client.responses.calls[0]["temperature"] == 0.25


@pytest.mark.asyncio
async def test_gateway_extracts_json_from_markdown_fence() -> None:
    client = FakeClient(
        '```json\n{"task_id":"t2","conclusion":"ok","evidence":[],"confidence":0.6,"risks":[]}\n```'
    )
    gateway = ModelGateway(client=client)

    result = await gateway.generate_structured(
        model="worker-model",
        system="work",
        prompt="task",
        schema=WorkerResult,
    )

    assert result.task_id == "t2"


@pytest.mark.asyncio
async def test_gateway_can_use_chat_completions_mode() -> None:
    client = FakeClient("CHAT_OK")
    gateway = ModelGateway(client=client, api_mode="chat_completions")

    result = await gateway.generate_text(
        model="worker-model",
        system="work",
        prompt="task",
        temperature=0.4,
    )

    assert result == "CHAT_OK"
    call = client.chat.completions.calls[0]
    assert call["temperature"] == 0.4
    assert call["messages"][0] == {"role": "system", "content": "work"}


class SequenceCompletions:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        message = type("Message", (), {"content": self.texts.pop(0)})()
        choice = type("Choice", (), {"message": message})()
        return type("Completion", (), {"choices": [choice]})()


@pytest.mark.asyncio
async def test_gateway_retries_once_when_model_breaks_schema() -> None:
    good = '{"task_id":"t3","conclusion":"ok","evidence":[],"confidence":0.7,"risks":[]}'
    completions = SequenceCompletions(['{"title": "wrong shape"}', good])
    client = type("C", (), {"chat": type("Chat", (), {"completions": completions})()})()
    gateway = ModelGateway(client=client, api_mode="chat_completions")

    result = await gateway.generate_structured(
        model="m", system="s", prompt="p", schema=WorkerResult
    )

    assert result.task_id == "t3"
    assert len(completions.calls) == 2
    assert "task_id" in completions.calls[1]["messages"][-1]["content"]
