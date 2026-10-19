import json

import pytest
from pydantic import ValidationError
from test_gateway import FakeClient

from roi_swarm.gateway import ModelGateway
from roi_swarm.schemas import WorkerResult

VALID = {"topic": "materials", "situation": "s", "recommendation": "r", "evidence": "e"}


@pytest.mark.parametrize(
    "notes", [None, {}, "bad", [{"topic": "invalid"}], [VALID, {"topic": "invalid"}, VALID, VALID]]
)
@pytest.mark.asyncio
async def test_optional_notes_never_trigger_worker_retries(notes):
    client = FakeClient(
        json.dumps(
            {"task_id": "t", "conclusion": "useful", "confidence": 0.9, "workflow_lessons": notes}
        )
    )
    result = await ModelGateway(client).generate_structured(
        model="fake",
        system="work",
        prompt="test",
        schema=WorkerResult,
    )
    assert result.conclusion == "useful" and result.confidence == 0.9
    assert len(result.workflow_lessons) == (2 if isinstance(notes, list) and len(notes) > 1 else 0)
    assert len(client.responses.calls) == 1


def test_primary_worker_fields_remain_strict():
    with pytest.raises(ValidationError):
        WorkerResult(task_id="t", conclusion="valid", confidence=2, workflow_lessons=None)
