import asyncio

import pytest
from fastapi.testclient import TestClient
from test_graph_e2e import FakeGateway, _settings

from roi_swarm.api import create_app
from roi_swarm.context import ContextPipeline
from roi_swarm.graph import build_graph


@pytest.mark.parametrize("large_field", ["constraints", "request"])
def test_router_pack_overflow_falls_back_with_full_original_context(tmp_path, large_field):
    pipeline = ContextPipeline(tmp_path, maximum_pack_chars=12000)
    text = "EARLY_RULE\n" + "required information " * 700 + "\nMIDDLE_RULE\n"
    text += "required information " * 700 + "\nLATE_RULE"
    messages = [
        {"role": "system", "content": text if large_field == "constraints" else "Never invent."},
        {"role": "user", "content": text if large_field == "request" else "Review this task."},
    ]
    prepared = pipeline.prepare(messages, run_id="large-router")
    with pytest.raises(ValueError, match="Router required context"):
        pipeline.pack_for_router(prepared)
    gateway = FakeGateway()

    result = asyncio.run(
        build_graph(_settings(10), gateway, pipeline).ainvoke(
            {
                "request": prepared.current_request,
                "prepared_context": prepared,
            }
        )
    )

    assert result["final_answer"] == "done"
    assert result["answered_directly"] is True
    assert result["assessment"]["task_type"] == "router-fallback"
    assert gateway.calls == {"text": 1}
    assert text in gateway.synth_prompt
    assert all(
        marker in gateway.synth_prompt for marker in ("EARLY_RULE", "MIDDLE_RULE", "LATE_RULE")
    )


def test_long_system_context_completes_through_json_and_sse(tmp_path, monkeypatch):
    monkeypatch.delenv("ROI_SWARM_API_KEY", raising=False)
    pipeline = ContextPipeline(tmp_path, maximum_pack_chars=12000)
    gateway = FakeGateway()
    graph = build_graph(_settings(10), gateway, pipeline)
    instructions = "EARLY_INSTRUCTION\n" + "required rule " * 2200 + "\nLATE_INSTRUCTION"

    class GraphRunner:
        async def run(self, request, run_id=None, messages=None):
            prepared = pipeline.prepare(messages, run_id=run_id)
            return await graph.ainvoke({"request": request, "prepared_context": prepared})

    client = TestClient(create_app(GraphRunner()))
    for stream in (False, True):
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": "Review this task."},
                ],
                "stream": stream,
            },
        )
        assert response.status_code == 200
        if stream:
            assert '"content":"done"' in response.text
            assert "data: [DONE]" in response.text
            assert "ROI Swarm error" not in response.text
        else:
            assert response.json()["choices"][0]["message"]["content"] == "done"
        assert instructions in gateway.synth_prompt
