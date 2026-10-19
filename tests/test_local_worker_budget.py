"""The worker's real Hermes call must receive the configured output reserve."""

import asyncio
import json
import sys

import roi_swarm.waves as waves
from roi_swarm.hermes_worker_host import generation_options
from roi_swarm.schemas import PlannedTask


def test_worker_generation_options_are_optional_and_validate_output_budget():
    assert generation_options({}) == {}
    assert generation_options({"max_output_tokens": 4096}) == {"max_tokens": 4096}


def test_pool_transmits_output_budget_to_host(tmp_path, monkeypatch):
    script = tmp_path / "host.py"
    script.write_text(
        "import json,sys\n"
        "task=json.loads(sys.stdin.readline())\n"
        "print(json.dumps({'type':'final','text':str(task['max_output_tokens'])}),flush=True)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(waves, "HOST_SCRIPT", script)
    pool = waves.HermesWorkerPool(
        python=sys.executable,
        hermes_root=str(tmp_path),
        base_url="http://127.0.0.1:1/v1",
        api_key="local",
        api_mode="chat_completions",
        model="local",
        disabled_toolsets=[],
        max_iterations=2,
        max_output_tokens=4096,
    )
    outcome = asyncio.run(
        pool.run(
            PlannedTask(id="budget", objective="x"),
            goal="",
            cwd=str(tmp_path),
            on_event=lambda event: None,
        )
    )
    assert outcome.ok and json.loads(outcome.report) == 4096
