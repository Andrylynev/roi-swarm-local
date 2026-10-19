"""Repeatable API benchmarks. Separate load latency, prompt time, decode, and tools."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def request(base: str, payload: dict, timeout: int = 600):
    started = time.perf_counter()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    return {"wall_seconds": time.perf_counter() - started, "response": result}


def run(base: str, model: str, *, rounds: int = 3, parallel: int = 1) -> dict:
    prompt = (
        "You are reviewing a small local Python application. Explain input validation, "
        "unit testing, exception handling, and clear interfaces. Use concrete examples. "
    ) * 8
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.2,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    cold = request(base, {**payload, "max_tokens": 16})
    observations = []
    for round_id in range(rounds):
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
            outputs = list(pool.map(lambda _: request(base, payload), range(parallel)))
        elapsed = time.perf_counter() - start
        for output in outputs:
            response = output["response"]
            usage = response.get("usage", {})
            timings = response.get("timings", {})
            observations.append(
                {
                    "round": round_id,
                    "wall_seconds": output["wall_seconds"],
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "tokens_per_second": timings.get("predicted_per_second"),
                    "prompt_tokens_per_second": timings.get("prompt_per_second"),
                    "aggregate_tokens_per_second": sum(
                        item["response"].get("usage", {}).get("completion_tokens", 0)
                        for item in outputs
                    )
                    / elapsed,
                    "finish_reason": response["choices"][0].get("finish_reason"),
                }
            )
    tool = {
        "type": "function",
        "function": {
            "name": "write_note",
            "description": "Save a local note",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["filename", "text"],
            },
        },
    }
    tool_result = request(
        base,
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Call write_note to save filename probe.txt with text exactly LOCAL_OK."
                    ),
                }
            ],
            "tools": [tool],
            "max_tokens": 512,
            "temperature": 0.1,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    calls = tool_result["response"]["choices"][0]["message"].get("tool_calls", [])
    valid = False
    if calls:
        call = calls[0]["function"]
        try:
            args = json.loads(call["arguments"])
            valid = call["name"] == "write_note" and args == {
                "filename": "probe.txt",
                "text": "LOCAL_OK",
            }
        except (ValueError, TypeError):
            pass
    return {
        "base": base,
        "model": model,
        "parallel": parallel,
        "cold_wall_seconds": cold["wall_seconds"],
        "samples": observations,
        "median_wall_seconds": statistics.median(x["wall_seconds"] for x in observations),
        "median_aggregate_tokens_per_second": statistics.median(
            x["aggregate_tokens_per_second"] for x in observations
        ),
        "tool_call_valid": valid,
        "tool_wall_seconds": tool_result["wall_seconds"],
        "tool_response": tool_result["response"]["choices"][0]["message"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8091/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.base, args.model, rounds=args.rounds, parallel=args.parallel)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
