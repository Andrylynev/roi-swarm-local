"""Exercise a real long input against the allocated 64K local worker slot."""

import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def post(path, payload):
    req = urllib.request.Request(
        "http://127.0.0.1:8091" + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def main():
    # Distinct numbered facts avoid a repeated-token prompt-cache shortcut.
    prompt = "\n".join(
        f"Record {i}: the local audit artifact is item_{i} and status is checked."
        for i in range(2600)
    )
    prompt += "\nIgnore the records for this test and reply exactly LONG_CONTEXT_OK."
    started = time.perf_counter()
    response = post(
        "/v1/chat/completions",
        {
            "model": "bonsai-4b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "temperature": 0.1,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    actual = response["choices"][0]["message"]["content"]
    result = {
        "wall_seconds": time.perf_counter() - started,
        "usage": response.get("usage"),
        "timings": response.get("timings"),
        "answer": actual,
        "correct": actual.strip() == "LONG_CONTEXT_OK",
    }
    (ROOT / "data/bench/long-context.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if not result["correct"]:
        raise SystemExit("Long-context instruction check failed")


if __name__ == "__main__":
    main()
