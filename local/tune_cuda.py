"""Compare native llama server MTP, batch, and parallel slots on the actual GPU."""

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from local.benchmark_http import run
from local.manage import child_environment

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def unload_router():
    with urllib.request.urlopen("http://127.0.0.1:8091/models", timeout=10) as response:
        models = json.load(response)["data"]
    for model in models:
        if model["status"]["value"] == "loaded":
            req = urllib.request.Request(
                "http://127.0.0.1:8091/models/unload",
                data=json.dumps({"model": model["id"]}).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=60).close()


def measure(size, *, mtp=False, slots=1, ubatch=128):
    port = free_port()
    filename = f"Ternary-Bonsai-2-{size}-PQ2_0-MTP-Q8_0.gguf"
    args = [
        str(ROOT / "runtime/llama-server.exe"),
        "-m",
        str(ROOT / "models" / filename),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-ngl",
        "99",
        "-fa",
        "on",
        "-c",
        str(65536 * slots),
        "-np",
        str(slots),
        "-t",
        "6",
        "-tb",
        "6",
        "-b",
        "512",
        "-ub",
        str(ubatch),
        "-ctk",
        "q8_0",
        "-ctv",
        "q8_0",
        "--cache-ram",
        "256",
        "--jinja",
        "--reasoning",
        "off",
        "--reasoning-format",
        "auto",
        "--temp",
        "0.7",
        "--top-p",
        "0.8",
        "--top-k",
        "20",
        "--min-p",
        "0",
        "--presence-penalty",
        "1.5",
        "--repeat-last-n",
        "64",
        "--no-webui",
    ]
    if mtp:
        args += [
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            "2",
            "--spec-draft-n-min",
            "0",
            "--spec-draft-p-min",
            "0",
        ]
    label = f"{size.lower()}-mtp{int(mtp)}-slots{slots}-ub{ubatch}"
    with (ROOT / f"data/bench/tune-{label}.log").open("wb") as log:
        process = subprocess.Popen(
            args,
            cwd=ROOT,
            env=child_environment(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            deadline = time.monotonic() + 120
            while True:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                        if r.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(
                            "Native tuning server did not become ready: " + label
                        ) from None
                    time.sleep(0.25)
            result = run(f"http://127.0.0.1:{port}/v1", "tuning", rounds=2, parallel=slots)
            memory = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"],
                text=True,
            ).strip()
            result.update({"label": label, "command": args, "gpu_memory": memory})
            (ROOT / f"data/bench/tune-{label}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in [
                            "label",
                            "median_aggregate_tokens_per_second",
                            "tool_call_valid",
                            "gpu_memory",
                        ]
                    }
                ),
                flush=True,
            )
            return result
        finally:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            process.wait(timeout=15)


if __name__ == "__main__":
    unload_router()
    for options in [
        {"size": "9B", "mtp": False},
        {"size": "9B", "mtp": True},
        {"size": "4B", "mtp": False},
        {"size": "4B", "mtp": True},
        {"size": "4B", "mtp": False, "slots": 2},
        {"size": "4B", "mtp": False, "slots": 2, "ubatch": 256},
    ]:
        measure(**options)
