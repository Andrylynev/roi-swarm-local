"""Exercise local planner -> real Hermes tools -> verifier, verify actual output files."""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inspect_files(folder: Path, expected: dict[str, str]) -> dict:
    raw = {
        name: (folder / name).read_bytes() if (folder / name).is_file() else None
        for name in expected
    }
    return {
        "actual": {
            name: value.decode("utf-8", errors="replace") if value is not None else None
            for name, value in raw.items()
        },
        "actual_hex": {
            name: value.hex() if value is not None else None for name, value in raw.items()
        },
        "verified": all(raw[name] == value.encode("utf-8") for name, value in expected.items()),
    }


async def main():
    from local.api import LocalSwarmRunner
    from roi_swarm.telemetry import RunStore

    runner = LocalSwarmRunner()
    (ROOT / "data/diagnostics").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/bench").mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="swarm-probe-", dir=ROOT / "data/diagnostics"))
    # A nested independent Git repository keeps artifacts trackable and avoids
    # the parent application's ignored diagnostic-data merge policy.
    subprocess.run(["git", "init", "-q", str(folder)], check=True)
    (folder / "README.md").write_text("Disposable local swarm fixture.\n", encoding="utf-8")
    events = []

    def event(node, status, data):
        row = {"node": node, "status": status, "data": data, "time": time.time()}
        events.append(row)
        print("EVENT", node, status, str(data)[:200], flush=True)

    goal = (
        f"Work ONLY in the existing folder {folder.as_posix()}. "
        "Create four independent UTF-8 text files: a.txt containing exactly ALPHA, "
        "b.txt containing exactly BETA, c.txt containing exactly GAMMA, "
        "d.txt containing exactly DELTA. Assign one file per worker, no duplicate ownership. "
        "Do not append newline characters, spaces, BOM, or quotes to any file. "
        "Use real file tools; do not merely describe the files. Verify exact bytes. "
        "Do not schedule a verification-only worker in the same wave as its prerequisites."
    )
    started = time.perf_counter()
    try:

        class ProbeStore(RunStore):
            def record_event(self, run_id, node, status, data):
                event(node, status, data)
                return super().record_event(run_id, node, status, data)

        runner.store = ProbeStore(ROOT / "runs")
        report = await runner.run_waves(
            [
                {"role": "system", "content": f"Current working directory: {folder.as_posix()}"},
                {"role": "user", "content": goal},
            ],
            run_id="local-probe-" + str(int(time.time())),
        )
        # runner method signature is checked before this script is first exercised.
        expected = {"a.txt": "ALPHA", "b.txt": "BETA", "c.txt": "GAMMA", "d.txt": "DELTA"}
        inspected = inspect_files(folder, expected)
        result = {
            "seconds": time.perf_counter() - started,
            "folder": str(folder),
            "expected": expected,
            **inspected,
            "report": report,
            "events": events,
        }
        (ROOT / "data/bench/local-swarm-probe.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps({k: v for k, v in result.items() if k != "events"}, ensure_ascii=False))
        if not result["verified"]:
            raise SystemExit("Actual local files did not pass verification")
    finally:
        await runner.close()


if __name__ == "__main__":
    os.environ["HERMES_HOME"] = str(ROOT / "hermes")
    asyncio.run(main())
