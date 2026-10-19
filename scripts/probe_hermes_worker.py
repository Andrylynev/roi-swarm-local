"""Live probe: run ONE real Hermes worker through the pool (paid call, bounded)."""

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

from roi_swarm.runner import PROJECT_ROOT, _hermes_home
from roi_swarm.schemas import PlannedTask
from roi_swarm.settings import load_settings
from roi_swarm.waves import HermesWorkerPool


async def main() -> None:
    settings = load_settings(PROJECT_ROOT / "config" / "settings.yaml")
    root = str(_hermes_home() / "hermes-agent")
    pool = HermesWorkerPool(
        python=str(Path(root) / "venv" / "Scripts" / "python.exe"),
        hermes_root=root,
        base_url=settings.provider.base_url,
        api_key=settings.provider.api_key,
        api_mode=settings.provider.api_mode,
        model=settings.model_for("worker").model,
        disabled_toolsets=settings.hermes_workers.disabled_toolsets,
        max_iterations=15,
    )
    work = Path(tempfile.mkdtemp(prefix="roi-wave-probe-", dir=os.environ.get("TEMP")))
    started = time.monotonic()
    outcome = await pool.run(
        PlannedTask(
            id="probe",
            objective=(
                "Create hello.py that prints the sum of 17 and 25, run it with python, "
                "and report the printed output."
            ),
        ),
        goal="live probe",
        cwd=str(work),
        on_event=lambda e: print("EVENT", e.get("type"), e.get("name"), flush=True),
    )
    print("elapsed", round(time.monotonic() - started, 1))
    print("ok", outcome.ok, "tools", outcome.tools)
    print("report", outcome.report[:800])
    target = work / "hello.py"
    print("file exists", target.exists(), target.read_text() if target.exists() else "")
    return 0 if outcome.ok and target.exists() else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
