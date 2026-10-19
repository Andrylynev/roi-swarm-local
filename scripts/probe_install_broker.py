"""Offline E2E: real Hermes dispatch + central uv editable install in a temporary project.

No model calls, external packages, user environment changes or network are required.
The disposable project has its own .venv and a zero-dependency PEP 517 backend.
"""

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import roi_swarm.waves as waves
from roi_swarm.environment_guard import EnvironmentGuard
from roi_swarm.runner import PROJECT_ROOT, _hermes_home
from roi_swarm.schemas import PlannedTask
from roi_swarm.waves import HermesWorkerPool, InstallBroker
from roi_swarm.worktrees import WaveWorkspace

BACKEND = """
import base64,csv,hashlib,io,zipfile
from pathlib import Path

def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    name = "roi_probe-0.0.1-py3-none-any.whl"
    files = {
        "roi_probe.pth": str(Path.cwd() / "src").encode(),
        "roi_probe-0.0.1.dist-info/METADATA": (
            b"Metadata-Version: 2.1\\nName: roi-probe\\nVersion: 0.0.1\\n"
        ),
        "roi_probe-0.0.1.dist-info/WHEEL": (
            b"Wheel-Version: 1.0\\nGenerator: probe\\nRoot-Is-Purelib: true\\n"
            b"Tag: py3-none-any\\n"
        ),
    }
    records = []
    with zipfile.ZipFile(Path(wheel_directory) / name,"w") as wheel:
        for path,data in files.items():
            wheel.writestr(path,data)
            digest=base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            records.append([path,"sha256="+digest,len(data)])
        record="roi_probe-0.0.1.dist-info/RECORD"
        records.append([record,"",""])
        buffer=io.StringIO(); csv.writer(buffer,lineterminator="\\n").writerows(records)
        wheel.writestr(record,buffer.getvalue())
    return name
"""

HOST = """
import importlib.util,json,os,sys,threading
from pathlib import Path
task=json.loads(sys.stdin.readline())
sys.path.insert(0,task["hermes_root"])
sys.path.insert(0,task["context"])
from hermes_worker_host import Channel,guard_tools
import model_tools
stream=sys.stdout
sys.stdout=sys.stderr
lock=threading.Lock()
def emit(kind,**data):
    with lock:
        stream.write(json.dumps({"type":kind,**data})+"\\n");stream.flush()
os.chdir(task["cwd"])
os.environ["TERMINAL_CWD"]=task["cwd"]
cleanup=guard_tools(model_tools,Channel(emit,sys.stdin))
model_tools.get_tool_definitions(quiet_mode=True)
try:
    ordinary=json.loads(model_tools.handle_function_call(
        "terminal", {"command":"echo pip install is-only-text"}, task_id=task["task_id"]
    ))
    assert ordinary["exit_code"]==0,ordinary
    python=Path(task["cwd"])/".venv"/"Scripts"/"python.exe"
    command=f'uv pip install --python "{python}" --no-build-isolation -e .'
    installed=json.loads(model_tools.handle_function_call("terminal",{"command":command},task_id=task["task_id"]))
    assert installed["exit_code"]==0,installed
    # Foreground and background commands both use the real Hermes tool registry.
    background=json.loads(model_tools.handle_function_call(
        "terminal", {"command":"echo background-ok", "background":True}, task_id=task["task_id"]
    ))
    assert background.get("session_id"),background
    cleanup()
    emit("final",text=json.dumps({"ordinary":ordinary["output"],"installed":installed}))
except BaseException as exc:
    cleanup()
    emit("error",error=f"{type(exc).__name__}: {exc}")
"""


def run(*args, cwd):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


async def main():
    diagnostics = PROJECT_ROOT / "data" / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="install-probe-", dir=diagnostics) as scratch:
        root = Path(scratch)
        project = root / "project"
        project.mkdir()
        (project / "src").mkdir()
        (project / "src" / "roi_probe.py").write_text("VALUE = 42\n")
        (project / "backend.py").write_text(BACKEND)
        (project / "pyproject.toml").write_text(
            '[build-system]\nrequires=[]\nbuild-backend="backend"\nbackend-path=["."]\n'
            '[project]\nname="roi-probe"\nversion="0.0.1"\n'
        )
        (project / ".gitignore").write_text(".venv/\n__pycache__/\n")
        run("uv", "venv", "--python", sys.executable, str(project / ".venv"), cwd=root)
        run("git", "init", "-q", cwd=project)
        run(
            "git",
            "-c",
            "user.name=probe",
            "-c",
            "user.email=probe@localhost",
            "add",
            ".",
            cwd=project,
        )
        run(
            "git",
            "-c",
            "user.name=probe",
            "-c",
            "user.email=probe@localhost",
            "commit",
            "-qm",
            "probe",
            cwd=project,
        )
        host = root / "host.py"
        host.write_text(HOST)
        real_host = waves.HOST_SCRIPT
        waves.HOST_SCRIPT = host
        broker = InstallBroker()
        await broker.watch(str(project))
        workspace = WaveWorkspace.open(str(project), root=root / "roi-waves")
        assert workspace is not None
        workspace.snapshot()
        folder = workspace.prepare("probe")
        events = []
        hermes = _hermes_home() / "hermes-agent"
        pool = HermesWorkerPool(
            python=str(hermes / "venv" / "Scripts" / "python.exe"),
            hermes_root=str(hermes),
            base_url="unused",
            api_key="",
            api_mode="chat_completions",
            model="unused",
            disabled_toolsets=[],
            max_iterations=1,
        )
        try:
            outcome = await pool.run(
                PlannedTask(id="probe", objective="offline", context=str(real_host.parent)),
                goal="offline",
                cwd=str(folder),
                on_event=events.append,
                broker=broker,
                to_shared=lambda text: workspace.to_shared(text, "probe"),
            )
            assert outcome.ok, outcome.report
        finally:
            workspace.close()
            waves.HOST_SCRIPT = real_host
        python = project / ".venv" / "Scripts" / "python.exe"
        result = run(str(python), "-c", "import roi_probe; print(roi_probe.VALUE)", cwd=project)
        assert result.stdout.strip() == "42"
        assert EnvironmentGuard(project).broken_editables() == []
        assert await broker.check_environment() == []
        statuses = [e["status"] for e in events if e["type"] == "install_status"]
        assert statuses == ["waiting", "running", "completed"], statuses
        print(
            json.dumps(
                {
                    "real_hermes_dispatch": "passed",
                    "editable_after_worktree_cleanup": "passed",
                    "import_output": result.stdout.strip(),
                    "install_statuses": statuses,
                    "environment_guard": "clean",
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
