"""Central installs use the target project's interpreter, not Hermes' environment."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from roi_swarm.waves import run_install_command


def test_central_install_activates_project_venv(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(project / ".venv")], check=True
    )
    source = 'import json,os,sys; print(json.dumps([sys.prefix, os.getenv("VIRTUAL_ENV")]))'
    result = asyncio.run(run_install_command(f"python -c '{source}'", str(project)))
    assert result["exit_code"] == 0, result
    prefix, active = json.loads(result["output"].strip())
    assert Path(prefix).resolve() == (project / ".venv").resolve()
    assert Path(active).resolve() == (project / ".venv").resolve()
    from roi_swarm.waves import _install_env

    assert (
        Path(_install_env(str(project))["UV_PYTHON"]).resolve()
        == (
            project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        ).resolve()
    )


def test_npm_is_local_despite_inherited_and_project_global_config(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "package.json").write_text('{"name":"probe","version":"1.0.0"}')
    (project / ".npmrc").write_text(f"global=true\nprefix={tmp_path / 'foreign'}\n")
    monkeypatch.setenv("NPM_CONFIG_GLOBAL", "true")
    monkeypatch.setenv("npm_config_prefix", str(tmp_path / "foreign"))
    global_result = asyncio.run(run_install_command("npm config get global", str(project)))
    prefix_result = asyncio.run(run_install_command("npm prefix", str(project)))
    assert global_result["exit_code"] == 0, global_result
    # npm may warn that prefix is invalid in a project .npmrc; the effective
    # setting still must be local despite both hostile configuration sources.
    assert global_result["output"].strip().splitlines()[-1] == "false"
    assert prefix_result["exit_code"] == 0, prefix_result
    assert Path(prefix_result["output"].strip().splitlines()[-1]).resolve() == project.resolve()


def test_central_install_drops_inherited_foreign_venv(tmp_path, monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "foreign"))
    source = 'import os; print(os.getenv("VIRTUAL_ENV", "none"))'
    result = asyncio.run(run_install_command(f"'{sys.executable}' -c '{source}'", str(tmp_path)))
    assert result["exit_code"] == 0, result
    assert result["output"].strip() == "none"
    assert os.environ["VIRTUAL_ENV"] == str(tmp_path / "foreign")
