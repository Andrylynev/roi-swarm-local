"""Actual bootstrap contracts, with no model downloads or system changes."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHELLS = ["powershell"] if os.name == "nt" else ["bash"]


@pytest.mark.parametrize("shell", SHELLS)
def test_bootstrap_help_is_local_and_has_no_cloud_credentials(shell):
    executable = shutil.which(shell) or shutil.which("pwsh")
    if not executable:
        pytest.skip(f"{shell} unavailable")
    script = ROOT / ("install.ps1" if shell == "powershell" else "install.sh")
    command = [executable]
    if shell == "powershell":
        command += ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]
    result = subprocess.run(
        [*command, str(script), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for flag in (
        "--dir",
        "--backend",
        "--profile",
        "--with-security",
        "--accept-whiterabbitneo-terms",
        "--no-start",
        "--non-interactive",
    ):
        assert flag in result.stdout
    assert "--provider-key" not in result.stdout
    assert "--api-key" not in result.stdout
    assert "roi-swarm-local" in result.stdout


def smoke_module():
    path = ROOT / "scripts/ci_local_installer_smoke.py"
    assert path.is_file(), "actual offline bootstrap smoke is missing"
    spec = importlib.util.spec_from_file_location("bootstrap_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("shell", SHELLS)
def test_actual_bootstrap_install_repeat_and_safe_refusals(tmp_path, shell):
    if sys.version_info[:2] != (3, 11):
        pytest.skip("bootstrap pins 3.11; the dedicated CI fixture jobs exercise it")
    if os.name == "nt":
        import ctypes

        if ctypes.windll.shell32.IsUserAnAdmin():
            pytest.skip("bootstrap requires non-admin; CI has a separate standard-user smoke")
    smoke_module().fixture_smoke(ROOT, tmp_path, shell)


def test_real_cpu_smoke_refuses_non_linux_hosts(tmp_path):
    if os.name != "nt":
        pytest.skip("negative platform contract is for native Windows")
    with pytest.raises(RuntimeError, match="Linux x86_64"):
        smoke_module().cpu_runtime_smoke(ROOT, tmp_path)
