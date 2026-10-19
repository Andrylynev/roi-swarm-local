"""Installs never fall through from a missing project pip to Hermes/global pip."""

import os

import pytest

from roi_swarm.install_targets import target_error


def project_env(tmp_path):
    venv = tmp_path / ".venv"
    scripts = venv / ("Scripts" if os.name == "nt" else "bin")
    scripts.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = test")
    (scripts / ("python.exe" if os.name == "nt" else "python")).touch()
    return venv, scripts


def test_missing_project_pip_never_falls_back(tmp_path):
    venv, scripts = project_env(tmp_path)
    error = target_error("pip install x", str(venv), str(scripts))
    assert "uv pip" in error
    assert target_error("uv pip install x", str(venv), str(scripts)) is None


@pytest.mark.parametrize(
    "command",
    [
        "pip install x",
        "uv sync",
        "python -m pip install x",
        "poetry install",
    ],
)
def test_python_install_requires_existing_project_environment(command):
    assert "project" in target_error(command, None, "").lower()


def test_node_install_does_not_require_python_environment():
    assert target_error("npm ci", None, "") is None


@pytest.mark.parametrize(
    "command",
    [
        "uv pip install --system x",
        "pip install --target out x",
        "uv --project other sync",
        "py -3.11 -m pip install x",
        "UV_PROJECT_ENVIRONMENT=other uv sync",
        "npm install -g x",
        "uv tool install x",
        "uv tool --offline install x",
        "uv tool --allow-insecure-host localhost install x",
        "uv --allow-insecure-host localhost tool install x",
        "npm --global true install x",
        "uv tool --color always install x",
        "python -m pip --py D:/foreign/python.exe install x",
        "python -m pip --pyth=D:/foreign/python.exe install x",
        "conda install numpy",
        "mamba install numpy",
        "micromamba install numpy",
        "pip install --prefix=elsewhere x",
    ],
)
def test_install_overrides_are_refused(command, tmp_path):
    venv, scripts = project_env(tmp_path)
    assert target_error(command, str(venv), str(scripts))


@pytest.mark.parametrize(
    "command",
    [
        "env --chdir=elsewhere uv sync",
        "env -C elsewhere uv sync",
        "env -i pip install x",
    ],
)
def test_shell_wrapper_cannot_redirect_the_install(command, tmp_path):
    venv, scripts = project_env(tmp_path)
    assert target_error(command, str(venv), str(scripts))


@pytest.mark.parametrize("flag", ["-p {python}", "-p{python}", "-p={python}", "--python={python}"])
@pytest.mark.parametrize("operation", ["uv pip install", "uv sync"])
def test_uv_interpreter_aliases_cannot_select_a_foreign_environment(tmp_path, flag, operation):
    venv, scripts = project_env(tmp_path)
    foreign = tmp_path / "foreign-python.exe"
    command = operation + " " + flag.format(python=foreign)
    assert target_error(command, str(venv), str(scripts))


def test_uv_short_python_option_can_select_the_project_interpreter(tmp_path):
    venv, scripts = project_env(tmp_path)
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    for flag in (f'-p "{python}"', f'-p"{python}"', f'--python="{python}"'):
        assert target_error(f"uv pip install {flag} x", str(venv), str(scripts)) is None


@pytest.mark.parametrize(
    "command",
    [
        "uv pip install -tforeign x",
        "pip install -t=foreign x",
        "npm install -gforeign x",
        "env -Cforeign uv sync",
        "env -uVIRTUAL_ENV uv sync",
        "py -V:3.11 -m pip install x",
        "py -V:PythonCore/3.11 -m pip install x",
        "uv --offline tool install x",
        "uv --color always --offline tool install x",
        "npm install --location=global x",
        "npm --location global install x",
        "npm install --loc=global x",
        "uv pip install --prefixforeign x",
    ],
)
def test_target_aliases_cannot_escape_the_shared_environment(tmp_path, command):
    venv, scripts = project_env(tmp_path)
    assert target_error(command, str(venv), str(scripts))


def test_relative_project_python_resolves_from_shared_cwd(tmp_path):
    venv, scripts = project_env(tmp_path)
    binary = "python.exe" if os.name == "nt" else "python"
    command = f'".venv/{scripts.name}/{binary}" -m pip install x'
    assert target_error(command, str(venv), str(scripts), cwd=str(tmp_path)) is None


def test_explicit_foreign_python_is_refused(tmp_path):
    venv, scripts = project_env(tmp_path)
    foreign = tmp_path / "other" / "python.exe"
    foreign.parent.mkdir()
    foreign.touch()
    assert target_error(f'"{foreign}" -m pip install x', str(venv), str(scripts))
    assert (
        target_error(f'"{scripts / "python.exe"}" -m pip install x', str(venv), str(scripts))
        is None
    )
