"""Keep a brokered package command scoped to the shared project's environment."""

import os
import re
import shutil
from pathlib import Path

from roi_swarm.worker_install_policy import _executable, _shell, _skip_options, _unwrap

_PYTHON_TOOLS = {"uv", "poetry", "pipenv", "conda", "mamba", "micromamba", "py"}
_OVERRIDES = {
    "--system",
    "--prefix",
    "--target",
    "--user",
    "--root",
    "--project",
    "--directory",
    "--cwd",
    "--dir",
    "--global",
    "-g",
    "-C",
    "--name",
    "-n",
    "-t",
    "--root-prefix",
    "--active",
    "--modules-folder",
}
_ENV_OVERRIDE = re.compile(
    r"(?:^|\s)(?:VIRTUAL_ENV|PYTHONHOME|PIP_\w+|UV_\w+|NPM_CONFIG_\w+|PATH|CONDA_PREFIX)=",
    re.IGNORECASE,
)


def target_error(
    command: str, venv: str | None, path: str, *, cwd: str | None = None
) -> str | None:
    parsed = _shell(command)
    if len(parsed.segments) != 1:
        return "Use one standalone installer command."
    literal = parsed.segments[0]
    words, _split = _unwrap(literal)
    wrappers = literal[: len(literal) - len(words)]
    if any(token.startswith("-") for token in wrappers):
        return "Do not redirect or clear the install environment through a shell wrapper."
    if not words:
        return "Missing installer command."
    executable, args = _executable(words[0]), words[1:]
    python_tool = executable in _PYTHON_TOOLS or bool(
        re.fullmatch(r"(?:python|pip)[\d.]*", executable)
    )
    if _ENV_OVERRIDE.search(command):
        return "Environment target overrides are not allowed in a worker install request."
    for token in args:
        option = token.partition("=")[0]
        # Short value options support attached values (-tpath, -Cpath, -g...).
        # Fail closed for abbreviated/extended target switches too; do not try to
        # emulate each package manager's prefix/alias resolution separately.
        forbidden = any(
            option.startswith(flag)
            if len(flag) == 2
            else (option.startswith(flag) or (len(option) >= 5 and flag.startswith(option)))
            for flag in _OVERRIDES
        )
        if executable in {"npm", "pnpm", "yarn", "bun"} and option.startswith("--loc"):
            forbidden = True
        if token.startswith("-") and forbidden:
            return (
                "Install into the shared project only; "
                "global/directory/target overrides are refused."
            )
    if executable == "py" and any(re.match(r"-\d|-V:|--?version", token) for token in args):
        return (
            "A versioned py launcher bypasses the project environment; use its python explicitly."
        )
    if executable in {"conda", "mamba", "micromamba"}:
        return "Project conda environments are not supported; use the project .venv with uv pip."
    if executable == "uv" and _skip_options(args, "uv")[:1] == ["tool"]:
        return "uv tool installs outside the project environment; use uv pip install instead."
    if not python_tool:
        return None
    if not venv:
        return (
            "No valid project .venv found; central coordinator must prepare it before installing."
        )
    root = Path(venv).resolve()
    if not (root / "pyvenv.cfg").is_file():
        return "The project .venv is invalid (missing pyvenv.cfg)."
    if executable.startswith(("python", "pip")) and ("/" in words[0] or "\\" in words[0]):
        interpreter = Path(words[0])
        if not interpreter.is_absolute():
            interpreter = Path(cwd or root.parent) / interpreter
        resolved = interpreter.resolve()
        if not resolved.is_relative_to(root):
            return "Explicit interpreter/pip is outside the project environment."
    for index, token in enumerate(args):
        value = None
        if token in {"--python", "-p"}:
            value = args[index + 1] if index + 1 < len(args) else ""
        elif (
            token.startswith("--")
            and len(token.partition("=")[0]) >= 4
            and (
                token.partition("=")[0] != "--python"
                and "--python".startswith(token.partition("=")[0])
            )
        ):
            return (
                "Use the full --python option naming the project interpreter; "
                "abbreviations are refused."
            )
        elif token.startswith("--python="):
            value = token.partition("=")[2]
        elif executable == "uv" and token.startswith("-p"):
            value = token[2:].removeprefix("=")
        if value is not None and (
            not Path(value).is_absolute() or not Path(value).resolve().is_relative_to(root)
        ):
            return "--python/-p must name the project environment's interpreter."
    if re.fullmatch(r"pip[\d.]*", executable):
        found = shutil.which(words[0], path=path)
        if not found or not Path(found).resolve().is_relative_to(root):
            python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            return f'Project pip is absent: use uv pip install --python "{python}" ... instead.'
    return None
