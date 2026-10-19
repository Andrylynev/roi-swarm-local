"""Single-user pinned installation; no elevation, global Hermes or cloud credentials."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
HERMES_SHA = "498abb677ec39ea3ae9f8f5ed60e7def6bc47e70"
ALIASES = ("bonsai-2b", "bonsai-4b", "bonsai-9b", "whiterabbit-7b")


def hermes_paths(root: Path, system: str) -> tuple[Path, Path]:
    source = root / "runtime/hermes-agent"
    return source, source / (
        ".venv/Scripts/python.exe" if system == "Windows" else ".venv/bin/python"
    )


def validate_managed_paths(root: Path) -> Path:
    """Check all installer mutation destinations before downloading or writing.

    Interpreter executables may legitimately be symlinks; check their writable
    venv directories, never the interpreter file itself.
    """
    from local.download_assets import mutation_path

    root = mutation_path(root)
    for relative in (
        "runtime/prism",
        "runtime/hermes-agent/.git",
        "runtime/hermes-agent/.venv/Scripts",
        "runtime/hermes-agent/.venv/bin",
        "runtime/python",
        "runtime/bin",
        "hermes/config.yaml",
        "local/settings.yaml",
        "local/models.ini",
        "data/local-install.json",
        "data/local-services.json",
        "data/local-spawn-recovery.json",
        "cache/tmp",
        "cache/uv",
        "config",
        "download",
        "models",
        "tools/uv",
    ):
        mutation_path(root / relative)
    # Git/uv/imports can mutate nested metadata/cache paths, not just their
    # top-level directories. Walk without following links; interpreter symlinks
    # in an ordinary venv are read/executed, not installer mutation targets.
    for relative in ("runtime/hermes-agent/.git", "cache", "hermes"):
        tree = root / relative
        if not tree.exists():
            continue
        for directory, children, files in os.walk(tree, followlinks=False):
            for name in (*children, *files):
                mutation_path(Path(directory) / name)
    return root.resolve()


def atomic_write(path: Path, content: bytes) -> None:
    from local.download_assets import mutation_path

    path = mutation_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".setup-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def configure(root: Path, profile: str, *, system: str, backend: str = "cuda") -> None:
    """Only create missing config files, never replace a user's saved settings."""
    from local.manage import read_install
    from local.profiles import profile_values
    from roi_swarm.config import AppSettings

    root = validate_managed_paths(root)
    installed = read_install(root)
    api_url = f"http://127.0.0.1:{installed.get('api_port', 8775)}/v1"
    inference_url = f"http://127.0.0.1:{installed.get('inference_port', 8091)}/v1"
    source, python = hermes_paths(root, system)
    settings_path = root / "local/settings.yaml"
    if not settings_path.exists():
        settings = profile_values(
            yaml.safe_load((root / "local/settings.example.yaml").read_text(encoding="utf-8")),
            profile,
            backend=backend,
        )
        if backend == "cpu":
            settings["swarm"]["maximum_concurrency"] = 1
            settings["hermes_workers"]["maximum_concurrency"] = 1
        settings["provider"].update(
            base_url=inference_url, api_key="local-only", api_mode="chat_completions"
        )
        settings["hermes_workers"].update(
            root=source.as_posix(),
            python=python.as_posix(),
            default_cwd=root.as_posix(),
            worktree_root=(root / "data/worktrees").as_posix(),
        )
        settings.setdefault("context", {}).update(
            worker_context_tokens=65536, main_context_tokens=65536
        )
        settings.setdefault("sleep", {}).update(
            enabled=False, auto_improve=False, auto_restart=False
        )
        AppSettings.model_validate(settings)
        atomic_write(settings_path, yaml.safe_dump(settings, sort_keys=False).encode("utf-8"))
    else:
        AppSettings.model_validate(yaml.safe_load(settings_path.read_text(encoding="utf-8")))
    config_path = root / "hermes/config.yaml"
    if config_path.exists():
        if not isinstance(yaml.safe_load(config_path.read_text(encoding="utf-8")), dict):
            raise ValueError(
                "Existing Hermes config must be a YAML mapping; it was not overwritten"
            )
        return
    # Fresh isolated profile only. Keys verified against pinned config_providers.py,
    # config_defaults.py, agent_init.py and cli-config.yaml.example. No live profile edit.
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    providers = [
        {
            "name": "roi-local",
            "base_url": api_url,
            "api_key": "local-only",
            "api_mode": "chat_completions",
            "context_length": 65536,
            "models": {"roi-swarm": {"context_length": 65536}},
            "extra_body": extra,
        },
        {
            "name": "prism-local",
            "base_url": inference_url,
            "api_key": "local-only",
            "api_mode": "chat_completions",
            "context_length": 65536,
            "models": {name: {"context_length": 65536} for name in ALIASES},
            "extra_body": extra,
        },
    ]
    config = {
        "model": {
            "default": "roi-swarm",
            "provider": "custom:roi-local",
            "base_url": api_url,
            "context_length": 65536,
            "streaming": True,
        },
        "custom_providers": providers,
        "agent": {"reasoning_effort": "none"},
        "streaming": {"enabled": True},
        "terminal": {"backend": "local", "cwd": root.as_posix()},
        "compression": {"enabled": True, "threshold": 0.75, "target_ratio": 0.2},
        "auxiliary": {
            "compression": {
                "provider": "custom:prism-local",
                "model": "bonsai-2b",
                "base_url": inference_url,
                "context_length": 65536,
                "reasoning_effort": "none",
                "extra_body": extra,
            },
            "background_review": {"enabled": False},
            "title_generation": {"enabled": False},
        },
        "curator": {"enabled": False},
        "skills": {"external_dirs": []},
    }
    atomic_write(config_path, yaml.safe_dump(config, sort_keys=False).encode("utf-8"))


def ensure_model_presets(root: Path, backend: str, aliases: list[str]) -> None:
    """Append missing managed sections without reserializing any saved bytes."""
    import re

    from local.download_assets import mutation_path
    from local.presets import model_presets

    path = mutation_path(root / "local/models.ini")
    try:
        original = path.read_bytes()
    except FileNotFoundError:
        if not aliases:
            atomic_write(path, b"version = 1\n")
            return
        original = b"version = 1\n"
    present = {name.strip() for name in re.findall(rb"(?m)^[ \t]*\[([^\]\r\n]+)\]", original)}
    missing = [name for name in aliases if name.encode("ascii") not in present]
    if not missing:
        return
    lines = []
    for name, options in model_presets(root, backend=backend, aliases=missing).items():
        lines.append(f"[{name}]")
        lines.extend(f"{key} = {value}" for key, value in options.items())
        lines.append("")
    atomic_write(path, original + b"\n\n" + "\n".join(lines).encode("utf-8"))


def preflight_space(
    root: Path, assets: list[dict], *, prepare_only: bool = False, disk_usage=None
) -> int:
    """Conservative peak budget: verified copies, staged extraction, venv and headroom."""
    import shutil

    from local.download_assets import safe_target, verified

    missing = sum(a["bytes"] for a in assets if not verified(safe_target(root, a["path"]), a))
    runtime_peak = sum(a["bytes"] * 6 for a in assets if a["kind"] == "runtime")
    hermes_budget = 0 if prepare_only else 2 * 1024**3
    required = missing + runtime_peak + hermes_budget + 256 * 1024**2
    existing = root
    while not existing.exists():
        existing = existing.parent
    free = (disk_usage or shutil.disk_usage)(existing).free
    if free < required:
        raise RuntimeError(
            f"Insufficient disk space: need {required} free bytes "
            f"(including staging/headroom), have {free}"
        )
    return required


def install(
    *,
    root: Path = ROOT,
    backend: str = "auto",
    profile: str = "balanced",
    with_security: bool = False,
    accept_whiterabbitneo_terms: bool = False,
    no_start: bool = False,
    prepare_only: bool = False,
    asset_cache: Path | None = None,
    assets: list[dict] | None = None,
    system: str | None = None,
    machine: str | None = None,
    run=subprocess.run,
    opener=None,
    starter=None,
    hermes_installer=None,
) -> dict:
    """Dependency-injected fixture API; the public CLI has no URL/manifest override."""
    import platform

    from local import download_assets as downloads
    from local import manage
    from local.presets import FILES

    if with_security and not accept_whiterabbitneo_terms:
        raise ValueError(
            "--with-security requires --accept-whiterabbitneo-terms after reviewing "
            "https://huggingface.co/WhiteRabbitNeo/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B/"
            "blob/4e7be6e25ccb4824f1dcce51c967e3e4e6e8d4e0/README.md ; "
            "acceptance is not a legal guarantee"
        )
    root = validate_managed_paths(root)
    system = system or platform.system()
    machine = machine or platform.machine()
    if system not in {"Windows", "Linux"} or machine.lower() not in {"amd64", "x86_64", "x64"}:
        raise RuntimeError("Pinned Prism binaries support Windows/Linux x86_64 only")
    if profile not in {"balanced", "fast"}:
        raise ValueError("Profile must be balanced or fast")
    from local.profiles import _require_stopped

    # Share lifecycle exclusion even for preparation/no-start public entrypoints.
    # Release reservations before start(), which acquires the same lock itself.
    with manage.lock_for_root(root), _require_stopped(root):
        validate_managed_paths(root)
        recovery = root / "data/local-spawn-recovery.json"
        try:
            recovery.stat()
        except FileNotFoundError:
            pass
        else:
            raise manage.LifecycleError(f"Unconfirmed spawn cleanup; inspect {recovery}")
        # A repeat invocation preserves the backend/profile represented by saved settings.
        previous = manage.read_install(root)
        notices = []
        if previous:
            if backend != "auto" and backend != previous["backend"]:
                raise RuntimeError(
                    "Backend change requires a separate install root; existing runtime preserved"
                )
            if profile != previous["profile"]:
                notices.append(
                    f"Requested profile {profile!r} was not applied; preserved saved profile "
                    f"{previous['profile']!r}. Use roi-local profile after setup to change it."
                )
            backend, profile = previous["backend"], previous["profile"]
        selected = select_backend(backend, run=run)
        planned = (
            assets
            if assets is not None
            else downloads.select_assets(system, selected, with_security=with_security)
        )
        preflight_space(root, planned, prepare_only=prepare_only)
        for asset in planned:
            downloads.fetch(asset, root=root, asset_cache=asset_cache, opener=opener)
        archives = [root / a["path"] for a in planned if a["kind"] == "runtime"]
        runtime = root / "runtime/prism"
        source, python = hermes_paths(root, system)
        receipt = {
            "version": 1,
            "root": str(root),
            "backend": selected,
            "profile": profile,
            "api_port": previous.get("api_port", 8775),
            "inference_port": previous.get("inference_port", 8091),
            "lock_port": previous.get("lock_port", 8776),
            "hermes_root": str(source),
            "hermes_python": str(python),
            "hermes_commit": HERMES_SHA,
            "hermes_verified": not prepare_only,
            "prism_release": "prism-b10777-e3de715",
            "state": "prepared" if prepare_only else "validated",
        }
        if notices:
            receipt["notices"] = notices
        files = [
            root / relative
            for relative in (
                "local/settings.yaml",
                "hermes/config.yaml",
                "local/models.ini",
                "data/local-install.json",
            )
        ]
        before = {path: path.read_bytes() if path.exists() else None for path in files}

        def prepare(stage):
            probe_runtime(stage, selected, run=run)
            configure(root, profile, system=system, backend=selected)
            if not prepare_only:
                (hermes_installer or install_hermes)(root, system, run=run)
            aliases = [
                alias for alias, filename in FILES.items() if (root / "models" / filename).is_file()
            ]
            ensure_model_presets(root, selected, aliases)
            atomic_write(
                root / "data/local-install.json",
                (json.dumps(receipt, indent=2) + "\n").encode("utf-8"),
            )

        try:
            downloads.extract_runtime(archives, runtime, validate=prepare)
        except BaseException:
            validate_managed_paths(root)
            for path, content in before.items():
                if content is None:
                    path.unlink(missing_ok=True)
                elif path.read_bytes() != content:
                    atomic_write(path, content)
            raise
    if not (no_start or prepare_only):
        (starter or manage.start)(root)
        receipt["state"] = "running"
    return receipt


def install_hermes(root: Path, system: str, *, run=subprocess.run) -> dict:
    """Install only our pinned checkout/venv; refuse to replace an existing checkout."""
    import shutil

    from local.manage import child_environment

    root = validate_managed_paths(root)
    source, python = hermes_paths(root, system)
    env = child_environment(root)
    # uv must never discover the app/global environment as its project target.
    env.pop("VIRTUAL_ENV", None)
    env["UV_PROJECT_ENVIRONMENT"] = str(source / ".venv")
    env["UV_PYTHON_INSTALL_DIR"] = str(root / "runtime/python")
    env["UV_PYTHON_BIN_DIR"] = str(root / "runtime/bin")
    for directory in (root / "cache/tmp", root / "hermes", root / "runtime"):
        directory.mkdir(parents=True, exist_ok=True)
    created = not source.exists()

    def command(args, timeout=300):
        result = run(
            args, cwd=root, env=env, capture_output=True, text=True, timeout=timeout, check=False
        )
        if result.returncode:
            raise RuntimeError(
                f"Installer subprocess failed ({args[0]}): "
                f"{(result.stderr or result.stdout or '')[-4000:]}"
            )
        return result.stdout.strip()

    try:
        if created:
            command(["git", "init", str(source)])
            command(
                [
                    "git",
                    "-C",
                    str(source),
                    "-c",
                    "http.sslVerify=true",
                    "fetch",
                    "--depth",
                    "1",
                    "https://github.com/NousResearch/hermes-agent.git",
                    HERMES_SHA,
                ]
            )
            command(
                [
                    "git",
                    "-C",
                    str(source),
                    "-c",
                    "core.hooksPath=",
                    "checkout",
                    "--detach",
                    "FETCH_HEAD",
                ]
            )
        if command(["git", "-C", str(source), "rev-parse", "HEAD"]) != HERMES_SHA:
            raise RuntimeError(
                "Existing isolated Hermes is not the pinned commit; refusing to overwrite it"
            )
        if command(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"]):
            raise RuntimeError(
                "Existing isolated Hermes has edited tracked files; refusing to overwrite it"
            )
        if not python.is_file():
            uv_path = root / "tools/uv" / ("uv.exe" if system == "Windows" else "uv")
            uv = str(uv_path) if uv_path.is_file() else "uv"
            command(
                [uv, "sync", "--project", str(source), "--frozen", "--no-dev", "--python", "3.11"],
                timeout=1800,
            )
        if not python.is_file():
            raise RuntimeError("uv did not create the isolated Hermes Python 3.11 environment")
        probe_hermes(root, system, run=run, env=env)
        return {"commit": HERMES_SHA, "root": str(source), "python": str(python), "verified": True}
    except BaseException:
        validate_managed_paths(root)
        if created and source.exists():
            shutil.rmtree(source)
        raise


def probe_hermes(root: Path, system: str, *, run=subprocess.run, env=None) -> None:
    """Import the installed AIAgent and supported guard seams, without inference."""
    from local.manage import child_environment, read_install

    installed = read_install(root)
    inference_url = f"http://127.0.0.1:{installed.get('inference_port', 8091)}/v1"
    source, python = hermes_paths(root, system)
    script = (
        "import sys,inspect; assert sys.version_info[:2] == (3,11), 'Hermes requires Python 3.11'; "
        "from run_agent import AIAgent; "
        "assert callable(AIAgent._create_request_openai_client); "
        "assert callable(AIAgent._ensure_primary_openai_client); "
        "assert 'reason' in inspect.signature(AIAgent._create_request_openai_client).parameters; "
        "assert 'reason' in inspect.signature(AIAgent._ensure_primary_openai_client).parameters; "
        f"sys.path.insert(0, {str(root / 'src')!r}); "
        "from roi_swarm.native_budget import guard_worker_clients; "
        "assert callable(guard_worker_clients); "
        "from hermes_cli.config import load_config,get_custom_provider_context_length; "
        "c=load_config(); "
        "assert all(get_custom_provider_context_length(model=m,"
        f"base_url={inference_url!r},config=c)==65536 "
        f"for m in {ALIASES!r}); "
        "print('AIAgent native guard seams and alias contexts verified')"
    )
    result = run(
        [str(python), "-c", script],
        cwd=source,
        env=env or child_environment(root),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "Isolated Hermes import/native guard/context validation failed: "
            + (result.stderr or result.stdout or "")[-4000:]
        )


def select_backend(requested: str, *, run=subprocess.run) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Unknown backend: " + requested)
    if requested == "cpu":
        return "cpu"
    try:
        result = run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        available = result.returncode == 0 and bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        available = False
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA requested but nvidia-smi cannot find a working NVIDIA driver/GPU")
    return "cuda" if available else "cpu"


def probe_runtime(directory: Path, backend: str, *, run=subprocess.run) -> dict:
    servers = sorted(
        p
        for p in Path(directory).rglob("*")
        if p.name in {"llama-server", "llama-server.exe"} and p.is_file()
    )
    if len(servers) != 1:
        raise RuntimeError("Expected exactly one pinned llama-server executable")
    output = {}
    for flag in ("--version", "--list-devices"):
        result = run(
            [str(servers[0]), flag],
            cwd=servers[0].parent,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        text = (result.stdout or "") + (result.stderr or "")
        if result.returncode:
            raise RuntimeError(f"Prism runtime probe {flag} failed: {text[-3000:]}")
        output[flag] = text
    cuda = bool(
        __import__("re").search(r"(?im)^\s*(?:\d+:\s*)?CUDA\d+\s*:", output["--list-devices"])
    )
    if backend == "cuda" and not cuda:
        raise RuntimeError(
            "Prism --list-devices did not report a usable CUDA device; refusing fallback"
        )
    return {
        "executable": str(servers[0]),
        "cuda": cuda,
        "version": output["--version"].strip(),
        "devices": output["--list-devices"].strip(),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--profile", choices=("balanced", "fast"), default="balanced")
    parser.add_argument(
        "--with-security",
        action="store_true",
        help="Optional restricted-license WhiteRabbitNeo text-only reviewer",
    )
    parser.add_argument(
        "--accept-whiterabbitneo-terms",
        action="store_true",
        help="Confirm you reviewed the optional model's publisher terms",
    )
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="Install and validate everything without starting services",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="CI preparation only: verify assets/runtime/config; skip Hermes, NEVER start services",
    )
    parser.add_argument(
        "--asset-cache",
        type=Path,
        help="Read-only source of files; every copy is SHA256/size verified",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never prompt (all setup operations already use explicit options)",
    )
    args = parser.parse_args(argv)
    options = vars(args)
    options.pop("non_interactive")
    try:
        result = install(**options)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
