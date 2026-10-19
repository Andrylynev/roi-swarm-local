"""Explicit local worker choices; model weights remain shared by logical agents."""

import argparse
import copy
import json
import os
import socket
import tempfile
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path

import yaml

from local import manage
from roi_swarm.config import AppSettings

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"fast": "bonsai-2b", "balanced": "bonsai-4b", "security": "bonsai-4b"}


def profile_values(source: dict, name: str, backend: str = "cuda") -> dict:
    if backend not in {"cpu", "cuda"}:
        raise ValueError("Unknown local backend: " + str(backend))
    if name not in MODELS:
        raise ValueError("Unknown local profile: " + name)
    if not isinstance(source, dict):
        raise ValueError("Local profile settings must be a mapping")
    for section in ("models", "swarm", "hermes_workers"):
        if not isinstance(source.get(section, {}), dict):
            raise ValueError(f"Local profile {section} must be a mapping")
    for role, value in source.get("models", {}).items():
        if not isinstance(value, dict):
            raise ValueError(f"Local profile models.{role} must be a mapping")
    data = copy.deepcopy(source)
    for role in ("router", "planner", "worker", "critic", "synthesizer"):
        data.setdefault("models", {}).setdefault(role, {})
        data["models"][role].update(
            {
                "model": "bonsai-2b"
                if role == "router"
                else (MODELS[name] if role == "worker" else "bonsai-9b"),
                "temperature": 0.7,
                "max_output_tokens": 4096,
            }
        )
    if name == "security":
        # WhiteRabbitNeo failed native tool-call smoke: use it only for textual
        # review, never as a Hermes tool worker. Direct consultation is also
        # available through the inference model ID whiterabbit-7b.
        data["models"]["critic"]["model"] = "whiterabbit-7b"
    concurrency = 1 if name == "security" or backend == "cpu" else 2
    data.setdefault("swarm", {})["maximum_concurrency"] = concurrency
    data.setdefault("hermes_workers", {})["maximum_concurrency"] = concurrency
    return data


@contextmanager
def _require_stopped(root: Path):
    """Require clean state and reserve listener-free ports throughout the write."""
    # Path.exists() suppresses access errors on Python 3.14. Only genuine
    # absence, never an unreadable recovery record, may imply a clean state.
    with suppress(FileNotFoundError):
        (root / "data/local-services.json").stat()
    state = manage.read_state(root)
    plan = manage.build_plan(root, state.get("models_max", 1))
    with ExitStack() as reserved:
        for name, service in plan["services"].items():
            if name in state["services"]:
                detail = manage.ownership(state["services"][name], service)["reason"]
                # Even an exited saved root may have surviving launcher children.
                raise manage.LifecycleError(
                    f"Stop the local swarm before changing the profile; {name}: {detail}"
                )
            port = service["port"]
            if manage.listener_pid(port) is not None:
                raise manage.LifecycleError(
                    f"Port {port} has a listener; stop the local swarm before switching"
                )
            probe = reserved.enter_context(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
            try:
                if os.name == "nt":
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                probe.bind((manage.HOST, port))
            except OSError as exc:
                raise manage.LifecycleError(
                    f"Cannot prove port {port} is stopped and available; refusing profile change"
                ) from exc
        yield


def _write_unique(directory: Path, prefix: str, suffix: str, content: bytes) -> Path:
    """Durably stage bytes using exclusive creation, never replacing an old backup."""
    fd, filename = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=directory)
    path = Path(filename)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _replace_profile(path: Path, override: Path, original: bytes, updated: bytes) -> None:
    """Back up before mutation; undo a config replacement if clearing overrides fails."""
    backup = None
    temporary = restore = None
    replaced = keep_recovery = False
    try:
        try:
            override_bytes = override.read_bytes()
        except FileNotFoundError:
            override_bytes = None
        if override_bytes is not None:
            diagnostics = override.parent / "diagnostics"
            diagnostics.mkdir(parents=True, exist_ok=True)
            backup = _write_unique(
                diagnostics, "model-settings.before-profile-", ".json", override_bytes
            )
        temporary = _write_unique(path.parent, "settings.profile-", ".tmp", updated)
        restore = _write_unique(path.parent, "settings.before-profile-", ".tmp", original)
        temporary.replace(path)
        replaced = True
        if backup is not None:
            override.unlink()
    except BaseException as exc:
        if replaced:
            try:
                restore.replace(path)
            except OSError as rollback_error:
                keep_recovery = True
                raise RuntimeError(
                    f"Profile change failed and settings rollback failed ({rollback_error}); "
                    f"original settings retained at {restore}; override backup: {backup}"
                ) from exc
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if restore is not None and not keep_recovery:
            restore.unlink(missing_ok=True)


def apply_profile(name: str, root: Path | None = None) -> None:
    root = Path(ROOT if root is None else root).resolve()
    with manage.lock_for_root(root), _require_stopped(root):
        installed = manage.read_install(root)
        path = root / "local/settings.yaml"
        original = path.read_bytes()
        updated = profile_values(
            yaml.safe_load(original), name, backend=installed.get("backend", "cuda")
        )
        if installed:
            updated.setdefault("provider", {})["base_url"] = (
                f"http://{manage.HOST}:{installed['inference_port']}/v1"
            )
        AppSettings.model_validate(updated)
        _replace_profile(
            path,
            root / "data/model-settings.json",
            original,
            yaml.safe_dump(updated, sort_keys=False).encode("utf-8"),
        )
    print(json.dumps({"profile": name, "worker": MODELS[name], "restart_required": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=sorted(MODELS))
    apply_profile(parser.parse_args().profile)
