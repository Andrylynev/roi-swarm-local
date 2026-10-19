"""Hidden, single-instance logon entry point (Windows); credentials come from .env."""

from __future__ import annotations

import contextlib
import os
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).parents[2]


def _acquire_single_instance():
    # Keep the guard bound for the entire service lifetime. Windows' exclusive
    # bind prevents concurrent logon/manual invocations, including startup races.
    guard = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        guard.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        guard.bind(("127.0.0.1", 8766))
        guard.listen(1)
    except OSError:
        guard.close()
        return None
    return guard


def _service_running() -> bool:
    try:
        with urlopen("http://127.0.0.1:8765/health", timeout=2) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def _run_server() -> bool:
    """Serve until stopped; True when sleep mode asked for a code-reload restart."""
    from roi_swarm import server

    server.main(supervised=True)
    return server.restart_requested()


def _relaunch() -> None:
    # A fresh interpreter is the only way to load merged code; reuse this one's path.
    import subprocess

    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(
        [sys.executable, "-m", "roi_swarm.startup"],
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
        close_fds=True,
    )


def main() -> int:
    guard = _acquire_single_instance()
    if guard is None:
        return 0
    restart = False
    try:
        log = PROJECT_ROOT / "data" / "diagnostics" / "autostart.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with (
            log.open("a", encoding="utf-8", buffering=1) as stream,
            contextlib.redirect_stdout(stream),
            contextlib.redirect_stderr(stream),
        ):
            print(f"{datetime.now(UTC).isoformat()} ROI startup pid={os.getpid()}", flush=True)
            if _service_running():
                print("Existing local service is healthy; no duplicate started.", flush=True)
                return 0
            try:
                restart = _run_server() is True
            except Exception as exc:
                print(f"Startup failed: {type(exc).__name__}", flush=True)
                return 1
            if restart:
                print("Restarting to load merged sleep-mode improvements.", flush=True)
        return 0
    finally:
        # Release the single-instance guard before the successor tries to bind it.
        guard.close()
        if restart:
            try:
                _relaunch()
            except Exception as exc:
                with contextlib.suppress(OSError):
                    log = PROJECT_ROOT / "data" / "diagnostics" / "autostart.log"
                    with log.open("a", encoding="utf-8") as stream:
                        stream.write(f"Relaunch failed: {type(exc).__name__}: {exc}\n")
                return 1


if __name__ == "__main__":
    sys.exit(main())
