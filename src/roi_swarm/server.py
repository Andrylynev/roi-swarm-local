from __future__ import annotations

import ipaddress
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from roi_swarm.security import provider_api_key

PROJECT_ROOT = Path(__file__).parents[2]

_server: uvicorn.Server | None = None
_restart = False
_supervised = False


def request_restart() -> bool:
    """Stop the running server so the startup loop relaunches it with new code."""
    global _restart
    # Without the startup supervisor nobody would relaunch us; keep serving.
    if _server is None or not _supervised:
        return False
    _restart = True
    _server.should_exit = True
    return True


def restart_requested() -> bool:
    return _restart


def bind_address() -> tuple[str, int]:
    host = os.getenv("ROI_HOST", "").strip() or "127.0.0.1"
    port = int(os.getenv("ROI_PORT", "").strip() or "8765")
    return host, port


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_exposure(host: str) -> None:
    """Refuse to publish the swarm (and the provider budget behind it) without a token."""
    if not _is_loopback(host) and not provider_api_key():
        raise SystemExit(
            f"ROI_HOST={host} is reachable from the network: set ROI_SWARM_API_KEY first"
        )


def main(supervised: bool | None = None) -> None:
    global _server, _restart, _supervised
    # Same .env the settings loader reads; real environment variables win.
    load_dotenv(PROJECT_ROOT / ".env")
    if supervised is None:
        # systemd (Restart=always) relaunches the process after a sleep-mode merge.
        supervised = os.getenv("ROI_SUPERVISED", "").strip() == "1"
    _restart = False
    _supervised = supervised
    host, port = bind_address()
    check_exposure(host)
    config = uvicorn.Config(
        "roi_swarm.api:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )
    _server = uvicorn.Server(config)
    try:
        _server.run()
    finally:
        _server = None
        _supervised = False


if __name__ == "__main__":
    main()
