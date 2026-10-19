#!/usr/bin/env python3
"""Real root/systemd installer acceptance test for a disposable Ubuntu CI runner.

The only OpenAI-compatible provider is an explicitly fake, deterministic HTTP
fixture bound to loopback. It serves a model catalog, not production model output;
no inference or paid API is called. Python/package downloads are still real.
Use --self-test to exercise just this fixture safely on any supported Python.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import stat
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

INSTALL = Path("/opt/roi-swarm-ci")
UNIT = Path("/etc/systemd/system/roi-swarm.service")
SERVICE = "roi-swarm"
CATALOG = ["fixture-main-v1", "fixture-worker-v1", "fixture-main-v2", "fixture-worker-v2"]
PROVIDER_KEY = 'fixture-provider-key with # apostrophe\' and "quotes" $literal\\path'
NEW_PROVIDER_KEY = "fixture-provider-key-v2"
TOKEN = 'roi-ci-fixture-token with # apostrophe\' and "quotes" $literal\\path'
NEW_TOKEN = "roi-ci-fixture-token-v2"
CUSTOM_VALUE = 'preserve # apostrophe\' and "quotes" $literal\\path'
HTTP = build_opener(ProxyHandler({}))


class ProviderFixture:
    """A catalog-only fake OpenAI server; every request is recorded for assertions."""

    def __init__(self) -> None:
        fixture = self
        self.seen: list[tuple[str, str, int]] = []
        self.lock = threading.Lock()
        self.keys = {
            "/v1/models": PROVIDER_KEY,
            "/reconfigured/v1/models": NEW_PROVIDER_KEY,
        }

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                expected = fixture.keys.get(self.path)
                if expected is None:
                    self.respond(404, {"error": "Unknown fixture path"})
                elif self.headers.get("Authorization") != f"Bearer {expected}":
                    self.respond(401, {"error": "Invalid fixture key"})
                else:
                    self.respond(
                        200,
                        {
                            "object": "list",
                            "data": [
                                {
                                    "id": model,
                                    "object": "model",
                                    "created": 0,
                                    "owned_by": "local-ci-fixture-not-a-real-provider",
                                }
                                for model in CATALOG
                            ],
                        },
                    )

            def do_POST(self) -> None:
                self.respond(405, {"error": "Fixture does not implement inference"})

            def respond(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode("utf-8")
                with fixture.lock:
                    fixture.seen.append((self.command, self.path, status))
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self) -> ProviderFixture:
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        assert not self.thread.is_alive(), "Fixture server failed to stop"

    def successful_probes(self, path: str) -> int:
        with self.lock:
            return self.seen.count(("GET", path, 200))


def request(
    url: str, token: str | None = None, *, expected: int = 200, method: str = "GET"
) -> bytes:
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    req = Request(url, headers=headers, method=method)
    try:
        response = HTTP.open(req, timeout=5)
    except HTTPError as exc:
        response = exc
    with response:
        body = response.read()
        assert response.code == expected, (
            f"{method} {url}: expected HTTP {expected}, got {response.code}: {body[:300]!r}"
        )
        return body


def self_test() -> None:
    """Exercise actual HTTP, including negative auth and unsupported inference."""
    with ProviderFixture() as provider:
        for path, key in provider.keys.items():
            request(provider.base + path, expected=401)
            request(provider.base + path, "wrong-fixture-key", expected=401)
            catalog = json.loads(request(provider.base + path, key))
            assert [model["id"] for model in catalog["data"]] == CATALOG
            assert provider.successful_probes(path) == 1
        request(provider.base + "/missing", expected=404)
        request(provider.base + "/v1/chat/completions", method="POST", expected=405)
        assert len(provider.seen) == 8
    print("PASS: deterministic loopback provider fixture (no paid API, no inference)")


def run(
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
    echo: bool = False,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if echo or (check and result.returncode):
        print(result.stdout, end="", flush=True)
    if check and result.returncode:
        raise RuntimeError(f"{argv[0]} failed with exit status {result.returncode}")
    return result


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def roles(main: str, worker: str) -> dict[str, str]:
    return {
        "planner": main,
        "synthesizer": main,
        "router": worker,
        "worker": worker,
        "critic": worker,
    }


def verify_install(
    *,
    provider: ProviderFixture,
    provider_path: str,
    provider_key: str,
    token: str,
    port: int,
    expected_roles: dict[str, str],
    custom: bool = False,
) -> None:
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 60
    while True:
        try:
            health = json.loads(request(base + "/health"))
            assert health == {"status": "ok"}, health
            break
        except (OSError, URLError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.5)

    models = json.loads(request(base + "/v1/models", token))
    assert [model["id"] for model in models["data"]] == ["roi-swarm"], models
    request(base + "/v1/models", expected=401)
    request(base + "/v1/models", "incorrect-fixture-token", expected=401)
    dashboard = request(base + "/dashboard")
    assert b"<html" in dashboard.lower(), "Dashboard response is not HTML"
    actual_roles = json.loads(request(base + "/api/model-settings", token))["roles"]
    assert actual_roles == expected_roles, actual_roles
    saved = json.loads((INSTALL / "data/model-settings.json").read_text(encoding="utf-8"))
    assert saved["roles"] == expected_roles, saved
    sleep = json.loads(request(base + "/api/sleep", token))
    assert sleep["enabled"] is True, sleep
    assert sleep["auto_improve"] is False, sleep
    assert sleep["daily_calls"] == 0, "Fresh idle installation must not infer without a source run"

    before = provider.successful_probes(provider_path + "/models")
    options = json.loads(request(base + "/api/model-options", token))
    assert options["source"] == "provider", options
    assert set(options["models"]) == set(CATALOG), options
    assert provider.successful_probes(provider_path + "/models") > before

    properties = run(
        ["systemctl", "show", SERVICE, "-p", "User", "-p", "MainPID", "-p", "ActiveState"]
    ).stdout
    unit = dict(line.split("=", 1) for line in properties.splitlines())
    assert unit["User"] == SERVICE and unit["ActiveState"] == "active", unit
    assert run(["systemctl", "is-enabled", SERVICE]).stdout.strip() == "enabled"
    pid = int(unit["MainPID"])
    assert pid > 0, unit
    uid = int(run(["id", "-u", SERVICE]).stdout)
    assert uid != 0, "Service account is root"
    status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    process_uids = next(line for line in status.splitlines() if line.startswith("Uid:"))
    assert all(int(value) == uid for value in process_uids.split()[1:]), process_uids
    executable = Path(f"/proc/{pid}/exe").resolve()
    assert executable.is_relative_to(INSTALL / ".python"), executable
    python = INSTALL / ".venv/bin/python"
    assert python.resolve() == executable, (python.resolve(), executable)

    # Run the installed interpreter as the actual non-root service user, not root.
    probe = run(
        [
            "runuser",
            "-u",
            SERVICE,
            "--",
            str(python),
            "-c",
            "import importlib.metadata as m, importlib.util as u, json, os, sys; "
            "names = ['numpy', 'onnx', 'onnxruntime', 'tokenizers', 'huggingface_hub']; "
            "print(json.dumps({'uid': os.geteuid(), 'base': sys.base_prefix, "
            "'installed': [d.metadata['Name'] for d in m.distributions()], "
            "'importable': {n: u.find_spec(n) is not None for n in names}}))",
        ]
    )
    runtime = json.loads(probe.stdout)
    assert runtime["uid"] == uid, runtime
    assert Path(runtime["base"]).is_relative_to(INSTALL / ".python"), runtime["base"]
    installed = {name.lower().replace("_", "-") for name in runtime["installed"]}
    heavy = {"numpy", "onnx", "tokenizers", "huggingface-hub"} & installed
    heavy.update(name for name in installed if name.startswith("onnx"))
    assert not heavy, f"Unexpected local model dependencies: {sorted(heavy)}"
    assert not any(runtime["importable"].values()), runtime["importable"]

    expected_env = {
        "ROI_PROVIDER_BASE_URL": provider.base + provider_path,
        "ROI_PROVIDER_API_KEY": provider_key,
        "ROI_SWARM_API_KEY": token,
        "ROI_HOST": "127.0.0.1",
        "ROI_PORT": str(port),
        "ROI_SLEEP_ENABLED": "true",
        "ROI_SLEEP_AUTO_IMPROVE": "false",
    }
    if custom:
        expected_env["ROI_CI_CUSTOM_SETTING"] = CUSTOM_VALUE
    env_file = INSTALL / ".env"
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600, ".env must be private (0600)"
    assert env_file.stat().st_uid == uid, ".env must be readable by the service account"
    values = json.loads(
        run(
            [
                str(python),
                "-c",
                "import json, sys; from dotenv import dotenv_values; "
                "print(json.dumps(dotenv_values(sys.argv[1], interpolate=False)))",
                str(env_file),
            ]
        ).stdout
    )
    # /proc/<pid>/environ exposes exec-time values, not load_dotenv's later setenv.
    # Read the same .env as the actual service user; HTTP assertions above verify
    # that the running process uses its provider, token and model settings.
    loaded = json.loads(
        run(
            [
                "runuser",
                "-u",
                SERVICE,
                "--",
                str(python),
                "-c",
                "import json, os, sys; from dotenv import load_dotenv; "
                "load_dotenv(sys.argv[1]); "
                "print(json.dumps({k:v for k,v in os.environ.items() if k.startswith('ROI_')}))",
                str(env_file),
            ]
        ).stdout
    )
    for name, value in expected_env.items():
        assert values.get(name) == value, f"Incorrect saved {name}"
        assert loaded.get(name) == value, f"Incorrect loaded environment {name}"
    print("PASS: real HTTP/auth/dashboard, roles, private env, non-root systemd, cloud-only deps")


def verify_uninstall(port: int) -> None:
    assert not INSTALL.exists(), "Uninstall left the installation directory"
    assert not INSTALL.with_name(INSTALL.name + "-worktrees").exists(), "Worktrees remain"
    assert not UNIT.exists(), "Uninstall left the systemd unit"
    assert run(["systemctl", "is-active", SERVICE], check=False).returncode != 0
    assert run(["systemctl", "is-enabled", SERVICE], check=False).returncode != 0
    assert run(["id", "-u", SERVICE], check=False).returncode != 0, "Service account remains"
    try:
        request(f"http://127.0.0.1:{port}/health")
    except (OSError, URLError):
        pass
    else:
        raise AssertionError("Uninstalled service still answers HTTP")
    print("PASS: uninstall removed service, account, installation and worktrees")


def installer_smoke(repo: Path) -> None:
    # Refuse all non-disposable environments before running destructive commands.
    assert sys.platform == "linux", "This acceptance test requires Linux/systemd"
    assert os.geteuid() == 0, "Run this acceptance test through sudo"
    assert os.environ.get("GITHUB_ACTIONS") == "true", "Only run on a disposable CI runner"
    assert Path("/run/systemd/system").is_dir(), "A real systemd instance is required"
    assert not INSTALL.exists() and not INSTALL.is_symlink(), "Installation already exists"
    assert not INSTALL.with_name(INSTALL.name + "-worktrees").exists(), "Worktrees already exist"
    assert not UNIT.exists(), "An existing ROI Swarm service must not be touched"
    assert run(["id", "-u", SERVICE], check=False).returncode != 0, "Service user already exists"
    # The service user must clone without traversing the runner's private HOME.
    repo = repo.resolve(strict=True)
    with TemporaryDirectory(prefix="roi-ci-source-", dir="/tmp") as temporary:
        source = Path(temporary) / "source"
        run(["git", "clone", "--no-hardlinks", "--branch", "main", str(repo), str(source)])
        os.chmod(temporary, 0o755)
        os.chmod(source, 0o755)
        for folder in (source, *[p for p in source.rglob("*") if p.is_dir()]):
            folder.chmod(folder.stat().st_mode | 0o055)
        # File protocol clone must regard this read-only root-owned source as safe
        # for the unprivileged user. Only the isolated fixture gets this setting.
        run(["git", "config", "--system", "--add", "safe.directory", str(source / ".git")])
        try:
            _installer_lifecycle(source)
        finally:
            run(
                [
                    "git",
                    "config",
                    "--system",
                    "--fixed-value",
                    "--unset-all",
                    "safe.directory",
                    str(source / ".git"),
                ],
                check=False,
            )


def _installer_lifecycle(repo: Path) -> None:
    installer = repo / "scripts" / "cloud-install.sh"
    assert installer.is_file(), installer
    head = run(["git", "-C", str(repo), "rev-parse", "HEAD"]).stdout.strip()
    main_head = run(["git", "-C", str(repo), "rev-parse", "refs/heads/main"]).stdout.strip()
    assert head == main_head, "Local main must point at the exact checkout under test"

    # Ignore developer/runner credentials and Python/uv overrides. Force a managed
    # Python so an accidentally root-private interpreter cannot pass this test.
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("ROI_", "UV_", "PYTHON", "OPENAI_", "ANTHROPIC_"))
        and name not in {"VIRTUAL_ENV", "CONDA_PREFIX"}
    }
    env.update({"UV_MANAGED_PYTHON": "1", "UV_LINK_MODE": "copy"})
    common = [
        "bash",
        str(installer),
        "--dir",
        str(INSTALL),
        "--repo",
        repo.as_uri(),
        "--branch",
        "main",
        "--non-interactive",
    ]
    uninstall = common + ["--uninstall"]
    cleaned = False
    port = free_port()
    print(
        "REAL installer acceptance test; upstream is LOCAL DETERMINISTIC FIXTURE ONLY", flush=True
    )
    with ProviderFixture() as provider:
        try:
            run(
                common
                + [
                    "--provider-url",
                    provider.base + "/v1",
                    "--provider-key",
                    PROVIDER_KEY,
                    "--main-model",
                    CATALOG[0],
                    "--worker-model",
                    CATALOG[1],
                    "--api-key",
                    TOKEN,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                env=env,
                echo=True,
                timeout=900,
            )
            assert provider.successful_probes("/v1/models") > 0, "Installer did not probe fixture"
            installed_head = run(
                ["runuser", "-u", SERVICE, "--", "git", "-C", str(INSTALL), "rev-parse", "HEAD"]
            ).stdout.strip()
            assert installed_head == head, "Installer did not clone the checkout under test"
            verify_install(
                provider=provider,
                provider_path="/v1",
                provider_key=PROVIDER_KEY,
                token=TOKEN,
                port=port,
                expected_roles=roles(CATALOG[0], CATALOG[1]),
            )

            with (INSTALL / ".env").open("a", encoding="utf-8") as stream:
                stream.write(f"ROI_CI_CUSTOM_SETTING={json.dumps(CUSTOM_VALUE)}\n")
            saved_roles = (INSTALL / "data/model-settings.json").read_bytes()
            before = provider.successful_probes("/v1/models")
            run(common, env=env, echo=True, timeout=900)
            assert provider.successful_probes("/v1/models") > before
            assert (INSTALL / "data/model-settings.json").read_bytes() == saved_roles
            verify_install(
                provider=provider,
                provider_path="/v1",
                provider_key=PROVIDER_KEY,
                token=TOKEN,
                port=port,
                expected_roles=roles(CATALOG[0], CATALOG[1]),
                custom=True,
            )
            print("PASS: repeat install preserves the custom .env key, credentials and model roles")

            run(
                common
                + [
                    "--reconfigure",
                    "--provider-url",
                    provider.base + "/reconfigured/v1",
                    "--provider-key",
                    NEW_PROVIDER_KEY,
                    "--main-model",
                    CATALOG[2],
                    "--worker-model",
                    CATALOG[3],
                    "--api-key",
                    NEW_TOKEN,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                env=env,
                echo=True,
                timeout=900,
            )
            assert provider.successful_probes("/reconfigured/v1/models") > 0
            verify_install(
                provider=provider,
                provider_path="/reconfigured/v1",
                provider_key=NEW_PROVIDER_KEY,
                token=NEW_TOKEN,
                port=port,
                expected_roles=roles(CATALOG[2], CATALOG[3]),
                custom=True,
            )
            request(f"http://127.0.0.1:{port}/v1/models", TOKEN, expected=401)
            assert all(method == "GET" and status == 200 for method, _, status in provider.seen)
            print("PASS: reconfiguration applied new provider/key/models/token; old token rejected")

            run(uninstall, env=env, echo=True, timeout=120)
            verify_uninstall(port)
            cleaned = True
        except BaseException:
            run(["systemctl", "status", SERVICE, "--no-pager", "-n", "40"], check=False, echo=True)
            run(["journalctl", "-u", SERVICE, "--no-pager", "-n", "100"], check=False, echo=True)
            raise
        finally:
            if not cleaned:
                run(uninstall, env=env, check=False, echo=True, timeout=120)
    print("PASS: full real Ubuntu installer lifecycle (no paid provider calls)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--self-test", action="store_true", help="Only test the loopback HTTP fixture"
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        installer_smoke(args.repo)


if __name__ == "__main__":
    main()
