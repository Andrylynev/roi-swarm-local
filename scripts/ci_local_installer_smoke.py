"""Run real user bootstraps against an offline Git/stdlib-only app fixture.

This verifies bootstrap safety/lifecycle, NOT inference. No fake model responses.
The optional --cpu-runtime mode is a separate real Prism/2B inference check.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PUBLIC_REPO = "https://github.com/Andrylynev/roi-swarm-local.git"


def run(command, *, cwd=None, env=None, ok=True, timeout=180):
    result = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if ok and result.returncode:
        raise AssertionError(f"Command failed: {command}\n{result.stdout}\n{result.stderr}")
    return result


def fixture_smoke(checkout: Path, work: Path, shell: str) -> dict:
    """Real Git and uv, isolated source, HOME, checkout, cache and Python env."""
    uv = shutil.which("uv")
    if not uv:
        raise AssertionError("Fixture requires uv on PATH; it never bootstraps tools")
    if shell == "bash" and os.name != "nt" and os.geteuid() == 0:
        raise AssertionError("Run fixture as a normal user (bootstrap refuses root)")
    work.mkdir(parents=True, exist_ok=True)
    source, home, target = work / "fixture source", work / "home", work / "install with spaces"
    source.mkdir()
    home.mkdir()
    (source / "local").mkdir()
    (source / ".gitignore").write_text(
        ".venv/\n.env\ndata/\nsetup-events.jsonl\n", encoding="utf-8"
    )
    (source / "pyproject.toml").write_text(
        '[project]\nname="bootstrap-offline-fixture"\nversion="0.0.0"\n'
        'requires-python=">=3.11"\ndependencies=[]\n[tool.uv]\npackage=false\n',
        encoding="utf-8",
    )
    (source / "local/__init__.py").write_text("", encoding="utf-8")
    (source / "local/setup.py").write_text(
        "import json, pathlib, sys\n"
        "root = pathlib.Path.cwd()\n"
        'with (root / "setup-events.jsonl").open("a") as f:\n'
        '    f.write(json.dumps(["setup", *sys.argv[1:]]) + "\\n")\n'
        'if not (root / ".env").exists():\n'
        '    (root / ".env").write_text("LOCAL_SETTING=preserve-me\\n")\n'
        'print("fixture setup (no inference)")\n',
        encoding="utf-8",
    )
    (source / "local/manage.py").write_text(
        "import json, pathlib, sys\n"
        "root = pathlib.Path.cwd()\n"
        'with (root / "setup-events.jsonl").open("a") as f:\n'
        '    f.write(json.dumps(["manage", *sys.argv[1:]]) + "\\n")\n'
        'stopped = not (root / "data/refuse-stop").exists()\n'
        'print(json.dumps({"stopped": stopped}))\n'
        "sys.exit(0 if stopped else 1)\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "HOME": str(home),
        "UV_CACHE_DIR": str(work / "uv-cache"),
        "UV_PYTHON_DOWNLOADS": "never",
        "UV_OFFLINE": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": str(work / "gitconfig"),
        "GIT_TERMINAL_PROMPT": "0",
    }
    # Keep real Python discoverable: uv sync is explicitly required to select 3.11.
    run([uv, "lock", "--offline", "--python", sys.executable], cwd=source, env=env)
    run(["git", "init", "-q", "-b", "main", source], env=env)
    run(["git", "-C", source, "add", "."], env=env)
    run(
        [
            "git",
            "-C",
            source,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        env=env,
    )
    # Native Git URL rewrite avoids a production backdoor/--repo override.
    env.update(
        {
            "GIT_CONFIG_COUNT": "2",
            "GIT_CONFIG_KEY_0": f"url.{source.as_posix()}.insteadOf",
            "GIT_CONFIG_VALUE_0": PUBLIC_REPO,
            "GIT_CONFIG_KEY_1": "protocol.file.allow",
            "GIT_CONFIG_VALUE_1": "always",
        }
    )
    executable = shutil.which(shell) or shutil.which("pwsh")
    if not executable:
        raise AssertionError(f"{shell} is required")
    prefix = [executable]
    script = checkout / ("install.ps1" if shell == "powershell" else "install.sh")
    if shell == "powershell":
        prefix += ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]
    prefix += [str(script)]

    def install(path=target, extra=(), ok=True):
        return run(
            [
                *prefix,
                "--dir",
                path,
                "--backend",
                "cpu",
                "--profile",
                "fast",
                "--with-security",
                "--accept-whiterabbitneo-terms",
                "--no-start",
                "--non-interactive",
                *extra,
            ],
            cwd=work,
            env=env,
            ok=ok,
        )

    install()
    python = target / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    assert python.is_file(), "real uv did not create the application environment"

    def events():
        return [json.loads(row) for row in (target / "setup-events.jsonl").read_text().splitlines()]

    expected = [
        "setup",
        "--backend",
        "cpu",
        "--profile",
        "fast",
        "--with-security",
        "--accept-whiterabbitneo-terms",
        "--no-start",
        "--non-interactive",
    ]
    assert events() == [expected], events()
    run(["git", "-C", target, "diff", "--exit-code"], env=env)
    marker = target / ".roi-swarm-local-install"
    assert marker.is_file(), "install needs an ownership marker"
    # Repeated install: fast-forward a real upstream commit, preserve tracked edits and settings.
    (target / "local/__init__.py").write_text("# client edit\n", encoding="utf-8")
    (target / ".env").write_text("LOCAL_SETTING=custom\n", encoding="utf-8")
    (source / "new-version.txt").write_text("upstream\n", encoding="utf-8")
    run(["git", "-C", source, "add", "new-version.txt"], env=env)
    run(
        [
            "git",
            "-C",
            source,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "update",
        ],
        env=env,
    )
    install()
    assert (target / "new-version.txt").read_text() == "upstream\n"
    assert (target / "local/__init__.py").read_text() == "# client edit\n"
    assert (target / ".env").read_text() == "LOCAL_SETTING=custom\n"
    assert events() == [expected, ["manage", "stop"], expected], events()
    # Incomplete stop must abort before Git mutation, uv sync or setup.
    (target / "data").mkdir(exist_ok=True)
    (target / "data/refuse-stop").touch()
    failed = install(ok=False)
    assert failed.returncode, "an incomplete stop was ignored"
    assert events()[-1] == ["manage", "stop"]
    assert events().count(expected) == 2
    assert "sync" not in failed.stdout.lower(), failed.stdout
    (target / "data/refuse-stop").unlink()
    # A linked venv directory must never be synchronized into another installation.
    sibling_env = work / "other environment"
    shutil.copytree(target / ".venv", sibling_env)
    original_env = target / ".venv.saved"
    (target / ".venv").rename(original_env)
    try:
        if os.name == "nt":
            import _winapi

            _winapi.CreateJunction(str(sibling_env), str(target / ".venv"))
        else:
            (target / ".venv").symlink_to(sibling_env, target_is_directory=True)
        assert install(ok=False).returncode, "redirected environment was accepted"
    finally:
        if os.name == "nt":
            os.rmdir(target / ".venv")
        else:
            (target / ".venv").unlink()
        original_env.rename(target / ".venv")
    # Unmarked directories and hostile origins must remain byte-for-byte intact.
    unrelated = work / "unrelated"
    unrelated.mkdir()
    sentinel = unrelated / "keep.txt"
    sentinel.write_bytes(b"user-owned\x00")
    assert install(unrelated, ok=False).returncode
    assert sentinel.read_bytes() == b"user-owned\x00"
    run(["git", "-C", target, "remote", "set-url", "origin", "ext::echo hostile"], env=env)
    before = events()
    assert install(ok=False).returncode
    assert events() == before, "hostile remote triggered code execution"
    run(["git", "-C", target, "remote", "set-url", "origin", PUBLIC_REPO], env=env)
    # Local commits must never be reset/cleaned when an upstream divergence appears.
    (target / "client.txt").write_text("local commit\n", encoding="utf-8")
    run(["git", "-C", target, "add", "client.txt"], env=env)
    run(
        [
            "git",
            "-C",
            target,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "local change",
        ],
        env=env,
    )
    head = run(["git", "-C", target, "rev-parse", "HEAD"], env=env).stdout
    (source / "diverged.txt").write_text("upstream divergence\n", encoding="utf-8")
    run(["git", "-C", source, "add", "diverged.txt"], env=env)
    run(
        [
            "git",
            "-C",
            source,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "divergent update",
        ],
        env=env,
    )
    assert install(ok=False).returncode
    assert run(["git", "-C", target, "rev-parse", "HEAD"], env=env).stdout == head
    assert (target / "client.txt").read_text() == "local commit\n"
    assert (target / ".env").read_text() == "LOCAL_SETTING=custom\n"
    # Symlinks/junctions are refused, including a symlink in an ancestor component.
    link = work / "linked"
    try:
        if os.name == "nt":
            import _winapi

            _winapi.CreateJunction(str(unrelated), str(link))
        else:
            link.symlink_to(unrelated, target_is_directory=True)
        assert install(link, ok=False).returncode
        assert install(link / "nested", ok=False).returncode
        assert not (unrelated / "nested").exists()
    finally:
        if link.exists():
            if os.name == "nt":
                os.rmdir(link)
            else:
                link.unlink()
    return {
        "shell": shell,
        "fixture": "passed",
        "real_git": True,
        "real_uv": True,
        "inference": "not exercised by offline fixture",
    }


def cpu_runtime_smoke(checkout: Path, work: Path) -> dict:
    """Download/load actual smallest Bonsai PQ2_0 on an actual CPU-only runtime.

    This deliberately tests 4K/one-slot inference, not full swarm throughput or 64K.
    Every failed download, model load or native tool assertion fails CI.
    """
    import platform
    import socket
    import time
    import urllib.error
    import urllib.request

    if sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("Real CPU smoke requires Linux x86_64 (Ubuntu 24.04)")
    sys.path[:0] = [str(checkout), str(checkout / "src")]
    from local.download_assets import extract_runtime, fetch
    from roi_swarm.native_budget import NativeBudget

    work.mkdir(parents=True, exist_ok=True)
    assets = json.loads((checkout / "local/assets.json").read_text(encoding="utf-8"))
    if isinstance(assets, dict):
        assets = assets["assets"]
    model = next(
        asset for asset in assets if asset["path"].endswith("Ternary-Bonsai-2-2B-PQ2_0.gguf")
    )
    runtime = {
        "path": "download/prism-cpu.tar.gz",
        "bytes": 17882710,
        "sha256": "3f97b6fbec0065fede2be92f05df531c1050636fc2613e1ba0d975249df19713",
        "url": "https://github.com/PrismML-Eng/llama.cpp/releases/download/"
        "prism-b10777-e3de715/llama-prism-b10777-e3de715-bin-ubuntu-x64.tar.gz",
    }
    fetch(runtime, root=work)
    fetch(model, root=work)
    destination = work / "runtime"
    # Shared extractor materializes safe in-archive .so symlinks as regular files.
    extract_runtime([work / runtime["path"]], destination)
    binaries = list(destination.rglob("llama-server"))
    if len(binaries) != 1:
        raise AssertionError("CPU archive must contain exactly one llama-server")
    binary = binaries[0]
    run([binary, "--version"], timeout=30)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [
        str(binary),
        "-m",
        str(work / model["path"]),
        "--alias",
        "bonsai-2b",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-ngl",
        "0",
        "-c",
        "4096",
        "-np",
        "1",
        "-t",
        "2",
        "-tb",
        "2",
        "--jinja",
        "--reasoning-format",
        "auto",
        "--reasoning",
        "off",
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
        "--no-webui",
    ]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            base + path, data=body, headers={"Content-Type": "application/json"}
        )
        with opener.open(req, timeout=180) as response:
            return json.load(response)

    log = work / "prism-cpu.log"
    with log.open("wb") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise AssertionError(f"Prism CPU server exited; inspect {log}")
                try:
                    if request("/health").get("status") == "ok":
                        break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() > deadline:
                    raise AssertionError(f"Prism CPU readiness deadline exceeded; inspect {log}")
                time.sleep(0.5)
            plain = {
                "model": "bonsai-2b",
                "messages": [{"role": "user", "content": "Reply with the single word READY."}],
                "max_tokens": 64,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            budget = NativeBudget(base + "/v1", context_limit=4096)
            native_tokens = budget.check(plain)
            assert native_tokens > 0
            answer = request("/v1/chat/completions", plain)
            assert answer["choices"][0]["message"].get("content"), answer
            assert answer["usage"]["completion_tokens"] > 0, answer
            tool_request = {
                "model": "bonsai-2b",
                "messages": [
                    {
                        "role": "system",
                        "content": "Call record_probe exactly once. Do not answer in text.",
                    },
                    {"role": "user", "content": 'Record status="OK" using record_probe now.'},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "record_probe",
                            "description": "Records a harmless CI probe result.",
                            "parameters": {
                                "type": "object",
                                "properties": {"status": {"type": "string", "enum": ["OK"]}},
                                "required": ["status"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                "tool_choice": "required",
                "max_tokens": 256,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            tool_tokens = budget.check(tool_request)
            tool_answer = request("/v1/chat/completions", tool_request)
            calls = tool_answer["choices"][0]["message"].get("tool_calls", [])
            assert len(calls) == 1, tool_answer
            assert calls[0]["function"]["name"] == "record_probe", tool_answer
            assert json.loads(calls[0]["function"]["arguments"]) == {"status": "OK"}
            result = {
                "runtime": "Prism prism-b10777-e3de715 Ubuntu CPU-only",
                "backend": "cpu",
                "gpu_layers": 0,
                "context": 4096,
                "slots": 1,
                "model_sha256": model["sha256"],
                "runtime_sha256": runtime["sha256"],
                "native_prompt_tokens": native_tokens,
                "native_tool_tokens": tool_tokens,
                "usage": answer["usage"],
                "native_tool_call": calls[0]["function"],
            }
            (work / "cpu-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shell",
        choices=("bash", "powershell"),
        default="powershell" if os.name == "nt" else "bash",
    )
    parser.add_argument("--cpu-runtime", action="store_true")
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    action = (
        cpu_runtime_smoke
        if args.cpu_runtime
        else (lambda root, work: fixture_smoke(root, work, args.shell))
    )
    if args.work_dir:
        print(json.dumps(action(checkout, args.work_dir), indent=2))
    else:
        with tempfile.TemporaryDirectory(prefix="roi-local-bootstrap-") as directory:
            print(json.dumps(action(checkout, Path(directory)), indent=2))


if __name__ == "__main__":
    main()
