"""Installer tests use tiny archives and injected subprocess seams, not global Hermes."""

import importlib
import subprocess
from pathlib import Path

import pytest
import yaml


def setup_module_api():
    assert importlib.util.find_spec("local.setup") is not None, "local.setup installer is missing"
    return importlib.import_module("local.setup")


def test_fresh_configuration_is_local_and_repeat_install_preserves_bytes(tmp_path):
    setup = setup_module_api()
    (tmp_path / "local").mkdir()
    template = Path(__file__).resolve().parents[1] / "local/settings.example.yaml"
    (tmp_path / "local/settings.example.yaml").write_bytes(template.read_bytes())
    setup.configure(tmp_path, "fast", system="Linux")
    settings = yaml.safe_load((tmp_path / "local/settings.yaml").read_text())
    workers = settings["hermes_workers"]
    assert workers["root"] == (tmp_path / "runtime/hermes-agent").as_posix()
    assert workers["python"] == (tmp_path / "runtime/hermes-agent/.venv/bin/python").as_posix()
    assert workers["default_cwd"] == tmp_path.as_posix()
    assert workers["worktree_root"] == (tmp_path / "data/worktrees").as_posix()
    assert settings["models"]["worker"]["model"] == "bonsai-2b"
    assert settings["context"]["worker_context_tokens"] == 65536
    assert settings["sleep"]["enabled"] is False
    assert settings["sleep"]["auto_improve"] is False
    hermes_path = tmp_path / "hermes/config.yaml"
    config = yaml.safe_load(hermes_path.read_text())
    assert config["model"]["provider"] == "custom:roi-local"
    assert config["model"]["streaming"] is True
    assert config["streaming"]["enabled"] is True
    assert config["agent"]["reasoning_effort"] == "none"
    cp = {p["name"]: p for p in config["custom_providers"]}
    aliases = cp["prism-local"]["models"]
    assert {name: a["context_length"] for name, a in aliases.items()} == {
        n: 65536 for n in ("bonsai-2b", "bonsai-4b", "bonsai-9b", "whiterabbit-7b")
    }
    assert config["auxiliary"]["compression"]["provider"] == "custom:prism-local"
    assert (
        config["auxiliary"]["compression"]["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        is False
    )
    (tmp_path / "local/settings.yaml").write_bytes(
        b"# user edits\n" + (tmp_path / "local/settings.yaml").read_bytes()
    )
    hermes_path.write_bytes(b"# user profile\n" + hermes_path.read_bytes())
    before = [(p, p.read_bytes()) for p in (tmp_path / "local/settings.yaml", hermes_path)]
    setup.configure(tmp_path, "balanced", system="Linux")
    assert all(p.read_bytes() == content for p, content in before)
    assert not (tmp_path / "hermes/memories").exists()


def test_hermes_checkout_pinned_venv_and_native_guard_import_probe(tmp_path):
    setup = setup_module_api()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[:2] == ["git", "init"]:
            Path(command[2]).mkdir()
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, setup.HERMES_SHA + "\n", "")
        if command[0] == "uv":
            python = tmp_path / "runtime/hermes-agent/.venv/Scripts/python.exe"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"fixture")
        return subprocess.CompletedProcess(command, 0, "", "")

    setup.install_hermes(tmp_path, "Windows", run=run)
    commands = [command for command, _ in calls]
    fetch = next(c for c in commands if "fetch" in c)
    assert "https://github.com/NousResearch/hermes-agent.git" in fetch
    assert fetch[-1] == "498abb677ec39ea3ae9f8f5ed60e7def6bc47e70"
    assert [
        "uv",
        "sync",
        "--project",
        str(tmp_path / "runtime/hermes-agent"),
        "--frozen",
        "--no-dev",
        "--python",
        "3.11",
    ] in commands
    probe = next(c for c in commands if c[0].endswith("python.exe"))
    assert "from run_agent import AIAgent" in probe[-1]
    assert "_create_request_openai_client" in probe[-1]
    assert "_ensure_primary_openai_client" in probe[-1]
    assert "guard_worker_clients" in probe[-1]
    assert all(k["env"]["HERMES_HOME"] == str(tmp_path / "hermes") for _, k in calls)


def test_prepare_only_real_http_archive_setup_receipt_and_rerun(tmp_path):
    import hashlib
    import json
    import urllib.request
    import zipfile

    from test_local_download_assets import fixture_server

    setup = setup_module_api()
    root = tmp_path / "install"
    (root / "local").mkdir(parents=True)
    template = Path(__file__).resolve().parents[1] / "local/settings.example.yaml"
    (root / "local/settings.example.yaml").write_bytes(template.read_bytes())
    http = tmp_path / "http"
    http.mkdir()
    with zipfile.ZipFile(http / "runtime.zip", "w") as archive:
        archive.writestr("llama-server.exe", b"fixture executable")
    (http / "model.gguf").write_bytes(b"fixture weights")

    def run(command, **kwargs):
        assert command[-1] in {"--version", "--list-devices"}
        return subprocess.CompletedProcess(
            command,
            0,
            "version b10777 e3de715" if command[-1] == "--version" else "Available devices:\n",
            "",
        )

    starts = []
    hermes = []
    with fixture_server(http) as url:
        assets = [
            {
                "path": "download/runtime.zip"
                if name.endswith("zip")
                else "models/Ternary-Bonsai-2-2B-PQ2_0.gguf",
                "bytes": (http / name).stat().st_size,
                "sha256": hashlib.sha256((http / name).read_bytes()).hexdigest(),
                "url": url + "/" + name,
                "kind": "runtime" if name.endswith("zip") else "model",
            }
            for name in ("runtime.zip", "model.gguf")
        ]
        kwargs = dict(
            root=root,
            backend="cpu",
            profile="balanced",
            prepare_only=True,
            assets=assets,
            system="Windows",
            machine="AMD64",
            run=run,
            opener=urllib.request.urlopen,
            starter=lambda r: starts.append(r),
            hermes_installer=lambda *a, **kw: hermes.append(a),
        )
        result = setup.install(**kwargs)
    assert result["state"] == "prepared"
    receipt = json.loads((root / "data/local-install.json").read_text())
    assert receipt["version"] == 1
    assert receipt["root"] == str(root.resolve())
    assert receipt["backend"] == "cpu"
    assert receipt["profile"] == "balanced"
    assert (receipt["api_port"], receipt["inference_port"], receipt["lock_port"]) == (
        8775,
        8091,
        8776,
    )
    assert receipt["hermes_python"] == str(root / "runtime/hermes-agent/.venv/Scripts/python.exe")
    assert receipt["hermes_verified"] is False
    assert not starts and not hermes
    ini = (root / "local/models.ini").read_text()
    assert "n-gpu-layers = 0" in ini
    settings = yaml.safe_load((root / "local/settings.yaml").read_text())
    assert settings["hermes_workers"]["maximum_concurrency"] == 1
    saved = (root / "local/settings.yaml").read_bytes()
    # The HTTP fixture is now offline: rerun must hash-verify the saved files.
    setup.install(**kwargs)
    assert (root / "local/settings.yaml").read_bytes() == saved


def test_cli_public_options_and_security_terms_are_explicit(monkeypatch, tmp_path):
    setup = setup_module_api()
    calls = []
    original = setup.install
    monkeypatch.setattr(setup, "install", lambda **kw: calls.append(kw) or {"state": "prepared"})
    assert (
        setup.main(
            [
                "--backend",
                "cpu",
                "--profile",
                "fast",
                "--prepare-only",
                "--non-interactive",
                "--asset-cache",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert calls[0]["prepare_only"] is True
    assert calls[0]["asset_cache"] == tmp_path
    with pytest.raises(SystemExit):
        setup.main(["--skip-hermes"])
    # The API also rejects optional security weights before any filesystem/network work.
    with pytest.raises(ValueError, match="accept-whiterabbitneo-terms"):
        original(root=tmp_path, with_security=True)


def test_preflight_refuses_low_disk_and_hashes_existing_files(tmp_path):
    from collections import namedtuple

    from local.download_assets import sha256

    setup = setup_module_api()
    model = tmp_path / "model"
    model.write_bytes(b"abc")
    assets = [{"path": "model", "bytes": 3, "sha256": sha256(model), "kind": "model"}]
    usage = namedtuple("usage", "total used free")
    required = setup.preflight_space(
        tmp_path, assets, prepare_only=True, disk_usage=lambda _: usage(10**10, 0, 10**10)
    )
    model.write_bytes(b"bad")
    assert (
        setup.preflight_space(
            tmp_path, assets, prepare_only=True, disk_usage=lambda _: usage(10**10, 0, 10**10)
        )
        > required
    )
    with pytest.raises(RuntimeError, match="disk space"):
        setup.preflight_space(tmp_path, assets, disk_usage=lambda _: usage(10, 9, 1))


def test_failed_hermes_install_preserves_previous_runtime_and_all_config(tmp_path):
    import hashlib
    import zipfile

    setup = setup_module_api()
    root = tmp_path / "root"
    (root / "local").mkdir(parents=True)
    (root / "local/settings.example.yaml").write_bytes(
        (Path(__file__).resolve().parents[1] / "local/settings.example.yaml").read_bytes()
    )
    setup.configure(root, "balanced", system="Windows", backend="cpu")
    runtime = root / "runtime/prism"
    runtime.mkdir(parents=True)
    (runtime / "old").write_bytes(b"working runtime")
    (root / "local/models.ini").write_bytes(b"user presets")
    cache = tmp_path / "cache"
    cache.mkdir()
    with zipfile.ZipFile(cache / "runtime.zip", "w") as z:
        z.writestr("llama-server.exe", b"new runtime")
    content = (cache / "runtime.zip").read_bytes()
    assets = [
        {
            "path": "download/runtime.zip",
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "url": "https://invalid.example",
            "kind": "runtime",
        }
    ]
    paths = [root / "local/settings.yaml", root / "hermes/config.yaml", root / "local/models.ini"]
    before = [p.read_bytes() for p in paths]

    def run(command, **kw):
        return subprocess.CompletedProcess(
            command,
            0,
            "version b10777 e3de715" if command[-1] == "--version" else "Available devices:",
            "",
        )

    def failure(*a, **kw):
        raise RuntimeError("Hermes dependencies failed")

    with pytest.raises(RuntimeError, match="dependencies failed"):
        setup.install(
            root=root,
            backend="cpu",
            assets=assets,
            asset_cache=cache,
            system="Windows",
            machine="AMD64",
            run=run,
            hermes_installer=failure,
        )
    assert (runtime / "old").read_bytes() == b"working runtime"
    assert not (runtime / "llama-server.exe").exists()
    assert [p.read_bytes() for p in paths] == before
    assert not (root / "data/local-install.json").exists()


def test_hermes_uses_bootstrap_uv_without_global_install(tmp_path):
    setup = setup_module_api()
    uv = tmp_path / "tools/uv/uv.exe"
    uv.parent.mkdir(parents=True)
    uv.write_bytes(b"fixture uv")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["git", "init"]:
            Path(command[2]).mkdir()
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, setup.HERMES_SHA, "")
        if command[0] == str(uv):
            python = tmp_path / "runtime/hermes-agent/.venv/Scripts/python.exe"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"fixture")
        return subprocess.CompletedProcess(command, 0, "", "")

    setup.install_hermes(tmp_path, "Windows", run=run)
    assert next(c for c in calls if "sync" in c)[0] == str(uv)


def test_backend_auto_requires_working_nvidia_smi():
    setup = setup_module_api()

    def working(command, **kwargs):
        assert command[0] == "nvidia-smi"
        assert kwargs["timeout"] <= 15
        return subprocess.CompletedProcess(command, 0, "NVIDIA RTX, 8192", "")

    def broken(command, **kwargs):
        raise FileNotFoundError("no driver")

    assert setup.select_backend("auto", run=working) == "cuda"
    assert setup.select_backend("auto", run=broken) == "cpu"
    assert setup.select_backend("cpu", run=working) == "cpu"
    with pytest.raises(RuntimeError, match="CUDA"):
        setup.select_backend("cuda", run=broken)


def test_runtime_probe_executes_version_and_list_devices(tmp_path):
    setup = setup_module_api()
    (tmp_path / "llama-server.exe").write_bytes(b"fixture")
    calls = []

    def run(command, **kwargs):
        calls.append(command[-1])
        return subprocess.CompletedProcess(
            command,
            0,
            "version: b10777 e3de715"
            if command[-1] == "--version"
            else "Available devices:\n  CUDA0: NVIDIA RTX",
            "",
        )

    result = setup.probe_runtime(tmp_path, "cuda", run=run)
    assert result["cuda"] is True
    assert calls == ["--version", "--list-devices"]

    def cpu(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            "version: b10777 e3de715" if command[-1] == "--version" else "Available devices:\n",
            "",
        )

    assert setup.probe_runtime(tmp_path, "cpu", run=cpu)["cuda"] is False
    with pytest.raises(RuntimeError, match="CUDA"):
        setup.probe_runtime(tmp_path, "cuda", run=cpu)
