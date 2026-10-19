"""Installer regressions use disposable roots, tiny assets and no model/service launches."""

import hashlib
import json
import os
import socket
import subprocess
import zipfile
from contextlib import ExitStack
from pathlib import Path

import pytest

from local import download_assets as downloads
from local import setup


def fixture_install(tmp_path):
    root = tmp_path / "install"
    (root / "local").mkdir(parents=True)
    template = Path(__file__).resolve().parents[1] / "local/settings.example.yaml"
    (root / "local/settings.example.yaml").write_bytes(template.read_bytes())
    cache = tmp_path / "asset-cache"
    cache.mkdir()
    with zipfile.ZipFile(cache / "runtime.zip", "w") as archive:
        archive.writestr("llama-server.exe", b"new fixture runtime")
    (cache / "Ternary-Bonsai-2-2B-PQ2_0.gguf").write_bytes(b"fixture model")
    assets = []
    for filename, kind, parent in (
        ("runtime.zip", "runtime", "download"),
        ("Ternary-Bonsai-2-2B-PQ2_0.gguf", "model", "models"),
    ):
        content = (cache / filename).read_bytes()
        assets.append(
            {
                "path": f"{parent}/{filename}",
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "url": "https://invalid.example/no-network",
                "kind": kind,
            }
        )

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "fixture CPU runtime", "")

    return {
        "root": root,
        "backend": "cpu",
        "prepare_only": True,
        "assets": assets,
        "asset_cache": cache,
        "system": "Windows",
        "machine": "AMD64",
        "run": run,
    }


def directory_link(target, link):
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


def remove_directory_link(link):
    if os.name == "nt":
        os.rmdir(link)
    else:
        link.unlink()


@pytest.mark.parametrize(
    "relative",
    [
        "runtime",
        "runtime/prism",
        "runtime/hermes-agent",
        "runtime/hermes-agent/.git",
        "runtime/hermes-agent/.git/objects",
        "runtime/hermes-agent/.venv",
        "cache/huggingface/hub",
        "cache/tiktoken",
        "hermes",
        "local",
        "data",
        "cache",
        "download",
        "models",
    ],
)
def test_install_rejects_redirected_managed_paths_before_any_write(tmp_path, monkeypatch, relative):
    options = fixture_install(tmp_path)
    root = options["root"]
    external = tmp_path / "external"
    (external / "prism").mkdir(parents=True)
    sentinel = external / "prism/keep"
    sentinel.write_bytes(b"external user data")
    link = root / relative
    if link.exists():
        link.rename(link.with_name(link.name + ".saved"))
    directory_link(external, link)
    writes = []
    original_fetch = downloads.fetch

    def fetch(*args, **kwargs):
        writes.append("fetch")
        return original_fetch(*args, **kwargs)

    monkeypatch.setattr(downloads, "fetch", fetch)
    try:
        with pytest.raises(ValueError, match="[Uu]nsafe|[Ll]ink|[Rr]eparse"):
            setup.install(**options)
        assert writes == []
        assert sentinel.read_bytes() == b"external user data"
        assert not (root / "data/local-install.json").exists()
    finally:
        remove_directory_link(link)


@pytest.mark.parametrize("linked", ["runtime", "runtime/prism", "ancestor"])
def test_standalone_extract_refuses_junction_destination_or_parent(tmp_path, linked):
    options = fixture_install(tmp_path)
    root = options["root"]
    external = tmp_path / "external"
    (external / "prism").mkdir(parents=True)
    sentinel = external / "prism/keep"
    sentinel.write_bytes(b"external user data")
    link = tmp_path / "ancestor" if linked == "ancestor" else root / linked
    directory_link(external, link)
    destination = link / "prism" if linked == "ancestor" else root / "runtime/prism"
    try:
        with pytest.raises(ValueError, match="[Uu]nsafe|[Ll]ink|[Rr]eparse"):
            downloads.extract_runtime([options["asset_cache"] / "runtime.zip"], destination)
        assert sentinel.read_bytes() == b"external user data"
        assert sorted(p.name for p in external.iterdir()) == ["prism"]
    finally:
        remove_directory_link(link)


def install_record(root, **overrides):
    from local import manage

    source, python = setup.hermes_paths(root, "Windows")
    record = {
        "version": 1,
        "root": str(root),
        "backend": "cpu",
        "profile": "balanced",
        "api_port": 18775,
        "inference_port": 18091,
        "lock_port": 18776,
        "hermes_root": str(source),
        "hermes_python": str(python),
        **overrides,
    }
    path = root / "data/local-install.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return manage.read_install(root)


def ephemeral_ports():
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(3)]
        for item in sockets:
            item.bind(("127.0.0.1", 0))
        return dict(
            zip(
                ("api_port", "inference_port", "lock_port"),
                (item.getsockname()[1] for item in sockets),
                strict=True,
            )
        )


@pytest.mark.parametrize("mode", ["prepare_only", "no_start"])
@pytest.mark.parametrize("occupied", ["api_port", "inference_port", "lock_port"])
def test_direct_install_refuses_real_tcp_listener_before_mutation(
    tmp_path, monkeypatch, mode, occupied
):
    from local import manage

    options = fixture_install(tmp_path)
    options["prepare_only"] = mode == "prepare_only"
    options["no_start"] = mode == "no_start"
    options["hermes_installer"] = lambda *a, **kw: None
    ports = ephemeral_ports()
    install_record(options["root"], **ports)
    mutations = []
    original_fetch = downloads.fetch

    def fetch(*args, **kwargs):
        mutations.append("fetch")
        return original_fetch(*args, **kwargs)

    monkeypatch.setattr(downloads, "fetch", fetch)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", ports[occupied]))
        listener.listen(1)
        before = (options["root"] / "data/local-install.json").read_bytes()
        with pytest.raises(manage.LifecycleError, match="[Pp]ort|[Ll]ock|listener"):
            setup.install(**options)
    assert mutations == []
    assert (options["root"] / "data/local-install.json").read_bytes() == before


def test_direct_install_refuses_saved_service_receipts_before_mutation(tmp_path, monkeypatch):
    from local import manage

    options = fixture_install(tmp_path)
    root = options["root"]
    install_record(root, **ephemeral_ports())
    manage.write_state(root, {"version": 1, "root": str(root), "services": {"api": None}})
    saved = (root / "data/local-services.json").read_bytes()
    mutations = []
    original_fetch = downloads.fetch

    def fetch(*args, **kwargs):
        mutations.append("fetch")
        return original_fetch(*args, **kwargs)

    monkeypatch.setattr(downloads, "fetch", fetch)
    with pytest.raises(manage.LifecycleError, match="[Ss]top|record|state"):
        setup.install(**options)
    assert mutations == []
    assert (root / "data/local-services.json").read_bytes() == saved


def test_install_holds_shared_lock_and_both_service_ports_then_releases_before_start(
    tmp_path, monkeypatch
):
    from local import manage

    options = fixture_install(tmp_path)
    root = options["root"]
    ports = ephemeral_ports()
    install_record(root, **ports)
    observations = []
    original_fetch = downloads.fetch

    def fetch(*args, **kwargs):
        for port in ports.values():
            with socket.socket() as competitor, pytest.raises(OSError):
                competitor.bind(("127.0.0.1", port))
        observations.append("protected")
        return original_fetch(*args, **kwargs)

    def start(path):
        assert path == root
        with manage.lock_for_root(root):
            for key in ("api_port", "inference_port"):
                with socket.socket() as available:
                    available.bind(("127.0.0.1", ports[key]))
        observations.append("start")

    monkeypatch.setattr(downloads, "fetch", fetch)
    options.update(prepare_only=False, hermes_installer=lambda *a, **kw: None, starter=start)
    setup.install(**options)
    assert observations == ["protected", "protected", "start"]


@pytest.mark.parametrize("regenerate", [False, True])
def test_repeat_install_preserves_custom_receipt_ports_and_config_endpoints(tmp_path, regenerate):
    import yaml

    options = fixture_install(tmp_path)
    root = options["root"]
    installed = install_record(root)
    setup.install(**options)
    paths = [root / "local/settings.yaml", root / "hermes/config.yaml"]
    saved = [p.read_bytes() for p in paths]
    if regenerate:
        for path in paths:
            path.unlink()
    result = setup.install(**options)
    receipt = json.loads((root / "data/local-install.json").read_bytes())
    for key in ("api_port", "inference_port", "lock_port"):
        assert result[key] == receipt[key] == installed[key]
    if not regenerate:
        assert [p.read_bytes() for p in paths] == saved
    settings = yaml.safe_load(paths[0].read_bytes())
    config = yaml.safe_load(paths[1].read_bytes())
    assert settings["provider"]["base_url"] == "http://127.0.0.1:18091/v1"
    assert config["model"]["base_url"] == "http://127.0.0.1:18775/v1"
    providers = {p["name"]: p["base_url"] for p in config["custom_providers"]}
    assert providers == {
        "roi-local": "http://127.0.0.1:18775/v1",
        "prism-local": "http://127.0.0.1:18091/v1",
    }
    assert config["auxiliary"]["compression"]["base_url"] == "http://127.0.0.1:18091/v1"


def test_hermes_probe_uses_validated_installed_inference_port(tmp_path):
    options = fixture_install(tmp_path)
    root = options["root"]
    install_record(root)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    setup.probe_hermes(root, "Windows", run=run)
    assert "http://127.0.0.1:18091/v1" in calls[0][-1]
    assert "http://127.0.0.1:8091/v1" not in calls[0][-1]


def test_requested_profile_change_is_reported_not_silently_claimed_applied(tmp_path):
    options = fixture_install(tmp_path)
    setup.install(**options)
    settings = options["root"] / "local/settings.yaml"
    original = settings.read_bytes()
    result = setup.install(**options, profile="fast")
    assert result["profile"] == "balanced"
    assert settings.read_bytes() == original
    assert any(
        "fast" in notice and "preserv" in notice.lower() for notice in result.get("notices", [])
    )


def security_asset(options):
    filename = "WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-Q4_K_M.gguf"
    content = b"optional fixture model"
    (options["asset_cache"] / filename).write_bytes(content)
    return {
        "path": f"models/{filename}",
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "url": "https://invalid.example/no-network",
        "kind": "model",
        "optional": True,
    }


@pytest.mark.parametrize("already_present", [False, True])
def test_second_install_adds_only_missing_security_alias_preserving_all_existing_bytes(
    tmp_path, already_present
):
    options = fixture_install(tmp_path)
    setup.install(**options)
    ini = options["root"] / "local/models.ini"
    custom = (
        b"version = 1\r\n# custom user settings must not be reformatted\r\n"
        b"[bonsai-2b]\r\nmodel = custom-model.gguf\r\nthreads = 3\r\n"
        b"[user-alias]\r\nctx-size = 12345\r\n"
    )
    if already_present:
        custom += b"[whiterabbit-7b]\r\nmodel = my-security-model.gguf\r\ntemp = 0.123\r\n"
    ini.write_bytes(custom)
    options["assets"].append(security_asset(options))
    setup.install(**options, with_security=True, accept_whiterabbitneo_terms=True)
    updated = ini.read_bytes()
    assert updated.startswith(custom)
    assert updated.count(b"[whiterabbit-7b]") == 1
    if already_present:
        assert updated == custom
    else:
        addition = updated[len(custom) :]
        assert b"n-gpu-layers = 0" in addition
        assert b"WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-Q4_K_M.gguf" in addition
        assert b"[bonsai-2b]" not in addition and b"[user-alias]" not in addition
    setup.install(**options, with_security=True, accept_whiterabbitneo_terms=True)
    assert ini.read_bytes() == updated


def test_optional_alias_append_rolls_back_exact_config_bytes_on_publication_failure(
    tmp_path, monkeypatch
):
    options = fixture_install(tmp_path)
    setup.install(**options)
    root = options["root"]
    ini = root / "local/models.ini"
    ini.write_bytes(b"version = 1\r\n[custom]\r\nthreads = 3\r\n")
    paths = [
        root / relative
        for relative in (
            "local/models.ini",
            "local/settings.yaml",
            "hermes/config.yaml",
            "data/local-install.json",
        )
    ]
    before = [p.read_bytes() for p in paths]
    options["assets"].append(security_asset(options))
    original_rename = Path.rename
    additions = []

    def denied(path, target):
        if path.name.startswith(".prism-stage-") and not path.name.endswith(".previous"):
            additions.append(ini.read_bytes())
            raise PermissionError("fixture runtime publication denied")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", denied)
    with pytest.raises(PermissionError, match="publication denied"):
        setup.install(**options, with_security=True, accept_whiterabbitneo_terms=True)
    assert b"[whiterabbit-7b]" in additions[0]
    assert [p.read_bytes() for p in paths] == before
    assert (root / "runtime/prism/llama-server.exe").read_bytes() == b"new fixture runtime"


def test_committed_runtime_cleanup_failure_keeps_matching_new_receipt_and_configs(
    tmp_path, monkeypatch, caplog
):
    options = fixture_install(tmp_path)
    root = options["root"]
    runtime = root / "runtime/prism"
    runtime.mkdir(parents=True)
    (runtime / "old").write_bytes(b"old runtime")
    original_rmtree = downloads.shutil.rmtree

    def denied(path, *args, **kwargs):
        if str(path).endswith(".previous"):
            raise PermissionError("fixture backup locked")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(downloads.shutil, "rmtree", denied)
    result = setup.install(**options)
    assert result["state"] == "prepared"
    assert json.loads((root / "data/local-install.json").read_bytes()) == result
    assert (root / "local/settings.yaml").is_file()
    assert (root / "hermes/config.yaml").is_file()
    assert (runtime / "llama-server.exe").read_bytes() == b"new fixture runtime"
    retained = list(runtime.parent.glob("*.previous"))
    assert len(retained) == 1
    assert (retained[0] / "old").read_bytes() == b"old runtime"
    assert str(retained[0]) in caplog.text


@pytest.mark.parametrize(
    "relative",
    ["local/settings.yaml", "local/models.ini", "hermes/config.yaml", "data/local-install.json"],
)
def test_linked_config_or_receipt_file_is_rejected_before_downloading(
    tmp_path, monkeypatch, relative
):
    options = fixture_install(tmp_path)
    target = tmp_path / "external-user-file"
    target.write_bytes(b"user-owned bytes")
    link = options["root"] / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"File symlinks unavailable: {exc}")
    writes = []
    monkeypatch.setattr(downloads, "fetch", lambda *a, **kw: writes.append("fetch"))
    with pytest.raises(ValueError, match="Unsafe|link|reparse"):
        setup.install(**options)
    assert writes == []
    assert target.read_bytes() == b"user-owned bytes"


@pytest.mark.parametrize("ancestor", [False, True])
def test_install_rejects_linked_root_or_ancestor_before_writes(tmp_path, ancestor):
    options = fixture_install(tmp_path)
    root = options["root"]
    link = tmp_path / "root-link"
    directory_link(root, link)
    try:
        options["root"] = link / "child" if ancestor else link
        with pytest.raises(ValueError, match="Unsafe|link|reparse"):
            setup.install(**options)
        assert not (root / "data").exists()
        assert not (root / "download").exists()
    finally:
        remove_directory_link(link)


def test_normal_venv_interpreter_symlinks_are_not_mutation_paths(tmp_path):
    options = fixture_install(tmp_path)
    root = options["root"]
    target = tmp_path / "real-python"
    target.write_bytes(b"external interpreter placeholder, never executed")
    interpreters = [root / ".venv/bin/python", root / "runtime/hermes-agent/.venv/bin/python"]
    for python in interpreters:
        python.parent.mkdir(parents=True)
        try:
            python.symlink_to(target)
        except OSError as exc:
            pytest.skip(f"File symlinks unavailable: {exc}")
    assert setup.install(**options)["state"] == "prepared"
    assert all(p.is_symlink() for p in interpreters)
    assert target.read_bytes() == b"external interpreter placeholder, never executed"


def test_interpreter_files_are_not_inspected_as_managed_mutation_targets(tmp_path, monkeypatch):
    options = fixture_install(tmp_path)
    root = options["root"]
    original_lstat = Path.lstat
    interpreter = root / ".venv/bin/python"
    hermes_python = root / "runtime/hermes-agent/.venv/bin/python"

    def checked(path, *args, **kwargs):
        assert path not in (interpreter, hermes_python), "Interpreter links are legitimate"
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", checked)
    assert setup.install(**options)["state"] == "prepared"


def test_runtime_only_preparation_still_creates_empty_models_preset(tmp_path):
    options = fixture_install(tmp_path)
    options["assets"] = [a for a in options["assets"] if a["kind"] == "runtime"]
    setup.install(**options)
    assert (options["root"] / "local/models.ini").read_bytes() == b"version = 1\n"


def test_public_setup_cli_uses_same_listener_guard_before_any_mutation(
    tmp_path, monkeypatch, capsys
):
    from local import cli

    options = fixture_install(tmp_path)
    ports = ephemeral_ports()
    install_record(options["root"], **ports)
    install = setup.install
    mutations = []

    def guarded(**kwargs):
        return install(**{**options, **kwargs})

    monkeypatch.setattr(setup, "install", guarded)
    monkeypatch.setattr(downloads, "fetch", lambda *a, **kw: mutations.append("fetch"))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", ports["api_port"]))
        listener.listen(1)
        assert cli.main(["setup", "--backend", "cpu", "--prepare-only"]) == 1
    assert mutations == []
    assert "listener" in json.loads(capsys.readouterr().out)["error"].lower()


def test_failed_spawn_recovery_record_refuses_setup_before_any_mutation(tmp_path, monkeypatch):
    from local import manage

    options = fixture_install(tmp_path)
    root = options["root"]
    install_record(root, **ephemeral_ports())
    recovery = root / "data/local-spawn-recovery.json"
    recovery.write_bytes(b'{"pid": 123, "stop_error": "unconfirmed"}')
    mutations = []
    monkeypatch.setattr(downloads, "fetch", lambda *a, **kw: mutations.append("fetch"))
    with pytest.raises(manage.LifecycleError, match="recovery|cleanup"):
        setup.install(**options)
    assert mutations == []
    assert recovery.read_bytes() == b'{"pid": 123, "stop_error": "unconfirmed"}'
