"""Small real HTTP fixtures; never fetch publisher weights in tests."""

import functools
import hashlib
import http.server
import io
import os
import stat
import tarfile
import threading
import urllib.request
import zipfile
from contextlib import contextmanager

import pytest

from local import download_assets as downloads


@contextmanager
def fixture_server(directory):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def asset_for(content, url, path="models/fixture.gguf", **extra):
    return {
        "path": path,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "url": url,
        "kind": "model",
        **extra,
    }


def test_download_real_http_atomic_verified_rerun(tmp_path):
    source = tmp_path / "http"
    source.mkdir()
    content = b"fixture model\x00" * 91
    (source / "model").write_bytes(content)
    root = tmp_path / "install"
    with fixture_server(source) as url:
        asset = asset_for(content, url + "/model")
        result = downloads.fetch(asset, root=root, opener=urllib.request.urlopen, attempts=1)
    assert result["verified"] is True
    assert (root / asset["path"]).read_bytes() == content
    # An offline rerun must verify local bytes rather than require the network.
    result = downloads.fetch(asset, root=root, attempts=1)
    assert result["verified"] is True
    assert not list(root.rglob("*.part"))


def test_bad_hash_retries_and_preserves_existing_target(tmp_path):
    source = tmp_path / "http"
    source.mkdir()
    (source / "model").write_bytes(b"wrong")
    root = tmp_path / "root"
    (root / "models").mkdir(parents=True)
    (root / "models/fixture.gguf").write_bytes(b"old model")
    calls = []

    def opening(*args, **kwargs):
        calls.append(kwargs["timeout"])
        return urllib.request.urlopen(*args, **kwargs)

    with fixture_server(source) as url, pytest.raises(ValueError, match="mismatch"):
        downloads.fetch(
            asset_for(b"right", url + "/model"),
            root=root,
            opener=opening,
            attempts=2,
            timeout=3,
            sleep=lambda _: None,
        )
    assert calls == [3, 3]
    assert (root / "models/fixture.gguf").read_bytes() == b"old model"
    assert not list(root.rglob("*.part"))


def test_oversized_http_response_is_bounded_before_publication(tmp_path):
    source = tmp_path / "http"
    source.mkdir()
    (source / "huge").write_bytes(b"x" * 65536)
    with fixture_server(source) as url, pytest.raises(ValueError, match="exceeds pinned byte size"):
        downloads.fetch(
            asset_for(b"x", url + "/huge"),
            root=tmp_path / "root",
            opener=urllib.request.urlopen,
            attempts=1,
        )
    assert not (tmp_path / "root/models/fixture.gguf").exists()


def test_production_download_forbids_plain_http(tmp_path):
    with pytest.raises(ValueError, match="HTTPS"):
        downloads.fetch(asset_for(b"x", "http://127.0.0.1:1/model"), root=tmp_path, attempts=1)


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_extract_runtime_atomically_with_executable_permissions(tmp_path, kind):
    archive = tmp_path / ("runtime." + kind)
    if kind == "zip":
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("build/bin/llama-server", b"binary")
    else:
        with tarfile.open(archive, "w") as t:
            member = tarfile.TarInfo("build/bin/llama-server")
            member.size = 6
            t.addfile(member, io.BytesIO(b"binary"))
    destination = tmp_path / "runtime/prism"
    destination.mkdir(parents=True)
    (destination / "old").write_text("old")
    downloads.extract_runtime([archive], destination)
    server = destination / "build/bin/llama-server"
    assert server.read_bytes() == b"binary"
    assert not (destination / "old").exists()
    if os.name != "nt":
        assert server.stat().st_mode & stat.S_IXUSR


@pytest.mark.parametrize("kind", ["zip", "symlink", "hardlink", "fifo"])
def test_unsafe_archive_preserves_old_runtime(tmp_path, kind):
    archive = tmp_path / ("bad.zip" if kind == "zip" else "bad.tar")
    if kind == "zip":
        with zipfile.ZipFile(archive, "w") as z:
            member = zipfile.ZipInfo("lib/link")
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(member, "../../escape")
    else:
        with tarfile.open(archive, "w") as t:
            member = tarfile.TarInfo("lib/link")
            member.type = {
                "symlink": tarfile.SYMTYPE,
                "hardlink": tarfile.LNKTYPE,
                "fifo": tarfile.FIFOTYPE,
            }[kind]
            member.linkname = "../../escape"
            t.addfile(member)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "old").write_bytes(b"old")
    with pytest.raises(ValueError, match="Unsafe|unsupported"):
        downloads.extract_runtime([archive], runtime)
    assert (runtime / "old").read_bytes() == b"old"
    assert not (tmp_path / "escape").exists()


def test_linux_tar_safe_library_links_materialized_without_symlinks(tmp_path):
    archive = tmp_path / "linux.tar"
    with tarfile.open(archive, "w") as t:
        t.addfile(tarfile.TarInfo(".")) if False else None
        directory = tarfile.TarInfo(".")
        directory.type = tarfile.DIRTYPE
        t.addfile(directory)
        member = tarfile.TarInfo("bin/lib.so.0.1")
        member.size = 3
        t.addfile(member, io.BytesIO(b"lib"))
        for name, target in [("bin/lib.so.0", "lib.so.0.1"), ("bin/lib.so", "lib.so.0")]:
            link = tarfile.TarInfo(name)
            link.type = tarfile.SYMTYPE
            link.linkname = target
            t.addfile(link)
    runtime = tmp_path / "runtime"
    downloads.extract_runtime([archive], runtime)
    assert (runtime / "bin/lib.so").read_bytes() == b"lib"
    assert not (runtime / "bin/lib.so").is_symlink()


def test_pinned_matrix_has_four_platform_backends_and_optional_security():
    for system, backend, expected in [
        ("Windows", "cpu", 1),
        ("Windows", "cuda", 2),
        ("Linux", "cpu", 1),
        ("Linux", "cuda", 1),
    ]:
        assets = downloads.select_assets(system, backend)
        runtime = [a for a in assets if a["kind"] == "runtime"]
        assert len(runtime) == expected
        assert all(a["release"] == "prism-b10777-e3de715" for a in runtime)
        models = [a for a in assets if a["kind"] == "model"]
        assert len(models) == 3
        assert all(a["revision"] in a["url"] and len(a["sha256"]) == 64 for a in models)
        assert not any("WhiteRabbit" in a["path"] for a in assets)
        assert len(downloads.select_assets(system, backend, with_security=True)) == len(assets) + 1


def test_cache_is_copied_only_after_verifying_hash(tmp_path):
    content = b"trusted bytes"
    cache = tmp_path / "cache"
    (cache / "models").mkdir(parents=True)
    (cache / "models/fixture.gguf").write_bytes(content)
    asset = asset_for(content, "https://invalid.example/no-network")
    downloads.fetch(asset, root=tmp_path / "root", asset_cache=cache, attempts=1)
    target = tmp_path / "root" / asset["path"]
    assert target.read_bytes() == content
    target.write_bytes(b"corrupted new copy")
    assert (cache / asset["path"]).read_bytes() == content


@pytest.mark.parametrize(
    "relative", ["../escape", "/absolute", "C:/escape", "models/../escape", "models\\..\\escape"]
)
def test_download_rejects_escaping_destinations_before_io(tmp_path, relative):
    with pytest.raises(ValueError, match="Unsafe"):
        downloads.fetch(
            asset_for(b"", "https://invalid.example", path=relative),
            root=tmp_path,
            attempts=1,
            opener=lambda *a, **k: (_ for _ in ()).throw(AssertionError("network must not run")),
        )


def test_backup_cleanup_denied_after_commit_warns_and_retains_previous_runtime(
    tmp_path, monkeypatch, caplog
):
    archive = tmp_path / "runtime.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("llama-server", b"new verified runtime")
    destination = tmp_path / "runtime/prism"
    destination.mkdir(parents=True)
    (destination / "old").write_bytes(b"old working runtime")
    original_rmtree = downloads.shutil.rmtree
    retained = []

    def denied(path, *args, **kwargs):
        if str(path).endswith(".previous"):
            retained.append(path)
            raise PermissionError("fixture backup locked")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(downloads.shutil, "rmtree", denied)
    assert downloads.extract_runtime([archive], destination) is None
    assert (destination / "llama-server").read_bytes() == b"new verified runtime"
    assert len(retained) == 1
    assert (retained[0] / "old").read_bytes() == b"old working runtime"
    assert str(retained[0]) in caplog.text
    assert "committed" in caplog.text.lower() and "cleanup" in caplog.text.lower()
