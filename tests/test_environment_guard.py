"""Offline safety checks for junction-shared package environments."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from roi_swarm.environment_guard import EnvironmentGuard


def _write(root: Path, relative: str, content: bytes = b"baseline") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_capture_detects_added_deleted_and_changed_files_without_accepting_them(tmp_path):
    changed = ".venv/Lib/site-packages/editable.pth"
    deleted = ".venv/lib/python3.11/site-packages/demo.dist-info/direct_url.json"
    added = "node_modules/.package-lock.json"
    _write(tmp_path, changed, b"old\r\n")
    deleted_path = _write(tmp_path, deleted, b"{}")
    guard = EnvironmentGuard(str(tmp_path))
    assert guard.capture() is None
    assert guard.check() == []

    _write(tmp_path, changed, b"old\n")  # Raw bytes, not newline-normalized text.
    deleted_path.unlink()
    _write(tmp_path, added, b'{"lockfileVersion": 3}')

    expected = sorted([changed, deleted, added])
    assert guard.check() == expected
    assert guard.check() == expected  # A check must never advance the baseline.

    guard.capture()  # The central broker explicitly accepts a legitimate install.
    assert guard.check() == []
    _write(tmp_path, changed, b"new unbrokered change")
    assert guard.check() == [changed]


def test_directory_scan_errors_propagate_and_do_not_replace_the_baseline(tmp_path, monkeypatch):
    relative = ".venv/Lib/site-packages/editable.pth"
    path = _write(tmp_path, relative)
    guard = EnvironmentGuard(tmp_path)
    guard.capture()
    path.write_bytes(b"modified")
    original_scandir = os.scandir

    def denied(directory):
        if Path(directory) == path.parent:
            raise PermissionError("metadata directory inaccessible")
        return original_scandir(directory)

    with monkeypatch.context() as patch:
        patch.setattr(os, "scandir", denied)
        with pytest.raises(PermissionError):
            guard.check()
        with pytest.raises(PermissionError):
            guard.capture()

    assert guard.check() == [relative]


def test_lockfile_lookup_does_not_enumerate_node_modules(tmp_path, monkeypatch):
    relative = "node_modules/.package-lock.json"
    path = _write(tmp_path, relative)
    original_scandir = os.scandir

    def checked_scandir(directory):
        assert Path(directory) != path.parent, "must not enumerate the installed package tree"
        return original_scandir(directory)

    monkeypatch.setattr(os, "scandir", checked_scandir)
    guard = EnvironmentGuard(tmp_path)
    guard.capture()
    path.write_bytes(b"changed lockfile")
    assert guard.check() == [relative]


def test_pth_flags_missing_or_wave_worktree_targets_but_never_executes_imports(tmp_path):
    root = tmp_path / "main"
    shared = tmp_path / "shared source"
    shared.mkdir()
    wave = tmp_path / "roi-waves" / "abc" / "src"
    wave.mkdir(parents=True)
    missing = tmp_path / "removed" / "src"
    marker = tmp_path / "must-not-exist"
    good_name = ".venv/Lib/site-packages/valid.pth"
    missing_name = ".venv/Lib/site-packages/missing.pth"
    wave_name = ".venv/lib/python3.14/site-packages/wave.pth"
    _write(
        root,
        good_name,
        (
            f"# Ignored comment\n{shared}\nrelative-package\n"
            f"import pathlib; pathlib.Path({str(marker)!r}).touch()\n"
            "import\tos; raise RuntimeError('must not execute')\n"
        ).encode(),
    )
    _write(root, missing_name, f"{shared}\n{missing}\n{missing}\n".encode())
    _write(root, wave_name, f"{wave}\n".encode())
    guard = EnvironmentGuard(root)
    guard.capture()

    assert guard.check() == []  # Bad editables may predate the captured fingerprint.
    assert guard.broken_editables() == sorted([missing_name, wave_name])
    assert not marker.exists()
    assert guard.broken_editables() == sorted([missing_name, wave_name])


def test_direct_url_editables_decode_file_urls_and_ignore_noneditable_metadata(tmp_path):
    root = tmp_path / "main"
    shared = tmp_path / "shared # source"
    shared.mkdir()
    wave = tmp_path / "roi-waves" / "abc"
    wave.mkdir(parents=True)
    missing = tmp_path / "removed source"
    payloads = {
        "valid": {"url": shared.as_uri(), "dir_info": {"editable": True}},
        "missing": {"url": missing.as_uri(), "dir_info": {"editable": True}},
        "wave": {"url": wave.as_uri(), "dir_info": {"editable": True}},
        "ordinary": {"url": missing.as_uri(), "dir_info": {"editable": False}},
        "archive": {"url": missing.as_uri(), "archive_info": {}},
        "remote": {"url": "https://example.invalid/project", "dir_info": {"editable": True}},
        "relative": {"url": "file:relative-source", "dir_info": {"editable": True}},
        "bad-shape": [],
        "bad-dir-info": {"url": missing.as_uri(), "dir_info": []},
    }
    names = {}
    for package, payload in payloads.items():
        names[package] = f".venv/Lib/site-packages/{package}.dist-info/direct_url.json"
        _write(root, names[package], json.dumps(payload).encode())
    _write(root, ".venv/Lib/site-packages/malformed.dist-info/direct_url.json", b"not json")
    guard = EnvironmentGuard(root)
    guard.capture()

    assert guard.broken_editables() == sorted([names["missing"], names["wave"]])
    assert guard.check() == []


@pytest.mark.parametrize(
    "payload",
    [
        None,
        12,
        {"url": None, "dir_info": {"editable": True}},
        {"url": "file:///unused", "dir_info": {"editable": "true"}},
        {"url": "file://[invalid", "dir_info": {"editable": True}},
    ],
)
def test_malformed_direct_url_shapes_are_not_executed_or_treated_as_paths(tmp_path, payload):
    _write(
        tmp_path,
        ".venv/lib/python3.11/site-packages/bad.dist-info/direct_url.json",
        json.dumps(payload).encode(),
    )

    assert EnvironmentGuard(tmp_path).broken_editables() == []


@pytest.mark.skipif(os.name != "nt", reason="Windows filenames are case-insensitive")
def test_windows_uppercase_pth_names_are_checked_for_broken_editables(tmp_path):
    relative = ".venv/Lib/site-packages/EDITABLE.PTH"
    _write(tmp_path, relative, f"{tmp_path / 'removed'}\n".encode())

    assert EnvironmentGuard(tmp_path).broken_editables() == [relative]


def test_editable_detection_follows_directory_aliases_to_temporary_worktrees(tmp_path):
    target = tmp_path / "roi-waves" / "abc"
    target.mkdir(parents=True)
    alias = tmp_path / "apparently-stable-source"
    if os.name == "nt":
        subprocess.run(
            ["cmd.exe", "/c", "mklink", "/J", str(alias), str(target)],
            check=True,
            capture_output=True,
        )
    else:
        alias.symlink_to(target, target_is_directory=True)
    pth = ".venv/Lib/site-packages/alias.pth"
    direct_url = ".venv/Lib/site-packages/alias.dist-info/direct_url.json"
    _write(tmp_path, pth, f"{alias}\n".encode())
    _write(
        tmp_path,
        direct_url,
        json.dumps({"url": alias.as_uri(), "dir_info": {"editable": True}}).encode(),
    )
    guard = EnvironmentGuard(tmp_path)

    assert guard.broken_editables() == sorted([pth, direct_url])


@pytest.mark.parametrize("operation", ["capture", "check", "broken_editables"])
def test_editable_target_stat_errors_propagate(tmp_path, monkeypatch, operation):
    target = tmp_path / "inaccessible-source"
    target.mkdir()
    _write(tmp_path, ".venv/Lib/site-packages/editable.pth", f"{target}\n".encode())
    guard = EnvironmentGuard(tmp_path)
    guard.capture()
    original_stat = Path.stat

    def denied(path, *args, **kwargs):
        if path == target:
            raise PermissionError("editable source inaccessible")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied)
    if operation == "broken_editables":
        with pytest.raises(PermissionError):
            guard.broken_editables()
    else:
        # Fingerprinting must not dereference editable targets or validate their contents.
        assert getattr(guard, operation)() in (None, [])


def test_missing_environment_directories_are_empty_and_not_created(tmp_path):
    root = tmp_path / "main"
    root.mkdir()
    guard = EnvironmentGuard(root)

    guard.capture()

    assert guard.check() == []
    assert guard.broken_editables() == []
    assert list(root.iterdir()) == []


def test_only_known_metadata_files_are_scanned_and_secret_env_is_never_read(tmp_path, monkeypatch):
    tracked = [
        ".venv/Lib/site-packages/a.pth",
        ".venv/Lib/site-packages/a.dist-info/direct_url.json",
        ".venv/lib/python3.11/site-packages/b.pth",
        ".venv/lib/python3.14/site-packages/b.dist-info/direct_url.json",
        "node_modules/.package-lock.json",
    ]
    omitted = [
        ".env",
        ".venv/.env",
        ".venv/Lib/site-packages/.env",
        ".venv/Lib/site-packages/pkg/code.py",
        ".venv/Lib/site-packages/pkg/nested.pth",
        ".venv/Lib/site-packages/pkg.dist-info/METADATA",
        ".venv/Lib/site-packages/pkg.dist-info/deeper/direct_url.json",
        "node_modules/pkg/.package-lock.json",
        "package-lock.json",
    ]
    for relative in tracked + omitted:
        _write(tmp_path, relative, b"{}")
    (tmp_path / ".venv/Lib/site-packages/directory.pth").mkdir()
    original_read = Path.read_bytes
    original_scandir = os.scandir
    reads = []

    def checked_read(path):
        relative = path.relative_to(tmp_path).as_posix()
        assert relative in tracked, "attempted reading an unrelated file"
        reads.append(relative)
        return original_read(path)

    def checked_scandir(directory):
        assert Path(directory).name not in ("pkg", "deeper", "directory.pth")
        return original_scandir(directory)

    monkeypatch.setattr(Path, "read_bytes", checked_read)
    monkeypatch.setattr("os.scandir", checked_scandir)
    guard = EnvironmentGuard(tmp_path)
    guard.capture()
    assert sorted(reads) == sorted(tracked)
    for relative in omitted:
        _write(tmp_path, relative, b"ignored change")
    assert guard.check() == []
    for relative in tracked:
        _write(tmp_path, relative, b"tracked change")
    assert guard.check() == sorted(tracked)


@pytest.mark.parametrize("operation", ["capture", "check", "broken_editables"])
def test_metadata_read_errors_propagate_without_accepting_changes(tmp_path, monkeypatch, operation):
    relative = ".venv/Lib/site-packages/editable.pth"
    path = _write(tmp_path, relative)
    guard = EnvironmentGuard(tmp_path)
    guard.capture()
    path.write_bytes(b"changed")
    original_bytes = Path.read_bytes
    original_text = Path.read_text

    def denied_bytes(candidate, *args, **kwargs):
        if candidate == path:
            raise OSError("metadata unreadable")
        return original_bytes(candidate, *args, **kwargs)

    def denied_text(candidate, *args, **kwargs):
        if candidate == path:
            raise OSError("metadata unreadable")
        return original_text(candidate, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", denied_bytes)
        patch.setattr(Path, "read_text", denied_text)
        with pytest.raises(OSError):
            getattr(guard, operation)()

    assert guard.check() == [relative]
