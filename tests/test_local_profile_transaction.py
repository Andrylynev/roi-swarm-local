"""Offline profile transactions: never inspect or mutate installed services."""

import socket
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from local import manage, profiles


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    root = tmp_path / "ROI local"
    settings = root / "local/settings.yaml"
    settings.parent.mkdir(parents=True)
    settings.write_bytes(
        b"# Keep exact original bytes on failure\r\n"
        b"provider: {base_url: 'http://127.0.0.1:8091/v1', api_key: local-only}\r\n"
        b"models: {}\r\nswarm: {}\r\nhermes_workers: {}\r\n"
    )
    diagnostics = root / "data/diagnostics"
    diagnostics.mkdir(parents=True)
    override = root / "data/model-settings.json"
    override.write_bytes(b'{"worker": "previous-model"}\r\n')
    state = SimpleNamespace(
        root=root,
        settings=settings,
        override=override,
        locked=False,
        ports={},
        blocked=set(),
        events=[],
    )

    @contextmanager
    def lock(port=8776):
        assert port == 8776, "Profile changes must share the launcher's fixed lock"
        assert not state.locked
        state.locked = True
        state.events.append("lock")
        try:
            yield
        finally:
            state.events.append("unlock")
            state.locked = False

    read_state = manage.read_state

    def checked_state(path):
        assert state.locked
        assert Path(path) == root
        state.events.append("state")
        return read_state(path)

    def listener(port):
        assert state.locked
        state.events.append(("listener", port))
        return state.ports.get(port)

    class PortProbe:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def setsockopt(self, *args):
            pass

        def bind(self, address):
            assert state.locked
            host, port = address
            assert host == "127.0.0.1"
            state.events.append(("bind", port))
            if port in state.blocked:
                raise PermissionError("port excluded or blocked")

        def close(self):
            state.events.append("close-probe")

    def no_connection(*args, **kwargs):
        raise urllib.error.URLError(ConnectionRefusedError("fixture service stopped"))

    monkeypatch.setattr(profiles, "ROOT", root)
    monkeypatch.setattr(manage, "lifecycle_lock", lock)
    monkeypatch.setattr(manage, "read_state", checked_state)
    monkeypatch.setattr(manage, "listener_pid", listener)
    monkeypatch.setattr(manage, "process_identity", lambda pid: None)
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: PortProbe())
    monkeypatch.setattr(urllib.request, "urlopen", no_connection)
    return state


def snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("failure", [401, 503, "timeout", "url-timeout"])
def test_live_unhealthy_service_cannot_be_treated_as_stopped(sandbox, monkeypatch, failure):
    sandbox.ports[8775] = 321
    before = snapshot(sandbox.root)

    def unhealthy(*args, **kwargs):
        if isinstance(failure, int):
            raise urllib.error.HTTPError(args[0], failure, "unhealthy", {}, None)
        if failure == "timeout":
            raise TimeoutError("health timed out")
        raise urllib.error.URLError(TimeoutError("health timed out"))

    monkeypatch.setattr(urllib.request, "urlopen", unhealthy)
    with pytest.raises(RuntimeError, match="[Ss]top|[Pp]ort|[Ll]istener"):
        profiles.apply_profile("fast")
    assert snapshot(sandbox.root) == before


@pytest.mark.parametrize("port", [8091, 8775])
def test_unavailable_port_fails_closed_without_writes(sandbox, port):
    sandbox.blocked.add(port)
    before = snapshot(sandbox.root)
    with pytest.raises(RuntimeError, match="[Pp]ort|stopped"):
        profiles.apply_profile("fast", root=sandbox.root)
    assert snapshot(sandbox.root) == before


@pytest.mark.parametrize("port", [8091, 8775])
def test_unknown_native_listener_ownership_fails_closed(sandbox, monkeypatch, port):
    before = snapshot(sandbox.root)

    def unknown(probed):
        if probed == port:
            raise manage.LifecycleError("Cannot inspect TCP listener ownership")
        return None

    monkeypatch.setattr(manage, "listener_pid", unknown)
    with pytest.raises(manage.LifecycleError, match="ownership"):
        profiles.apply_profile("balanced", root=sandbox.root)
    assert snapshot(sandbox.root) == before


@pytest.mark.parametrize("live", [True, False])
def test_retained_process_record_is_not_proof_of_stop(sandbox, monkeypatch, live):
    command = manage.build_plan(sandbox.root)["services"]["api"]["command"]
    identity = {"pid": 902, "created": "12345", "image": command[0]}
    record = {**identity, "command": command}
    manage.write_state(
        sandbox.root, {"version": 1, "root": str(sandbox.root), "services": {"api": record}}
    )
    monkeypatch.setattr(manage, "process_identity", lambda pid: identity if live else None)
    before = snapshot(sandbox.root)
    with pytest.raises(manage.LifecycleError, match="Stop"):
        profiles.apply_profile("fast", root=sandbox.root)
    assert snapshot(sandbox.root) == before


def test_null_process_record_fails_closed(sandbox):
    manage.write_state(
        sandbox.root, {"version": 1, "root": str(sandbox.root), "services": {"api": None}}
    )
    before = snapshot(sandbox.root)
    with pytest.raises(manage.LifecycleError, match="Stop|state"):
        profiles.apply_profile("fast", root=sandbox.root)
    assert snapshot(sandbox.root) == before


def test_unreadable_lifecycle_state_is_not_proof_of_stop(sandbox, monkeypatch):
    state_path = sandbox.root / "data/local-services.json"
    state_path.write_text("{}", encoding="utf-8")
    before = snapshot(sandbox.root)
    original_exists = Path.exists
    original_stat = Path.stat

    def hidden(path, *args, **kwargs):
        return False if path == state_path else original_exists(path, *args, **kwargs)

    def denied(path, *args, **kwargs):
        if path == state_path:
            raise PermissionError("lifecycle state access denied")
        return original_stat(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "exists", hidden)
        patch.setattr(Path, "stat", denied)
        with pytest.raises(PermissionError, match="state access"):
            profiles.apply_profile("fast", root=sandbox.root)
    assert snapshot(sandbox.root) == before


def test_malformed_lifecycle_state_fails_closed(sandbox):
    (sandbox.root / "data/local-services.json").write_text("{broken", encoding="utf-8")
    before = snapshot(sandbox.root)
    with pytest.raises(manage.LifecycleError, match="state"):
        profiles.apply_profile("fast", root=sandbox.root)
    assert snapshot(sandbox.root) == before


def test_stopped_switch_checks_and_writes_under_same_lifecycle_lock(sandbox, monkeypatch):
    original_replace = Path.replace
    original_unlink = Path.unlink
    original_stage = profiles._write_unique
    writes = []

    def checked_replace(path, target):
        assert sandbox.locked
        assert "close-probe" not in sandbox.events
        writes.append(Path(target))
        return original_replace(path, target)

    def checked_unlink(path, *args, **kwargs):
        assert sandbox.locked
        assert "close-probe" not in sandbox.events
        return original_unlink(path, *args, **kwargs)

    def checked_stage(*args, **kwargs):
        assert sandbox.locked
        assert "close-probe" not in sandbox.events
        return original_stage(*args, **kwargs)

    monkeypatch.setattr(Path, "replace", checked_replace)
    monkeypatch.setattr(Path, "unlink", checked_unlink)
    monkeypatch.setattr(profiles, "_write_unique", checked_stage)
    profiles.apply_profile("fast", root=sandbox.root)
    assert sandbox.settings in writes
    assert sandbox.events[0] == "lock"
    assert sandbox.events[-1] == "unlock"
    assert {("listener", 8091), ("listener", 8775)} <= set(sandbox.events)
    assert {("bind", 8091), ("bind", 8775)} <= set(sandbox.events)
    assert not sandbox.locked


def test_installed_profile_honors_backend_ports_and_lock(sandbox, monkeypatch):
    import yaml
    from test_local_manage_portable import install_record

    install_record(sandbox.root)
    original = manage.lifecycle_lock
    observed = []

    def installed_lock(port=8776):
        observed.append(port)
        return original()

    monkeypatch.setattr(manage, "lifecycle_lock", installed_lock)
    profiles.apply_profile("fast", root=sandbox.root)
    updated = yaml.safe_load(sandbox.settings.read_bytes())
    assert updated["swarm"]["maximum_concurrency"] == 1
    assert updated["provider"]["base_url"] == "http://127.0.0.1:18091/v1"
    assert observed == [18776]
    assert {("bind", 18091), ("bind", 18775)} <= set(sandbox.events)


def test_repeated_switches_preserve_unique_override_backups(sandbox):
    diagnostics = sandbox.root / "data/diagnostics"
    legacy = diagnostics / "model-settings.before-profile.json"
    legacy.write_bytes(b"previous recovery copy")
    originals = [sandbox.override.read_bytes(), b'{"worker": "second-model"}\n']
    profiles.apply_profile("fast", root=sandbox.root)
    sandbox.override.write_bytes(originals[1])
    profiles.apply_profile("balanced", root=sandbox.root)
    backups = list(diagnostics.glob("model-settings.before-profile*.json"))
    assert len(backups) == 3
    assert legacy.read_bytes() == b"previous recovery copy"
    assert sorted(path.read_bytes() for path in backups if path != legacy) == sorted(originals)
    assert not sandbox.override.exists()


def test_backup_permission_error_keeps_original_settings_and_override(sandbox, monkeypatch):
    before = snapshot(sandbox.root)
    original_mkstemp = profiles.tempfile.mkstemp

    def denied(*args, **kwargs):
        if Path(kwargs.get("dir", ".")) == sandbox.root / "data/diagnostics":
            raise PermissionError("backup directory denied")
        return original_mkstemp(*args, **kwargs)

    monkeypatch.setattr(profiles.tempfile, "mkstemp", denied)
    with pytest.raises(PermissionError, match="backup directory"):
        profiles.apply_profile("fast", root=sandbox.root)
    assert snapshot(sandbox.root) == before


def test_unreadable_override_cannot_be_treated_as_absent(sandbox, monkeypatch):
    before = snapshot(sandbox.root)
    original_exists = Path.exists
    original_read = Path.read_bytes

    def hidden(path, *args, **kwargs):
        # Python 3.14's Path.exists suppresses access errors and returns False.
        return False if path == sandbox.override else original_exists(path, *args, **kwargs)

    def denied(path):
        if path == sandbox.override:
            raise PermissionError("override read denied")
        return original_read(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "exists", hidden)
        patch.setattr(Path, "read_bytes", denied)
        with pytest.raises(PermissionError, match="override read"):
            profiles.apply_profile("fast", root=sandbox.root)
    assert snapshot(sandbox.root) == before


def test_config_replace_failure_preserves_both_originals(sandbox, monkeypatch):
    settings = sandbox.settings.read_bytes()
    override = sandbox.override.read_bytes()
    original_replace = Path.replace

    def denied(path, target):
        if Path(target) == sandbox.settings:
            raise PermissionError("settings replace denied")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", denied)
    with pytest.raises(PermissionError, match="settings replace"):
        profiles.apply_profile("security", root=sandbox.root)
    assert sandbox.settings.read_bytes() == settings
    assert sandbox.override.read_bytes() == override
    assert not list(sandbox.settings.parent.glob("*.tmp"))


def test_override_removal_failure_rolls_back_settings_bytes(sandbox, monkeypatch):
    settings = sandbox.settings.read_bytes()
    override = sandbox.override.read_bytes()
    original_unlink = Path.unlink

    def denied(path, *args, **kwargs):
        if path == sandbox.override:
            raise PermissionError("override removal denied")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", denied)
    with pytest.raises(PermissionError, match="override removal"):
        profiles.apply_profile("fast", root=sandbox.root)
    assert sandbox.settings.read_bytes() == settings
    assert sandbox.override.read_bytes() == override
    backups = list((sandbox.root / "data/diagnostics").glob("model-settings.before-profile*.json"))
    assert [path.read_bytes() for path in backups] == [override]
    assert not list(sandbox.settings.parent.glob("*.tmp"))


@pytest.mark.parametrize(
    "content",
    [
        b"null\n",
        b"models: []\n",
        b"models: {worker: null}\n",
        b"provider: {base_url: 'http://127.0.0.1:8091/v1', api_key: local-only, max_retries: -1}\n",
        b"provider: {base_url: 'http://127.0.0.1:8091/v1', api_key: local-only}\n"
        b"context: {recent_messages: -1}\n",
    ],
)
def test_malformed_profile_is_validated_before_any_write(sandbox, monkeypatch, content):
    sandbox.settings.write_bytes(content)
    before = snapshot(sandbox.root)

    def no_staging(*args, **kwargs):
        pytest.fail("Malformed profile reached the first transaction write")

    monkeypatch.setattr(profiles, "_write_unique", no_staging)
    with pytest.raises(ValueError):
        profiles.apply_profile("fast", root=sandbox.root)
    assert snapshot(sandbox.root) == before


def test_backup_flush_failure_keeps_originals_and_removes_partial_backup(sandbox, monkeypatch):
    before = snapshot(sandbox.root)

    def denied(fd):
        raise PermissionError("backup flush denied")

    monkeypatch.setattr(profiles.os, "fsync", denied)
    with pytest.raises(PermissionError, match="backup flush"):
        profiles.apply_profile("balanced", root=sandbox.root)
    assert snapshot(sandbox.root) == before


def test_failed_rollback_keeps_recoverable_original_and_names_it(sandbox, monkeypatch):
    settings = sandbox.settings.read_bytes()
    override = sandbox.override.read_bytes()
    original_unlink = Path.unlink
    original_replace = Path.replace

    def denied_unlink(path, *args, **kwargs):
        if path == sandbox.override:
            raise PermissionError("override removal denied")
        return original_unlink(path, *args, **kwargs)

    def denied_restore(path, target):
        if path.name.startswith("settings.before-profile-"):
            raise PermissionError("settings rollback denied")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "unlink", denied_unlink)
    monkeypatch.setattr(Path, "replace", denied_restore)
    with pytest.raises(RuntimeError, match="rollback failed") as error:
        profiles.apply_profile("fast", root=sandbox.root)
    recoveries = list(sandbox.settings.parent.glob("settings.before-profile-*.tmp"))
    assert len(recoveries) == 1
    assert recoveries[0].read_bytes() == settings
    assert str(recoveries[0]) in str(error.value)
    assert sandbox.override.read_bytes() == override
    backups = list((sandbox.root / "data/diagnostics").glob("model-settings.before-profile*.json"))
    assert [path.read_bytes() for path in backups] == [override]


def test_switch_without_overrides_needs_no_diagnostics_directory(sandbox):
    sandbox.override.unlink()
    (sandbox.root / "data/diagnostics").rmdir()
    profiles.apply_profile("security", root=sandbox.root)
    assert not sandbox.override.exists()
    assert not (sandbox.root / "data/diagnostics").exists()
    assert not list(sandbox.settings.parent.glob("*.tmp"))


def test_busy_lifecycle_lock_refuses_before_reading_or_writing(sandbox, monkeypatch):
    before = snapshot(sandbox.root)

    @contextmanager
    def busy():
        raise manage.LifecycleError("another lifecycle operation is running")
        yield

    monkeypatch.setattr(manage, "lifecycle_lock", busy)
    with pytest.raises(manage.LifecycleError, match="another lifecycle"):
        profiles.apply_profile("fast", root=sandbox.root)
    assert snapshot(sandbox.root) == before
    assert sandbox.events == []
