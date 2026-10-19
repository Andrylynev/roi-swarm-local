"""Concurrent artifact writes must own separate staging files."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest

from roi_swarm import artifacts
from roi_swarm.artifacts import ArtifactStore


@pytest.mark.parametrize("existing", [False, True])
def test_concurrent_atomic_writes_do_not_share_staging_files(tmp_path, monkeypatch, existing):
    destination = tmp_path / "artifact.txt"
    if existing:
        destination.write_bytes(b"original")
    ready = Barrier(2)
    original_replace = Path.replace
    contents = ["first\r\nα\n", "second\rβ\n"]
    staging_files = {}
    staging_lock = Lock()

    def synchronized_replace(source, target):
        if target == destination:
            with staging_lock:
                first_attempt = source not in staging_files
                if first_attempt:
                    staging_files[source] = source.read_bytes()
            # Synchronize only the first attempt; one writer may need a retry.
            if first_attempt:
                ready.wait(timeout=5)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", synchronized_replace)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(ArtifactStore._atomic_write, destination, text) for text in contents]
        for future in futures:
            assert future.result(timeout=10) is None

    expected = [text.encode("utf-8") for text in contents]
    assert len(staging_files) == 2
    assert set(staging_files.values()) == set(expected)
    assert destination.read_bytes() in expected
    assert list(tmp_path.iterdir()) == [destination]


def test_failed_replacement_removes_only_its_staging_file(tmp_path, monkeypatch):
    destination = tmp_path / "artifact.txt"
    destination.write_bytes(b"original")
    unrelated = tmp_path / ".another-writer.tmp"
    unrelated.write_bytes(b"keep")
    original_replace = Path.replace

    def failed_replace(source, target):
        if target == destination:
            raise OSError("fixture replacement failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", failed_replace)
    with pytest.raises(OSError, match="fixture replacement failure"):
        ArtifactStore._atomic_write(destination, "replacement\r\n")

    assert destination.read_bytes() == b"original"
    assert unrelated.read_bytes() == b"keep"
    assert set(tmp_path.iterdir()) == {destination, unrelated}


@pytest.mark.parametrize("winerror", [5, 32, 33])
@pytest.mark.parametrize("failures", [1, 4])
def test_windows_transient_replacement_retries_owned_staging(
    tmp_path, monkeypatch, winerror, failures
):
    destination = tmp_path / "artifact.txt"
    destination.write_bytes(b"original")
    unrelated = tmp_path / ".another-writer.tmp"
    unrelated.write_bytes(b"keep")
    content = "replacement\r\nα\nβ\r"
    expected = content.encode("utf-8")
    original_replace = Path.replace
    attempts = []
    delays = []
    error = PermissionError("fixture transient Windows failure")
    error.winerror = winerror

    def transient_replace(source, target):
        assert target == destination
        assert source.read_bytes() == expected
        assert destination.read_bytes() == b"original"
        assert unrelated.read_bytes() == b"keep"
        attempts.append(source)
        if len(attempts) <= failures:
            raise error
        return original_replace(source, target)

    # Patch module-local platform/time bindings, not process-wide os.name.
    monkeypatch.setattr(artifacts, "os", SimpleNamespace(name="nt"), raising=False)
    monkeypatch.setattr(artifacts, "time", SimpleNamespace(sleep=delays.append), raising=False)
    monkeypatch.setattr(Path, "replace", transient_replace)

    ArtifactStore._atomic_write(destination, content)

    assert len(attempts) == failures + 1
    assert len(set(attempts)) == 1
    assert delays == [0.01 * 2**attempt for attempt in range(failures)]
    assert destination.read_bytes() == expected
    assert unrelated.read_bytes() == b"keep"
    assert set(tmp_path.iterdir()) == {destination, unrelated}


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_persistent_windows_replacement_failure_is_bounded(tmp_path, monkeypatch, winerror):
    destination = tmp_path / "artifact.txt"
    destination.write_bytes(b"original")
    unrelated = tmp_path / ".another-writer.tmp"
    unrelated.write_bytes(b"keep")
    attempts = []
    errors = []
    delays = []

    def failed_replace(source, target):
        assert target == destination
        assert source.read_bytes() == b"replacement"
        assert destination.read_bytes() == b"original"
        assert unrelated.read_bytes() == b"keep"
        attempts.append(source)
        error = PermissionError(f"persistent failure {len(attempts)}")
        error.winerror = winerror
        errors.append(error)
        # Fail fast rather than hanging the suite if the retry bound regresses.
        assert len(attempts) <= 5
        raise error

    monkeypatch.setattr(artifacts, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(artifacts, "time", SimpleNamespace(sleep=delays.append))
    monkeypatch.setattr(Path, "replace", failed_replace)

    with pytest.raises(PermissionError) as raised:
        ArtifactStore._atomic_write(destination, "replacement")

    assert len(attempts) == 5
    assert raised.value is errors[-1]
    assert len(set(attempts)) == 1
    assert delays == [0.01, 0.02, 0.04, 0.08]
    assert destination.read_bytes() == b"original"
    assert unrelated.read_bytes() == b"keep"
    assert set(tmp_path.iterdir()) == {destination, unrelated}


@pytest.mark.parametrize(
    ("platform", "winerror"),
    [("posix", code) for code in (5, 32, 33, None)]
    + [("nt", code) for code in (None, 2, 3, 19, 112)],
)
def test_non_retryable_replacement_failure_propagates_immediately(
    tmp_path, monkeypatch, platform, winerror
):
    destination = tmp_path / "artifact.txt"
    destination.write_bytes(b"original")
    unrelated = tmp_path / ".another-writer.tmp"
    unrelated.write_bytes(b"keep")
    attempts = []
    delays = []
    error = PermissionError(13, "fixture non-retryable replacement failure")
    if winerror is not None:
        error.winerror = winerror

    def failed_replace(source, target):
        assert target == destination
        attempts.append(source)
        raise error

    monkeypatch.setattr(artifacts, "os", SimpleNamespace(name=platform))
    monkeypatch.setattr(artifacts, "time", SimpleNamespace(sleep=delays.append))
    monkeypatch.setattr(Path, "replace", failed_replace)

    with pytest.raises(OSError) as raised:
        ArtifactStore._atomic_write(destination, "replacement")

    assert raised.value is error
    assert len(attempts) == 1
    assert delays == []
    assert destination.read_bytes() == b"original"
    assert unrelated.read_bytes() == b"keep"
    assert set(tmp_path.iterdir()) == {destination, unrelated}


def test_hard_failure_after_transient_failure_stops_retrying(tmp_path, monkeypatch):
    destination = tmp_path / "artifact.txt"
    destination.write_bytes(b"original")
    transient = PermissionError("fixture transient failure")
    transient.winerror = 5
    permanent = OSError("fixture disk full")
    permanent.winerror = 112
    attempts = []
    delays = []

    def failed_replace(source, target):
        assert target == destination
        attempts.append(source)
        if len(attempts) == 1:
            raise transient
        raise permanent

    monkeypatch.setattr(artifacts, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(artifacts, "time", SimpleNamespace(sleep=delays.append))
    monkeypatch.setattr(Path, "replace", failed_replace)

    with pytest.raises(OSError) as raised:
        ArtifactStore._atomic_write(destination, "replacement")

    assert raised.value is permanent
    assert len(attempts) == 2
    assert len(set(attempts)) == 1
    assert delays == [0.01]
    assert destination.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [destination]


def test_staging_write_failure_does_not_retry_and_cleans_partial_file(tmp_path, monkeypatch):
    destination = tmp_path / "artifact.txt"
    destination.write_bytes(b"original")
    unrelated = tmp_path / ".another-writer.tmp"
    unrelated.write_bytes(b"keep")
    original_write = Path.write_bytes
    error = PermissionError("fixture staging write failure")
    error.winerror = 5
    writes = []
    delays = []

    def failed_write(source, data):
        writes.append(source)
        original_write(source, b"partial")
        raise error

    def unexpected_replace(source, target):
        pytest.fail("Must not replace after a failed staging write")

    monkeypatch.setattr(artifacts, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(artifacts, "time", SimpleNamespace(sleep=delays.append))
    monkeypatch.setattr(Path, "write_bytes", failed_write)
    monkeypatch.setattr(Path, "replace", unexpected_replace)

    with pytest.raises(PermissionError) as raised:
        ArtifactStore._atomic_write(destination, "replacement")

    assert raised.value is error
    assert len(writes) == 1
    assert delays == []
    assert destination.read_bytes() == b"original"
    assert unrelated.read_bytes() == b"keep"
    assert set(tmp_path.iterdir()) == {destination, unrelated}
