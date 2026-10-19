"""An owned process may exit between its liveness and metadata queries."""

from types import SimpleNamespace

import pytest

from local import manage
from local.manage import LifecycleError, identity_from_handle


class FakeFunction:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


@pytest.mark.parametrize("stage", ["times", "image"])
@pytest.mark.parametrize("exits", [False, True])
def test_metadata_query_failure_rechecks_process_exit(stage, exits):
    failed = False

    def query_times(*args):
        nonlocal failed
        if stage == "times":
            failed = True
            return 0
        return 1

    def query_image(*args):
        nonlocal failed
        failed = True
        return 0

    kernel = SimpleNamespace(
        WaitForSingleObject=FakeFunction(lambda *args: 0 if failed and exits else 0x102),
        GetProcessTimes=FakeFunction(query_times),
        QueryFullProcessImageNameW=FakeFunction(query_image),
    )
    if exits:
        assert identity_from_handle(123, (kernel, "pinned process")) is None
    else:
        with pytest.raises(LifecycleError, match="Cannot read"):
            identity_from_handle(123, (kernel, "pinned process"))


@pytest.mark.parametrize("wait_status", [0xFFFFFFFF, 0x80, 17])
def test_failed_initial_liveness_check_is_not_reported_as_exited(wait_status):
    kernel = SimpleNamespace(WaitForSingleObject=FakeFunction(lambda *args: wait_status))
    with pytest.raises(LifecycleError, match="state"):
        identity_from_handle(123, (kernel, "pinned process"))


@pytest.mark.skipif(manage.os.name != "nt", reason="Win32 process handle")
def test_open_process_access_denied_is_not_reported_as_missing(monkeypatch):
    kernel = SimpleNamespace(
        OpenProcess=FakeFunction(lambda *args: None),
        CloseHandle=FakeFunction(lambda *args: 1),
    )
    monkeypatch.setattr(manage.ctypes, "WinDLL", lambda *a, **k: kernel)
    monkeypatch.setattr(manage.ctypes, "get_last_error", lambda: 5)
    with pytest.raises(LifecycleError, match="inspect"), manage.process_handle(123):
        pytest.fail("access denied must not be presented as a missing process")


@pytest.mark.skipif(manage.os.name != "nt", reason="Win32 process handle")
def test_successful_tree_kill_waits_on_pinned_handle_without_requerying_image(monkeypatch):
    from contextlib import nullcontext

    record = {"pid": 123, "created": "42", "image": "D:/local/llama-server.exe"}
    service = {"command": [record["image"]]}
    calls = 0

    def identity(*args):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise LifecycleError("image unavailable while termination is pending")
        return record

    kernel = SimpleNamespace(WaitForSingleObject=FakeFunction(lambda *args: 0))
    monkeypatch.setattr(manage, "ownership", lambda *a, **k: {"owned": True})
    monkeypatch.setattr(manage, "process_handle", lambda pid: nullcontext((kernel, "handle")))
    monkeypatch.setattr(manage, "identity_from_handle", identity)
    monkeypatch.setattr(manage.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    assert manage.terminate_owned(record, service)
    assert calls == 1
