"""Autostart contracts without changing Windows registry or starting a server."""


import pytest

from roi_swarm import startup


class Lock:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.mark.parametrize("already_running", [False, True])
def test_startup_keeps_api_key_logs_and_never_duplicates(
    tmp_path, monkeypatch, already_running
):
    events = []
    lock = Lock()
    monkeypatch.setenv("ROI_SWARM_API_KEY", "fixture-only")
    monkeypatch.setattr(startup, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(startup, "_acquire_single_instance", lambda: lock)
    monkeypatch.setattr(startup, "_service_running", lambda: already_running)

    def run():
        import os

        # Client installs protect /v1 and the dashboard: the key must reach the server.
        assert os.environ["ROI_SWARM_API_KEY"] == "fixture-only"
        events.append("started")
        print("fixture startup reached server")

    monkeypatch.setattr(startup, "_run_server", run)
    assert startup.main() == 0
    assert events == ([] if already_running else ["started"])
    assert lock.closed
    assert (tmp_path / "data/diagnostics/autostart.log").exists()


@pytest.mark.parametrize("restart", [False, True])
def test_startup_relaunches_after_guard_release_only_on_restart(tmp_path, monkeypatch, restart):
    events = []

    class Guard:
        def close(self):
            events.append("guard_closed")

    monkeypatch.setattr(startup, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(startup, "_acquire_single_instance", Guard)
    monkeypatch.setattr(startup, "_service_running", lambda: False)
    monkeypatch.setattr(startup, "_run_server", lambda: restart)
    monkeypatch.setattr(startup, "_relaunch", lambda: events.append("relaunched"))
    assert startup.main() == 0
    assert events == (["guard_closed", "relaunched"] if restart else ["guard_closed"])


def test_unsupervised_server_ignores_restart_request(monkeypatch):
    from roi_swarm import server

    class Fake:
        should_exit = False

    fake = Fake()
    monkeypatch.setattr(server, "_server", fake)
    monkeypatch.setattr(server, "_supervised", False)
    assert server.request_restart() is False
    assert fake.should_exit is False
    monkeypatch.setattr(server, "_supervised", True)
    assert server.request_restart() is True
    assert fake.should_exit is True


def test_second_startup_process_does_not_start_server(tmp_path, monkeypatch):
    monkeypatch.setattr(startup, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(startup, "_acquire_single_instance", lambda: None)
    monkeypatch.setattr(startup, "_run_server", lambda: pytest.fail("duplicate server"))
    assert startup.main() == 0
