"""The installed entry point always uses local, never cloud, commands."""

from local.cli import main


def test_status_is_forwarded_to_local_lifecycle(monkeypatch):
    seen = []
    monkeypatch.setattr("local.manage.main", lambda argv: seen.append(argv) or 0)
    assert main(["status"]) == 0
    assert seen == [["status"]]


def test_chat_uses_the_dedicated_hermes_home(monkeypatch):
    seen = []
    monkeypatch.setattr("local.run_hermes.main", lambda argv=None: seen.append(argv) or 0)
    assert main(["chat", "--oneshot", "-q", "test"]) == 0
    assert seen == [["--oneshot", "-q", "test"]]
