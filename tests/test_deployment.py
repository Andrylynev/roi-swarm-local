"""Client-server deployment contracts: env-driven config, safe network exposure."""

import pytest

from roi_swarm import server
from roi_swarm.settings import _expand_environment


def test_env_default_applies_only_when_variable_is_unset_or_empty(monkeypatch):
    monkeypatch.delenv("ROI_FIXTURE_FLAG", raising=False)
    assert _expand_environment("${ROI_FIXTURE_FLAG:-false}") == "false"
    monkeypatch.setenv("ROI_FIXTURE_FLAG", "")
    assert _expand_environment("${ROI_FIXTURE_FLAG:-false}") == "false"
    monkeypatch.setenv("ROI_FIXTURE_FLAG", "true")
    assert _expand_environment({"a": ["${ROI_FIXTURE_FLAG:-false}"]}) == {"a": ["true"]}


def test_required_env_without_default_still_fails(monkeypatch):
    monkeypatch.delenv("ROI_FIXTURE_REQUIRED", raising=False)
    with pytest.raises(ValueError, match="ROI_FIXTURE_REQUIRED"):
        _expand_environment("${ROI_FIXTURE_REQUIRED}")


def test_shipped_settings_are_cloud_only_with_sleep_enabled_by_default(monkeypatch):
    from roi_swarm.runner import PROJECT_ROOT
    from roi_swarm.settings import load_settings

    monkeypatch.setenv("ROI_PROVIDER_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("ROI_PROVIDER_API_KEY", "secret")
    for name in ("ROI_RAG_EMBEDDINGS", "ROI_SLEEP_ENABLED", "ROI_SLEEP_AUTO_IMPROVE"):
        monkeypatch.delenv(name, raising=False)
    settings = load_settings(PROJECT_ROOT / "config" / "settings.yaml")
    assert settings.rag.embeddings_enabled is False
    assert settings.sleep.enabled is True
    assert settings.sleep.auto_improve is False


def test_shipped_sleep_respects_explicit_disabled_environment(monkeypatch):
    from roi_swarm.runner import PROJECT_ROOT
    from roi_swarm.settings import load_settings

    monkeypatch.setenv("ROI_PROVIDER_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("ROI_PROVIDER_API_KEY", "fixture-only")
    monkeypatch.setenv("ROI_SLEEP_ENABLED", "false")
    settings = load_settings(PROJECT_ROOT / "config" / "settings.yaml")
    assert settings.sleep.enabled is False


@pytest.mark.parametrize("host", ["0.0.0.0", "192.0.2.10", "::"])
def test_network_bind_requires_api_key(monkeypatch, host):
    monkeypatch.delenv("ROI_SWARM_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        server.check_exposure(host)
    monkeypatch.setenv("ROI_SWARM_API_KEY", "fixture-token")
    server.check_exposure(host)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_bind_works_without_api_key(monkeypatch, host):
    monkeypatch.delenv("ROI_SWARM_API_KEY", raising=False)
    server.check_exposure(host)


def test_bind_address_reads_env(monkeypatch):
    monkeypatch.delenv("ROI_HOST", raising=False)
    monkeypatch.delenv("ROI_PORT", raising=False)
    assert server.bind_address() == ("127.0.0.1", 8765)
    monkeypatch.setenv("ROI_HOST", "0.0.0.0")
    monkeypatch.setenv("ROI_PORT", "9000")
    assert server.bind_address() == ("0.0.0.0", 9000)


def test_hermes_python_prefers_existing_venv(tmp_path):
    import os

    from roi_swarm.runner import _hermes_python

    folder, name = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    missing = _hermes_python(str(tmp_path))
    assert missing.endswith(os.path.join("venv", folder, name))
    target = tmp_path / ".venv" / folder / name
    target.parent.mkdir(parents=True)
    target.write_text("")
    assert _hermes_python(str(tmp_path)) == str(target)
