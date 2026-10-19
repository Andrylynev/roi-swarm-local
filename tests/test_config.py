import pytest
from pydantic import ValidationError

from roi_swarm.config import AppSettings, ModelRoleConfig, ProviderConfig, SwarmConfig


def test_roles_are_resolved_independently_from_provider() -> None:
    settings = AppSettings(
        provider=ProviderConfig(base_url="https://example.test/v1", api_key="secret"),
        models={
            "router": ModelRoleConfig(model="fast-model"),
            "planner": ModelRoleConfig(model="strong-model"),
            "worker": ModelRoleConfig(model="fast-model"),
            "critic": ModelRoleConfig(model="fast-model"),
            "synthesizer": ModelRoleConfig(model="strong-model"),
        },
    )

    assert settings.model_for("worker").model == "fast-model"
    assert settings.model_for("planner").model == "strong-model"


def test_base_url_is_normalized_without_trailing_slash() -> None:
    provider = ProviderConfig(base_url="https://example.test/v1/", api_key="secret")

    assert provider.base_url == "https://example.test/v1"


@pytest.mark.parametrize(
    "values",
    [
        {"minimum_workers": 9},
        {"maximum_workers": 201},
        {"minimum_workers": 50, "maximum_workers": 20},
        {"maximum_concurrency": 0},
        {"expansion_batch": 0},
        {"maximum_rounds": 0},
    ],
)
def test_swarm_config_rejects_invalid_limits(values: dict) -> None:
    with pytest.raises(ValidationError):
        SwarmConfig(**values)


def test_provider_accepts_chat_completions_mode() -> None:
    provider = ProviderConfig(
        base_url="https://example.test/v1",
        api_key="secret",
        api_mode="chat_completions",
    )

    assert provider.api_mode == "chat_completions"


def test_provider_timeout_is_disabled_by_default() -> None:
    provider = ProviderConfig(base_url="https://example.test/v1", api_key="secret")
    assert provider.timeout_seconds is None


def test_shipped_settings_use_opus_for_main_roles_without_timeout() -> None:
    import os

    from roi_swarm.runner import PROJECT_ROOT
    from roi_swarm.settings import load_settings

    os.environ.setdefault("ROI_PROVIDER_BASE_URL", "https://example.test/v1")
    os.environ.setdefault("ROI_PROVIDER_API_KEY", "secret")
    settings = load_settings(PROJECT_ROOT / "config" / "settings.yaml")
    assert settings.models["planner"].model == "claude-opus-5-5"
    assert settings.models["synthesizer"].model == "claude-opus-5-5"
    assert settings.provider.timeout_seconds is None
    assert settings.provider.api_mode == "chat_completions"
    for role in ("router", "worker", "critic"):
        assert settings.models[role].model == "gpt-6-luna"


def test_settings_require_every_runtime_model_role() -> None:
    with pytest.raises(ValidationError):
        AppSettings(
            provider=ProviderConfig(base_url="https://example.test/v1", api_key="secret"),
            models={
                "router": ModelRoleConfig(model="fast-model"),
                "planner": ModelRoleConfig(model="strong-model"),
                "worker": ModelRoleConfig(model="fast-model"),
                "synthesizer": ModelRoleConfig(model="strong-model"),
            },
        )
