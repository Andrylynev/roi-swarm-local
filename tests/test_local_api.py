"""The local fork must not inherit cloud endpoints or the production service."""

from pathlib import Path

from roi_swarm.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]


def test_local_settings_are_isolated_and_inference_only():
    settings = load_settings(ROOT / "local/settings.example.yaml")
    assert settings.provider.base_url == "http://127.0.0.1:8091/v1"
    assert settings.provider.api_mode == "chat_completions"
    assert settings.provider.timeout_seconds is None
    assert settings.models["planner"].model == "bonsai-9b"
    assert settings.models["synthesizer"].model == "bonsai-9b"
    assert settings.models["worker"].model == "bonsai-4b"
    assert settings.context.main_context_tokens == 65536
    assert settings.context.worker_context_tokens == 65536
    assert not settings.sleep.enabled
    assert not settings.sleep.auto_improve
    assert settings.hermes_workers.default_cwd == ""
    assert settings.hermes_workers.maximum_concurrency == 2


def test_local_inference_presets_are_all_local_files():
    from local.presets import model_presets

    models = model_presets(ROOT)
    assert set(models) == {"bonsai-9b", "bonsai-4b", "bonsai-2b", "whiterabbit-7b"}
    for model in models.values():
        assert Path(model["model"]).is_relative_to(ROOT)
        assert int(model["ctx-size"]) // int(model["parallel"]) == 65536
        assert model["jinja"] == "true"
        assert model["load-on-startup"] == "false"
    assert models["bonsai-9b"]["spec-type"] == "draft-mtp"
    assert models["bonsai-4b"]["spec-type"] == "none"
    assert models["bonsai-4b"]["parallel"] == "2"
    assert models["bonsai-4b"]["ubatch-size"] == "256"
    assert models["bonsai-2b"]["parallel"] == "2"
    assert models["whiterabbit-7b"]["n-gpu-layers"] == "99"
