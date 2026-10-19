"""Public local build fails closed instead of inheriting a cloud dotenv route."""

from pathlib import Path

import pytest
import yaml

from local.api import validate_local_settings
from roi_swarm.config import AppSettings

TEMPLATE = Path(__file__).resolve().parents[1] / "local/settings.example.yaml"


def settings():
    return AppSettings.model_validate(yaml.safe_load(TEMPLATE.read_text(encoding="utf-8")))


def test_local_route_accepts_loopback_only():
    validate_local_settings(settings())


@pytest.mark.parametrize("url", ["https://example.com/v1", "http://0.0.0.0:8091/v1"])
def test_local_route_refuses_nonloopback_model_endpoint(url):
    config = settings()
    config.provider.base_url = url
    with pytest.raises(ValueError, match="loopback"):
        validate_local_settings(config)


@pytest.mark.parametrize("field", ["worker_context_tokens", "main_context_tokens"])
def test_local_route_refuses_context_above_native_guard_window(field):
    config = settings()
    setattr(config.context, field, 1000000)
    with pytest.raises(ValueError, match="65536"):
        validate_local_settings(config)
