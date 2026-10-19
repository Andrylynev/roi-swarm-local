from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from roi_swarm.config import AppSettings

# ${NAME} is required; ${NAME:-default} falls back when NAME is unset or empty.
_ENV_PATTERN = re.compile(r"^\$\{([A-Z0-9_]+)(?::-(.*))?\}$")


def load_settings(path: str | Path) -> AppSettings:
    config_path = Path(path)
    load_dotenv(config_path.parent.parent / ".env")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return AppSettings.model_validate(_expand_environment(raw))


def load_effective_settings(path: str | Path) -> AppSettings:
    """Load base configuration plus the operator's model-only overrides."""
    from roi_swarm.model_settings import ModelSelection

    settings = load_settings(path)
    selection_path = Path(path).parent.parent / "data" / "model-settings.json"
    if selection_path.exists():
        selection = ModelSelection.model_validate_json(selection_path.read_text(encoding="utf-8"))
        for role, model_id in selection.roles.items():
            settings.models[role].model = model_id
    return settings


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        match = _ENV_PATTERN.fullmatch(value)
        if match:
            name, default = match.group(1), match.group(2)
            resolved = os.getenv(name)
            if default is not None:
                return resolved or default
            if resolved is None:
                raise ValueError(f"Required environment variable is missing: {name}")
            return resolved
    return value
