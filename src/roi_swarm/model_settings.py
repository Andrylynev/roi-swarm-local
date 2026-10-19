"""Validated model-only overrides; provider credentials never enter this file."""

from __future__ import annotations

import json
import os
import unicodedata
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ModelRole = Literal["router", "planner", "worker", "critic", "synthesizer"]
MODEL_ROLES = ("router", "planner", "worker", "critic", "synthesizer")


class ModelSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    roles: dict[ModelRole, str] = Field(min_length=5, max_length=5)

    @field_validator("roles")
    @classmethod
    def validate_roles(cls, roles: dict[str, str]) -> dict[str, str]:
        for value in roles.values():
            if not value or value != value.strip() or len(value) > 256:
                raise ValueError("Model IDs must be non-empty, trimmed and at most 256 characters")
            if any(char.isspace() or unicodedata.category(char).startswith("C") for char in value):
                raise ValueError("Model IDs cannot contain whitespace or control characters")
        return roles


def save_selection(path: Path, selection: ModelSelection) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(selection.model_dump(), ensure_ascii=False, indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
