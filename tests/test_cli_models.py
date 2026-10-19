"""CLI uses the same operator model selection as the HTTP service."""

import json
from types import SimpleNamespace

import roi_swarm.cli as cli


async def test_cli_uses_saved_model_roles(tmp_path, monkeypatch):
    import yaml

    roles = ("router", "planner", "worker", "critic", "synthesizer")
    config = tmp_path / "config/settings.yaml"
    config.parent.mkdir()
    config.write_text(
        yaml.safe_dump(
            {
                "provider": {"base_url": "https://example.test/v1", "api_key": "fixture-only"},
                "models": {role: {"model": "base"} for role in roles},
            }
        ),
        encoding="utf-8",
    )
    chosen = {role: f"chosen-{role}" for role in roles}
    (tmp_path / "data").mkdir()
    (tmp_path / "data/model-settings.json").write_text(json.dumps({"roles": chosen}))
    captured = {}

    class Client:
        def __init__(self, **kwargs):
            self.closed = False
            captured["client"] = self

        async def close(self):
            self.closed = True

    async def invoke(*args, **kwargs):
        return {"final_answer": "fixture answer"}

    def build(settings, gateway):
        captured["models"] = {role: settings.models[role].model for role in roles}
        return SimpleNamespace(ainvoke=invoke)

    monkeypatch.setattr(cli, "AsyncOpenAI", Client)
    monkeypatch.setattr(cli, "build_graph", build)
    await cli._run("fixture", config)
    assert captured["models"] == chosen
    assert captured["client"].closed
