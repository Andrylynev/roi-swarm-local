"""Offline resolution using the installed Hermes code, not a guessed config schema."""

import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HERMES_HOME"] = str(ROOT / "hermes")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
source = ROOT / "runtime/hermes-agent"
sys.path.insert(0, str(source))

from agent.agent_init import _resolve_context_length  # noqa: E402
from hermes_cli.config import get_compatible_custom_providers, load_config  # noqa: E402


def main():
    cfg = load_config()
    providers = get_compatible_custom_providers(cfg)
    results = {}
    for name in ("bonsai-2b", "bonsai-4b", "bonsai-9b", "whiterabbit-7b"):
        agent = types.SimpleNamespace(
            model=name,
            provider="custom",
            base_url="http://127.0.0.1:8091/v1",
            request_overrides={},
            _ensure_lmstudio_runtime_loaded=lambda context: None,
            _lmstudio_load_was_unverified=lambda context: False,
            _effective_lmstudio_context_length=lambda configured, runtime: configured,
        )
        actual, _, _, _ = _resolve_context_length(agent, cfg, agent.base_url)
        assert actual == 65536, (name, actual)
        assert agent._aux_compression_context_length_config == 65536
        results[name] = actual
    assert any(p.get("base_url") == "http://127.0.0.1:8091/v1" for p in providers)
    print(
        json.dumps({"resolved_contexts": results, "auxiliary_context": 65536, "network_calls": 0})
    )


if __name__ == "__main__":
    main()
