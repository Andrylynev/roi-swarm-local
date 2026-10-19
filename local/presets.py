"""Portable native llama-server routing presets; no extra Python inference proxy."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

FILES = {
    "bonsai-9b": "Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf",
    "bonsai-4b": "Ternary-Bonsai-2-4B-PQ2_0-MTP-Q8_0.gguf",
    "bonsai-2b": "Ternary-Bonsai-2-2B-PQ2_0.gguf",
    "whiterabbit-7b": "WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-Q4_K_M.gguf",
}


def model_presets(
    root: Path, backend: str = "cuda", aliases: Iterable[str] | None = None
) -> dict[str, dict[str, str]]:
    if backend not in {"cpu", "cuda"}:
        raise ValueError("Unknown inference backend: " + str(backend))
    names = list(FILES) if aliases is None else list(dict.fromkeys(aliases))
    if any(name not in FILES for name in names):
        raise ValueError("Unknown model alias in presets")
    result = {}
    for name in names:
        filename = FILES[name]
        result[name] = {
            "model": (root / "models" / filename).as_posix(),
            "ctx-size": "65536",
            "parallel": "1",
            "n-gpu-layers": "99",
            "flash-attn": "on",
            "threads": "6",
            "threads-batch": "6",
            "batch-size": "512",
            "ubatch-size": "128",
            "cache-type-k": "q8_0",
            "cache-type-v": "q8_0",
            "cache-ram": "0",
            "jinja": "true",
            "load-on-startup": "false",
            "temp": "0.7",
            "top-p": "0.8",
            "top-k": "20",
            "min-p": "0",
            "presence-penalty": "1.5",
            "repeat-penalty": "1.0",
            "repeat-last-n": "64",
            "chat-template-kwargs": '{"enable_thinking":false}',
            "reasoning-budget": "0",
            "reasoning-format": "none",
            # Benchmark MTP before enabling: faster decode can cost prompt reuse.
            "spec-type": "none",
        }
        if name == "bonsai-9b":
            # Measured on this RTX3050: ~62 vs52 aggregate tok/s, tools passed.
            result[name].update(
                {
                    "spec-type": "draft-mtp",
                    "spec-draft-n-max": "2",
                    "spec-draft-n-min": "0",
                    "spec-draft-p-min": "0",
                }
            )
        if name in {"bonsai-4b", "bonsai-2b"}:
            # Keep 64K PER slot, not 64K divided between both workers.
            result[name].update(
                {
                    "ctx-size": "131072",
                    "parallel": "2",
                    "ubatch-size": "256",
                }
            )
        result[name]["reasoning-format"] = "auto"
        result[name]["reasoning"] = "off"
        result[name]["cache-ram"] = "256"
        if name == "whiterabbit-7b":
            result[name].update({"temp": "0.2", "presence-penalty": "0", "top-p": "0.95"})
        if backend == "cpu":
            result[name].update(
                {
                    "n-gpu-layers": "0",
                    "flash-attn": "off",
                    "spec-type": "none",
                    # Quantized V requires flash-attention in llama.cpp.
                    "cache-type-k": "f16",
                    "cache-type-v": "f16",
                }
            )
            for key in tuple(result[name]):
                if key.startswith("spec-draft-"):
                    del result[name][key]
    return result


def write_presets(root: Path, backend: str = "cuda", aliases: Iterable[str] | None = None) -> Path:
    path = root / "local/models.ini"
    lines = ["version = 1", ""]
    for name, options in model_presets(root, backend=backend, aliases=aliases).items():
        lines.append(f"[{name}]")
        lines.extend(f"{key} = {value}" for key, value in options.items())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write_presets(Path(__file__).resolve().parents[1]))
