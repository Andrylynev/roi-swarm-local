"""Local token budgeting for provider aliases without published tokenizers.

Use the configurable o200k_base reference tokenizer, not a chars/4 estimate.
Provider-native accounting can differ for aliases such as Luna and Opus.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import tiktoken


@lru_cache(maxsize=4)
def _encoding(name: str):
    os.environ.setdefault(
        "TIKTOKEN_CACHE_DIR", str(Path(__file__).parents[2] / "data" / "models" / "tiktoken")
    )
    return tiktoken.get_encoding(name)


def count_tokens(text: str, encoding: str = "o200k_base") -> int:
    return len(_encoding(encoding).encode_ordinary(text))


def check_context(
    value: Any, *, limit: int, output_tokens: int = 0, encoding: str = "o200k_base"
) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    # Count serialized messages/tools too, with a small message-framing allowance.
    used = count_tokens(text, encoding) + (32 if not isinstance(value, str) else 0)
    if used + output_tokens > limit:
        raise ValueError(
            f"Context exceeds {limit} tokens: input={used}, output reserve={output_tokens} "
            f"(reference tokenizer {encoding}). Split the task; no input was silently truncated."
        )
    return used
