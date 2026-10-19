from __future__ import annotations

import threading
from typing import Any

# Int8/quantized Granite ONNX exports collapse every text to nearly the same vector
# (cosine >= 0.99 for unrelated sentences), so the fp32 export is the default.
DEFAULT_MODEL_ID = "onnx-community/granite-embedding-97m-multilingual-r2-ONNX"
DEFAULT_MODEL_FILE = "onnx/model.onnx"
DEFAULT_TOKENIZER_FILE = "tokenizer.json"


class GraniteOnnxEmbedder:
    """Lazy local ONNX embedder; the model is downloaded only on the first semantic query."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        filename: str = DEFAULT_MODEL_FILE,
        tokenizer_filename: str = DEFAULT_TOKENIZER_FILE,
        max_length: int = 512,
        cache_dir: str | None = None,
        pooling: str = "cls",
    ) -> None:
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")
        self.model_id = model_id
        self.filename = filename
        self.tokenizer_filename = tokenizer_filename
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.pooling = pooling
        self._lock = threading.Lock()
        self._tokenizer: Any = None
        self._session: Any = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load()
        encoded = self._tokenizer(texts)
        input_names = {item.name for item in self._session.get_inputs()}
        feeds = {name: encoded[name] for name in input_names if name in encoded}
        hidden = self._session.run(None, feeds)[0]
        if len(hidden.shape) == 3:
            if self.pooling == "cls":
                vectors = hidden[:, 0]
            else:
                mask = encoded["attention_mask"].astype("float32")[:, :, None]
                vectors = (hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1.0)
        else:
            vectors = hidden
        return [self._normalize(row.tolist()) for row in vectors]

    def _load(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            from huggingface_hub import hf_hub_download
            from onnxruntime import InferenceSession
            from tokenizers import Tokenizer

            model_path = hf_hub_download(
                repo_id=self.model_id, filename=self.filename, cache_dir=self.cache_dir
            )
            tokenizer_path = hf_hub_download(
                repo_id=self.model_id, filename=self.tokenizer_filename, cache_dir=self.cache_dir
            )
            tokenizer = Tokenizer.from_file(tokenizer_path)
            tokenizer.enable_padding()
            tokenizer.enable_truncation(max_length=self.max_length)
            self._tokenizer = _TokenizerAdapter(tokenizer)
            self._session = InferenceSession(model_path, providers=["CPUExecutionProvider"])

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        magnitude = sum(value * value for value in vector) ** 0.5
        return [value / magnitude for value in vector] if magnitude else vector


class _TokenizerAdapter:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, texts: list[str]) -> dict[str, Any]:
        import numpy as np

        batch = self.tokenizer.encode_batch(texts)
        return {
            "input_ids": np.asarray([item.ids for item in batch], dtype="int64"),
            "attention_mask": np.asarray([item.attention_mask for item in batch], dtype="int64"),
        }
