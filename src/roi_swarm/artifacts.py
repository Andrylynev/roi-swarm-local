from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

_SAFE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")


@dataclass(frozen=True)
class ArtifactRef:
    id: str
    sha256: str
    kind: str
    size_chars: int
    path: str


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.content_dir = self.root / "content"
        self.meta_dir = self.root / "metadata"
        self.content_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def put_text(self, run_id: str, text: str, *, kind: str = "text") -> ArtifactRef:
        """Serialize content and metadata updates by callers sharing this instance.

        Atomic replacement alone cannot protect the metadata read-modify-write
        sequence from lost run IDs. Separate store instances and processes still
        require external coordination when writing to the same root.
        """
        self._validate_id(run_id)
        with self._lock:
            return self._put_text(run_id, text, kind=kind)

    def _put_text(self, run_id: str, text: str, *, kind: str) -> ArtifactRef:
        content = text.encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        artifact_id = f"sha256-{digest}"
        path = self.content_dir / f"{artifact_id}.txt"
        if not path.exists() or path.read_bytes() != content:
            self._atomic_write(path, text)
        ref = ArtifactRef(
            id=artifact_id,
            sha256=digest,
            kind=kind,
            size_chars=len(text),
            path=str(path),
        )
        metadata = self.meta_dir / f"{artifact_id}.json"
        if not metadata.exists():
            self._atomic_write(
                metadata,
                json.dumps({**asdict(ref), "run_ids": [run_id]}, ensure_ascii=False, indent=2),
            )
        else:
            data = json.loads(metadata.read_text(encoding="utf-8"))
            if run_id not in data.get("run_ids", []):
                data.setdefault("run_ids", []).append(run_id)
                self._atomic_write(metadata, json.dumps(data, ensure_ascii=False, indent=2))
        return ref

    def get_text(self, artifact_id: str) -> str:
        self._validate_id(artifact_id)
        path = self.content_dir / f"{artifact_id}.txt"
        resolved = path.resolve()
        if resolved.parent != self.content_dir:
            raise ValueError("Artifact path escapes store")
        if not resolved.exists():
            raise KeyError(f"Unknown artifact: {artifact_id}")
        raw = resolved.read_bytes()
        if f"sha256-{hashlib.sha256(raw).hexdigest()}" == artifact_id:
            return raw.decode("utf-8")
        # Old Windows text writes expanded every LF, including those in CRLF.
        # Recover once, only if the complete original content address proves it.
        recovered = raw.replace(b"\r\n", b"\n")
        if f"sha256-{hashlib.sha256(recovered).hexdigest()}" == artifact_id:
            return recovered.decode("utf-8")
        raise ValueError(f"Artifact content hash mismatch: {artifact_id}")

    @staticmethod
    def _validate_id(value: str) -> None:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("Unsafe identifier")

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        # Concurrent writers must not replace or remove each other's staging file.
        # This isolates individual writes; callers using separate store instances
        # must still coordinate metadata read-modify-write operations externally.
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(content.encode("utf-8"))
            # Windows can briefly deny simultaneous replacements of one target.
            # Retry only access/sharing/lock violations, retaining our own staging
            # file. Persistent denials still raise after five total attempts.
            for attempt in range(5):
                try:
                    temporary.replace(path)
                    break
                except OSError as error:
                    if (
                        os.name != "nt"
                        or getattr(error, "winerror", None) not in {5, 32, 33}
                        or attempt == 4
                    ):
                        raise
                    time.sleep(0.01 * 2**attempt)
        finally:
            temporary.unlink(missing_ok=True)
