from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any


class CanonicalStateStore:
    def __init__(self, root: str | Path, snapshot_interval: int = 5) -> None:
        if type(snapshot_interval) is not int or snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be a positive integer")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshot_interval = snapshot_interval

    def load_latest(self, session_id: str) -> dict[str, Any] | None:
        """Restore the newest durable state, with legacy snapshot fallback."""
        session = self._session_dir(session_id)
        # Padding is a minimum width: sequence 1000000 sorts before 999999
        # lexically. Compare numeric suffixes so restoration remains current.
        snapshots = self._record_paths(session, "snapshot-")
        snapshot = json.loads(snapshots[-1].read_text(encoding="utf-8")) if snapshots else None
        snapshot_sequence = snapshot["sequence"] if snapshot else 0
        for path in reversed(self._record_paths(session, "delta-")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["sequence"] <= snapshot_sequence:
                break
            if "state" in record:
                return record["state"]
        return snapshot["state"] if snapshot else None

    @staticmethod
    def _record_paths(session: Path, prefix: str) -> list[Path]:
        # Match the sequence namespace used by _next_sequence. Backup names and
        # non-ASCII numeric spellings are not durable records. Valid record
        # contents are still read normally; corruption is not silently ignored.
        paths = []
        for path in session.glob(f"{prefix}*.json"):
            suffix = path.stem[len(prefix) :]
            if suffix.isascii() and suffix.isdecimal():
                paths.append(path)
        return sorted(paths, key=lambda path: int(path.stem[len(prefix) :]))

    def save(
        self,
        session_id: str,
        state: dict[str, Any],
        delta: dict[str, Any],
    ) -> dict[str, Any]:
        session = self._session_dir(session_id)
        session.mkdir(parents=True, exist_ok=True)
        sequence = self._next_sequence(session)
        record = {
            "sequence": sequence,
            "base_hash": self._state_hash(self.load_latest(session_id)),
            "state_hash": self._state_hash(state),
            "delta": delta,
            "state": state,
        }
        delta_path = session / f"delta-{sequence:06d}.json"
        self._atomic_write(delta_path, record)
        if sequence == 1 or sequence % self.snapshot_interval == 0:
            snapshot_path = session / f"snapshot-{sequence:06d}.json"
            self._atomic_write(snapshot_path, {"sequence": sequence, "state": state})
        return record

    def _session_dir(self, session_id: str) -> Path:
        safe = hashlib.sha256(session_id.encode()).hexdigest()[:24]
        return self.root / safe

    @staticmethod
    def _next_sequence(session: Path) -> int:
        # Counting files can reuse an existing sequence after an older delta is
        # removed. Snapshots also reserve their sequence if their delta is gone.
        highest = 0
        for prefix in ("delta-", "snapshot-"):
            for path in session.glob(f"{prefix}*.json"):
                suffix = path.stem[len(prefix) :]
                if suffix.isascii() and suffix.isdecimal():
                    highest = max(highest, int(suffix))
        return highest + 1

    @staticmethod
    def _state_hash(state: dict[str, Any] | None) -> str | None:
        if state is None:
            return None
        encoded = json.dumps(state, ensure_ascii=False, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _atomic_write(path: Path, value: dict[str, Any]) -> None:
        # Own a distinct staging file so failure cleanup cannot remove another
        # writer's staging data. This does not serialize save sequence allocation.
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
