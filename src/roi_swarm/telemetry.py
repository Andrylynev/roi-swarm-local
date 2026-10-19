from __future__ import annotations

import heapq
import json
import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


_SAFE_RUN_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")


class RunStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def create_run(self, request: str, run_id: str | None = None) -> str:
        run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        self._validate_run_id(run_id)
        # Keep the existence check and initial write in the same critical section.
        # Otherwise concurrent callers sharing this store can both accept an ID
        # and the later write can erase the first caller's run history.
        with self._lock:
            if self._path_for(run_id).exists():
                raise ValueError(f"Run already exists: {run_id}")
            now = _now()
            data = {
                "id": run_id,
                "request": request,
                "status": "running",
                "created_at": now,
                "updated_at": now,
                "answer": None,
                "worker_target": None,
                "nodes": {},
                "events": [],
            }
            self._write(data)
        return run_id

    def record_event(
        self,
        run_id: str,
        node: str,
        status: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            snapshot = self.get_run(run_id)
            now = _now()
            payload = data or {}
            event = {
                "seq": len(snapshot["events"]) + 1,
                "time": now,
                "node": node,
                "status": status,
                "data": payload,
            }
            snapshot["events"].append(event)
            previous = snapshot["nodes"].get(node, {})
            snapshot["nodes"][node] = {
                **previous,
                **payload,
                "status": status,
                "updated_at": now,
            }
            if "worker_target" in payload:
                snapshot["worker_target"] = payload["worker_target"]
            snapshot["updated_at"] = now
            self._write(snapshot)

    def complete_run(self, run_id: str, answer: str) -> None:
        self._finish(run_id, "completed", answer=answer)

    def fail_run(self, run_id: str, error: str) -> None:
        self._finish(run_id, "failed", error=error)

    def get_run(self, run_id: str) -> dict[str, Any]:
        path = self._path_for(run_id)
        if not path.exists():
            raise KeyError(f"Unknown run: {run_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_runs(self, limit: int = 30) -> list[dict[str, Any]]:
        """Return newest runs, retaining only up to limit parsed histories.

        Every history is still inspected; equal timestamps retain scan order,
        matching a stable newest-first sort. Zero avoids scanning entirely.
        """
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        if limit == 0:
            return []

        def records():
            for path in self.root.glob("*.json"):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    # Leave unreadable histories intact for direct inspection.
                    continue
                if not isinstance(record, dict) or not isinstance(record.get("created_at"), str):
                    continue
                yield record

        # Consume lazily: histories can contain large event arrays, so collecting
        # all of them before sorting wastes memory for a small dashboard page.
        return heapq.nlargest(limit, records(), key=lambda item: item["created_at"])

    def _finish(self, run_id: str, status: str, **values: Any) -> None:
        with self._lock:
            snapshot = self.get_run(run_id)
            snapshot.update(values)
            snapshot["status"] = status
            snapshot["updated_at"] = _now()
            self._write(snapshot)

    def _write(self, data: dict[str, Any]) -> None:
        with self._lock:
            destination = self._path_for(data["id"])
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                temporary.replace(destination)
            finally:
                # A failed write/replace must not accumulate full run histories
                # in staging files. Remove only this write's unique staging path.
                temporary.unlink(missing_ok=True)

    def _path_for(self, run_id: str) -> Path:
        self._validate_run_id(run_id)
        path = (self.root / f"{run_id}.json").resolve()
        if path.parent != self.root.resolve():
            raise ValueError("Run path escapes store")
        return path

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not _SAFE_RUN_ID.fullmatch(run_id):
            raise ValueError("Unsafe run id")
