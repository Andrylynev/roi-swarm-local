"""Isolated, project-scoped workflow memory; observations are not established truth."""

import hashlib
import json
import math
import re
import sqlite3
import threading
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WorkflowLesson(BaseModel):
    """A bounded workflow suggestion requiring explicit operator review."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    topic: Literal["decomposition", "materials", "verification", "coordination"]
    situation: str = Field(min_length=1, max_length=240)
    recommendation: str = Field(min_length=1, max_length=800)
    evidence: str = Field(min_length=1, max_length=500)


_CREDENTIAL = re.compile(
    r"\bsk-[A-Za-z0-9_-]+|\bbearer\s+\S+|"
    r"\b(?:api[_-]?key|password)\b[\"']?\s*[:=]\s*\S+",
    re.IGNORECASE,
)
_SOURCE_PATH = re.compile(
    r"\b[a-z]:[\\/]|\\\\[^\\\s]+\\|\bfile://|(?<![\w:/])/(?:[^/\s]+/)+[^\s]*",
    re.IGNORECASE,
)


_COUNT_METRICS = frozenset(
    {
        "worker_count",
        "task_count",
        "completed_count",
        "failed_count",
        "retry_count",
        "artifact_count",
        "tool_call_count",
        "verification_count",
        "error_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    }
)
_DURATION_METRICS = frozenset({"duration_seconds", "elapsed_seconds"})


def _identifier(value: str) -> None:
    # Scope identifiers are preserved exactly; never casefold or strip them.
    if not isinstance(value, str) or not value.strip():
        raise ValueError("project_id and run_id must be nonempty strings")
    if _CREDENTIAL.search(value):
        raise ValueError("credential-like content is not allowed")


class ExperienceStore:
    """Dedicated SQLite memory, serialized across threads sharing this instance."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock, self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute(
                """CREATE TABLE IF NOT EXISTS experience_notes (
                    id INTEGER PRIMARY KEY,
                    lesson_hash TEXT NOT NULL UNIQUE,
                    project_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    situation TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'candidate'
                        CHECK(status IN ('candidate', 'approved', 'rejected')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    seen_count INTEGER NOT NULL DEFAULT 1,
                    origin TEXT NOT NULL CHECK(origin IN ('model', 'operator'))
                )"""
            )

            self._connection.execute(
                """CREATE TABLE IF NOT EXISTS experience_observations (
                    note_id INTEGER NOT NULL REFERENCES experience_notes(id),
                    run_id TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    origin TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(note_id, run_id, evidence)
                )"""
            )

            self._connection.execute(
                """CREATE TABLE IF NOT EXISTS experience_runs (
                    project_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(project_id, run_id)
                )"""
            )

    def record(
        self, lesson: WorkflowLesson, *, project_id: str, run_id: str, origin: str = "model"
    ) -> dict:
        _identifier(project_id)
        _identifier(run_id)
        if origin not in ("model", "operator"):
            raise ValueError("origin must be model or operator")
        if project_id == "global" and origin != "operator":
            raise ValueError("global is reserved for operator-published notes")
        if not isinstance(lesson, WorkflowLesson):
            raise ValueError("lesson must be a WorkflowLesson")
        # Revalidate model_construct()/mutated instances. Do not echo untrusted inputs.
        try:
            lesson = WorkflowLesson.model_validate(lesson.model_dump())
        except ValueError:
            raise ValueError("invalid workflow lesson") from None
        fields = (lesson.situation, lesson.recommendation, lesson.evidence)
        if any(_CREDENTIAL.search(value) for value in fields):
            raise ValueError("credential-like content is not allowed")
        if origin == "model" and any(_SOURCE_PATH.search(value) for value in fields):
            raise ValueError("automatic lessons must not contain source-specific paths")
        normalized = [
            " ".join(unicodedata.normalize("NFKC", value).casefold().split())
            for value in (lesson.topic, lesson.situation, lesson.recommendation)
        ]
        digest = hashlib.sha256(json.dumps([project_id, *normalized]).encode()).hexdigest()
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO experience_notes (
                    lesson_hash, project_id, run_id, topic, situation, recommendation, evidence,
                    created_at, updated_at, origin
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(lesson_hash) DO UPDATE SET updated_at = excluded.updated_at""",
                (
                    digest,
                    project_id,
                    run_id,
                    lesson.topic,
                    lesson.situation,
                    lesson.recommendation,
                    lesson.evidence,
                    now,
                    now,
                    origin,
                ),
            )
            note_id = self._connection.execute(
                "SELECT id FROM experience_notes WHERE lesson_hash = ?", (digest,)
            ).fetchone()[0]
            self._connection.execute(
                """INSERT INTO experience_observations(
                    note_id, run_id, evidence, origin, created_at
                ) VALUES (?, ?, ?, ?, ?) ON CONFLICT(note_id, run_id, evidence) DO NOTHING""",
                (note_id, run_id, lesson.evidence, origin, now),
            )
            self._connection.execute(
                """UPDATE experience_notes SET seen_count = (
                    SELECT COUNT(DISTINCT run_id) FROM experience_observations WHERE note_id = ?
                ) WHERE id = ?""",
                (note_id, note_id),
            )
            return dict(
                self._connection.execute(
                    "SELECT * FROM experience_notes WHERE id = ?", (note_id,)
                ).fetchone()
            )

    def list_notes(self, project_id: str, status: str | None = None, limit: int = 50) -> list[dict]:
        _identifier(project_id)
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        if status is not None and status not in ("candidate", "approved", "rejected"):
            raise ValueError("invalid note status")
        with self._lock:
            return [
                dict(row)
                for row in self._connection.execute(
                    """SELECT * FROM experience_notes WHERE project_id = ?
                    AND (? IS NULL OR status = ?) ORDER BY id DESC LIMIT ?""",
                    (project_id, status, status, limit),
                )
            ]

    def review(self, note_id: int, status: Literal["approved", "rejected"]) -> dict:
        """Explicit operator decision; repeated observations never change this status."""
        if status not in ("approved", "rejected"):
            raise ValueError("review status must be approved or rejected")
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "UPDATE experience_notes SET status = ?, updated_at = ? WHERE id = ?",
                (status, datetime.now(UTC).isoformat(), note_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(note_id)
            return dict(
                self._connection.execute(
                    "SELECT * FROM experience_notes WHERE id = ?", (note_id,)
                ).fetchone()
            )

    def publish(self, note_id: int) -> dict:
        """Approve and publish an operator-attributed global copy in one transaction."""
        with self._lock, self._connection:
            source = self._connection.execute(
                "SELECT * FROM experience_notes WHERE id = ?", (note_id,)
            ).fetchone()
            if source is None:
                raise KeyError(note_id)
            lesson = WorkflowLesson(**{key: source[key] for key in WorkflowLesson.model_fields})
            values = (lesson.situation, lesson.recommendation, lesson.evidence)
            if any(_CREDENTIAL.search(value) for value in values):
                raise ValueError("credential-like content is not allowed")
            normalized = [
                " ".join(unicodedata.normalize("NFKC", value).casefold().split())
                for value in (lesson.topic, lesson.situation, lesson.recommendation)
            ]
            digest = hashlib.sha256(json.dumps(["global", *normalized]).encode()).hexdigest()
            now = datetime.now(UTC).isoformat()
            run_id = f"review-{note_id}"
            self._connection.execute(
                """INSERT INTO experience_notes (
                    lesson_hash, project_id, run_id, topic, situation, recommendation, evidence,
                    status, created_at, updated_at, origin
                ) VALUES (?, 'global', ?, ?, ?, ?, ?, 'approved', ?, ?, 'operator')
                ON CONFLICT(lesson_hash) DO UPDATE SET
                    origin='operator', status='approved', updated_at=excluded.updated_at,
                    evidence=excluded.evidence, run_id=excluded.run_id""",
                (digest, run_id, lesson.topic, *values, now, now),
            )
            shared_id = self._connection.execute(
                "SELECT id FROM experience_notes WHERE lesson_hash=?", (digest,)
            ).fetchone()[0]
            self._connection.execute(
                """INSERT OR IGNORE INTO experience_observations
                (note_id,run_id,evidence,origin,created_at) VALUES (?,?,?,'operator',?)""",
                (shared_id, run_id, lesson.evidence, now),
            )
            self._connection.execute(
                """UPDATE experience_notes SET seen_count=(
                SELECT COUNT(DISTINCT run_id) FROM experience_observations WHERE note_id=?)
                WHERE id=?""",
                (shared_id, shared_id),
            )
            self._connection.execute(
                "UPDATE experience_notes SET status='approved', updated_at=? WHERE id=?",
                (now, note_id),
            )
            return dict(
                self._connection.execute(
                    "SELECT * FROM experience_notes WHERE id=?", (shared_id,)
                ).fetchone()
            )

    def lessons(self, project_id: str, limit: int = 6) -> list[dict]:
        """Retrieve only reviewed suggestions in the exact requested project."""
        notes = self.list_notes(project_id, status="approved", limit=limit)
        return (
            [note for note in notes if note["origin"] == "operator"]
            if project_id == "global"
            else notes
        )

    def observe_run(self, *, run_id: str, project_id: str, metrics: dict) -> None:
        """Persist allowlisted telemetry, never raw requests, scores or inferred lessons.

        Counts must be nonnegative integers; durations finite nonnegative numbers.
        Unknown fields, booleans, nested values and numeric strings are discarded.
        An observation replaces the prior telemetry for the same project/run pair.
        """
        _identifier(project_id)
        _identifier(run_id)
        if not isinstance(metrics, dict):
            raise ValueError("metrics must be a dictionary")
        safe = {}
        for key in _COUNT_METRICS:
            value = metrics.get(key)
            if type(value) is int and value >= 0:
                safe[key] = value
        for key in _DURATION_METRICS:
            value = metrics.get(key)
            if (
                type(value) in (int, float)
                and value >= 0
                and (type(value) is int or math.isfinite(value))
            ):
                safe[key] = value
        payload = json.dumps(safe, sort_keys=True, allow_nan=False)
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO experience_runs(
                    project_id, run_id, metrics_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(project_id, run_id) DO UPDATE SET
                    metrics_json = excluded.metrics_json, updated_at = excluded.updated_at""",
                (project_id, run_id, payload, now, now),
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()
