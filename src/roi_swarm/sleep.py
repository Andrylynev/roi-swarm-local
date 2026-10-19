"""Offline checkpointed reflection over completed run summaries only."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import math
import sqlite3
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class SleepManager:
    """Checkpointed, single-flight idle reflection; no inference or tools of its own.

    ``None`` budgets leave reflection running until foreground work, disable, or
    shutdown. Completed sources rotate FIFO after each step, including ``done``
    results. Explicit step budgets retire a source on ``done`` or exhaustion.

    Summaries retain the last 64 entries per field, at most 2048 characters each;
    source metrics are finite numbers only. Callbacks receive detached public
    checkpoints, never the original request. All lifecycle calls belong to the
    same event-loop thread. Synchronous save callbacks must be short/nonblocking.

    Cancellation fences results and retains a single callback lease until even a
    cancellation-suppressing callback exits. Close detaches it without waiting;
    a callback already executing must cooperate to stop its own external effects.
    """

    def __init__(
        self,
        path: Path,
        reflect: Callable[[dict], Awaitable[dict]],
        save_lessons: Callable[[dict, dict], Any],
        enabled: bool = False,
        idle_seconds: float = 15,
        step_delay_seconds: float = 30,
        max_steps_per_cycle: int | None = None,
        daily_call_limit: int | None = None,
    ) -> None:
        for name, value in (
            ("idle_seconds", idle_seconds),
            ("step_delay_seconds", step_delay_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite nonnegative number")
        for name, value in (
            ("max_steps_per_cycle", max_steps_per_cycle),
            ("daily_call_limit", daily_call_limit),
        ):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.reflect = reflect
        self.save_lessons = save_lessons
        self.enabled = enabled
        self.idle_seconds = idle_seconds
        self.step_delay_seconds = step_delay_seconds
        self.max_steps_per_cycle = max_steps_per_cycle
        self.daily_call_limit = daily_call_limit
        self._db = sqlite3.connect(self.path)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS checkpoint (id INTEGER PRIMARY KEY, data TEXT)"
        )
        self._db.execute("CREATE TABLE IF NOT EXISTS seen (run_id TEXT PRIMARY KEY)")
        self._current: dict | None = None
        self._queue: list[dict] = []
        self._pending_result: dict | None = None
        self._daily_calls = 0
        self._day = self._today()
        self._step_count = 0
        self._last_error: str | None = None
        self._foreground_count = 0
        self._task: asyncio.Task | None = None
        self._active: asyncio.Task | None = None
        self._epoch = 0
        self._closed = False
        self._last_activity = 0.0
        self._next_step_at = 0.0
        self._wake = asyncio.Event()
        row = self._db.execute("SELECT data FROM checkpoint WHERE id=1").fetchone()
        if row:
            data = json.loads(row[0])
            self.enabled = data["enabled"]
            self._current = data["current"]
            self._queue = data["queue"]
            self._pending_result = data.get("pending_result")
            self._day = data["day"]
            self._daily_calls = data["daily_calls"]
            self._step_count = data["step_count"]
            self._last_error = data["last_error"]
        self._checkpoint()

    @staticmethod
    def _metrics(value: Any) -> dict:
        if not isinstance(value, dict):
            return {}
        return {
            key: number
            for key, number in list(value.items())[:64]
            if isinstance(key, str)
            and len(key) <= 128
            and isinstance(number, (int, float))
            and math.isfinite(number)
        }

    @staticmethod
    def _notes(previous: list, incoming: Any) -> list:
        if not isinstance(incoming, list):
            return previous
        # Checkpoint summaries only: bounded strings or bounded JSON summaries.
        notes = []
        for item in incoming[-64:]:
            if isinstance(item, str):
                notes.append(item[:2048])
            elif isinstance(item, dict):
                notes.append(json.dumps(item, ensure_ascii=False, default=str)[:2048])
        return (previous + notes)[-64:]

    @staticmethod
    def _today() -> str:
        return datetime.now(UTC).date().isoformat()

    def _checkpoint(self) -> None:
        if self._closed:
            return
        data = {
            "enabled": self.enabled,
            "current": self._current,
            "queue": self._queue,
            "pending_result": self._pending_result,
            "day": self._day,
            "daily_calls": self._daily_calls,
            "step_count": self._step_count,
            "last_error": self._last_error,
            "state": self._state(),
            "foreground_count": self._foreground_count,
        }
        self._db.execute("INSERT OR REPLACE INTO checkpoint VALUES (1, ?)", (json.dumps(data),))
        self._db.commit()
        self._wake.set()

    async def start(self) -> None:
        if not self._closed and self._task is None:
            self._last_activity = asyncio.get_running_loop().time()
            self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        if self._closed:
            return
        self._checkpoint()
        self._epoch += 1
        self._closed = True
        if self._active and not self._active.done():
            self._active.cancel()
        if self._task:
            self._task.cancel()
        self._db.close()

    async def foreground_started(self) -> None:
        if self._closed:
            return
        self._foreground_count += 1
        self._last_activity = asyncio.get_running_loop().time()
        self._epoch += 1
        if self._active and not self._active.done():
            self._active.cancel()
        self._checkpoint()

    def foreground_finished(self, completed_run: dict | None = None) -> None:
        if self._closed:
            return
        self._foreground_count = max(0, self._foreground_count - 1)
        self._last_activity = asyncio.get_running_loop().time()
        if completed_run and completed_run.get("run_id"):
            run_id = completed_run["run_id"]
            inserted = self._db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (run_id,)).rowcount
            if inserted:
                self._queue.append(
                    {
                        "source_run_id": run_id,
                        "scope": completed_run.get("scope") or f"run:{run_id}",
                        "metrics": self._metrics(completed_run.get("metrics")),
                        "observations": [],
                        "hypotheses": [],
                        "completed_checks": [],
                        "next_step": None,
                        "attempts": 0,
                        "steps_completed": 0,
                    }
                )
        self._checkpoint()

    async def set_enabled(self, enabled: bool) -> None:
        if self._closed:
            return
        if self.enabled != enabled:
            self.enabled = enabled
            self._epoch += 1
            if not enabled and self._active and not self._active.done():
                self._active.cancel()
        self._checkpoint()

    def _refresh_day(self) -> None:
        today = self._today()
        if today != self._day:
            self._day = today
            self._daily_calls = 0
            self._checkpoint()

    def _daily_limited(self) -> bool:
        return self.daily_call_limit is not None and self._daily_calls >= self.daily_call_limit

    def _state(self) -> str:
        if self._closed:
            return "closed"
        if not self.enabled:
            return "disabled"
        if self._foreground_count:
            return "foreground"
        if self._active and not self._active.done():
            return "reflecting"
        if self._daily_limited():
            return "daily_limit"
        return "idle"

    def status(self) -> dict:
        self._refresh_day()
        return {
            "enabled": self.enabled,
            "state": self._state(),
            "foreground_count": self._foreground_count,
            "daily_calls": self._daily_calls,
            "daily_call_limit": self.daily_call_limit,
            "max_steps_per_cycle": self.max_steps_per_cycle,
            "pending_cycles": len(self._queue) + int(self._current is not None),
            "step_count": self._step_count,
            "last_error": self._last_error,
        }

    def snapshot(self) -> dict:
        return copy.deepcopy({"current": self._current, "queue": self._queue})

    async def _wait(self, delay: float | None = None) -> None:
        try:
            if delay is None:
                await self._wake.wait()
            else:
                await asyncio.wait_for(self._wake.wait(), max(0.001, delay))
        except TimeoutError:
            pass

    async def _run(self) -> None:
        while not self._closed:
            self._wake.clear()
            self._refresh_day()
            if not self.enabled or self._foreground_count:
                await self._wait()
                continue
            if self._daily_limited() and self._pending_result is None:
                await self._wait(60)
                continue
            if self._current is None and not self._queue:
                await self._wait()
                continue
            now = asyncio.get_running_loop().time()
            ready_at = max(self._last_activity + self.idle_seconds, self._next_step_at)
            if now < ready_at:
                await self._wait(ready_at - now)
                continue
            if self._current is None:
                self._current = self._queue.pop(0)
            current = self._current
            if (
                self._pending_result is None
                and self.max_steps_per_cycle is not None
                and current["steps_completed"] >= self.max_steps_per_cycle
            ):
                self._current = None
                self._checkpoint()
                await asyncio.sleep(0)
                continue
            if self._pending_result is None:
                current["attempts"] += 1
                self._daily_calls += 1
            epoch = self._epoch
            self._active = asyncio.create_task(self._step(copy.deepcopy(current), epoch))
            self._active.add_done_callback(self._consume_exception)
            self._checkpoint()
            try:
                # The supervisor may exit while a non-cooperative callback unwinds.
                await asyncio.shield(self._active)
            except asyncio.CancelledError:
                if self._closed:
                    raise
                continue
            except Exception as exc:
                if not self._closed and epoch == self._epoch:
                    self._last_error = f"Reflection failed: {type(exc).__name__}"
                    self._checkpoint()
            finally:
                self._active = None
                self._next_step_at = asyncio.get_running_loop().time() + max(
                    0.001, self.step_delay_seconds
                )

    @staticmethod
    def _consume_exception(task: asyncio.Task) -> None:
        if not task.cancelled():
            task.exception()

    async def _step(self, current: dict, epoch: int) -> None:
        if self._pending_result is None:
            result = await self.reflect(copy.deepcopy(current))
            if self._closed or epoch != self._epoch:
                return
            if not isinstance(result, dict):
                raise TypeError("Reflection must return a dictionary")
            current["steps_completed"] += 1
            for key in ("observations", "hypotheses", "completed_checks"):
                current[key] = self._notes(current[key], result.get(key))
            if "next_step" in result:
                value = result["next_step"]
                current["next_step"] = value[:2048] if isinstance(value, str) else None
            self._current = current
            self._step_count += 1
            # Persist successful reflection before the potentially long save callback.
            # Only public fields and bounded lesson candidates survive a restart.
            self._pending_result = {
                key: copy.deepcopy(current[key])
                for key in ("observations", "hypotheses", "completed_checks", "next_step")
            }
            self._pending_result["done"] = bool(result.get("done"))
            lessons = result.get("lessons")
            if isinstance(lessons, list):
                self._pending_result["lessons"] = [
                    lesson
                    for lesson in lessons[:2]
                    if isinstance(lesson, dict) and len(json.dumps(lesson, default=str)) <= 8192
                ]
            self._last_error = None
            self._checkpoint()
        else:
            result = copy.deepcopy(self._pending_result)
        if self._closed or epoch != self._epoch:
            return
        saved = self.save_lessons(copy.deepcopy(current), result)
        if inspect.isawaitable(saved):
            await saved
        if self._closed or epoch != self._epoch:
            return
        self._pending_result = None
        self._last_error = None
        if self.max_steps_per_cycle is None or (
            not result.get("done") and current["steps_completed"] < self.max_steps_per_cycle
        ):
            # Rotate rather than spin on a done source; new FIFO work gets a turn.
            self._queue.append(current)
        self._current = None
        self._checkpoint()
