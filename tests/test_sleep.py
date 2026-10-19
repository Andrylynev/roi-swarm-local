"""Offline, event-driven contracts: never make provider or tool calls."""

import asyncio
import json
import sqlite3
from contextlib import closing

import pytest

from roi_swarm.sleep import SleepManager


async def test_completion_is_required_and_fifo_runs_are_deduplicated(tmp_path):
    calls = []
    saved = []
    called = asyncio.Event()

    async def reflect(snapshot):
        calls.append(snapshot)
        called.set()
        return {"observations": ["checked"], "done": True, "lessons": [{"topic": "materials"}]}

    manager = SleepManager(
        tmp_path / "sleep.db",
        reflect=reflect,
        save_lessons=lambda snapshot, result: saved.append((snapshot, result)),
        enabled=True,
        idle_seconds=0,
        step_delay_seconds=0,
        max_steps_per_cycle=1,
    )
    try:
        await manager.start()
        await asyncio.sleep(0.01)
        assert calls == []
        await manager.foreground_started()
        manager.foreground_finished({"run_id": "one", "scope": "project", "metrics": {"n": 1}})
        manager.foreground_finished({"run_id": "one"})
        manager.foreground_finished({"run_id": "two"})
        await asyncio.wait_for(called.wait(), 1)
        # Short sleeps may finish before the next Windows/Python 3.11 clock tick.
        # Wait for scheduler completion, not a fixed number of polling turns.
        await eventually(lambda: len(calls) == 2 and manager.status()["pending_cycles"] == 0)
        assert [item["source_run_id"] for item in calls] == ["one", "two"]
        assert calls[0]["scope"] == "project"
        assert calls[0]["metrics"] == {"n": 1}
        assert calls[1]["scope"] == "run:two"
        assert calls[0]["attempts"] == 1
        assert calls[0]["steps_completed"] == 0
        assert manager.status()["foreground_count"] == 0
        assert manager.status()["daily_calls"] == 2
        assert manager.status()["pending_cycles"] == 0
        assert len(saved) == 2
        assert saved[0][0]["steps_completed"] == 1
        manager.foreground_finished({"run_id": "one"})
        await asyncio.sleep(0.01)
        assert len(calls) == 2
        with closing(sqlite3.connect(manager.path)) as db:
            assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        await manager.close()
        await manager.close()


async def eventually(predicate, timeout=5):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.001)


async def test_null_budgets_keep_reflecting_done_sources_without_starving_fifo(tmp_path):
    calls = []
    saved = []

    async def reflect(snapshot):
        calls.append(snapshot)
        return {"observations": [f"check {len(calls)}"], "done": True}

    manager = SleepManager(
        tmp_path / "sleep.db",
        reflect,
        lambda s, r: saved.append(s),
        enabled=True,
        idle_seconds=0,
        step_delay_seconds=0.001,
    )
    try:
        await manager.start()
        manager.foreground_finished({"run_id": "one"})
        await eventually(lambda: len(saved) >= 4)
        manager.foreground_finished({"run_id": "two"})
        await eventually(lambda: len(saved) >= 28)
        await manager.set_enabled(False)
        assert "two" in [item["source_run_id"] for item in calls[4:7]]
        assert max(item["steps_completed"] for item in saved) > 3
        assert manager.status()["daily_calls"] >= 28
        assert manager.status()["step_count"] >= 28
        assert manager.status()["daily_call_limit"] is None
        assert manager.status()["max_steps_per_cycle"] is None
        assert manager.status()["pending_cycles"] == 2
    finally:
        await manager.close()


async def test_daily_limit_is_persistent_and_resets_on_new_utc_day(tmp_path):
    calls = []

    async def reflect(snapshot):
        calls.append(snapshot)
        return {"done": False}

    path = tmp_path / "sleep.db"
    manager = SleepManager(
        path,
        reflect,
        lambda s, r: None,
        enabled=True,
        idle_seconds=0,
        step_delay_seconds=0.001,
        daily_call_limit=2,
    )
    try:
        await manager.start()
        manager.foreground_finished({"run_id": "one"})
        await eventually(lambda: len(calls) == 2)
        await asyncio.sleep(0.03)
        assert len(calls) == 2
        assert manager.status()["state"] == "daily_limit"
    finally:
        await manager.close()
    manager = SleepManager(
        path,
        reflect,
        lambda s, r: None,
        enabled=False,
        idle_seconds=0,
        step_delay_seconds=0.001,
        daily_call_limit=2,
    )
    try:
        await manager.start()
        await asyncio.sleep(0.02)
        assert manager.status()["enabled"] is True
        assert len(calls) == 2
        assert manager.status()["daily_calls"] == 2
        manager._today = lambda: "2099-01-02"
        assert manager.status()["daily_calls"] == 0
        await eventually(lambda: len(calls) == 4)
        await asyncio.sleep(0.02)
        assert len(calls) == 4
        assert manager.status()["daily_calls"] == 2
    finally:
        await manager.close()


async def test_explicit_step_limit_stops_incomplete_cycle(tmp_path):
    calls = []

    async def reflect(snapshot):
        calls.append(snapshot)
        return {"done": False}

    manager = SleepManager(
        tmp_path / "sleep.db",
        reflect,
        lambda s, r: None,
        enabled=True,
        idle_seconds=0,
        step_delay_seconds=0.001,
        max_steps_per_cycle=3,
    )
    try:
        await manager.start()
        manager.foreground_finished({"run_id": "one"})
        await eventually(lambda: manager.status()["pending_cycles"] == 0)
        await asyncio.sleep(0.02)
        assert len(calls) == 3
        assert manager.status()["step_count"] == 3
    finally:
        await manager.close()


async def test_async_saver_receives_completed_public_checkpoint(tmp_path):
    saved = []

    async def reflect(snapshot):
        return {
            "observations": ["observed"],
            "hypotheses": ["hypothesis"],
            "completed_checks": ["verified"],
            "next_step": "follow up",
            "done": True,
        }

    async def save(snapshot, result):
        await asyncio.sleep(0)
        saved.append((snapshot, result))

    manager = SleepManager(
        tmp_path / "sleep.db",
        reflect,
        save,
        enabled=True,
        idle_seconds=0,
        step_delay_seconds=0,
        max_steps_per_cycle=1,
    )
    try:
        await manager.start()
        manager.foreground_finished({"run_id": "one"})
        await eventually(lambda: len(saved) == 1)
        snapshot, _ = saved[0]
        assert snapshot["observations"] == ["observed"]
        assert snapshot["hypotheses"] == ["hypothesis"]
        assert snapshot["completed_checks"] == ["verified"]
        assert snapshot["next_step"] == "follow up"
        assert snapshot["attempts"] == snapshot["steps_completed"] == 1
        assert manager.status()["pending_cycles"] == 0
    finally:
        await manager.close()


async def test_foreground_fences_late_result_without_overlapping_calls(tmp_path):
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()
    returned = asyncio.Event()
    calls = []
    saved = []

    async def reflect(snapshot):
        calls.append(snapshot)
        if len(calls) == 1:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()  # Deliberately non-cooperative provider.
            returned.set()
            return {"observations": ["stale"], "done": True}
        return {"observations": ["fresh"], "done": True}

    manager = SleepManager(
        tmp_path / "sleep.db",
        reflect,
        lambda s, r: saved.append(s),
        enabled=True,
        idle_seconds=0,
        step_delay_seconds=0.001,
        max_steps_per_cycle=1,
    )
    try:
        await manager.start()
        await manager.start()
        manager.foreground_finished({"run_id": "one"})
        await asyncio.wait_for(entered.wait(), 1)
        assert manager.status()["state"] == "reflecting"
        await asyncio.wait_for(manager.foreground_started(), 0.2)
        await asyncio.wait_for(cancelled.wait(), 0.2)
        checkpoint = manager.snapshot()
        assert checkpoint["current"]["attempts"] == 1
        assert checkpoint["current"]["steps_completed"] == 0
        assert manager.status()["state"] == "foreground"
        manager.foreground_finished()
        await asyncio.sleep(0.02)
        assert len(calls) == 1  # A cancelled but live call owns the paid-call lease.
        release.set()
        await asyncio.wait_for(returned.wait(), 1)
        await eventually(lambda: len(saved) == 1)
        assert saved[0]["observations"] == ["fresh"]
        assert saved[0]["attempts"] == 2
        assert saved[0]["steps_completed"] == 1
        assert manager.status()["daily_calls"] == 2
        assert manager.status()["step_count"] == 1
        assert manager.status()["pending_cycles"] == 0
    finally:
        release.set()
        await asyncio.wait_for(manager.close(), 0.2)


async def test_close_detaches_stubborn_callback_and_prevents_restart_writes(tmp_path):
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()
    returned = asyncio.Event()
    saved = []

    async def reflect(snapshot):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        returned.set()
        return {"observations": ["late close"], "done": True}

    path = tmp_path / "sleep.db"
    manager = SleepManager(
        path,
        reflect,
        lambda s, r: saved.append(s),
        enabled=True,
        idle_seconds=0,
        step_delay_seconds=0.001,
    )
    try:
        await manager.start()
        manager.foreground_finished({"run_id": "one"})
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(manager.close(), 0.2)
        await asyncio.wait_for(cancelled.wait(), 0.2)
        with closing(sqlite3.connect(path)) as db:
            before = db.execute("SELECT data FROM checkpoint").fetchone()[0]
        await manager.start()
        await manager.set_enabled(False)
        await manager.foreground_started()
        manager.foreground_finished({"run_id": "ignored"})
        release.set()
        await asyncio.wait_for(returned.wait(), 1)
        await asyncio.sleep(0.02)
        assert saved == []
        assert manager.status()["state"] == "closed"
        with closing(sqlite3.connect(path)) as db:
            after = db.execute("SELECT data FROM checkpoint").fetchone()[0]
        assert before == after
        assert manager.snapshot()["current"]["steps_completed"] == 0
        await manager.close()
    finally:
        release.set()
        await asyncio.wait_for(manager.close(), 0.2)


async def test_disable_cancels_reflection_and_persists_resume_checkpoint(tmp_path):
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    saved = []
    calls = []

    async def reflect(snapshot):
        calls.append(snapshot)
        if len(calls) == 1:
            return {
                "observations": ["one"],
                "hypotheses": ["two"],
                "completed_checks": ["three"],
                "next_step": "continue",
                "done": False,
            }
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    path = tmp_path / "sleep.db"
    manager = SleepManager(
        path,
        reflect,
        lambda s, r: saved.append(s),
        enabled=True,
        idle_seconds=0,
        step_delay_seconds=0.001,
    )
    await manager.start()
    manager.foreground_finished({"run_id": "one"})
    manager.foreground_finished({"run_id": "two"})
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await manager.set_enabled(False)
        await asyncio.wait_for(cancelled.wait(), 0.2)
        assert manager.status()["state"] == "disabled"
        before = manager.snapshot()
        await asyncio.sleep(0.02)
        assert len(calls) == 2
    finally:
        await manager.close()

    resumed = []

    async def resume(snapshot):
        resumed.append(snapshot)
        return {"done": True}

    manager = SleepManager(
        path,
        resume,
        lambda s, r: None,
        enabled=True,
        idle_seconds=0,
        step_delay_seconds=0.001,
        max_steps_per_cycle=2,
    )
    try:
        assert manager.snapshot() == before
        assert manager.status()["enabled"] is False
        assert manager.status()["daily_calls"] == 2
        assert manager.status()["step_count"] == 1
        await manager.start()
        await asyncio.sleep(0.02)
        assert resumed == []
        await manager.set_enabled(True)
        await eventually(lambda: manager.status()["pending_cycles"] == 0)
        assert [s["source_run_id"] for s in resumed] == ["two", "one"]
        assert resumed[0]["attempts"] == 2
        assert resumed[1]["observations"] == ["one"]
        assert resumed[1]["hypotheses"] == ["two"]
        assert resumed[1]["completed_checks"] == ["three"]
        assert resumed[1]["next_step"] == "continue"
        manager.foreground_finished({"run_id": "one"})
        await asyncio.sleep(0.02)
        assert len(resumed) == 2
    finally:
        await manager.close()


async def test_checkpoints_keep_bounded_rolling_notes_not_raw_requests(tmp_path):
    saved = []

    async def reflect(snapshot):
        return {
            "observations": [f"obs-{i}" for i in range(200)],
            "hypotheses": ["h" * 10000] * 200,
            "completed_checks": [f"check-{i}" for i in range(200)],
            "next_step": "n" * 10000,
            "request": "RAW_REQUEST_SHOULD_NOT_PERSIST",
            "done": False,
        }

    manager = SleepManager(
        tmp_path / "sleep.db",
        reflect,
        lambda s, r: saved.append(s),
        enabled=True,
        idle_seconds=0,
        step_delay_seconds=0.001,
    )
    try:
        await manager.start()
        manager.foreground_finished(
            {
                "run_id": "one",
                "request": "RAW_REQUEST_SHOULD_NOT_PERSIST",
                "messages": ["RAW_REQUEST_SHOULD_NOT_PERSIST"],
                "metrics": {"workers": 2, "request": "RAW_REQUEST_SHOULD_NOT_PERSIST"},
            }
        )
        await eventually(lambda: len(saved) >= 4)
        await manager.set_enabled(False)
        public = manager.snapshot()
        current = public["current"] or public["queue"][0]
        assert 0 < len(current["observations"]) <= 64
        assert current["observations"][-1] == "obs-199"
        assert len(current["hypotheses"]) <= 64
        assert len(current["hypotheses"][-1]) <= 2048
        assert current["completed_checks"][-1] == "check-199"
        assert len(current["next_step"]) <= 2048
        assert current["metrics"] == {"workers": 2}
        assert "RAW_REQUEST_SHOULD_NOT_PERSIST" not in json.dumps(public)
        assert len(json.dumps(public)) < 200000
        with closing(sqlite3.connect(manager.path)) as db:
            stored = db.execute("SELECT data FROM checkpoint").fetchone()[0]
        assert "RAW_REQUEST_SHOULD_NOT_PERSIST" not in stored
        current["observations"].append("mutated external snapshot")
        assert "mutated external snapshot" not in json.dumps(manager.snapshot())
    finally:
        await manager.close()


async def test_reflection_is_checkpointed_before_preemptible_async_save(tmp_path):
    saving = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()
    returned = asyncio.Event()
    calls = []
    saved = []

    async def reflect(snapshot):
        calls.append(snapshot)
        return {"observations": ["checked"], "done": True}

    save_attempts = []

    async def save(snapshot, result):
        save_attempts.append(snapshot)
        if len(save_attempts) == 1:
            saving.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
            returned.set()
        else:
            saved.append(snapshot)

    manager = SleepManager(
        tmp_path / "sleep.db",
        reflect,
        save,
        enabled=True,
        idle_seconds=0,
        step_delay_seconds=0.001,
        max_steps_per_cycle=1,
    )
    try:
        await manager.start()
        manager.foreground_finished({"run_id": "one"})
        await asyncio.wait_for(saving.wait(), 1)
        await asyncio.wait_for(manager.foreground_started(), 0.2)
        await asyncio.wait_for(cancelled.wait(), 0.2)
        assert manager.snapshot()["current"]["steps_completed"] == 1
        assert manager.snapshot()["current"]["observations"] == ["checked"]
        assert manager.status()["step_count"] == 1
        manager.foreground_finished()
        await asyncio.sleep(0.02)
        assert len(calls) == 1
        release.set()
        await asyncio.wait_for(returned.wait(), 1)
        await eventually(lambda: len(saved) == 1)
        assert saved[0]["attempts"] == 1
        assert saved[0]["steps_completed"] == 1
        assert len(calls) == 1
        assert manager.status()["daily_calls"] == 1
        assert manager.status()["step_count"] == 1
    finally:
        release.set()
        await asyncio.wait_for(manager.close(), 0.2)


async def test_reflection_failure_is_counted_sanitized_and_retried_with_delay(tmp_path):
    calls = []
    saved = []
    failed = asyncio.Event()

    async def reflect(snapshot):
        calls.append(snapshot)
        if len(calls) == 1:
            failed.set()
            raise RuntimeError("private provider request must not reach storage")
        return {"observations": ["recovered"], "done": True}

    manager = SleepManager(
        tmp_path / "sleep.db",
        reflect,
        lambda s, r: saved.append(s),
        enabled=True,
        idle_seconds=0,
        step_delay_seconds=0.03,
        max_steps_per_cycle=1,
    )
    try:
        await manager.start()
        manager.foreground_finished({"run_id": "one"})
        await asyncio.wait_for(failed.wait(), 1)
        await eventually(lambda: manager.status()["last_error"] is not None)
        assert "RuntimeError" in manager.status()["last_error"]
        assert "private provider request" not in json.dumps(manager.status())
        assert manager.status()["daily_calls"] == 1
        assert manager.snapshot()["current"]["steps_completed"] == 0
        await eventually(lambda: len(saved) == 1)
        assert saved[0]["attempts"] == 2
        assert saved[0]["steps_completed"] == 1
        assert manager.status()["step_count"] == 1
        assert manager.status()["daily_calls"] == 2
        assert manager.status()["last_error"] is None
    finally:
        await manager.close()


async def test_saver_failure_keeps_checkpoint_for_retry_without_exposing_exception(tmp_path):
    saved = []
    failed = asyncio.Event()

    async def reflect(snapshot):
        return {"observations": ["checked"], "done": True}

    save_attempts = []

    def save(snapshot, result):
        save_attempts.append(snapshot)
        if len(save_attempts) == 1:
            failed.set()
            raise ValueError("private save payload")
        saved.append(snapshot)

    manager = SleepManager(
        tmp_path / "sleep.db",
        reflect,
        save,
        enabled=True,
        idle_seconds=0,
        step_delay_seconds=0.03,
        max_steps_per_cycle=1,
    )
    try:
        await manager.start()
        manager.foreground_finished({"run_id": "one"})
        await asyncio.wait_for(failed.wait(), 1)
        await eventually(lambda: manager.status()["last_error"] is not None)
        assert "ValueError" in manager.status()["last_error"]
        assert "private save payload" not in json.dumps(manager.status())
        assert manager.snapshot()["current"]["observations"] == ["checked"]
        assert manager.status()["step_count"] == 1
        await eventually(lambda: len(saved) == 1)
        assert saved[0]["attempts"] == 1
        assert saved[0]["steps_completed"] == 1
        assert manager.status()["daily_calls"] == 1
        assert manager.status()["last_error"] is None
    finally:
        await manager.close()


async def test_idle_grace_does_not_wait_for_between_step_delay(tmp_path):
    called = asyncio.Event()
    calls = []
    call_times = []

    async def reflect(snapshot):
        calls.append(snapshot)
        call_times.append(asyncio.get_running_loop().time())
        called.set()
        return {"done": False}

    manager = SleepManager(
        tmp_path / "sleep.db",
        reflect,
        lambda s, r: None,
        enabled=True,
        idle_seconds=0.02,
        step_delay_seconds=60,
    )
    try:
        await manager.start()
        await manager.foreground_started()
        await manager.foreground_started()
        manager.foreground_finished({"run_id": "one"})
        await asyncio.sleep(0.03)
        assert calls == []
        assert manager.status()["foreground_count"] == 1
        finished_at = asyncio.get_running_loop().time()
        manager.foreground_finished()
        # A requested 5ms sleep can overshoot 20ms under Windows load. Check the
        # scheduler's actual start time, not when this test happens to resume.
        await asyncio.wait_for(called.wait(), 2)
        assert call_times[0] - finished_at >= manager.idle_seconds
        assert len(calls) == 1
        await asyncio.sleep(0.03)
        assert len(calls) == 1
    finally:
        await manager.close()


@pytest.mark.parametrize("budget", ["max_steps_per_cycle", "daily_call_limit"])
async def test_explicit_zero_budget_makes_no_reflection_calls(tmp_path, budget):
    calls = []

    async def reflect(snapshot):
        calls.append(snapshot)
        return {"done": False}

    manager = SleepManager(
        tmp_path / "sleep.db",
        reflect,
        lambda s, r: None,
        enabled=True,
        idle_seconds=0,
        step_delay_seconds=0.001,
        **{budget: 0},
    )
    try:
        await manager.start()
        manager.foreground_finished({"run_id": "one"})
        await asyncio.sleep(0.03)
        assert calls == []
        assert manager.status()["daily_calls"] == 0
    finally:
        await manager.close()


async def test_zero_delay_yields_and_retains_recent_observations_across_steps(tmp_path):
    saved = []

    async def reflect(snapshot):
        return {"observations": [str(snapshot["attempts"])], "done": True}

    manager = SleepManager(
        tmp_path / "sleep.db",
        reflect,
        lambda s, r: saved.append(s),
        enabled=True,
        idle_seconds=0,
        step_delay_seconds=0,
    )
    try:
        await manager.start()
        manager.foreground_finished({"run_id": "one"})
        await eventually(lambda: len(saved) >= 80)
        await manager.foreground_started()
        before = len(saved)
        await asyncio.sleep(0.02)
        assert len(saved) == before
        latest = saved[-1]
        assert len(latest["observations"]) == 64
        assert latest["observations"][-1] == str(latest["steps_completed"])
        assert latest["observations"][0] != "1"
        assert manager.status()["daily_calls"] > 24
        assert manager.status()["step_count"] > 3
    finally:
        await manager.close()


async def test_persisted_status_records_checkpoint_phase_and_safe_restart(tmp_path):
    entered = asyncio.Event()

    async def reflect(snapshot):
        entered.set()
        await asyncio.Event().wait()
        return {}

    path = tmp_path / "sleep.db"
    manager = SleepManager(
        path,
        reflect,
        lambda s, r: None,
        enabled=True,
        idle_seconds=0,
        step_delay_seconds=0.001,
    )
    try:
        await manager.start()
        manager.foreground_finished({"run_id": "one"})
        await asyncio.wait_for(entered.wait(), 1)
        with closing(sqlite3.connect(path)) as db:
            running = json.loads(db.execute("SELECT data FROM checkpoint").fetchone()[0])
        assert running["state"] == "reflecting"
        await manager.foreground_started()
        with closing(sqlite3.connect(path)) as db:
            checkpoint = json.loads(db.execute("SELECT data FROM checkpoint").fetchone()[0])
        assert checkpoint["state"] == "foreground"
        assert checkpoint["foreground_count"] == 1
    finally:
        await manager.close()
    restarted = SleepManager(path, reflect, lambda s, r: None)
    try:
        assert restarted.status()["foreground_count"] == 0
        assert restarted.status()["pending_cycles"] == 1
        assert restarted.status()["enabled"] is True
    finally:
        await restarted.close()


@pytest.mark.parametrize(
    "option,value",
    [
        ("idle_seconds", -1),
        ("step_delay_seconds", float("nan")),
        ("max_steps_per_cycle", -1),
        ("daily_call_limit", 1.5),
        ("daily_call_limit", True),
    ],
)
def test_invalid_limits_are_rejected_before_opening_storage(tmp_path, option, value):
    async def reflect(snapshot):
        return {}

    with pytest.raises(ValueError):
        SleepManager(tmp_path / "sleep.db", reflect, lambda s, r: None, **{option: value})
    assert not (tmp_path / "sleep.db").exists()


async def test_restart_resumes_pending_save_without_repeating_reflection(tmp_path):
    saving = asyncio.Event()
    path = tmp_path / "sleep.db"
    reflected = []

    async def reflect(snapshot):
        reflected.append(snapshot)
        return {"observations": ["durable"], "done": True, "lessons": [{"topic": "test"}]}

    async def save(snapshot, result):
        saving.set()
        await asyncio.Event().wait()

    manager = SleepManager(
        path,
        reflect,
        save,
        enabled=True,
        idle_seconds=0,
        step_delay_seconds=0.001,
        max_steps_per_cycle=1,
        daily_call_limit=1,
    )
    try:
        await manager.start()
        manager.foreground_finished({"run_id": "one"})
        await asyncio.wait_for(saving.wait(), 1)
        assert manager.snapshot()["current"]["observations"] == ["durable"]
    finally:
        await manager.close()
    saved = []
    manager = SleepManager(
        path,
        reflect,
        lambda s, r: saved.append((s, r)),
        enabled=True,
        idle_seconds=0,
        step_delay_seconds=0.001,
        max_steps_per_cycle=1,
        daily_call_limit=1,
    )
    try:
        await manager.start()
        await eventually(lambda: manager.status()["pending_cycles"] == 0)
        assert len(reflected) == 1
        assert len(saved) == 1
        assert saved[0][0]["observations"] == ["durable"]
        assert saved[0][1]["lessons"] == [{"topic": "test"}]
        assert manager.status()["step_count"] == 1
        assert manager.status()["daily_calls"] == 1
    finally:
        await manager.close()


async def test_snapshot_is_available_when_completion_arrives_before_start(tmp_path):
    async def reflect(snapshot):
        return {"done": True}

    manager = SleepManager(tmp_path / "sleep.db", reflect, lambda s, r: None)
    try:
        manager.foreground_finished({"run_id": "one"})
        assert manager.snapshot()["queue"][0]["source_run_id"] == "one"
        assert manager.status()["enabled"] is False
        assert manager.status()["daily_calls"] == 0
        await manager.close()
        await manager.start()
        assert manager.status()["state"] == "closed"
    finally:
        await manager.close()
