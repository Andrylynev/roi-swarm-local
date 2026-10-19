from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Protocol

import anyio
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from roi_swarm.conversation import current_task
from roi_swarm.experience import WorkflowLesson
from roi_swarm.model_settings import ModelSelection
from roi_swarm.native_budget_errors import NativeContextBudgetError
from roi_swarm.runner import SwarmRunner
from roi_swarm.security import require_provider_auth

STATIC_DIR = Path(__file__).parent / "static"
PROGRESS_POLL_SECONDS = 5.0


async def _settle_stream_task(task: asyncio.Task) -> None:
    """Drain cleanup through AnyIO level cancellation and repeated asyncio cancellation."""
    cancelled = False
    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not task.done():
                    cancelled = True
        with contextlib.suppress(asyncio.CancelledError):
            task.result()
    if cancelled:
        raise asyncio.CancelledError


class _ReservedToolResponse(StreamingResponse):
    """Release admission even if sending headers fails before iteration starts."""

    def __init__(self, *args, reservation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reservation = reservation

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await _settle_stream_task(asyncio.create_task(self.body_iterator.aclose()))
            finally:
                if self.reservation is not None:
                    self.reservation.close()


_NODE_LABELS = {
    "context": "Контекст подготовлен",
    "router": "Router оценил задачу",
    "planner": "Планировщик разбил задачу",
    "retry-planner": "Повтор неудачных подзадач",
    "verifier": "Проверка результатов",
    "direct": "Старшая модель отвечает сама",
    "synthesizer": "Synthesizer собрал ответ",
}


def _describe_event(event: dict[str, Any]) -> str | None:
    node = str(event.get("node", ""))
    status = event.get("status")
    data = event.get("data") or {}
    if node.startswith("merge:"):
        name = node.split(":", 1)[1]
        if status == "merged":
            files = data.get("files") or []
            return f"  ⇄ {name}: слито в общую папку ({len(files)} файлов)" if files else None
        if status == "conflict":
            return f"  ⚠ {name}: конфликт слияния, изменения не применены"
        return None
    if node == "environment" and status == "failed":
        return "⛔ Общее окружение изменено в обход установки: рой остановлен до восстановления"
    if node.startswith("install:"):
        name = node.split(":", 1)[1]
        if status == "waiting":
            others = ", ".join(data.get("waiting_for") or []) or "другие установки"
            return f"⏳ {name}: установка ожидает завершения инструментов — {others}"
        if status == "running":
            command = str(data.get("command", ""))[:100]
            return f"⏸ {name}: центральный агент ставит пакеты, воркеры на паузе — {command}"
        if status == "failed":
            return f"✗ установка для {name} не удалась; ошибка передана воркеру"
        return f"✓ установка для {name} завершена, воркеры продолжают"
    if node.startswith("worker:"):
        name = node.split(":", 1)[1]
        if status == "queued":
            return None
        if status == "running":
            return f"▶ воркер {name}: {str(data.get('objective', ''))[:120]}"
        if status == "tool":
            return f"  · {name} → {data.get('tool')} {str(data.get('args', ''))[:100]}"
        if status == "failed" and data.get("error") == "merge conflict":
            return None
        mark = "✓" if status == "completed" else "✗"
        return f"{mark} воркер {name}"
    if node == "wave":
        if status == "bypass":
            return f"  ⚠ Волна {data.get('wave')}: файлы изменены напрямую в общей папке"
        if status == "isolation":
            if "error" in data:
                return f"Волна {data.get('wave')}: worktree не создан, общая папка"
            return f"Волна {data.get('wave')}: у каждого воркера свой git worktree"
        return f"Волна {data.get('wave')}: {data.get('task_count')} задач для воркеров Hermes"
    if node == "verifier" and "wave" in data:
        verdict = "задача решена" if status == "completed" else "нужна ещё волна"
        return f"Проверка волны {data['wave']}: принято {data.get('accepted', 0)}, {verdict}"
    if node == "swarm":
        return None
    if node == "coordinator":
        if status == "delegated":
            return "Координатор передал задачу рою"
        if status == "running":
            return "Координатор работает с инструментами Hermes"
        return None
    label = _NODE_LABELS.get(node, node)
    if node == "router":
        target = data.get("worker_target")
        return f"{label}: до {target} воркеров" if target else label
    if node in {"planner", "retry-planner"} and "task_count" in data:
        return f"{label}: {data['task_count']} подзадач"
    return label


def _progress_lines(store: Any, run_id: str, seen: int) -> tuple[list[str], int]:
    if store is None:
        return [], seen
    try:
        events = store.get_run(run_id).get("events", [])
    except (KeyError, ValueError, OSError):
        return [], seen
    lines = [line for event in events[seen:] if (line := _describe_event(event))]
    return lines, len(events)


def _family_progress(store: Any, run_id: str, seen: dict[str, int]) -> list[str]:
    """Progress of a run plus the swarm runs it delegated (``<run_id>-s1``, ``-s2``...)."""
    lines, seen[run_id] = _progress_lines(store, run_id, seen.get(run_id, 0))
    children = set()
    if store is not None:
        try:
            events = store.get_run(run_id).get("events", [])
        except (KeyError, ValueError, OSError):
            events = []
        children.update(
            event["data"]["run"] for event in events
            if event.get("node") == "coordinator" and event.get("status") == "delegated"
            and isinstance(event.get("data", {}).get("run"), str)
        )
    index = 1
    while True:
        child = f"{run_id}-s{index}"
        child_lines, count = _progress_lines(store, child, seen.get(child, 0))
        if count == 0 and child not in seen:
            break
        seen[child] = count
        lines.extend(child_lines)
        children.discard(child)
        index += 1
    for child in sorted(children):
        child_lines, seen[child] = _progress_lines(store, child, seen.get(child, 0))
        lines.extend(child_lines)
    return lines


def _reasoning_chunk(completion_id: str, created: int, text: str) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "roi-swarm",
        "choices": [{"index": 0, "delta": {"reasoning_content": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _chunk(completion_id: str, created: int, delta: dict, finish: str | None = None) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "roi-swarm",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _forward_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Hermes history as-is for the coordinator model (no null fields)."""
    return [message.model_dump(exclude_none=True) for message in messages]


class RunnerProtocol(Protocol):
    async def run(
        self,
        request: str,
        run_id: str | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]: ...


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ProjectScope(BaseModel):
    project_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

    @field_validator("project_id")
    @classmethod
    def reserve_global(cls, value: str | None) -> str | None:
        if value == "global":
            raise ValueError("global is reserved for operator-published notes")
        return value


class ChatRequest(ProjectScope):
    model: str = "roi-swarm"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    stream_options: dict[str, Any] | None = None


class PromptRequest(ProjectScope):
    prompt: str = Field(min_length=1)


class ExperienceRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    lesson: WorkflowLesson


class SleepToggle(BaseModel):
    enabled: bool


class ExperienceReview(BaseModel):
    status: Literal["approved", "rejected"]
    publish_global: bool = False


def create_app(runner: RunnerProtocol | None = None) -> FastAPI:
    supplied_runner = runner

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if supplied_runner is None:
            app.state.swarm = SwarmRunner()
        else:
            app.state.swarm = supplied_runner
        try:
            start = getattr(app.state.swarm, "start", None)
            if start is not None:
                await start()
            yield
        finally:
            close = getattr(app.state.swarm, "close", None)
            if close is not None:
                await close()

    app = FastAPI(title="ROI Swarm Provider", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(ValueError)
    async def input_error(_: Request, exc: ValueError):
        message = str(exc)
        if _is_context_limit_error(exc):
            return JSONResponse(status_code=413, content={"detail": message})
        return JSONResponse(status_code=422, content={"detail": "Invalid request input"})

    def swarm_runner() -> RunnerProtocol:
        current = getattr(app.state, "swarm", None)
        if current is None:
            if supplied_runner is not None:
                return supplied_runner
            current = SwarmRunner()
            app.state.swarm = current
        return current

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[Depends(require_provider_auth)])
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "roi-swarm",
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    @app.post(
        "/v1/chat/completions",
        response_model=None,
        dependencies=[Depends(require_provider_auth)],
    )
    async def chat_completions(request: ChatRequest):
        prompt = _last_user_text(request.messages)
        if not prompt:
            raise HTTPException(status_code=422, detail="A non-empty user message is required")
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:18]}"
        swarm = swarm_runner()
        scope = {"project_id": request.project_id} if request.project_id else {}
        if request.tools and hasattr(swarm, "respond_with_tools"):
            forwarded = _forward_messages(request.messages)
            resolve = getattr(swarm, "coordinator_run_id", None)
            if resolve is not None:
                # Validate a complete tool batch before opening an SSE response.
                resolve(forwarded, completion_id, **scope)
            waves = hasattr(swarm, "run_waves") and await swarm.should_use_waves(
                forwarded, request.tools
            )
            if waves and request.stream:
                return StreamingResponse(
                    _stream_wave_turn(swarm, completion_id, forwarded, **scope),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            if waves:
                answer = await swarm.run_waves(forwarded, completion_id, **scope)
                return {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "roi-swarm",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": answer},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
            if request.stream:
                reserve = getattr(swarm, "reserve_tool_turn", None)
                reservation = reserve(forwarded, completion_id, **scope) if reserve else None
                return _ReservedToolResponse(
                    _stream_tool_turn(
                        swarm, completion_id, forwarded, request.tools,
                        reservation=reservation, **scope,
                    ),
                    reservation=reservation,
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            turn = await swarm.respond_with_tools(forwarded, request.tools, completion_id, **scope)
            message: dict[str, Any] = {"role": "assistant", "content": turn.content or None}
            if turn.tool_calls:
                message["tool_calls"] = turn.tool_calls
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "roi-swarm",
                "choices": [{"index": 0, "message": message, "finish_reason": turn.finish_reason}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        serialized_messages = [
            message.model_dump(exclude_none=False) for message in request.messages
        ]
        if request.stream:
            return StreamingResponse(
                _stream_swarm_completion(
                    swarm, completion_id, prompt, serialized_messages, **scope
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        result = await swarm.run(
            prompt,
            run_id=completion_id,
            messages=serialized_messages,
            **scope,
        )
        answer = result["final_answer"]
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "roi-swarm",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "roi_swarm": {
                "run_id": result.get("run_id", completion_id),
                "worker_target": result.get("worker_target"),
            },
        }

    @app.get("/api/model-settings", dependencies=[Depends(require_provider_auth)])
    async def model_settings() -> dict[str, Any]:
        swarm = swarm_runner()
        if not hasattr(swarm, "model_settings"):
            raise HTTPException(status_code=503, detail="Model settings unavailable")
        return swarm.model_settings()

    @app.get("/api/model-options", dependencies=[Depends(require_provider_auth)])
    async def model_options() -> dict[str, Any]:
        swarm = swarm_runner()
        if not hasattr(swarm, "model_options"):
            raise HTTPException(status_code=503, detail="Model catalog unavailable")
        return await swarm.model_options()

    @app.put("/api/model-settings", dependencies=[Depends(require_provider_auth)])
    async def update_model_settings(request: ModelSelection) -> dict[str, Any]:
        swarm = swarm_runner()
        if not hasattr(swarm, "update_model_settings"):
            raise HTTPException(status_code=503, detail="Model settings unavailable")
        try:
            return swarm.update_model_settings(request.roles)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=409, detail="Дождитесь завершения активных запросов."
            ) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=503, detail="Не удалось сохранить настройки моделей."
            ) from exc

    @app.get("/api/sleep", dependencies=[Depends(require_provider_auth)])
    async def sleep_status() -> dict[str, Any]:
        current = swarm_runner()
        if not hasattr(current, "sleep_status"):
            raise HTTPException(status_code=503, detail="Sleep unavailable")
        return current.sleep_status()

    @app.get("/api/sleep/snapshot", dependencies=[Depends(require_provider_auth)])
    async def sleep_snapshot() -> dict[str, Any]:
        current = swarm_runner()
        if not hasattr(current, "sleep_snapshot"):
            raise HTTPException(status_code=503, detail="Sleep unavailable")
        return current.sleep_snapshot()

    @app.put("/api/sleep", dependencies=[Depends(require_provider_auth)])
    async def sleep_toggle(request: SleepToggle) -> dict[str, Any]:
        current = swarm_runner()
        if not hasattr(current, "set_sleep_enabled"):
            raise HTTPException(status_code=503, detail="Sleep unavailable")
        try:
            return await current.set_sleep_enabled(request.enabled)
        except (OSError, sqlite3.Error) as exc:
            raise HTTPException(status_code=503, detail="Sleep storage unavailable") from exc

    @app.post("/api/sleep/release-tools", dependencies=[Depends(require_provider_auth)])
    async def release_abandoned_tools() -> dict[str, Any]:
        current = swarm_runner()
        if not hasattr(current, "release_tool_leases"):
            raise HTTPException(status_code=503, detail="Tool lease management unavailable")
        try:
            return await current.release_tool_leases()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail="Requests are active") from exc
        except (OSError, sqlite3.Error) as exc:
            raise HTTPException(status_code=503, detail="Sleep storage unavailable") from exc

    def experience_store():
        store = getattr(swarm_runner(), "experience", None)
        if store is None:
            raise HTTPException(status_code=503, detail="Experience memory disabled")
        return store

    @app.get("/api/experience", dependencies=[Depends(require_provider_auth)])
    async def experience_notes(
        project_id: str,
        status: Literal["candidate", "approved", "rejected"] | None = None,
    ) -> list[dict[str, Any]]:
        try:
            return experience_store().list_notes(project_id, status=status)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Invalid experience scope") from exc
        except (OSError, sqlite3.Error) as exc:
            raise HTTPException(status_code=503, detail="Experience storage unavailable") from exc

    @app.post("/api/experience", dependencies=[Depends(require_provider_auth)])
    async def add_experience(request: ExperienceRequest) -> dict[str, Any]:
        try:
            return experience_store().record(
                request.lesson,
                project_id=request.project_id,
                run_id=request.run_id,
                origin="operator",
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail="Invalid or sensitive workflow note"
            ) from exc
        except (OSError, sqlite3.Error) as exc:
            raise HTTPException(status_code=503, detail="Experience storage unavailable") from exc

    @app.patch("/api/experience/{note_id}", dependencies=[Depends(require_provider_auth)])
    async def review_experience(note_id: int, request: ExperienceReview) -> dict[str, Any]:
        store = experience_store()
        if request.publish_global and request.status != "approved":
            raise HTTPException(status_code=422, detail="Only approved notes can be shared")
        try:
            if request.publish_global:
                return store.publish(note_id)
            return store.review(note_id, request.status)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Experience note not found") from exc
        except (OSError, sqlite3.Error) as exc:
            raise HTTPException(status_code=503, detail="Experience storage unavailable") from exc

    @app.post("/api/run", dependencies=[Depends(require_provider_auth)])
    async def run_prompt(request: PromptRequest) -> dict[str, Any]:
        swarm = swarm_runner()
        messages = [{"role": "user", "content": request.prompt}]
        scope = {"project_id": request.project_id} if request.project_id else {}
        return await swarm.run(request.prompt, messages=messages, **scope)

    @app.get("/api/runs", dependencies=[Depends(require_provider_auth)])
    async def list_runs() -> list[dict[str, Any]]:
        swarm = swarm_runner()
        store = getattr(swarm, "store", None)
        if store is None:
            return []
        return store.list_runs()

    @app.get("/api/runs/{run_id}", dependencies=[Depends(require_provider_auth)])
    async def get_run(run_id: str) -> dict[str, Any]:
        swarm = swarm_runner()
        store = getattr(swarm, "store", None)
        if store is None:
            raise HTTPException(status_code=404, detail="Run store unavailable")
        try:
            return store.get_run(run_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/dashboard", response_class=FileResponse)
    async def dashboard() -> FileResponse:
        return FileResponse(STATIC_DIR / "dashboard.html")

    return app


def _is_context_limit_error(exc: Exception) -> bool:
    if isinstance(exc, NativeContextBudgetError):
        return True
    message = str(exc)
    return (
        isinstance(exc, ValueError)
        and message.startswith(("Planner input is ", "Context exceeds "))
        and "Split the task" in message
    )


def _safe_error(exc: Exception) -> str:
    if _is_context_limit_error(exc):
        return f"ROI Swarm error: {exc}"
    return f"ROI Swarm error: {type(exc).__name__}"


async def _stream_swarm_completion(
    swarm: RunnerProtocol,
    completion_id: str,
    prompt: str,
    messages: list[dict[str, Any]],
    *,
    project_id: str | None = None,
):
    created = int(time.time())
    initial = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "roi-swarm",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(initial, ensure_ascii=False, separators=(',', ':'))}\n\n"
    scope = {"project_id": project_id} if project_id else {}
    task = asyncio.create_task(swarm.run(prompt, run_id=completion_id, messages=messages, **scope))
    store = getattr(swarm, "store", None)
    seen_events = 0
    try:
        while True:
            try:
                result = await asyncio.wait_for(asyncio.shield(task), timeout=PROGRESS_POLL_SECONDS)
                break
            except TimeoutError:
                # SSE comments are invisible to OpenAI clients, so Hermes treated a long
                # swarm as a dead stream. Real reasoning deltas keep the stream alive and
                # show the user what the swarm is doing.
                lines, seen_events = _progress_lines(store, completion_id, seen_events)
                text = "".join(f"{line}\n" for line in lines) or "·"
                yield _reasoning_chunk(completion_id, created, text)
        answer = result["final_answer"]
    except asyncio.CancelledError:
        task.cancel()
        raise
    except Exception as exc:
        answer = _safe_error(exc)
    for chunk in _stream_chat_completion(completion_id, answer, include_role=False):
        yield chunk


async def _stream_wave_turn(
    swarm: Any,
    completion_id: str,
    messages: list[dict[str, Any]],
    *,
    project_id: str | None = None,
):
    """Single stream for the whole task: progress as reasoning, final summary as content."""
    created = int(time.time())
    yield _chunk(completion_id, created, {"role": "assistant", "content": ""})
    scope = {"project_id": project_id} if project_id else {}
    deltas: asyncio.Queue[str] = asyncio.Queue()
    task = asyncio.create_task(
        swarm.run_waves(messages, completion_id, on_delta=deltas.put_nowait, **scope)
    )
    store = getattr(swarm, "store", None)
    seen_events = 0
    streamed = False
    try:
        while not task.done() or not deltas.empty():
            getter = asyncio.ensure_future(deltas.get())
            done, _ = await asyncio.wait(
                {getter, task}, timeout=PROGRESS_POLL_SECONDS, return_when=asyncio.FIRST_COMPLETED
            )
            if getter not in done:
                getter.cancel()
                if task.done():
                    continue
                lines, seen_events = _progress_lines(store, completion_id, seen_events)
                yield _reasoning_chunk(
                    completion_id, created, "".join(f"{line}\n" for line in lines) or "·"
                )
                continue
            text = getter.result()
            if not streamed:
                lines, seen_events = _progress_lines(store, completion_id, seen_events)
                if lines:
                    yield _reasoning_chunk(completion_id, created, "\n".join(lines) + "\n")
            streamed = True
            yield _chunk(completion_id, created, {"content": text})
        try:
            answer = task.result()
        except Exception as exc:
            error = _safe_error(exc)
            answer = f"\n\n{error}" if streamed else error
            streamed = False
        if not streamed:
            yield _chunk(completion_id, created, {"content": answer})
        yield _chunk(completion_id, created, {}, "stop")
        yield "data: [DONE]\n\n"
    finally:
        # Client gone (GeneratorExit at a yield) or cancelled: stop paid workers too.
        if not task.done():
            task.cancel()


async def _stream_tool_turn(
    swarm: Any,
    completion_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    project_id: str | None = None,
    reservation: Any = None,
):
    created = int(time.time())
    scope = {"project_id": project_id} if project_id else {}
    resolve = getattr(swarm, "coordinator_run_id", None)
    logical_id = reservation.logical_id if reservation is not None else (
        resolve(messages, completion_id, **scope) if resolve is not None else completion_id
    )
    task = None
    store = getattr(swarm, "store", None)
    seen: dict[str, int] = {}
    try:
        yield _chunk(completion_id, created, {"role": "assistant", "content": ""})
        kwargs = {**scope, "_reservation": reservation} if reservation is not None else scope
        task = asyncio.create_task(
            swarm.respond_with_tools(messages, tools, completion_id, **kwargs)
        )
        while True:
            try:
                turn = await asyncio.wait_for(asyncio.shield(task), timeout=PROGRESS_POLL_SECONDS)
                break
            except TimeoutError:
                lines = _family_progress(store, logical_id, seen)
                text = "".join(f"{line}\n" for line in lines) or "·"
                yield _reasoning_chunk(completion_id, created, text)
    except Exception as exc:
        yield _chunk(completion_id, created, {"content": _safe_error(exc)})
        yield _chunk(completion_id, created, {}, "stop")
        yield "data: [DONE]\n\n"
        return
    finally:
        if task is not None and not task.done():
            task.cancel()
            await _settle_stream_task(task)
    if turn.content:
        yield _chunk(completion_id, created, {"content": turn.content})
    for index, call in enumerate(turn.tool_calls):
        yield _chunk(
            completion_id,
            created,
            {"tool_calls": [{"index": index, **call}]},
        )
    yield _chunk(completion_id, created, {}, turn.finish_reason)
    yield "data: [DONE]\n\n"


def _stream_chat_completion(
    completion_id: str,
    answer: str,
    *,
    include_role: bool = True,
):
    created = int(time.time())
    chunks = []
    if include_role:
        chunks.append(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "roi-swarm",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
        )
    chunks.extend(
        [
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "roi-swarm",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": answer},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "roi-swarm",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
    )
    for chunk in chunks:
        yield f"data: {json.dumps(chunk, ensure_ascii=False, separators=(',', ':'))}\n\n"
    yield "data: [DONE]\n\n"


def _last_user_text(messages: list[ChatMessage]) -> str:
    return current_task(_forward_messages(messages))[1]


app = create_app()
