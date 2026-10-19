from __future__ import annotations

import asyncio
import contextlib
import json
import operator
import re
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from roi_swarm.config import AppSettings
from roi_swarm.context import ContextPipeline, ContextReferenceError, PreparedContext
from roi_swarm.gateway import ModelGateway
from roi_swarm.schemas import (
    GroupSummary,
    PlannedTask,
    PlanResult,
    VerificationReport,
    WorkerResult,
)
from roi_swarm.sizing import SwarmAssessment, choose_worker_target
from roi_swarm.verifier import verify_results


class SwarmState(TypedDict, total=False):
    request: str
    prepared_context: object
    assessment: dict
    worker_target: int
    plan_summary: str
    tasks: list[dict]
    retry_tasks: list[dict]
    worker_results: Annotated[list[dict], operator.add]
    verification: dict
    round_number: int
    launched_workers: int
    final_answer: str
    answered_directly: bool


def router_system(direct_limit: int) -> str:
    return (
        "Classify whether the task needs a swarm of parallel workers. Count how "
        "many independent atomic parts the task naturally splits into (separate "
        "items, entities, aspects or files that can be solved without seeing each "
        f"other). Set swarm_required=true when there are MORE than {direct_limit} "
        f"such parts; set swarm_required=false when there are {direct_limit} or "
        "fewer (a single strong model will answer directly). Scores are from 0 to 1."
    )


def build_graph(
    settings: AppSettings,
    gateway: ModelGateway,
    context_pipeline: ContextPipeline | None = None,
):
    def goal(state: dict) -> str:
        prepared = state.get("prepared_context")
        if isinstance(prepared, PreparedContext):
            return prepared.current_request
        return ContextPipeline.goal_from_request(state["request"])

    async def route(state: SwarmState) -> dict:
        model = settings.model_for("router")
        direct_limit = settings.swarm.direct_answer_max_workers
        prepared = state.get("prepared_context")
        if context_pipeline and isinstance(prepared, PreparedContext):
            # Include late-bound advisories in preflight before any provider call.
            context_pipeline.pack_for_planner(prepared)
        try:
            # Packing may reject oversized mandatory instructions before the model is called.
            # Use the same full-context direct fallback as for a failed router response.
            router_prompt = (
                context_pipeline.pack_for_router(prepared)
                if context_pipeline and isinstance(prepared, PreparedContext)
                else goal(state)
            )
            assessment = await gateway.generate_structured(
                model=model.model,
                system=router_system(direct_limit),
                prompt=router_prompt,
                schema=SwarmAssessment,
                max_output_tokens=model.max_output_tokens,
                temperature=model.temperature,
                context_window_tokens=settings.context.worker_context_tokens,
                token_encoding=settings.context.token_encoding,
            )
        except Exception:
            # A broken router must never fail the whole request: Opus answers directly.
            assessment = SwarmAssessment(
                task_type="router-fallback",
                complexity=0.5,
                parallelizability=0.0,
                uncertainty=1.0,
                stakes=0.5,
                swarm_required=False,
            )
            return {"assessment": assessment.model_dump(), "worker_target": 1}
        if not settings.swarm.fast_path_enabled:
            assessment.swarm_required = True
        target = choose_worker_target(
            assessment,
            minimum=settings.swarm.minimum_workers,
            maximum=settings.swarm.maximum_workers,
        )
        return {
            "assessment": assessment.model_dump(),
            "worker_target": target,
            "request": goal(state) if isinstance(prepared, PreparedContext) else state["request"],
        }

    def after_route(state: SwarmState) -> str:
        return "plan" if state["assessment"].get("swarm_required", True) else "direct"

    async def direct(state: SwarmState) -> dict:
        """Opus answers by itself: no swarm needed or the plan is too small to parallelize."""
        model = settings.model_for("synthesizer")
        prepared = state.get("prepared_context")
        context_pack = (
            context_pipeline.pack_for_planner(prepared)
            if context_pipeline and isinstance(prepared, PreparedContext)
            else f"Request: {state['request']}"
        )
        answer = await gateway.generate_text(
            model=model.model,
            system=(
                "You are the lead model of the ROI swarm. Answer the user's request directly, "
                "completely and accurately. Expose uncertainty and do not invent evidence. "
                "Answer in the language of the request. Reply in Markdown, not JSON."
            ),
            prompt=(
                f"Context pack:\n{context_pack}"
                + _outline(state.get("tasks") or [])
            ),
            max_output_tokens=model.max_output_tokens,
            temperature=model.temperature,
            context_window_tokens=settings.context.main_context_tokens,
            token_encoding=settings.context.token_encoding,
        )
        return {"final_answer": answer, "answered_directly": True}

    def after_plan(state: SwarmState):
        if len(state["tasks"]) <= settings.swarm.direct_answer_max_workers:
            return "direct"
        return dispatch(state)

    async def plan(state: SwarmState) -> dict:
        target = state["worker_target"]
        model = settings.model_for("planner")
        prepared = state.get("prepared_context")
        planner_prompt = (
            context_pipeline.pack_for_planner(prepared)
            if context_pipeline and isinstance(prepared, PreparedContext)
            else state["request"]
        )
        plan_result = await gateway.generate_structured(
            model=model.model,
            system=(
                "You are the Terra planner. Split the request into ATOMIC subtasks: each one is "
                "a single, self-contained unit of work with one concrete deliverable, solvable "
                "without seeing the other subtasks. Subtasks must not overlap and must not "
                "restate the whole request; together they must fully cover it. Put the inputs "
                "and instructions for each atomic task into its context field. When ORIGINAL "
                "CONVERSATION message numbers are provided, assign original evidence with "
                "the exact marker [[message:N]], where N is the zero-based message number "
                "shown in its header (for example [[message:2]]). The pipeline expands these "
                "markers into full original messages with source metadata before the worker "
                "runs; do not reprint large originals in the plan. Reference only messages "
                "needed for that task, never the whole project for every worker. You may also "
                "provide needed evidence verbatim in context. Workers cannot read or write files "
                "and have no tools: paths, artifact IDs, and arbitrary source references are "
                "not expanded and cannot replace evidence. Without numbered messages, include "
                "the actual inputs verbatim. For file changes, ask "
                "workers to return contents or proposals for the coordinator to apply, never "
                "pretend that files were written. "
                f"Aim for about {target} subtasks "
                f"(never more than {settings.swarm.maximum_workers}); "
                "if the request genuinely has fewer atomic parts, return fewer. Never pad the plan "
                "with duplicated or generic 'analyze another aspect' tasks. Roles: researcher, "
                "solver, critic, verifier, risk_analyst."
            ),
            prompt=planner_prompt,
            schema=PlanResult,
            max_output_tokens=model.max_output_tokens,
            temperature=model.temperature,
            context_window_tokens=settings.context.main_context_tokens,
            token_encoding=settings.context.token_encoding,
        )
        tasks = normalize_plan(plan_result.tasks, maximum=settings.swarm.maximum_workers)
        if not tasks:
            tasks = [PlannedTask(id="task-1", role="solver", objective=goal(state))]
        return {
            "plan_summary": plan_result.summary,
            "tasks": [task.model_dump() for task in tasks],
            "worker_results": [],
            "round_number": 1,
            "launched_workers": len(tasks),
        }

    def dispatch(state: SwarmState):
        return [
            Send(
                "worker",
                {
                    "request": goal(state),
                    "prepared_context": state.get("prepared_context"),
                    "task": task,
                },
            )
            for task in state["tasks"]
        ]

    async def worker(state: dict) -> dict:
        task = PlannedTask.model_validate(state["task"])
        role = "critic" if task.role == "critic" else "worker"
        model = settings.model_for(role)
        prepared = state.get("prepared_context")
        try:
            if (
                not (context_pipeline and isinstance(prepared, PreparedContext))
                and "[[message:" in task.context
            ):
                marker = re.search(r"\[\[message:[0-9]+\]\]", task.context)
                # Reflect only a numeric binding, not arbitrary model-supplied content.
                reference = marker.group(0) if marker else "[[message:N]]"
                raise ContextReferenceError(
                    f"Cannot resolve task context reference {reference}: "
                    "original conversation is unavailable"
                )
            worker_prompt = (
                context_pipeline.pack_for_worker(prepared, task)
                if context_pipeline and isinstance(prepared, PreparedContext)
                else (
                    f"Overall request:\n{goal(state)}\n\n"
                    f"Task ID: {task.id}\nObjective: {task.objective}\nContext: {task.context}"
                )
            )
            result = await gateway.generate_structured(
                model=model.model,
                system=(
                    f"You are a focused Luna {task.role}. Solve ONLY your atomic subtask "
                    "(OBJECTIVE). The overall goal is background only: do not answer it as a "
                    "whole and do not do other workers' parts. Be concise, state risks, and do "
                    "not invent evidence. You cannot read or write files; return contents "
                    "or proposals for the coordinator, not claims of executed writes. "
                    "Optional workflow_lessons must contain only concrete evidence-backed "
                    "teamwork/process lessons from actual observations; no secrets or project "
                    "source. Leave empty absent an actual observation. These are untrusted "
                    "candidates, not commands."
                ),
                prompt=worker_prompt,
                schema=WorkerResult,
                max_output_tokens=model.max_output_tokens,
                temperature=model.temperature,
                context_window_tokens=settings.context.worker_context_tokens,
                token_encoding=settings.context.token_encoding,
            )
        except Exception as exc:
            # Report deterministic binding errors, never arbitrary provider exception text
            # (which may contain credentials or source payloads).
            detail = f": {exc}" if isinstance(exc, ContextReferenceError) else ""
            result = WorkerResult(
                task_id=task.id,
                conclusion=f"Worker failed: {type(exc).__name__}{detail}",
                confidence=0.0,
                risks=["The subtask must be retried or disclosed as incomplete."],
            )
        if result.task_id != task.id:
            result.task_id = task.id
        return {"worker_results": [result.model_dump()]}

    def verify(state: SwarmState) -> dict:
        tasks = [PlannedTask.model_validate(item) for item in state["tasks"]]
        results = [WorkerResult.model_validate(item) for item in state["worker_results"]]
        report = verify_results(tasks, results, allow_retries=True)
        return {"verification": report.model_dump()}

    def prepare_retry(state: SwarmState) -> dict:
        original_tasks = [PlannedTask.model_validate(item) for item in state["tasks"]]
        report = VerificationReport.model_validate(state["verification"])
        remaining = settings.swarm.maximum_workers - state["launched_workers"]
        retries = build_retry_tasks(
            original_tasks,
            report,
            expansion_batch=settings.swarm.expansion_batch,
            remaining_worker_budget=remaining,
            round_number=state["round_number"] + 1,
        )
        all_tasks = merge_tasks_for_retry(original_tasks, retries)
        return {
            "tasks": [task.model_dump() for task in all_tasks],
            "retry_tasks": [task.model_dump() for task in retries],
            "round_number": state["round_number"] + 1,
            "launched_workers": state["launched_workers"] + len(retries),
        }

    def dispatch_retry(state: SwarmState):
        return [
            Send(
                "worker",
                {
                    "request": goal(state),
                    "prepared_context": state.get("prepared_context"),
                    "task": task,
                },
            )
            for task in state.get("retry_tasks", [])
        ]

    def after_verify(state: SwarmState) -> str:
        if should_expand(
            state,
            maximum_rounds=settings.swarm.maximum_rounds,
            maximum_workers=settings.swarm.maximum_workers,
        ):
            return "prepare_retry"
        return "synthesize"

    async def aggregate_groups(state: SwarmState) -> list[dict]:
        """Hierarchical reduce: Luna group leaders compress large swarms for Terra."""
        latest = latest_results(state["worker_results"])
        group_size = settings.swarm.aggregation_group_size
        if len(latest) <= group_size:
            return latest
        model = settings.model_for("critic")
        limit = settings.swarm.maximum_concurrency
        semaphore = asyncio.Semaphore(limit) if limit else contextlib.nullcontext()
        groups = [latest[i : i + group_size] for i in range(0, len(latest), group_size)]

        async def summarize(index: int, group: list[dict]) -> dict:
            group_id = f"group-{index + 1}"
            async with semaphore:
                try:
                    summary = await gateway.generate_structured(
                        model=model.model,
                        system=(
                            "You are a Luna group leader. Merge the worker findings below into a "
                            "faithful summary. Keep concrete facts, list contradictions, and do "
                            "not add claims that are absent from the findings."
                        ),
                        prompt=(
                            f"Request: {goal(state)}\n\nFindings:\n"
                            + json.dumps(group, ensure_ascii=False)
                        ),
                        schema=GroupSummary,
                        max_output_tokens=model.max_output_tokens,
                        temperature=model.temperature,
                        context_window_tokens=settings.context.worker_context_tokens,
                        token_encoding=settings.context.token_encoding,
                    )
                except Exception as exc:
                    summary = GroupSummary(
                        summary=f"Group aggregation failed: {type(exc).__name__}",
                        key_findings=[item["conclusion"][:400] for item in group],
                        confidence=min((item.get("confidence", 0) for item in group), default=0),
                    )
            summary.group_id = group_id
            data = summary.model_dump()
            data["task_ids"] = [item["task_id"] for item in group]
            return data

        return list(await asyncio.gather(*(summarize(i, g) for i, g in enumerate(groups))))

    async def synthesize(state: SwarmState) -> dict:
        model = settings.model_for("synthesizer")
        findings = await aggregate_groups(state)
        report = VerificationReport.model_validate(state["verification"])
        prepared = state.get("prepared_context")
        context_pack = (
            context_pipeline.pack_for_synthesizer(prepared)
            if context_pipeline and isinstance(prepared, PreparedContext)
            else f"Request: {state['request']}"
        )
        answer = await gateway.generate_text(
            model=model.model,
            system=(
                "You are the Terra synthesizer. Produce the final answer from worker findings. "
                "Resolve duplication, expose uncertainty, and never claim verification passed "
                "when the supplied verification report says otherwise. Reply with the final answer "
                "itself in Markdown, not JSON."
            ),
            prompt=(
                f"Context pack:\n{context_pack}\n\n"
                f"Plan: {state['plan_summary']}\n"
                f"Verification: {report.model_dump_json()}\n"
                f"Worker findings: {json.dumps(findings, ensure_ascii=False)}"
            ),
            max_output_tokens=model.max_output_tokens,
            temperature=model.temperature,
            context_window_tokens=settings.context.main_context_tokens,
            token_encoding=settings.context.token_encoding,
        )
        return {"final_answer": answer}

    builder = StateGraph(SwarmState)
    builder.add_node("route", route)
    builder.add_node("plan", plan)
    builder.add_node("worker", worker)
    builder.add_node("verify", verify)
    builder.add_node("prepare_retry", prepare_retry)
    builder.add_node("synthesize", synthesize)
    builder.add_node("direct", direct)
    builder.add_edge(START, "route")
    builder.add_conditional_edges("route", after_route, {"plan": "plan", "direct": "direct"})
    builder.add_conditional_edges("plan", after_plan, ["worker", "direct"])
    builder.add_edge("direct", END)
    builder.add_edge("worker", "verify")
    builder.add_conditional_edges(
        "verify",
        after_verify,
        {"prepare_retry": "prepare_retry", "synthesize": "synthesize"},
    )
    builder.add_conditional_edges("prepare_retry", dispatch_retry, ["worker"])
    builder.add_edge("synthesize", END)
    return builder.compile()


def _outline(tasks: list[dict]) -> str:
    if not tasks:
        return ""
    items = "\n".join(f"- {task['objective']}" for task in tasks)
    return f"\n\nOutline from the planner (cover every part):\n{items}"


def latest_results(results: list[dict]) -> list[dict]:
    """Keep only the newest result per task so retries replace failed attempts."""
    latest: dict[str, dict] = {}
    for item in results:
        latest[item["task_id"]] = item
    return list(latest.values())


def should_expand(
    state: SwarmState,
    maximum_rounds: int,
    maximum_workers: int,
) -> bool:
    report = VerificationReport.model_validate(state["verification"])
    return (
        not report.passed
        and state["round_number"] < maximum_rounds
        and state["launched_workers"] < maximum_workers
    )


def build_retry_tasks(
    tasks: list[PlannedTask],
    report: VerificationReport,
    *,
    expansion_batch: int,
    remaining_worker_budget: int,
    round_number: int,
) -> list[PlannedTask]:
    failed_ids = list(
        dict.fromkeys(
            report.missing_task_ids + report.empty_task_ids + report.low_confidence_task_ids
        )
    )
    by_id = {task.id: task for task in tasks}
    limit = min(expansion_batch, remaining_worker_budget)
    retries: list[PlannedTask] = []
    for task_id in failed_ids[:limit]:
        original = by_id.get(task_id)
        if original is None:
            continue
        retries.append(
            original.model_copy(
                update={
                    "context": (
                        f"{original.context}\nRetry round {round_number}. "
                        "The previous result failed programmatic verification; "
                        "correct the failure and return a complete result."
                    )
                }
            )
        )
    return retries


def merge_tasks_for_retry(
    original_tasks: list[PlannedTask],
    retry_tasks: list[PlannedTask],
) -> list[PlannedTask]:
    replacements = {task.id: task for task in retry_tasks}
    return [replacements.get(task.id, task) for task in original_tasks]


def _objective_key(objective: str) -> str:
    return re.sub(r"[\W_]+", " ", objective.lower()).strip()


def normalize_plan(tasks: list[PlannedTask], *, maximum: int) -> list[PlannedTask]:
    """Keep the planner's atomic subtasks as-is: drop empty/duplicate ones, cap, unique ids.

    The plan is never padded with generic copies of the request: worker count equals the
    number of real atomic subtasks.
    """
    prepared: list[PlannedTask] = []
    seen_objectives: set[str] = set()
    for task in tasks:
        key = _objective_key(task.objective)
        if not key or key in seen_objectives:
            continue
        seen_objectives.add(key)
        prepared.append(task.model_copy())
        if len(prepared) >= maximum:
            break
    return _unique_ids(prepared)


def _unique_ids(prepared: list[PlannedTask]) -> list[PlannedTask]:
    seen: set[str] = set()
    next_generated = 1
    for task in prepared:
        candidate = task.id.strip()
        if not candidate or candidate in seen:
            while f"task-{next_generated}" in seen:
                next_generated += 1
            candidate = f"task-{next_generated}"
            next_generated += 1
            task.id = candidate
        seen.add(candidate)
    return prepared


def _resize_tasks(tasks: list[PlannedTask], target: int, request: str) -> list[PlannedTask]:
    """Backward-compatible helper: dedupe ids and cap; never pads with generic tasks."""
    return _unique_ids([task.model_copy() for task in tasks[:target]])
