from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from roi_swarm.experience import WorkflowLesson


class PlannedTask(BaseModel):
    id: str
    objective: str
    role: str = "researcher"
    context: str = ""


class PlanResult(BaseModel):
    summary: str
    tasks: list[PlannedTask]


class WorkerResult(BaseModel):
    task_id: str
    conclusion: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    risks: list[str] = Field(default_factory=list)
    workflow_lessons: list[WorkflowLesson] = Field(default_factory=list, max_length=2)

    @field_validator("workflow_lessons", mode="before")
    @classmethod
    def optional_lessons(cls, value: Any) -> list[WorkflowLesson]:
        if not isinstance(value, list):
            return []
        valid = []
        for item in value:
            try:
                valid.append(WorkflowLesson.model_validate(item))
            except ValueError:
                continue
            if len(valid) == 2:
                break
        return valid


class VerificationReport(BaseModel):
    passed: bool
    missing_task_ids: list[str] = Field(default_factory=list)
    empty_task_ids: list[str] = Field(default_factory=list)
    low_confidence_task_ids: list[str] = Field(default_factory=list)
    duplicate_task_ids: list[str] = Field(default_factory=list)
    duplicate_result_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class GroupSummary(BaseModel):
    group_id: str = ""
    summary: str
    key_findings: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class FinalResult(BaseModel):
    answer: str
