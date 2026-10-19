"""Structured public research checkpoint, not private model reasoning."""

from pydantic import BaseModel, ConfigDict, Field

from roi_swarm.experience import WorkflowLesson


class ReflectionStep(BaseModel):
    observations: list[str] = Field(default_factory=list, max_length=12)
    hypotheses: list[str] = Field(default_factory=list, max_length=12)
    completed_checks: list[str] = Field(default_factory=list, max_length=12)
    next_step: str = Field(default="", max_length=800)
    done: bool = False
    lessons: list[WorkflowLesson] = Field(default_factory=list, max_length=2)


class FileReplacement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=256)
    content: str = Field(max_length=100000)


class ImprovementProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=120)
    changes: list[FileReplacement] = Field(default_factory=list, max_length=20)


class ImprovementReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool = False
    concerns: list[str] = Field(default_factory=list, max_length=20)
