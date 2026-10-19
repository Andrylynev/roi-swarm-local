from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str
    api_key: str = Field(repr=False)
    # None = без таймаута на запрос к провайдеру
    timeout_seconds: float | None = Field(default=None, gt=0)
    max_retries: int = Field(default=2, ge=0)
    api_mode: str = "responses"

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("api_mode")
    @classmethod
    def validate_api_mode(cls, value: str) -> str:
        allowed = {"responses", "chat_completions"}
        if value not in allowed:
            raise ValueError(f"api_mode must be one of: {sorted(allowed)}")
        return value


class ModelRoleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=4096, gt=0)


class SwarmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_workers: int = Field(default=10, ge=10, le=200)
    maximum_workers: int = Field(default=70, ge=10, le=200)
    # None = без ограничения: сколько подзадач, столько воркеров работает одновременно
    maximum_concurrency: int | None = Field(default=None, gt=0)
    # Если план требует не больше стольких воркеров, Opus отвечает сам без роя
    direct_answer_max_workers: int = Field(default=3, ge=0)
    expansion_batch: int = Field(default=10, gt=0)
    maximum_rounds: int = Field(default=5, gt=0)
    fast_path_enabled: bool = True
    aggregation_group_size: int = Field(default=15, ge=5, le=50)

    @model_validator(mode="after")
    def validate_worker_range(self) -> SwarmConfig:
        if self.minimum_workers > self.maximum_workers:
            raise ValueError("minimum_workers cannot exceed maximum_workers")
        return self


class ContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_threshold_chars: int = Field(default=8000, gt=0)
    recent_messages: int = Field(default=8, gt=0)
    # Legacy explicit character caps are optional, disabled in the shipped configuration.
    maximum_pack_chars: int | None = Field(default=None, gt=1000)
    planner_input_limit_chars: int | None = Field(default=None, gt=1000)
    worker_context_tokens: int = Field(default=800000, gt=0)
    main_context_tokens: int = Field(default=1000000, gt=0)
    token_encoding: str = "o200k_base"


class RagConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    database_path: str = "data/rag/rag.sqlite"
    top_k: int = Field(default=6, gt=0, le=30)
    collections: list[str] = Field(default_factory=list)
    project_collections: dict[str, list[str]] = Field(default_factory=dict)
    embeddings_enabled: bool = False
    embedding_model: str = "ibm-granite/granite-embedding-97m-multilingual-r2"
    embedding_dimension: int = Field(default=384, gt=0)


class ExperienceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    database_path: str = "data/experience/experience.sqlite"
    maximum_lessons: int = Field(default=6, ge=0, le=12)


class SleepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    idle_seconds: float = Field(default=15, ge=0)
    step_delay_seconds: float = Field(default=30, ge=0)
    # None means continuous reflection while idle, without a call/step quota.
    max_steps_per_cycle: int | None = Field(default=None, ge=1)
    daily_call_limit: int | None = Field(default=None, ge=1)
    auto_improve: bool = False
    # Verified improvements fast-forward only into this branch (None: current branch).
    merge_branch: str | None = None
    # After a merge, restart the hidden service once idle so the new code is loaded.
    auto_restart: bool = False


class HermesWorkersConfig(BaseModel):
    """Streaming wave mode: every atomic task runs as a full Hermes agent with tools."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # Hermes' own interpreter and checkout (Hermes may run on another Python than ROI).
    python: str = ""
    root: str = ""
    # Fallback working directory when Hermes' system prompt does not state one.
    default_cwd: str = ""
    # Workers never spawn new tasks, ask the user or schedule jobs: that is the
    # central agent's job. Memory writes from parallel workers would race.
    disabled_toolsets: list[str] = Field(
        default_factory=lambda: [
            "delegation",
            "clarify",
            "cronjob",
            "kanban",
            "memory",
            "desktop_ui",
            "project",
        ]
    )
    max_iterations: int = Field(default=60, gt=0)
    maximum_waves: int = Field(default=4, gt=0)
    maximum_concurrency: int | None = Field(default=None, gt=0)
    # Each worker of a multi-task wave gets its own git worktree; results are merged back.
    isolate_worktrees: bool = True
    # Where worker worktrees live (empty = %TEMP%/roi-waves).
    worktree_root: str = ""
    # Big ignored dirs linked (junction) into every worktree instead of being rebuilt.
    linked_dirs: list[str] = Field(default_factory=lambda: [".venv", "node_modules"])


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderConfig
    models: dict[str, ModelRoleConfig]
    hermes_workers: HermesWorkersConfig = Field(default_factory=HermesWorkersConfig)
    swarm: SwarmConfig = Field(default_factory=SwarmConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    experience: ExperienceConfig = Field(default_factory=ExperienceConfig)
    sleep: SleepConfig = Field(default_factory=SleepConfig)

    @model_validator(mode="after")
    def validate_required_model_roles(self) -> AppSettings:
        required = {"router", "planner", "worker", "critic", "synthesizer"}
        missing = sorted(required - self.models.keys())
        if missing:
            raise ValueError(f"Missing required model roles: {', '.join(missing)}")
        return self

    def model_for(self, role: str) -> ModelRoleConfig:
        try:
            return self.models[role]
        except KeyError as exc:
            raise ValueError(f"Model role is not configured: {role}") from exc
