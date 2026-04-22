"""Pydantic schemas for run-level outputs."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.agent_schemas import (
    AgentStatus,
    BaseAgentOutput,
    validate_agent_output,
)


class FinalOutput(BaseModel):
    """Final assembled response shape for a completed run."""

    task_summary: str = ""
    key_files: list[str] = Field(default_factory=list)
    final_response: str = ""
    reviewer_notes: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: AgentStatus = AgentStatus.PARTIAL


class RunSummaryStatus(str, Enum):
    """Compatibility status values for the scaffolded pipeline summary."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    PLACEHOLDER = "placeholder"
    NOT_STARTED = "not_started"


class RunSummary(FinalOutput):
    """Compatibility summary returned by the scaffolded pipeline."""

    model_config = ConfigDict(populate_by_name=True)

    run_id: str = "scaffold-run"
    status: RunSummaryStatus = RunSummaryStatus.NOT_STARTED
    final_message: str = ""

    @model_validator(mode="after")
    def sync_message_fields(self) -> "RunSummary":
        """Keep compatibility message fields aligned."""

        if self.final_message and not self.final_response:
            self.final_response = self.final_message
        if self.final_response and not self.final_message:
            self.final_message = self.final_response
        return self


class RunLifecycleStatus(str, Enum):
    """Lifecycle states for an in-memory run."""

    INITIALIZED = "initialized"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolHistoryEntry(BaseModel):
    """Lightweight record of one tool invocation."""

    tool_name: str
    status: str
    input_summary: str = ""
    error: str | None = None
    recorded_at: datetime = Field(default_factory=datetime.utcnow)


class RunStateSchema(BaseModel):
    """Shared in-memory structure for one task execution."""

    run_id: str = "scaffold-run"
    user_task: str = ""
    repo_path: str = ""
    task_type: str = "unknown"
    status: RunLifecycleStatus = RunLifecycleStatus.INITIALIZED
    inspected_files: list[str] = Field(default_factory=list)
    notes_path: str | None = None
    agent_outputs: dict[str, BaseAgentOutput] = Field(default_factory=dict)
    tool_history: list[ToolHistoryEntry] = Field(default_factory=list)
    final_output: FinalOutput | None = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


def validate_final_output(payload: dict[str, object]) -> FinalOutput:
    """Validate a final assembled output payload."""

    return FinalOutput.model_validate(payload)


def validate_run_state(payload: dict[str, object]) -> RunStateSchema:
    """Validate a run-state payload."""

    return RunStateSchema.model_validate(payload)


def validate_agent_outputs(
    payload: dict[str, dict[str, object]],
) -> dict[str, BaseAgentOutput]:
    """Validate a mapping of named agent outputs."""

    return {
        agent_name: validate_agent_output(agent_payload)
        for agent_name, agent_payload in payload.items()
    }
