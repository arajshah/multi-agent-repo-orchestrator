"""Pydantic schemas for agent-facing outputs."""

from enum import Enum
from typing import TypeVar

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class AgentStatus(str, Enum):
    """Execution status shared by all agent outputs."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


class BaseAgentOutput(BaseModel):
    """Base output shape shared by future agent modules."""

    model_config = ConfigDict(populate_by_name=True)

    agent_name: str
    task_summary: str = Field(
        default="Scaffold response only.",
        validation_alias=AliasChoices("task_summary", "summary"),
    )
    reasoning_summary: str = ""
    key_findings: list[str] = Field(default_factory=list)
    next_action: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: AgentStatus = AgentStatus.PARTIAL

    @property
    def summary(self) -> str:
        """Compatibility accessor for older scaffold code."""

        return self.task_summary


class PlannerOutput(BaseAgentOutput):
    """Structured output for the planner agent."""

    task_type: str = "unknown"
    workflow_steps: list[str] = Field(default_factory=list)
    recommended_tools: list[str] = Field(default_factory=list)
    output_mode: str = "analysis"


class EvidenceSnippet(BaseModel):
    """Small structured evidence excerpt used by the analyst."""

    file_path: str = ""
    snippet: str
    line_reference: str = ""


class AnalystOutput(BaseAgentOutput):
    """Structured output for the analyst agent."""

    relevant_files: list[str] = Field(default_factory=list)
    relevant_symbols: list[str] = Field(default_factory=list)
    evidence_snippets: list[EvidenceSnippet] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class ImplementationPlannerOutput(BaseAgentOutput):
    """Structured output for the implementation planner agent."""

    proposed_changes: list[str] = Field(default_factory=list)
    minimal_files_to_touch: list[str] = Field(default_factory=list)
    implementation_steps: list[str] = Field(default_factory=list)
    risks_or_assumptions: list[str] = Field(default_factory=list)


class ReviewIssue(BaseModel):
    """A focused issue raised by the reviewer."""

    severity: str = "medium"
    description: str


class ReviewerOutput(BaseAgentOutput):
    """Structured output for the reviewer agent."""

    issues_found: list[ReviewIssue] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    revisions_requested: list[str] = Field(default_factory=list)
    final_assessment: str = ""


AgentOutputT = TypeVar("AgentOutputT", bound=BaseAgentOutput)


def validate_agent_output(
    payload: dict[str, object],
    schema: type[AgentOutputT] = BaseAgentOutput,
) -> AgentOutputT:
    """Validate an agent payload against a chosen schema."""

    return schema.model_validate(payload)
