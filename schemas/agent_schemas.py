"""Pydantic schemas for agent-facing outputs."""

from enum import Enum

from pydantic import BaseModel


class AgentStatus(str, Enum):
    """Minimal lifecycle states for scaffolded agent responses."""

    PLACEHOLDER = "placeholder"


class BaseAgentOutput(BaseModel):
    """Base output shape shared by future agent modules."""

    agent_name: str
    status: AgentStatus = AgentStatus.PLACEHOLDER
    summary: str = "Scaffold response only."
