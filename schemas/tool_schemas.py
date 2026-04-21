"""Pydantic schemas for tool-facing outputs."""

from enum import Enum
from typing import Dict

from pydantic import BaseModel


class ToolStatus(str, Enum):
    """Minimal lifecycle states for scaffolded tool responses."""

    PLACEHOLDER = "placeholder"


class BaseToolOutput(BaseModel):
    """Base output shape shared by future tool modules."""

    tool_name: str
    status: ToolStatus = ToolStatus.PLACEHOLDER
    message: str = "Tool scaffold only."
    metadata: Dict[str, str] = {}
