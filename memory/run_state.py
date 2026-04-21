"""Placeholder state model for orchestrator runs."""

from datetime import datetime

from pydantic import BaseModel, Field


class RunState(BaseModel):
    """Track minimal scaffold state for a run."""

    run_id: str = "scaffold-run"
    status: str = "initialized"
    created_at: datetime = Field(default_factory=datetime.utcnow)
