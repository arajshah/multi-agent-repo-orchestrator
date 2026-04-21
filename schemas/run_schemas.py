"""Pydantic schemas for run-level outputs."""

from pydantic import BaseModel


class RunSummary(BaseModel):
    """Minimal summary returned by the future orchestrator."""

    run_id: str = "scaffold-run"
    status: str = "not_started"
    final_message: str = "Pipeline scaffold only."
