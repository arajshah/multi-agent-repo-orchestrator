"""Placeholder pipeline entrypoints for future orchestration."""

from memory.run_state import RunState
from schemas.run_schemas import RunSummary


class Pipeline:
    """Future orchestration pipeline for RepoPilot runs."""

    def start(self) -> RunSummary:
        """Return a minimal scaffold summary without executing work."""

        state = RunState()
        return RunSummary(
            run_id=state.run_id,
            status="placeholder",
            final_message="Pipeline orchestration is not implemented yet.",
        )
