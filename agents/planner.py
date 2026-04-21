"""Planner agent scaffold."""

from schemas.agent_schemas import BaseAgentOutput


class PlannerAgent:
    """Future agent responsible for high-level task planning."""

    def run(self) -> BaseAgentOutput:
        """Return a placeholder response for the planner agent."""

        return BaseAgentOutput(
            agent_name="planner",
            summary="Planner agent scaffold only.",
        )
