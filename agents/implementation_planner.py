"""Implementation planner agent scaffold."""

from schemas.agent_schemas import BaseAgentOutput


class ImplementationPlannerAgent:
    """Future agent responsible for converting plans into implementation steps."""

    def run(self) -> BaseAgentOutput:
        """Return a placeholder response for the implementation planner."""

        return BaseAgentOutput(
            agent_name="implementation_planner",
            summary="Implementation planner scaffold only.",
        )
