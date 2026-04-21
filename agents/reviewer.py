"""Reviewer agent scaffold."""

from schemas.agent_schemas import BaseAgentOutput


class ReviewerAgent:
    """Future agent responsible for review feedback and quality checks."""

    def run(self) -> BaseAgentOutput:
        """Return a placeholder response for the reviewer agent."""

        return BaseAgentOutput(
            agent_name="reviewer",
            summary="Reviewer agent scaffold only.",
        )
