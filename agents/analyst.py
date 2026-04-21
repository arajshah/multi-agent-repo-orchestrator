"""Analyst agent scaffold."""

from schemas.agent_schemas import BaseAgentOutput


class AnalystAgent:
    """Future agent responsible for repository analysis and context synthesis."""

    def run(self) -> BaseAgentOutput:
        """Return a placeholder response for the analyst agent."""

        return BaseAgentOutput(
            agent_name="analyst",
            summary="Analyst agent scaffold only.",
        )
