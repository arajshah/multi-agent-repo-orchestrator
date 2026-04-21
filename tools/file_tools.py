"""Placeholder file-oriented tools for future repository operations."""

from schemas.tool_schemas import BaseToolOutput


def inspect_repository() -> BaseToolOutput:
    """Return a placeholder result for future repository inspection tools."""

    return BaseToolOutput(
        tool_name="inspect_repository",
        message="Repository inspection is not implemented in the scaffold phase.",
    )
