"""Placeholder note-oriented tools for future workflow memory."""

from schemas.tool_schemas import BaseToolOutput


def record_note() -> BaseToolOutput:
    """Return a placeholder result for future note-taking helpers."""

    return BaseToolOutput(
        tool_name="record_note",
        message="Note writing is not implemented in the scaffold phase.",
    )
