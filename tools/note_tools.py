"""Writable note tool for scratch run output.

Provides write_note, which appends text to a plain-text or markdown
file, creating parent directories as needed, and record_note, a
compatibility wrapper kept for the scaffolded package export.
"""

from pathlib import Path

from schemas.tool_schemas import ToolStatus, WriteNoteResult


def write_note(note_path: str, content: str) -> WriteNoteResult:
    """Append note content to a plain-text or markdown file."""

    path = Path(note_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as note_file:
            note_file.write(content)
    except OSError as exc:
        return WriteNoteResult(
            tool_name="write_note",
            status=ToolStatus.ERROR,
            note_path=str(path),
            error=f"Unable to write note: {exc}",
        )

    return WriteNoteResult(
        tool_name="write_note",
        status=ToolStatus.SUCCESS,
        note_path=str(path),
        characters_written=len(content),
        mode="append",
    )


def record_note(note_path: str, content: str) -> WriteNoteResult:
    """Compatibility wrapper for the scaffolded package export."""

    return write_note(note_path=note_path, content=content)
