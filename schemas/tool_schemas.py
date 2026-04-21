"""Pydantic schemas for tool-facing outputs."""

from enum import Enum

from pydantic import BaseModel, Field


class ToolStatus(str, Enum):
    """Execution status for tool results."""

    SUCCESS = "success"
    ERROR = "error"


class BaseToolOutput(BaseModel):
    """Base output shape shared by tool modules."""

    tool_name: str
    status: ToolStatus
    error: str | None = None


class FileListResult(BaseToolOutput):
    """Structured result for recursive file listing."""

    repo_path: str
    files: list[str] = Field(default_factory=list)
    total_count: int = 0
    excluded_directories: list[str] = Field(default_factory=list)


class SearchResultEntry(BaseModel):
    """A single text match found in a repository search."""

    file_path: str
    line_number: int
    line_text: str


class SearchCodeResult(BaseToolOutput):
    """Structured result for repository text search."""

    repo_path: str
    query: str
    matches: list[SearchResultEntry] = Field(default_factory=list)
    total_matches: int = 0


class FileMetadata(BaseModel):
    """Simple metadata for a text file."""

    line_count: int
    character_count: int


class ReadFileResult(BaseToolOutput):
    """Structured result for reading a full text file."""

    file_path: str
    content: str = ""
    metadata: FileMetadata | None = None


class LineRange(BaseModel):
    """1-based inclusive line range."""

    start_line: int
    end_line: int


class ReadFileChunkResult(BaseToolOutput):
    """Structured result for reading a bounded line range."""

    file_path: str
    requested_range: LineRange
    actual_range: LineRange | None = None
    content_lines: list[str] = Field(default_factory=list)


class WriteNoteResult(BaseToolOutput):
    """Structured result for appending a note file."""

    note_path: str
    characters_written: int = 0
    mode: str = "append"
