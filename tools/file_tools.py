"""Read-only repository analysis tools."""

from pathlib import Path

from schemas.tool_schemas import (
    FileListResult,
    FileMetadata,
    LineRange,
    ReadFileChunkResult,
    ReadFileResult,
    SearchCodeResult,
    SearchResultEntry,
    ToolStatus,
)


NOISE_DIRECTORIES = {".git", "__pycache__", ".venv", "node_modules", "runs"}


def list_files(repo_path: str, limit: int | None = None) -> FileListResult:
    """Return a recursive list of relative file paths under a repository path."""

    repo = Path(repo_path)
    if not repo.exists():
        return FileListResult(
            tool_name="list_files",
            status=ToolStatus.ERROR,
            repo_path=str(repo),
            excluded_directories=sorted(NOISE_DIRECTORIES),
            error=f"Repository path does not exist: {repo}",
        )

    if not repo.is_dir():
        return FileListResult(
            tool_name="list_files",
            status=ToolStatus.ERROR,
            repo_path=str(repo),
            excluded_directories=sorted(NOISE_DIRECTORIES),
            error=f"Repository path is not a directory: {repo}",
        )

    if limit is not None and limit < 0:
        return FileListResult(
            tool_name="list_files",
            status=ToolStatus.ERROR,
            repo_path=str(repo),
            excluded_directories=sorted(NOISE_DIRECTORIES),
            error="Limit must be greater than or equal to 0.",
        )

    if limit == 0:
        return FileListResult(
            tool_name="list_files",
            status=ToolStatus.SUCCESS,
            repo_path=str(repo),
            files=[],
            total_count=0,
            excluded_directories=sorted(NOISE_DIRECTORIES),
        )

    files: list[str] = []
    for file_path in _iter_files(repo):
        files.append(str(file_path.relative_to(repo)))
        if limit is not None and len(files) >= limit:
            break

    return FileListResult(
        tool_name="list_files",
        status=ToolStatus.SUCCESS,
        repo_path=str(repo),
        files=files,
        total_count=len(files),
        excluded_directories=sorted(NOISE_DIRECTORIES),
    )


def search_code(repo_path: str, query: str, limit: int | None = None) -> SearchCodeResult:
    """Search for a plain-text query across repository files."""

    repo = Path(repo_path)
    if not repo.exists():
        return SearchCodeResult(
            tool_name="search_code",
            status=ToolStatus.ERROR,
            repo_path=str(repo),
            query=query,
            error=f"Repository path does not exist: {repo}",
        )

    if not repo.is_dir():
        return SearchCodeResult(
            tool_name="search_code",
            status=ToolStatus.ERROR,
            repo_path=str(repo),
            query=query,
            error=f"Repository path is not a directory: {repo}",
        )

    if not query:
        return SearchCodeResult(
            tool_name="search_code",
            status=ToolStatus.ERROR,
            repo_path=str(repo),
            query=query,
            error="Query must not be empty.",
        )

    if limit is not None and limit < 0:
        return SearchCodeResult(
            tool_name="search_code",
            status=ToolStatus.ERROR,
            repo_path=str(repo),
            query=query,
            error="Limit must be greater than or equal to 0.",
        )

    if limit == 0:
        return SearchCodeResult(
            tool_name="search_code",
            status=ToolStatus.SUCCESS,
            repo_path=str(repo),
            query=query,
            matches=[],
            total_matches=0,
        )

    matches: list[SearchResultEntry] = []
    for file_path in _iter_files(repo):
        content = _read_text(file_path)
        if content is None:
            continue

        for line_number, line_text in enumerate(content.splitlines(), start=1):
            if query in line_text:
                matches.append(
                    SearchResultEntry(
                        file_path=str(file_path.relative_to(repo)),
                        line_number=line_number,
                        line_text=line_text,
                    )
                )
                if limit is not None and len(matches) >= limit:
                    return SearchCodeResult(
                        tool_name="search_code",
                        status=ToolStatus.SUCCESS,
                        repo_path=str(repo),
                        query=query,
                        matches=matches,
                        total_matches=len(matches),
                    )

    return SearchCodeResult(
        tool_name="search_code",
        status=ToolStatus.SUCCESS,
        repo_path=str(repo),
        query=query,
        matches=matches,
        total_matches=len(matches),
    )


def read_file(file_path: str) -> ReadFileResult:
    """Read a text file safely and return content plus basic metadata."""

    file = Path(file_path)
    if not file.exists():
        return ReadFileResult(
            tool_name="read_file",
            status=ToolStatus.ERROR,
            file_path=str(file),
            error=f"File does not exist: {file}",
        )

    if not file.is_file():
        return ReadFileResult(
            tool_name="read_file",
            status=ToolStatus.ERROR,
            file_path=str(file),
            error=f"Path is not a file: {file}",
        )

    content = _read_text(file)
    if content is None:
        return ReadFileResult(
            tool_name="read_file",
            status=ToolStatus.ERROR,
            file_path=str(file),
            error=f"File is not readable as UTF-8 text: {file}",
        )

    return ReadFileResult(
        tool_name="read_file",
        status=ToolStatus.SUCCESS,
        file_path=str(file),
        content=content,
        metadata=FileMetadata(
            line_count=len(content.splitlines()),
            character_count=len(content),
        ),
    )


def read_file_chunk(file_path: str, start_line: int, end_line: int) -> ReadFileChunkResult:
    """Read a bounded inclusive line range from a text file."""

    requested_range = LineRange(start_line=start_line, end_line=end_line)
    file = Path(file_path)

    if start_line <= 0 or end_line <= 0:
        return ReadFileChunkResult(
            tool_name="read_file_chunk",
            status=ToolStatus.ERROR,
            file_path=str(file),
            requested_range=requested_range,
            error="Line numbers must be greater than or equal to 1.",
        )

    if start_line > end_line:
        return ReadFileChunkResult(
            tool_name="read_file_chunk",
            status=ToolStatus.ERROR,
            file_path=str(file),
            requested_range=requested_range,
            error="Start line must be less than or equal to end line.",
        )

    full_result = read_file(str(file))
    if full_result.status == ToolStatus.ERROR:
        return ReadFileChunkResult(
            tool_name="read_file_chunk",
            status=ToolStatus.ERROR,
            file_path=str(file),
            requested_range=requested_range,
            error=full_result.error,
        )

    lines = full_result.content.splitlines()
    if not lines:
        return ReadFileChunkResult(
            tool_name="read_file_chunk",
            status=ToolStatus.SUCCESS,
            file_path=str(file),
            requested_range=requested_range,
            actual_range=LineRange(start_line=0, end_line=0),
            content_lines=[],
        )

    if start_line > len(lines):
        return ReadFileChunkResult(
            tool_name="read_file_chunk",
            status=ToolStatus.SUCCESS,
            file_path=str(file),
            requested_range=requested_range,
            actual_range=LineRange(start_line=0, end_line=0),
            content_lines=[],
        )

    actual_start = start_line
    actual_end = min(end_line, len(lines))
    chunk = lines[actual_start - 1 : actual_end]

    return ReadFileChunkResult(
        tool_name="read_file_chunk",
        status=ToolStatus.SUCCESS,
        file_path=str(file),
        requested_range=requested_range,
        actual_range=LineRange(start_line=actual_start, end_line=actual_end),
        content_lines=chunk,
    )


def inspect_repository(repo_path: str, limit: int | None = None) -> FileListResult:
    """Compatibility wrapper for the scaffolded package export."""

    return list_files(repo_path=repo_path, limit=limit)


def _iter_files(root: Path) -> list[Path]:
    """Yield files beneath a root path while skipping noisy directories."""

    files: list[Path] = []
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError:
        return files

    for entry in entries:
        if entry.is_dir():
            if entry.name in NOISE_DIRECTORIES:
                continue
            files.extend(_iter_files(entry))
            continue
        if entry.is_file():
            files.append(entry)
    return files


def _read_text(file_path: Path) -> str | None:
    """Read a UTF-8 text file and return None for binary or unreadable files."""

    try:
        return file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
