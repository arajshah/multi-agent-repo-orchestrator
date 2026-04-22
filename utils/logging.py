"""Lightweight helpers for console logging and run artifacts."""

from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any


def build_log_message(message: str) -> str:
    """Format a timestamped message for console progress output."""

    return f"[{datetime.utcnow().isoformat()}] {message}"


def build_run_slug(user_task: str, max_length: int = 48) -> str:
    """Return a filesystem-safe slug for a run folder."""

    lowered = user_task.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    if not normalized:
        normalized = "run"
    return normalized[:max_length].rstrip("-") or "run"


def build_run_folder_name(user_task: str, started_at: datetime) -> str:
    """Return a unique run folder name."""

    timestamp = started_at.strftime("%Y-%m-%dT%H-%M-%S-%f")
    return f"{timestamp}_{build_run_slug(user_task)}"


def create_run_artifact_dir(
    runs_directory: Path,
    user_task: str,
    started_at: datetime,
) -> Path:
    """Create and return the artifact directory for one run."""

    runs_directory.mkdir(parents=True, exist_ok=True)
    artifact_dir = runs_directory / build_run_folder_name(user_task, started_at)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    return artifact_dir


def write_json_file(path: Path, payload: Any) -> None:
    """Write a JSON file with stable readable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def write_text_file(path: Path, content: str) -> None:
    """Write a UTF-8 text file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def ensure_notes_file(path: Path) -> None:
    """Ensure the notes file exists even if no stage wrote to it."""

    if not path.exists():
        write_text_file(path, "# Run Notes\n\nNo notes were recorded for this run.\n")


def read_text_if_exists(path: Path) -> str:
    """Read a text file if it exists, otherwise return an empty string."""

    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def build_summary_markdown(
    *,
    run_id: str,
    repo_path: str,
    started_at: str,
    ended_at: str,
    duration_ms: int | None,
    task_summary: str,
    key_files: list[str],
    final_response: str,
    reviewer_notes: list[str],
    confidence: float,
    status: str,
    stage_outcomes: list[dict[str, Any]],
) -> str:
    """Build a concise human-readable run summary."""

    key_files_lines = "\n".join(f"- `{file_path}`" for file_path in key_files) or "- none"
    reviewer_note_lines = "\n".join(f"- {note}" for note in reviewer_notes) or "- none"
    stage_lines = "\n".join(
        f"- `{stage['name']}`: {stage['status']} ({stage['duration_ms']} ms)"
        if stage.get("duration_ms") is not None
        else f"- `{stage['name']}`: {stage['status']}"
        for stage in stage_outcomes
    ) or "- none"

    duration_text = f"{duration_ms} ms" if duration_ms is not None else "unknown"

    return (
        "# Run Summary\n\n"
        f"- Run ID: `{run_id}`\n"
        f"- Repo: `{repo_path}`\n"
        f"- Started at: `{started_at}`\n"
        f"- Ended at: `{ended_at}`\n"
        f"- Duration: `{duration_text}`\n"
        f"- Status: `{status}`\n"
        f"- Confidence: `{confidence:.2f}`\n\n"
        "## Task\n\n"
        f"{task_summary}\n\n"
        "## Key Files\n\n"
        f"{key_files_lines}\n\n"
        "## Final Response\n\n"
        f"{final_response}\n\n"
        "## Reviewer Notes\n\n"
        f"{reviewer_note_lines}\n\n"
        "## Stage Outcomes\n\n"
        f"{stage_lines}\n"
    )


def _json_default(value: Any) -> Any:
    """Convert common Python objects into JSON-safe values."""

    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)
