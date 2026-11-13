"""Project configuration scaffold."""

from dataclasses import dataclass
import os
from pathlib import Path


# Paths are derived from the repo root so runs and the demo repo resolve
# regardless of the working directory the CLI is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIRECTORY = PROJECT_ROOT / "runs"
DEMO_REPO_DIRECTORY = PROJECT_ROOT / "demo_repo"


@dataclass(frozen=True)
class AppConfig:
    """Static configuration for the scaffolded project."""

    # Set once at startup from defaults and environment variables.
    project_name: str
    ollama_base_url: str
    default_model_name: str
    runs_directory: Path
    demo_repo_directory: Path


def get_config() -> AppConfig:
    """Return the default application configuration."""

    return AppConfig(
        project_name="RepoPilot",
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        default_model_name=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
        runs_directory=RUNS_DIRECTORY,
        demo_repo_directory=DEMO_REPO_DIRECTORY,
    )
