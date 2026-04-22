"""CLI entrypoint for RepoPilot."""

import typer

from config import get_config
from orchestrator.pipeline import Pipeline
from schemas.agent_schemas import AgentStatus
from schemas.run_schemas import FinalOutput
from utils.logging import build_log_message
from utils.ollama_client import (
    OllamaClientError,
    check_ollama_server,
    generate_text,
    list_models,
)


app = typer.Typer(
    help="RepoPilot CLI for local Ollama health checks and future orchestration."
)


@app.callback()
def main() -> None:
    """Expose the root CLI application."""


@app.command()
def health() -> None:
    """Run local Ollama readiness checks."""

    config = get_config()
    typer.echo(f"{config.project_name} health check")

    try:
        server_info = check_ollama_server(config.ollama_base_url)
        typer.echo(
            f"[OK] Ollama server reachable at {server_info['base_url']} "
            f"(version: {server_info['version']})."
        )
    except OllamaClientError as exc:
        typer.echo(f"[FAIL] {exc}")
        raise typer.Exit(code=1)

    try:
        installed_models = list_models(config.ollama_base_url)
    except OllamaClientError as exc:
        typer.echo(f"[FAIL] {exc}")
        raise typer.Exit(code=1)

    if config.default_model_name not in installed_models:
        typer.echo(
            "[FAIL] Configured model not installed: "
            f"{config.default_model_name}."
        )
        if installed_models:
            typer.echo(f"Installed models: {', '.join(installed_models)}")
        else:
            typer.echo("Installed models: none reported by Ollama.")
        raise typer.Exit(code=1)

    typer.echo(f"[OK] Configured model found: {config.default_model_name}.")

    try:
        result = generate_text(
            prompt="Say OK.",
            model_name=config.default_model_name,
            base_url=config.ollama_base_url,
        )
    except OllamaClientError as exc:
        typer.echo(f"[FAIL] {exc}")
        raise typer.Exit(code=1)

    preview = result["response"] or "<empty response>"
    typer.echo(f"[OK] Test generation succeeded: {preview}")


@app.command()
def run(
    repo: str = typer.Option(..., "--repo", "-r", help="Repository path to inspect."),
    task: str = typer.Option(..., "--task", "-t", help="User task to execute."),
    verbose: bool = typer.Option(False, "--verbose", help="Print concise stage progress."),
) -> None:
    """Run the fixed multi-agent pipeline."""

    pipeline = Pipeline(progress_callback=lambda message: typer.echo(build_log_message(message)))
    result = pipeline.run(repo_path=repo, user_task=task, verbose=verbose)
    typer.echo(_render_final_output(result))

    if result.status == AgentStatus.FAILURE:
        raise typer.Exit(code=1)


def _render_final_output(result: FinalOutput) -> str:
    """Format the final pipeline output for terminal display."""

    key_files = ", ".join(result.key_files) if result.key_files else "none"
    reviewer_notes = "\n".join(f"- {note}" for note in result.reviewer_notes) or "- none"

    return (
        f"Task summary: {result.task_summary}\n"
        f"Key files: {key_files}\n\n"
        "Final response:\n"
        f"{result.final_response}\n\n"
        "Reviewer notes:\n"
        f"{reviewer_notes}\n\n"
        f"Confidence: {result.confidence:.2f}\n"
        f"Status: {result.status.value}"
    )


if __name__ == "__main__":
    app()
