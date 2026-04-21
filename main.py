"""CLI entrypoint for RepoPilot."""

from pathlib import Path

import typer

from config import get_config
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
def run() -> None:
    """Print a placeholder run message."""

    config = get_config()
    typer.echo(
        "Run pipeline is not implemented yet. "
        f"Future outputs will be stored in {Path(config.runs_directory).name}/."
    )


if __name__ == "__main__":
    app()
