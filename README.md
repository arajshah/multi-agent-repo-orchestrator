# RepoPilot

RepoPilot is a local Python CLI project that will grow into a multi-agent repository assistant. This repository is the bootstrap for the GitHub project `multi-agent-repo-orchestrator`, with a clean package layout, a minimal runnable CLI, and an Ollama readiness check.

Current status: fixed-pipeline phase. The codebase currently includes a minimal CLI, local Ollama health checks, structured repository tools, shared Pydantic contracts, and a fixed end-to-end Planner -> Analyst -> Implementation Planner -> Reviewer pipeline.

## Setup

Use Python 3.11 and install the existing dependencies:

```bash
pip install -r requirements.txt
python main.py --help
python main.py health
python main.py run --repo . --task "Generate the minimal implementation plan for improving the Ollama health check flow in this repo."
```

Each pipeline run writes artifacts under `runs/<timestamp>_<task-slug>/`, including structured trace data, a markdown summary, the final output JSON, and any collected notes.

## Ollama Requirements

RepoPilot expects a local Ollama server running at `http://127.0.0.1:11434` by default and uses `qwen2.5-coder:7b` as the default model. You can override these with `OLLAMA_BASE_URL` and `OLLAMA_MODEL` if needed.

Run the startup health check with:

```bash
python main.py health
```

## Tool Layer

The repository now includes a small structured tool layer for file listing, code search, file reads, bounded file chunk reads, and note writing. These tools are intended for later agent phases and are read-only except for note appends.

## Shared Schemas

The repository also includes shared agent-output, final-output, and run-state schemas so later pipeline phases can exchange structured data consistently.

## Planned Architecture

The project is planned around four agent roles:

- Planner
- Analyst
- Implementation Planner
- Reviewer
