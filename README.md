# RepoPilot

RepoPilot is a local Python CLI project that will grow into a multi-agent repository assistant. This repository is the bootstrap for the GitHub project `multi-agent-repo-orchestrator`, with a clean package layout, a minimal runnable CLI, and an Ollama readiness check.

Current status: local runtime and tool-layer phase. The codebase currently includes a minimal CLI, local Ollama health checks, and a first pass of structured repository tools; agent logic and orchestration are not implemented yet.

## Setup

Use Python 3.11 and install the existing dependencies:

```bash
pip install -r requirements.txt
python main.py --help
python main.py health
```

## Ollama Requirements

RepoPilot expects a local Ollama server running at `http://127.0.0.1:11434` by default and uses `qwen2.5-coder:7b` as the default model. You can override these with `OLLAMA_BASE_URL` and `OLLAMA_MODEL` if needed.

Run the startup health check with:

```bash
python main.py health
```

## Tool Layer

The repository now includes a small structured tool layer for file listing, code search, file reads, bounded file chunk reads, and note writing. These tools are intended for later agent phases and are read-only except for note appends.

## Planned Architecture

The project is planned around four agent roles:

- Planner
- Analyst
- Implementation Planner
- Reviewer
