# Architecture Notes

RepoPilot is a fixed local multi-agent pipeline for repository analysis and planning. The system is intentionally explicit and deterministic: there is no dynamic agent graph, no background framework, and no hidden orchestration layer.

## Pipeline Stages

### 1. Planner

The `Planner` accepts a user task string and classifies it into one of the supported task types:

- `explain_code_path`
- `find_feature_location`
- `minimal_files_to_change`
- `implementation_plan`
- `unsupported`

It returns a structured `PlannerOutput` with workflow steps, recommended tools, output mode, confidence, and status.

### 2. Codebase Analyst

The `Codebase Analyst` consumes the planner output plus the target repo path. It uses the controlled tool layer to:

- list candidate files
- search for task-relevant terms
- read bounded file chunks
- collect short evidence snippets
- write concise run notes

Its output is an `AnalystOutput` containing relevant files, symbols, evidence snippets, open questions, and grounded confidence.

### 3. Implementation Planner

The `Implementation Planner` does not traverse the repo directly. It consumes:

- the original user task
- the `PlannerOutput`
- the `AnalystOutput`

It turns grounded evidence into:

- `proposed_changes`
- `minimal_files_to_touch`
- `implementation_steps`
- `risks_or_assumptions`

Its output is an `ImplementationPlannerOutput`.

### 4. Reviewer

The `Reviewer` inspects the outputs of the three earlier stages and checks:

- task consistency across stages
- whether the implementation plan follows from the analyst evidence
- whether the proposed file set is grounded
- whether missing evidence or weak assumptions should block confidence

It returns a `ReviewerOutput` with issues, missing evidence, revision requests, and a final assessment.

## Tool Layer

The pipeline uses a controlled local tool layer instead of arbitrary execution. The approved tools are:

- `list_files`
- `search_code`
- `read_file`
- `read_file_chunk`
- `write_note`

This keeps repo access explicit, inspectable, and aligned with saved run artifacts.

## Data Contracts

Shared Pydantic schemas define:

- agent-stage outputs
- run-state structure
- final assembled output

This makes the pipeline easy to inspect in-memory and easy to serialize into artifact files.

## Run State And Artifacts

Each run initializes a `RunState` object that records:

- user task
- repo path
- task type
- inspected files
- agent outputs
- tool history
- final output

The pipeline also creates a per-run artifact folder under `runs/` containing:

- `input.json`
- `trace.json`
- `summary.md`
- `final_output.json`
- `notes.md`

## Retry Behavior

Retry behavior is intentionally limited:

- agents can internally perform one repair attempt if structured model output is malformed
- the pipeline can trigger one broadened retry for the Analyst stage when evidence is clearly weak
- unsupported tasks short-circuit early instead of forcing the full pipeline

There is no generalized retry engine in v1.

## Final Output Assembly

The orchestrator assembles a `FinalOutput` from the stage outputs. The final response is task-aware:

- explanation tasks are rendered as code-path summaries
- implementation-oriented tasks are rendered as grounded engineering plans

Reviewer notes, confidence, and status are included in both CLI output and saved artifacts.
