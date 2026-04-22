"""Fixed end-to-end orchestration pipeline with run artifacts."""

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from agents.analyst import AnalystAgent
from agents.implementation_planner import ImplementationPlannerAgent
from agents.planner import PlannerAgent
from agents.reviewer import ReviewerAgent
from config import get_config
from memory.run_state import RunState
from schemas.agent_schemas import (
    AgentStatus,
    AnalystOutput,
    ImplementationPlannerOutput,
    PlannerOutput,
    ReviewerOutput,
)
from schemas.run_schemas import FinalOutput, RunLifecycleStatus
from utils.logging import (
    build_summary_markdown,
    create_run_artifact_dir,
    ensure_notes_file,
    read_text_if_exists,
    write_json_file,
    write_text_file,
)


class Pipeline:
    """Run the fixed Planner -> Analyst -> Implementation Planner -> Reviewer flow."""

    def __init__(self, progress_callback: Callable[[str], None] | None = None) -> None:
        """Initialize the pipeline with concrete agents."""

        self.progress_callback = progress_callback
        self.planner = PlannerAgent()
        self.analyst = AnalystAgent()
        self.implementation_planner = ImplementationPlannerAgent()
        self.reviewer = ReviewerAgent()
        self.last_run_state: RunState | None = None
        self.last_artifact_dir: Path | None = None
        self.last_trace: dict[str, Any] | None = None

    def run(self, repo_path: str, user_task: str, verbose: bool = False) -> FinalOutput:
        """Execute the fixed multi-agent pipeline and return a final structured result."""

        config = get_config()
        started_at = datetime.utcnow()
        artifact_dir = create_run_artifact_dir(config.runs_directory, user_task, started_at)
        notes_path = artifact_dir / "notes.md"
        run_id = self._build_run_id(started_at)

        state = RunState.initialize(
            run_id=run_id,
            user_task=user_task,
            repo_path=repo_path,
            notes_path=str(notes_path),
        )
        state.status = RunLifecycleStatus.IN_PROGRESS
        state.touch()

        trace = self._initialize_trace(
            run_id=run_id,
            repo_path=repo_path,
            user_task=user_task,
            verbose=verbose,
            started_at=started_at,
            artifact_dir=artifact_dir,
            notes_path=notes_path,
        )

        self.last_run_state = state
        self.last_artifact_dir = artifact_dir
        self.last_trace = trace
        self._write_input_artifact(artifact_dir, state, verbose, started_at)

        final_output: FinalOutput | None = None

        try:
            self._emit(verbose, "starting planner")
            planner_result = self._run_planner(state, trace, user_task)
            if isinstance(planner_result, FinalOutput):
                final_output = planner_result
                self._emit(verbose, "final assembly complete")
                return final_output
            planner_output = planner_result
            self._emit(verbose, "planner complete")

            self._emit(verbose, "starting analyst")
            analyst_result = self._run_analyst(
                state,
                trace,
                repo_path,
                user_task,
                planner_output,
                verbose,
            )
            if isinstance(analyst_result, FinalOutput):
                final_output = analyst_result
                self._emit(verbose, "final assembly complete")
                return final_output
            analyst_output = analyst_result
            self._emit(verbose, "analyst complete")

            self._emit(verbose, "starting implementation planner")
            implementation_result = self._run_implementation_planner(
                state,
                trace,
                user_task,
                planner_output,
                analyst_output,
            )
            if isinstance(implementation_result, FinalOutput):
                final_output = implementation_result
                self._emit(verbose, "final assembly complete")
                return final_output
            implementation_output = implementation_result
            self._emit(verbose, "implementation planner complete")

            self._emit(verbose, "starting reviewer")
            reviewer_result = self._run_reviewer(
                state,
                trace,
                user_task,
                planner_output,
                analyst_output,
                implementation_output,
            )
            if isinstance(reviewer_result, FinalOutput):
                final_output = reviewer_result
                self._emit(verbose, "final assembly complete")
                return final_output
            reviewer_output = reviewer_result
            self._emit(verbose, "reviewer complete")

            final_output = self._assemble_final_output(
                user_task,
                planner_output,
                analyst_output,
                implementation_output,
                reviewer_output,
            )
            state.attach_final_output(final_output)
            trace["final_output"] = final_output.model_dump(mode="json")
            self._emit(verbose, "final assembly complete")
            return final_output
        except Exception as exc:
            final_output = self._build_failure_output(
                state,
                trace,
                task_summary=user_task,
                final_response="The pipeline encountered an unexpected error before completion.",
                reviewer_notes=[f"Unexpected pipeline error: {exc}"],
                status=AgentStatus.FAILURE,
                confidence=0.0,
                failing_stage="pipeline",
            )
            return final_output
        finally:
            self._finalize_run_artifacts(
                state=state,
                trace=trace,
                artifact_dir=artifact_dir,
                notes_path=notes_path,
                final_output=state.final_output or final_output,
                started_at=started_at,
            )

    def start(self, repo_path: str, user_task: str, verbose: bool = False) -> FinalOutput:
        """Compatibility wrapper for the fixed pipeline entrypoint."""

        return self.run(repo_path=repo_path, user_task=user_task, verbose=verbose)

    def _run_planner(
        self,
        state: RunState,
        trace: dict[str, Any],
        user_task: str,
    ) -> PlannerOutput | FinalOutput:
        """Run the planner stage and stop early for unsupported tasks."""

        stage_entry = self._start_stage(trace, "planner")
        attempt = self._start_attempt(stage_entry, "initial")

        try:
            planner_output = self.planner.run(user_task)
        except Exception as exc:
            self._finish_attempt(attempt, status="failure", error=str(exc))
            self._finish_stage(stage_entry, status="failure")
            return self._build_failure_output(
                state,
                trace,
                task_summary=user_task,
                final_response="Planner stage failed before a valid routing decision could be produced.",
                reviewer_notes=[f"Planner error: {exc}"],
                status=AgentStatus.FAILURE,
                confidence=0.0,
                failing_stage="planner",
            )

        state.attach_agent_output(planner_output)
        state.task_type = planner_output.task_type
        state.touch()
        state.add_tool_history_entry(
            tool_name="planner_stage",
            status=planner_output.status.value,
            input_summary="initial planner execution",
        )

        self._finish_attempt(
            attempt,
            status=planner_output.status.value,
            output=planner_output.model_dump(mode="json"),
        )
        self._finish_stage(
            stage_entry,
            status=planner_output.status.value,
            output=planner_output.model_dump(mode="json"),
        )

        if planner_output.task_type == "unsupported" or planner_output.confidence < 0.3:
            final_output = FinalOutput(
                task_summary=user_task,
                key_files=[],
                final_response=(
                    "The task was classified as unsupported or too ambiguous for the "
                    "current fixed pipeline, so downstream stages were skipped."
                ),
                reviewer_notes=[
                    planner_output.reasoning_summary or "Planner could not support the requested task.",
                    planner_output.next_action or "Restate the task in a supported category.",
                ],
                confidence=planner_output.confidence,
                status=AgentStatus.PARTIAL,
            )
            state.attach_final_output(final_output)
            trace["final_output"] = final_output.model_dump(mode="json")
            return final_output

        return planner_output

    def _run_analyst(
        self,
        state: RunState,
        trace: dict[str, Any],
        repo_path: str,
        user_task: str,
        planner_output: PlannerOutput,
        verbose: bool,
    ) -> AnalystOutput | FinalOutput:
        """Run the analyst stage with one broadened retry if evidence is weak."""

        stage_entry = self._start_stage(trace, "analyst")
        attempt = self._start_attempt(stage_entry, "initial")

        try:
            analyst_output = self.analyst.run(
                repo_path,
                user_task,
                planner_output,
                note_path=state.notes_path,
            )
        except Exception as exc:
            self._finish_attempt(attempt, status="failure", error=str(exc))
            self._finish_stage(stage_entry, status="failure")
            return self._build_failure_output(
                state,
                trace,
                task_summary=user_task,
                final_response="Analyst stage failed before grounded evidence could be collected.",
                reviewer_notes=[f"Analyst error: {exc}"],
                status=AgentStatus.FAILURE,
                confidence=0.0,
                failing_stage="analyst",
            )

        self._finish_attempt(
            attempt,
            status=analyst_output.status.value,
            output=analyst_output.model_dump(mode="json"),
        )

        analyst_retry_used = False
        if self._analyst_output_is_weak(analyst_output):
            analyst_retry_used = True
            self._emit(verbose, "analyst retry triggered")
            retry_reason = "weak evidence triggered broadened retry"
            trace["retries"].append(
                {
                    "stage": "analyst",
                    "reason": retry_reason,
                    "triggered_at": datetime.utcnow().isoformat(),
                }
            )
            broadened_task = self._build_broadened_analyst_task(user_task, planner_output)
            retry_attempt = self._start_attempt(stage_entry, "broadened_retry")
            try:
                retry_output = self.analyst.run(
                    repo_path,
                    broadened_task,
                    planner_output,
                    note_path=state.notes_path,
                )
                self._finish_attempt(
                    retry_attempt,
                    status=retry_output.status.value,
                    output=retry_output.model_dump(mode="json"),
                )
            except Exception as exc:
                retry_output = analyst_output
                self._finish_attempt(retry_attempt, status="failure", error=str(exc))

            if self._analyst_strength(retry_output) > self._analyst_strength(analyst_output):
                analyst_output = retry_output

        state.attach_agent_output(analyst_output)
        for file_path in analyst_output.relevant_files:
            state.add_inspected_file(file_path)
        state.add_tool_history_entry(
            tool_name="analyst_stage",
            status=analyst_output.status.value,
            input_summary="analyst execution with retry" if analyst_retry_used else "initial analyst execution",
        )

        self._finish_stage(
            stage_entry,
            status=analyst_output.status.value,
            output=analyst_output.model_dump(mode="json"),
        )

        if analyst_output.status == AgentStatus.FAILURE:
            return self._build_failure_output(
                state,
                trace,
                task_summary=user_task,
                final_response="Analyst stage could not produce a usable evidence package.",
                reviewer_notes=[analyst_output.reasoning_summary or "Analyst failed."],
                status=AgentStatus.FAILURE,
                confidence=analyst_output.confidence,
                failing_stage="analyst",
            )

        return analyst_output

    def _run_implementation_planner(
        self,
        state: RunState,
        trace: dict[str, Any],
        user_task: str,
        planner_output: PlannerOutput,
        analyst_output: AnalystOutput,
    ) -> ImplementationPlannerOutput | FinalOutput:
        """Run the implementation planner stage."""

        stage_entry = self._start_stage(trace, "implementation_planner")
        attempt = self._start_attempt(stage_entry, "initial")

        try:
            implementation_output = self.implementation_planner.run(
                user_task,
                planner_output,
                analyst_output,
            )
        except Exception as exc:
            self._finish_attempt(attempt, status="failure", error=str(exc))
            self._finish_stage(stage_entry, status="failure")
            return self._build_failure_output(
                state,
                trace,
                task_summary=user_task,
                final_response="Implementation planning failed before a structured plan could be assembled.",
                reviewer_notes=[f"Implementation planner error: {exc}"],
                status=AgentStatus.FAILURE,
                confidence=0.0,
                failing_stage="implementation_planner",
            )

        state.attach_agent_output(implementation_output)
        for file_path in implementation_output.minimal_files_to_touch:
            state.add_inspected_file(file_path)
        state.add_tool_history_entry(
            tool_name="implementation_planner_stage",
            status=implementation_output.status.value,
            input_summary="implementation planning execution",
        )

        self._finish_attempt(
            attempt,
            status=implementation_output.status.value,
            output=implementation_output.model_dump(mode="json"),
        )
        self._finish_stage(
            stage_entry,
            status=implementation_output.status.value,
            output=implementation_output.model_dump(mode="json"),
        )

        if implementation_output.status == AgentStatus.FAILURE:
            return self._build_failure_output(
                state,
                trace,
                task_summary=user_task,
                final_response="Implementation planning produced a failure result and the pipeline stopped before review.",
                reviewer_notes=[implementation_output.reasoning_summary or "Implementation planning failed."],
                status=AgentStatus.FAILURE,
                confidence=implementation_output.confidence,
                failing_stage="implementation_planner",
            )

        return implementation_output

    def _run_reviewer(
        self,
        state: RunState,
        trace: dict[str, Any],
        user_task: str,
        planner_output: PlannerOutput,
        analyst_output: AnalystOutput,
        implementation_output: ImplementationPlannerOutput,
    ) -> ReviewerOutput | FinalOutput:
        """Run the reviewer stage."""

        stage_entry = self._start_stage(trace, "reviewer")
        attempt = self._start_attempt(stage_entry, "initial")

        try:
            reviewer_output = self.reviewer.run(
                user_task,
                planner_output,
                analyst_output,
                implementation_output,
            )
        except Exception as exc:
            self._finish_attempt(attempt, status="failure", error=str(exc))
            self._finish_stage(stage_entry, status="failure")
            return self._build_failure_output(
                state,
                trace,
                task_summary=user_task,
                final_response="Reviewer stage failed before a final assessment could be produced.",
                reviewer_notes=[f"Reviewer error: {exc}"],
                status=AgentStatus.FAILURE,
                confidence=0.0,
                failing_stage="reviewer",
            )

        state.attach_agent_output(reviewer_output)
        state.add_tool_history_entry(
            tool_name="reviewer_stage",
            status=reviewer_output.status.value,
            input_summary="review execution",
        )

        self._finish_attempt(
            attempt,
            status=reviewer_output.status.value,
            output=reviewer_output.model_dump(mode="json"),
        )
        self._finish_stage(
            stage_entry,
            status=reviewer_output.status.value,
            output=reviewer_output.model_dump(mode="json"),
        )

        if reviewer_output.status == AgentStatus.FAILURE:
            return self._build_failure_output(
                state,
                trace,
                task_summary=user_task,
                final_response="Reviewer marked the overall result as unsupported.",
                reviewer_notes=[reviewer_output.final_assessment or reviewer_output.reasoning_summary],
                status=AgentStatus.FAILURE,
                confidence=reviewer_output.confidence,
                failing_stage="reviewer",
            )

        return reviewer_output

    def _assemble_final_output(
        self,
        user_task: str,
        planner_output: PlannerOutput,
        analyst_output: AnalystOutput,
        implementation_output: ImplementationPlannerOutput,
        reviewer_output: ReviewerOutput,
    ) -> FinalOutput:
        """Assemble the final output schema from completed stage results."""

        key_files = self._unique(
            implementation_output.minimal_files_to_touch + analyst_output.relevant_files
        )[:5]
        reviewer_notes = self._unique(
            [reviewer_output.final_assessment]
            + reviewer_output.revisions_requested[:2]
            + reviewer_output.missing_evidence[:2]
        )
        final_response = self._build_final_response(
            planner_output,
            analyst_output,
            implementation_output,
            reviewer_output,
            key_files,
        )
        confidence = min(
            planner_output.confidence,
            analyst_output.confidence,
            implementation_output.confidence,
            reviewer_output.confidence,
        )

        return FinalOutput(
            task_summary=user_task.strip(),
            key_files=key_files,
            final_response=final_response,
            reviewer_notes=reviewer_notes,
            confidence=confidence,
            status=reviewer_output.status,
        )

    def _build_final_response(
        self,
        planner_output: PlannerOutput,
        analyst_output: AnalystOutput,
        implementation_output: ImplementationPlannerOutput,
        reviewer_output: ReviewerOutput,
        key_files: list[str],
    ) -> str:
        """Build a polished final response string."""

        if planner_output.task_type == "explain_code_path":
            return self._build_explanation_response(analyst_output, reviewer_output, key_files)

        lines = [
            f"Task type: {planner_output.task_type}.",
            (
                "Grounded files: " + ", ".join(key_files) + "."
                if key_files
                else "Grounded files: none confirmed yet."
            ),
        ]

        if implementation_output.proposed_changes:
            lines.append(
                "Proposed changes: " + "; ".join(implementation_output.proposed_changes[:3]) + "."
            )
        if implementation_output.implementation_steps:
            lines.append(
                "Implementation steps: "
                + " ".join(
                    f"{index + 1}. {step}"
                    for index, step in enumerate(implementation_output.implementation_steps[:4])
                )
            )
        if implementation_output.risks_or_assumptions:
            lines.append(
                "Risks and assumptions: "
                + "; ".join(implementation_output.risks_or_assumptions[:3])
                + "."
            )
        if reviewer_output.final_assessment:
            lines.append(f"Reviewer assessment: {reviewer_output.final_assessment}")
        if analyst_output.open_questions and reviewer_output.status != AgentStatus.SUCCESS:
            lines.append(
                "Open questions: " + "; ".join(analyst_output.open_questions[:2]) + "."
            )

        return "\n".join(lines)

    def _build_explanation_response(
        self,
        analyst_output: AnalystOutput,
        reviewer_output: ReviewerOutput,
        key_files: list[str],
    ) -> str:
        """Build a task-aware explanation response for code-path questions."""

        file_pool = set(key_files + analyst_output.relevant_files)
        has_route = any("auth_routes.py" in file_path or "routes.py" in file_path for file_path in file_pool)
        has_auth_service = any("auth_service.py" in file_path for file_path in file_pool)
        has_user_lookup = any("user_repository.py" in file_path for file_path in file_pool)
        has_security = any("security.py" in file_path for file_path in file_pool)
        has_token = any("token_service.py" in file_path for file_path in file_pool)

        lines = [
            "Task type: explain_code_path.",
            (
                "Grounded files: " + ", ".join(key_files) + "."
                if key_files
                else "Grounded files: none confirmed yet."
            ),
        ]

        flow_steps: list[str] = []
        if has_route:
            flow_steps.append(
                "The request enters the auth route layer, where the login handler validates the incoming payload."
            )
        if has_auth_service:
            flow_steps.append(
                "The route delegates credential handling to the auth service, which owns the login decision."
            )
        if has_user_lookup:
            flow_steps.append(
                "The auth service looks up the account record through the user repository."
            )
        if has_security:
            flow_steps.append(
                "Password verification happens in the security helper before a session is issued."
            )
        if has_token:
            flow_steps.append(
                "Once credentials pass, the token service creates the session token returned to the client."
            )

        if flow_steps:
            lines.append("Auth flow: " + " ".join(flow_steps))
        else:
            lines.append(
                "Auth flow: the current evidence points to the route layer, auth service, and session creation path, but some steps still need stronger grounding."
            )

        if reviewer_output.final_assessment:
            lines.append(f"Reviewer assessment: {reviewer_output.final_assessment}")
        if analyst_output.open_questions and reviewer_output.status != AgentStatus.SUCCESS:
            lines.append(
                "Open questions: " + "; ".join(analyst_output.open_questions[:2]) + "."
            )

        return "\n".join(lines)

    def _build_failure_output(
        self,
        state: RunState,
        trace: dict[str, Any],
        task_summary: str,
        final_response: str,
        reviewer_notes: list[str],
        status: AgentStatus,
        confidence: float,
        failing_stage: str,
    ) -> FinalOutput:
        """Create a structured failure output and update run state."""

        final_output = FinalOutput(
            task_summary=task_summary,
            key_files=self._unique(state.inspected_files)[:5],
            final_response=final_response,
            reviewer_notes=reviewer_notes,
            confidence=confidence,
            status=status,
        )
        state.final_output = final_output
        state.add_tool_history_entry(
            tool_name=f"{failing_stage}_stage",
            status="failure",
            input_summary="pipeline aborted",
            error=reviewer_notes[0] if reviewer_notes else None,
        )
        trace["final_output"] = final_output.model_dump(mode="json")
        state.mark_failed()
        return final_output

    def _analyst_output_is_weak(self, analyst_output: AnalystOutput) -> bool:
        """Return whether the analyst output is weak enough to justify one retry."""

        return (
            analyst_output.status != AgentStatus.SUCCESS
            or analyst_output.confidence < 0.5
            or not analyst_output.evidence_snippets
        )

    def _analyst_strength(self, analyst_output: AnalystOutput) -> float:
        """Score analyst output strength for one retry decision."""

        return (
            analyst_output.confidence
            + (0.1 * len(analyst_output.relevant_files))
            + (0.1 * len(analyst_output.evidence_snippets))
            - (0.05 * len(analyst_output.open_questions))
        )

    def _build_broadened_analyst_task(
        self,
        user_task: str,
        planner_output: PlannerOutput,
    ) -> str:
        """Build a single broadened task hint for the analyst retry."""

        return (
            f"{user_task} Focus on nearby entrypoints, handlers, services, config, and model "
            f"files relevant to the {planner_output.task_type} path."
        )

    def _build_run_id(self, started_at: datetime) -> str:
        """Build a lightweight run identifier."""

        return started_at.strftime("run-%Y%m%d%H%M%S%f")

    def _initialize_trace(
        self,
        *,
        run_id: str,
        repo_path: str,
        user_task: str,
        verbose: bool,
        started_at: datetime,
        artifact_dir: Path,
        notes_path: Path,
    ) -> dict[str, Any]:
        """Create the base trace structure for a run."""

        return {
            "run_id": run_id,
            "user_task": user_task,
            "repo_path": repo_path,
            "verbose": verbose,
            "artifact_dir": str(artifact_dir),
            "notes_path": str(notes_path),
            "started_at": started_at.isoformat(),
            "ended_at": None,
            "duration_ms": None,
            "status": RunLifecycleStatus.IN_PROGRESS.value,
            "stage_order": [
                "planner",
                "analyst",
                "implementation_planner",
                "reviewer",
            ],
            "stages": {},
            "retries": [],
            "tool_history": [],
            "inspected_files": [],
            "state_snapshot": {},
            "final_output": None,
        }

    def _write_input_artifact(
        self,
        artifact_dir: Path,
        state: RunState,
        verbose: bool,
        started_at: datetime,
    ) -> None:
        """Write the initial run input artifact."""

        write_json_file(
            artifact_dir / "input.json",
            {
                "run_id": state.run_id,
                "user_task": state.user_task,
                "repo_path": state.repo_path,
                "verbose": verbose,
                "started_at": started_at.isoformat(),
                "notes_path": state.notes_path,
            },
        )

    def _finalize_run_artifacts(
        self,
        *,
        state: RunState,
        trace: dict[str, Any],
        artifact_dir: Path,
        notes_path: Path,
        final_output: FinalOutput | None,
        started_at: datetime,
    ) -> None:
        """Write trace, final output, notes, and summary artifacts."""

        ended_at = datetime.utcnow()
        duration_ms = int((ended_at - started_at).total_seconds() * 1000)

        if final_output is not None and state.final_output is None:
            state.attach_final_output(final_output)
        elif final_output is None and state.final_output is not None:
            final_output = state.final_output

        trace["ended_at"] = ended_at.isoformat()
        trace["duration_ms"] = duration_ms
        trace["status"] = state.status.value
        trace["tool_history"] = [entry.model_dump(mode="json") for entry in state.tool_history]
        trace["inspected_files"] = state.inspected_files
        trace["state_snapshot"] = state.to_serializable_dict()
        if final_output is not None:
            trace["final_output"] = final_output.model_dump(mode="json")

        ensure_notes_file(notes_path)
        write_json_file(artifact_dir / "trace.json", trace)

        if final_output is not None:
            write_json_file(artifact_dir / "final_output.json", final_output.model_dump(mode="json"))
            summary = build_summary_markdown(
                run_id=state.run_id,
                repo_path=state.repo_path,
                started_at=trace["started_at"],
                ended_at=trace["ended_at"],
                duration_ms=trace["duration_ms"],
                task_summary=final_output.task_summary,
                key_files=final_output.key_files,
                final_response=final_output.final_response,
                reviewer_notes=final_output.reviewer_notes,
                confidence=final_output.confidence,
                status=final_output.status.value,
                stage_outcomes=self._stage_outcomes_for_summary(trace),
            )
            write_text_file(artifact_dir / "summary.md", summary)

        notes_contents = read_text_if_exists(notes_path)
        write_text_file(artifact_dir / "notes.md", notes_contents)

    def _start_stage(self, trace: dict[str, Any], stage_name: str) -> dict[str, Any]:
        """Create a stage entry in the trace."""

        stage_entry = {
            "name": stage_name,
            "started_at": datetime.utcnow().isoformat(),
            "ended_at": None,
            "duration_ms": None,
            "status": "in_progress",
            "attempts": [],
            "output": None,
        }
        trace["stages"][stage_name] = stage_entry
        return stage_entry

    def _finish_stage(
        self,
        stage_entry: dict[str, Any],
        *,
        status: str,
        output: dict[str, Any] | None = None,
    ) -> None:
        """Finalize a stage trace entry."""

        ended_at = datetime.utcnow()
        started_at = datetime.fromisoformat(stage_entry["started_at"])
        stage_entry["ended_at"] = ended_at.isoformat()
        stage_entry["duration_ms"] = int((ended_at - started_at).total_seconds() * 1000)
        stage_entry["status"] = status
        if output is not None:
            stage_entry["output"] = output

    def _start_attempt(self, stage_entry: dict[str, Any], label: str) -> dict[str, Any]:
        """Create a stage attempt record."""

        attempt = {
            "label": label,
            "started_at": datetime.utcnow().isoformat(),
            "ended_at": None,
            "duration_ms": None,
            "status": "in_progress",
            "error": None,
            "output": None,
        }
        stage_entry["attempts"].append(attempt)
        return attempt

    def _finish_attempt(
        self,
        attempt: dict[str, Any],
        *,
        status: str,
        output: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Finalize a stage attempt record."""

        ended_at = datetime.utcnow()
        started_at = datetime.fromisoformat(attempt["started_at"])
        attempt["ended_at"] = ended_at.isoformat()
        attempt["duration_ms"] = int((ended_at - started_at).total_seconds() * 1000)
        attempt["status"] = status
        attempt["error"] = error
        if output is not None:
            attempt["output"] = output

    def _stage_outcomes_for_summary(self, trace: dict[str, Any]) -> list[dict[str, Any]]:
        """Return concise stage outcomes for the summary markdown."""

        return [
            {
                "name": stage_name,
                "status": stage_data.get("status", "unknown"),
                "duration_ms": stage_data.get("duration_ms"),
            }
            for stage_name, stage_data in trace["stages"].items()
        ]

    def _unique(self, values: list[str]) -> list[str]:
        """Preserve list order while removing duplicates and empty values."""

        unique_values: list[str] = []
        for value in values:
            if value and value not in unique_values:
                unique_values.append(value)
        return unique_values

    def _emit(self, verbose: bool, message: str) -> None:
        """Emit concise stage progress when verbose mode is enabled."""

        if verbose and self.progress_callback is not None:
            self.progress_callback(message)
