"""Fixed end-to-end orchestration pipeline."""

from datetime import datetime
from typing import Callable

from agents.analyst import AnalystAgent
from agents.implementation_planner import ImplementationPlannerAgent
from agents.planner import PlannerAgent
from agents.reviewer import ReviewerAgent
from memory.run_state import RunState
from schemas.agent_schemas import (
    AgentStatus,
    AnalystOutput,
    ImplementationPlannerOutput,
    PlannerOutput,
    ReviewerOutput,
)
from schemas.run_schemas import FinalOutput, RunLifecycleStatus


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

    def run(self, repo_path: str, user_task: str, verbose: bool = False) -> FinalOutput:
        """Execute the fixed multi-agent pipeline and return a final structured result."""

        state = RunState.initialize(
            run_id=self._build_run_id(),
            user_task=user_task,
            repo_path=repo_path,
        )
        state.status = RunLifecycleStatus.IN_PROGRESS
        state.touch()
        self.last_run_state = state

        self._emit(verbose, "starting planner")
        planner_result = self._run_planner(state, user_task)
        if isinstance(planner_result, FinalOutput):
            self._emit(verbose, "final assembly complete")
            return planner_result
        planner_output = planner_result
        self._emit(verbose, "planner complete")

        self._emit(verbose, "starting analyst")
        analyst_result = self._run_analyst(state, repo_path, user_task, planner_output, verbose)
        if isinstance(analyst_result, FinalOutput):
            self._emit(verbose, "final assembly complete")
            return analyst_result
        analyst_output = analyst_result
        self._emit(verbose, "analyst complete")

        self._emit(verbose, "starting implementation planner")
        implementation_result = self._run_implementation_planner(
            state,
            user_task,
            planner_output,
            analyst_output,
        )
        if isinstance(implementation_result, FinalOutput):
            self._emit(verbose, "final assembly complete")
            return implementation_result
        implementation_output = implementation_result
        self._emit(verbose, "implementation planner complete")

        self._emit(verbose, "starting reviewer")
        reviewer_result = self._run_reviewer(
            state,
            user_task,
            planner_output,
            analyst_output,
            implementation_output,
        )
        if isinstance(reviewer_result, FinalOutput):
            self._emit(verbose, "final assembly complete")
            return reviewer_result
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
        self._emit(verbose, "final assembly complete")
        return final_output

    def start(self, repo_path: str, user_task: str, verbose: bool = False) -> FinalOutput:
        """Compatibility wrapper for the fixed pipeline entrypoint."""

        return self.run(repo_path=repo_path, user_task=user_task, verbose=verbose)

    def _run_planner(self, state: RunState, user_task: str) -> PlannerOutput | FinalOutput:
        """Run the planner stage and stop early for unsupported tasks."""

        try:
            planner_output = self.planner.run(user_task)
        except Exception as exc:
            return self._build_failure_output(
                state,
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
            return final_output

        return planner_output

    def _run_analyst(
        self,
        state: RunState,
        repo_path: str,
        user_task: str,
        planner_output: PlannerOutput,
        verbose: bool,
    ) -> AnalystOutput | FinalOutput:
        """Run the analyst stage with one broadened retry if evidence is weak."""

        try:
            analyst_output = self.analyst.run(repo_path, user_task, planner_output)
        except Exception as exc:
            return self._build_failure_output(
                state,
                task_summary=user_task,
                final_response="Analyst stage failed before grounded evidence could be collected.",
                reviewer_notes=[f"Analyst error: {exc}"],
                status=AgentStatus.FAILURE,
                confidence=0.0,
                failing_stage="analyst",
            )

        analyst_retry_used = False
        if self._analyst_output_is_weak(analyst_output):
            analyst_retry_used = True
            self._emit(verbose, "analyst retry triggered")
            broadened_task = self._build_broadened_analyst_task(user_task, planner_output)
            try:
                retry_output = self.analyst.run(repo_path, broadened_task, planner_output)
            except Exception:
                retry_output = analyst_output
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

        if analyst_output.status == AgentStatus.FAILURE:
            return self._build_failure_output(
                state,
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
        user_task: str,
        planner_output: PlannerOutput,
        analyst_output: AnalystOutput,
    ) -> ImplementationPlannerOutput | FinalOutput:
        """Run the implementation planner stage."""

        try:
            implementation_output = self.implementation_planner.run(
                user_task,
                planner_output,
                analyst_output,
            )
        except Exception as exc:
            return self._build_failure_output(
                state,
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

        if implementation_output.status == AgentStatus.FAILURE:
            return self._build_failure_output(
                state,
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
        user_task: str,
        planner_output: PlannerOutput,
        analyst_output: AnalystOutput,
        implementation_output: ImplementationPlannerOutput,
    ) -> ReviewerOutput | FinalOutput:
        """Run the reviewer stage."""

        try:
            reviewer_output = self.reviewer.run(
                user_task,
                planner_output,
                analyst_output,
                implementation_output,
            )
        except Exception as exc:
            return self._build_failure_output(
                state,
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

        if reviewer_output.status == AgentStatus.FAILURE:
            return self._build_failure_output(
                state,
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

    def _build_failure_output(
        self,
        state: RunState,
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

    def _build_run_id(self) -> str:
        """Build a lightweight in-memory run identifier."""

        return datetime.utcnow().strftime("run-%Y%m%d%H%M%S")

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
