"""Lightweight helpers for in-memory run state."""

from datetime import datetime

from schemas.agent_schemas import BaseAgentOutput
from schemas.run_schemas import (
    FinalOutput,
    RunLifecycleStatus,
    RunStateSchema,
    ToolHistoryEntry,
)


class RunState(RunStateSchema):
    """Mutable helper model used to assemble one run in memory."""

    @classmethod
    def initialize(
        cls,
        run_id: str,
        user_task: str,
        repo_path: str,
        task_type: str = "unknown",
        notes_path: str | None = None,
    ) -> "RunState":
        """Create a new initialized run state."""

        now = datetime.utcnow()
        return cls(
            run_id=run_id,
            user_task=user_task,
            repo_path=repo_path,
            task_type=task_type,
            notes_path=notes_path,
            started_at=now,
            updated_at=now,
        )

    def touch(self) -> None:
        """Update the run timestamp."""

        self.updated_at = datetime.utcnow()

    def add_inspected_file(self, file_path: str) -> None:
        """Record an inspected file once."""

        if file_path not in self.inspected_files:
            self.inspected_files.append(file_path)
            self.touch()

    def attach_agent_output(self, output: BaseAgentOutput) -> None:
        """Attach a validated agent output to the run state."""

        self.agent_outputs[output.agent_name] = output
        self.touch()

    def add_tool_history_entry(
        self,
        tool_name: str,
        status: str,
        input_summary: str = "",
        error: str | None = None,
    ) -> ToolHistoryEntry:
        """Append a lightweight tool invocation record."""

        entry = ToolHistoryEntry(
            tool_name=tool_name,
            status=status,
            input_summary=input_summary,
            error=error,
        )
        self.tool_history.append(entry)
        self.touch()
        return entry

    def attach_final_output(self, final_output: FinalOutput) -> None:
        """Attach the final assembled output and mark the run complete."""

        self.final_output = final_output
        self.status = RunLifecycleStatus.COMPLETED
        self.touch()

    def mark_failed(self) -> None:
        """Mark the run as failed."""

        self.status = RunLifecycleStatus.FAILED
        self.touch()

    def to_serializable_dict(self) -> dict[str, object]:
        """Serialize the run state into standard Python types."""

        return self.model_dump(mode="json")
