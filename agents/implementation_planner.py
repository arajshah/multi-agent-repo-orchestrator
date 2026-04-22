"""Implementation planner agent for evidence-based engineering plans."""

import json
from typing import Any

from config import get_config
from schemas.agent_schemas import (
    AgentStatus,
    AnalystOutput,
    ImplementationPlannerOutput,
    PlannerOutput,
)
from utils.ollama_client import OllamaClient, OllamaClientError, OllamaRequest


class ImplementationPlannerAgent:
    """Turn planner routing and analyst evidence into a concrete change plan."""

    def __init__(self, base_url: str | None = None, model_name: str | None = None) -> None:
        """Initialize the implementation planner with the configured Ollama client."""

        config = get_config()
        self.model_name = model_name or config.default_model_name
        self.client = OllamaClient(base_url or config.ollama_base_url)

    def run(
        self,
        user_task: str,
        planner_output: PlannerOutput | dict[str, Any],
        analyst_output: AnalystOutput | dict[str, Any],
    ) -> ImplementationPlannerOutput:
        """Build a grounded implementation plan from structured prior outputs."""

        planner = self._validate_planner_output(planner_output)
        analyst = self._validate_analyst_output(analyst_output)
        evidence = self._build_evidence_context(user_task, planner, analyst)

        try:
            response_text = self.client.generate(
                OllamaRequest(
                    model_name=self.model_name,
                    prompt=self._build_prompt(user_task, planner, analyst, evidence),
                )
            )
            return self._parse_and_validate(response_text, user_task, planner, analyst, evidence)
        except (OllamaClientError, ValueError):
            try:
                repaired_text = self.client.generate(
                    OllamaRequest(
                        model_name=self.model_name,
                        prompt=self._build_repair_prompt(user_task, planner, analyst, evidence),
                    )
                )
                return self._parse_and_validate(
                    repaired_text,
                    user_task,
                    planner,
                    analyst,
                    evidence,
                )
            except (OllamaClientError, ValueError):
                return self._fallback_output(user_task, planner, analyst, evidence)

    def _validate_planner_output(
        self,
        planner_output: PlannerOutput | dict[str, Any],
    ) -> PlannerOutput:
        """Validate a planner output input."""

        if isinstance(planner_output, PlannerOutput):
            return planner_output
        return PlannerOutput.model_validate(planner_output)

    def _validate_analyst_output(
        self,
        analyst_output: AnalystOutput | dict[str, Any],
    ) -> AnalystOutput:
        """Validate an analyst output input."""

        if isinstance(analyst_output, AnalystOutput):
            return analyst_output
        return AnalystOutput.model_validate(analyst_output)

    def _build_evidence_context(
        self,
        user_task: str,
        planner: PlannerOutput,
        analyst: AnalystOutput,
    ) -> dict[str, Any]:
        """Assemble a compact evidence package for planning."""

        grounded_code_files = [
            file_path
            for file_path in analyst.relevant_files
            if not file_path.endswith(".md")
        ]
        evidence_is_strong = (
            analyst.confidence >= 0.65
            and len(grounded_code_files) > 0
            and len(analyst.evidence_snippets) > 0
        )

        return {
            "user_task": user_task,
            "task_type": planner.task_type,
            "planner_key_findings": planner.key_findings,
            "planner_workflow_steps": planner.workflow_steps,
            "analyst_relevant_files": analyst.relevant_files,
            "grounded_code_files": grounded_code_files,
            "analyst_relevant_symbols": analyst.relevant_symbols,
            "analyst_evidence_snippets": [
                snippet.model_dump() for snippet in analyst.evidence_snippets
            ],
            "analyst_open_questions": analyst.open_questions,
            "evidence_is_strong": evidence_is_strong,
        }

    def _build_prompt(
        self,
        user_task: str,
        planner: PlannerOutput,
        analyst: AnalystOutput,
        evidence: dict[str, Any],
    ) -> str:
        """Build the implementation-planning prompt."""

        return (
            "You are the Implementation Planner agent for RepoPilot.\n"
            "Your job is to turn grounded analyst evidence into a concrete engineering plan.\n"
            "Do not invent files outside the evidence package.\n"
            "Do not write code. Do not emit diffs. Do not perform review logic.\n"
            "If evidence is weak, say so clearly and produce a cautious partial plan.\n"
            "Respond with JSON only. No markdown fences.\n"
            "Return exactly these keys:\n"
            "agent_name, task_summary, reasoning_summary, key_findings, next_action, "
            "confidence, status, proposed_changes, minimal_files_to_touch, "
            "implementation_steps, risks_or_assumptions\n"
            "Use agent_name='implementation_planner'.\n"
            "Status must be one of: success, partial, failure.\n"
            "Confidence must be a float between 0 and 1.\n"
            "minimal_files_to_touch must be a subset of the analyst's relevant files.\n"
            f"User task: {user_task}\n"
            f"Planner task type: {planner.task_type}\n"
            f"Planner workflow steps: {planner.workflow_steps}\n"
            f"Analyst reasoning summary: {analyst.reasoning_summary}\n"
            f"Evidence package: {json.dumps(evidence)}"
        )

    def _build_repair_prompt(
        self,
        user_task: str,
        planner: PlannerOutput,
        analyst: AnalystOutput,
        evidence: dict[str, Any],
    ) -> str:
        """Build a stricter repair prompt for malformed model output."""

        return (
            "Return valid JSON only.\n"
            "Required keys: agent_name, task_summary, reasoning_summary, key_findings, "
            "next_action, confidence, status, proposed_changes, minimal_files_to_touch, "
            "implementation_steps, risks_or_assumptions.\n"
            "Use agent_name='implementation_planner'.\n"
            "Status must be success, partial, or failure.\n"
            "minimal_files_to_touch must only use analyst relevant files.\n"
            f"User task: {user_task}\n"
            f"Planner task type: {planner.task_type}\n"
            f"Analyst reasoning summary: {analyst.reasoning_summary}\n"
            f"Evidence package: {json.dumps(evidence)}"
        )

    def _parse_and_validate(
        self,
        response_text: str,
        user_task: str,
        planner: PlannerOutput,
        analyst: AnalystOutput,
        evidence: dict[str, Any],
    ) -> ImplementationPlannerOutput:
        """Parse model output, normalize grounded fields, and validate the schema."""

        payload = self._extract_json_object(response_text)
        payload["agent_name"] = "implementation_planner"
        payload["task_summary"] = str(payload.get("task_summary") or user_task).strip()
        payload["reasoning_summary"] = str(payload.get("reasoning_summary") or "").strip()
        payload["next_action"] = str(payload.get("next_action") or "").strip()
        payload["key_findings"] = self._normalize_string_list(payload.get("key_findings"))
        payload["proposed_changes"] = self._normalize_string_list(payload.get("proposed_changes"))
        payload["implementation_steps"] = self._normalize_string_list(
            payload.get("implementation_steps")
        )
        payload["risks_or_assumptions"] = self._normalize_string_list(
            payload.get("risks_or_assumptions")
        )
        payload["minimal_files_to_touch"] = self._normalize_minimal_files(
            payload.get("minimal_files_to_touch"),
            analyst.relevant_files,
            evidence,
            planner.task_type,
        )
        payload["confidence"] = self._normalize_confidence(
            payload.get("confidence"),
            evidence,
        )
        payload["status"] = self._normalize_status(payload.get("status"), evidence)

        if not payload["key_findings"]:
            payload["key_findings"] = self._default_key_findings(planner, analyst, evidence)
        if not payload["proposed_changes"]:
            payload["proposed_changes"] = self._default_proposed_changes(planner, analyst, evidence)
        if not payload["implementation_steps"]:
            payload["implementation_steps"] = self._default_steps(planner, analyst, evidence)
        if not payload["risks_or_assumptions"]:
            payload["risks_or_assumptions"] = self._default_risks(planner, analyst, evidence)
        if not payload["reasoning_summary"]:
            payload["reasoning_summary"] = self._default_reasoning(planner, analyst, evidence)
        if not payload["next_action"]:
            payload["next_action"] = self._default_next_action(planner, evidence)

        return ImplementationPlannerOutput.model_validate(payload)

    def _extract_json_object(self, response_text: str) -> dict[str, Any]:
        """Extract a JSON object from a model response."""

        start = response_text.find("{")
        end = response_text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("Implementation planner response did not contain a JSON object.")

        try:
            payload = json.loads(response_text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("Implementation planner response was not valid JSON.") from exc

        if not isinstance(payload, dict):
            raise ValueError("Implementation planner response JSON must be an object.")

        return payload

    def _normalize_string_list(self, value: Any) -> list[str]:
        """Normalize an arbitrary list-like value into unique strings."""

        if not isinstance(value, list):
            return []

        normalized: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in normalized:
                normalized.append(text)
        return normalized

    def _normalize_minimal_files(
        self,
        value: Any,
        analyst_files: list[str],
        evidence: dict[str, Any],
        task_type: str,
    ) -> list[str]:
        """Restrict file suggestions to analyst-grounded candidates."""

        allowed_files = set(analyst_files)
        normalized = self._normalize_string_list(value)
        filtered = [file_path for file_path in normalized if file_path in allowed_files]

        if filtered:
            return filtered[: self._max_files_for_task(task_type)]

        if not evidence["evidence_is_strong"]:
            return []

        return evidence["grounded_code_files"][: self._max_files_for_task(task_type)]

    def _max_files_for_task(self, task_type: str) -> int:
        """Return the maximum file count to keep plans minimal."""

        if task_type == "minimal_files_to_change":
            return 2
        if task_type == "find_feature_location":
            return 2
        if task_type == "implementation_plan":
            return 3
        return 2

    def _normalize_confidence(self, value: Any, evidence: dict[str, Any]) -> float:
        """Normalize confidence based on evidence strength."""

        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.75 if evidence["evidence_is_strong"] else 0.35

        confidence = max(0.0, min(1.0, confidence))
        if evidence["evidence_is_strong"]:
            return max(confidence, 0.65)
        return min(confidence, 0.5)

    def _normalize_status(self, value: Any, evidence: dict[str, Any]) -> AgentStatus:
        """Normalize planner status based on evidence strength."""

        raw_status = str(value).strip().lower()
        if raw_status == AgentStatus.FAILURE.value:
            status = AgentStatus.FAILURE
        elif raw_status == AgentStatus.SUCCESS.value:
            status = AgentStatus.SUCCESS
        else:
            status = AgentStatus.PARTIAL

        if evidence["evidence_is_strong"]:
            return AgentStatus.SUCCESS if status != AgentStatus.FAILURE else status
        return AgentStatus.PARTIAL if status != AgentStatus.FAILURE else status

    def _default_reasoning(
        self,
        planner: PlannerOutput,
        analyst: AnalystOutput,
        evidence: dict[str, Any],
    ) -> str:
        """Return a grounded reasoning summary."""

        if evidence["evidence_is_strong"]:
            return (
                "The implementation plan is grounded in analyst-identified files, symbols, "
                "and evidence snippets rather than fresh repo exploration."
            )
        return (
            "Analyst evidence was incomplete, so the plan stays cautious and focuses on "
            "minimal next steps plus explicit assumptions."
        )

    def _default_key_findings(
        self,
        planner: PlannerOutput,
        analyst: AnalystOutput,
        evidence: dict[str, Any],
    ) -> list[str]:
        """Return core planning findings."""

        findings = [f"Planner task type: {planner.task_type}."]
        if analyst.relevant_files:
            findings.append(
                f"Analyst surfaced candidate files: {', '.join(analyst.relevant_files[:3])}."
            )
        if not evidence["evidence_is_strong"]:
            findings.append("Evidence is incomplete, so the plan should be treated as provisional.")
        return findings

    def _default_proposed_changes(
        self,
        planner: PlannerOutput,
        analyst: AnalystOutput,
        evidence: dict[str, Any],
    ) -> list[str]:
        """Return grounded high-level change proposals."""

        if not evidence["evidence_is_strong"]:
            return [
                "Confirm the concrete implementation files before making code changes.",
                "Limit the eventual change set to the narrowest entrypoint and service files once identified.",
            ]

        files = evidence["grounded_code_files"]
        if planner.task_type == "implementation_plan":
            return [
                f"Extend the primary feature logic in `{files[0]}` to support the requested behavior.",
                (
                    f"Update adjacent logic in `{files[1]}` to keep control flow consistent."
                    if len(files) > 1
                    else "Update adjacent validation or state handling near the primary change point."
                ),
            ]
        if planner.task_type == "minimal_files_to_change":
            return [
                f"Constrain the change set to `{files[0]}` as the most likely primary edit location.",
                (
                    f"Touch `{files[1]}` only if a supporting dependency or interface update is required."
                    if len(files) > 1
                    else "Avoid secondary file changes unless the primary edit reveals a required dependency."
                ),
            ]
        if planner.task_type == "find_feature_location":
            return [
                f"Use `{files[0]}` as the likely insertion point for the requested feature change.",
                "Validate whether a second supporting file is required before broadening the implementation scope.",
            ]
        return [
            "Preserve the current code path while documenting the likely change points for future implementation.",
            "Translate the identified evidence into a small follow-up change plan only if the task shifts toward implementation.",
        ]

    def _default_steps(
        self,
        planner: PlannerOutput,
        analyst: AnalystOutput,
        evidence: dict[str, Any],
    ) -> list[str]:
        """Return ordered implementation steps."""

        if not evidence["evidence_is_strong"]:
            return [
                "Resolve the analyst open questions and confirm the exact feature entrypoint.",
                "Re-run targeted analysis on the confirmed files before proposing code changes.",
                "Keep the eventual implementation limited to the smallest validated file set.",
            ]

        files = evidence["grounded_code_files"]
        steps = [f"Start with `{files[0]}` as the primary implementation file."]
        if len(files) > 1:
            steps.append(f"Update `{files[1]}` only if the primary change requires a supporting adjustment.")
        steps.append("Implement the feature logic at a high level without expanding the change surface unnecessarily.")
        steps.append("Re-check assumptions and any test or documentation impact before execution.")
        return steps

    def _default_risks(
        self,
        planner: PlannerOutput,
        analyst: AnalystOutput,
        evidence: dict[str, Any],
    ) -> list[str]:
        """Return grounded risks and assumptions."""

        risks = list(analyst.open_questions[:2])
        if not evidence["evidence_is_strong"]:
            risks.append("The repository evidence does not yet confirm the exact implementation files.")
        if planner.task_type == "implementation_plan":
            risks.append("Supporting model, config, or state changes may be required once the real extension points are confirmed.")
        elif planner.task_type == "minimal_files_to_change":
            risks.append("The smallest file set could expand if the primary entrypoint depends on shared abstractions.")
        return risks

    def _default_next_action(self, planner: PlannerOutput, evidence: dict[str, Any]) -> str:
        """Return the next planning handoff."""

        if evidence["evidence_is_strong"]:
            return "Use this file-scoped plan as the execution brief for the coding phase."
        return "Resolve the missing evidence first, then tighten the implementation plan around confirmed files."

    def _fallback_output(
        self,
        user_task: str,
        planner: PlannerOutput,
        analyst: AnalystOutput,
        evidence: dict[str, Any],
    ) -> ImplementationPlannerOutput:
        """Return a deterministic plan if model output is unusable."""

        return ImplementationPlannerOutput(
            agent_name="implementation_planner",
            task_summary=user_task,
            reasoning_summary=self._default_reasoning(planner, analyst, evidence),
            key_findings=self._default_key_findings(planner, analyst, evidence),
            next_action=self._default_next_action(planner, evidence),
            confidence=0.7 if evidence["evidence_is_strong"] else 0.35,
            status=AgentStatus.SUCCESS if evidence["evidence_is_strong"] else AgentStatus.PARTIAL,
            proposed_changes=self._default_proposed_changes(planner, analyst, evidence),
            minimal_files_to_touch=self._normalize_minimal_files(
                [],
                analyst.relevant_files,
                evidence,
                planner.task_type,
            ),
            implementation_steps=self._default_steps(planner, analyst, evidence),
            risks_or_assumptions=self._default_risks(planner, analyst, evidence),
        )
