"""Planner agent for task classification and workflow planning."""

import json
import re
from typing import Any

from config import get_config
from schemas.agent_schemas import AgentStatus, PlannerOutput
from utils.ollama_client import OllamaClient, OllamaClientError, OllamaRequest


SUPPORTED_TASK_TYPES = {
    "explain_code_path",
    "find_feature_location",
    "minimal_files_to_change",
    "implementation_plan",
    "unsupported",
}

SUPPORTED_TOOLS = {
    "list_files",
    "search_code",
    "read_file",
    "read_file_chunk",
    "write_note",
}

OUTPUT_MODES = {
    "explanation-oriented",
    "localization-oriented",
    "planning-oriented",
}


class PlannerAgent:
    """Interpret a user task and return a structured planning result."""

    def __init__(self, base_url: str | None = None, model_name: str | None = None) -> None:
        """Initialize the planner with the configured Ollama client."""

        config = get_config()
        self.model_name = model_name or config.default_model_name
        self.client = OllamaClient(base_url or config.ollama_base_url)

    def run(self, user_task: str) -> PlannerOutput:
        """Classify a user task and return a structured workflow plan."""

        normalized_task = user_task.strip()
        if not normalized_task:
            return self._unsupported_output(
                user_task=normalized_task,
                reason="User task is empty.",
            )

        hint = self._classify_task_hint(normalized_task)
        if hint == "unsupported":
            return self._unsupported_output(
                user_task=normalized_task,
                reason=(
                    "The task is ambiguous or outside the current supported planner "
                    "categories."
                ),
            )

        prompt = self._build_prompt(normalized_task, hint)

        try:
            response_text = self.client.generate(
                OllamaRequest(model_name=self.model_name, prompt=prompt)
            )
            output = self._parse_and_validate(response_text, normalized_task, hint)
        except (OllamaClientError, ValueError):
            repair_prompt = self._build_repair_prompt(normalized_task, hint)
            try:
                repaired_text = self.client.generate(
                    OllamaRequest(model_name=self.model_name, prompt=repair_prompt)
                )
                output = self._parse_and_validate(repaired_text, normalized_task, hint)
            except (OllamaClientError, ValueError) as exc:
                return self._fallback_output(normalized_task, hint, str(exc))

        if output.task_type == "unsupported":
            return self._unsupported_output(
                user_task=normalized_task,
                reason=output.reasoning_summary or "Task does not fit the supported planner scope.",
                confidence=min(output.confidence, 0.35),
                key_findings=output.key_findings,
            )

        return output

    def _build_prompt(self, user_task: str, hint: str) -> str:
        """Build the planner prompt for an initial structured response."""

        return (
            "You are the Planner agent for RepoPilot.\n"
            "Your job is only to classify the user's task and produce a structured plan.\n"
            "Do not analyze any repository contents. Do not mention files you have not inspected.\n"
            "Valid task_type values are exactly:\n"
            "- explain_code_path\n"
            "- find_feature_location\n"
            "- minimal_files_to_change\n"
            "- implementation_plan\n"
            "- unsupported\n"
            "Valid output_mode values are exactly:\n"
            "- explanation-oriented\n"
            "- localization-oriented\n"
            "- planning-oriented\n"
            "Valid recommended_tools values are chosen only from:\n"
            "- list_files\n"
            "- search_code\n"
            "- read_file\n"
            "- read_file_chunk\n"
            "- write_note\n"
            "Respond with JSON only. No markdown fences. No extra prose.\n"
            "Return an object with exactly these keys:\n"
            "agent_name, task_summary, reasoning_summary, key_findings, next_action, "
            "confidence, status, task_type, workflow_steps, recommended_tools, output_mode\n"
            "Use agent_name='planner'.\n"
            "Status must be one of: success, partial, failure.\n"
            "Confidence must be a float between 0 and 1.\n"
            "Prefer concise arrays of short strings.\n"
            f"Heuristic classification hint: {hint}\n"
            f"User task: {user_task}"
        )

    def _build_repair_prompt(self, user_task: str, hint: str) -> str:
        """Build a stricter prompt for one repair attempt."""

        return (
            "Return valid JSON only.\n"
            "Do not include markdown. Do not include commentary.\n"
            "Required keys: agent_name, task_summary, reasoning_summary, key_findings, "
            "next_action, confidence, status, task_type, workflow_steps, recommended_tools, output_mode.\n"
            "Allowed task_type values: explain_code_path, find_feature_location, "
            "minimal_files_to_change, implementation_plan, unsupported.\n"
            "Allowed output_mode values: explanation-oriented, localization-oriented, planning-oriented.\n"
            "Allowed status values: success, partial, failure.\n"
            "Allowed tools: list_files, search_code, read_file, read_file_chunk, write_note.\n"
            "Use agent_name='planner'.\n"
            f"Heuristic classification hint: {hint}\n"
            f"User task: {user_task}"
        )

    def _parse_and_validate(self, response_text: str, user_task: str, hint: str) -> PlannerOutput:
        """Parse model output, normalize fields, and validate the planner schema."""

        payload = self._extract_json_object(response_text)
        payload["agent_name"] = "planner"
        payload["task_summary"] = str(payload.get("task_summary") or user_task)

        task_type = str(payload.get("task_type") or hint).strip()
        if task_type not in SUPPORTED_TASK_TYPES:
            task_type = hint if hint in SUPPORTED_TASK_TYPES else "unsupported"
        payload["task_type"] = task_type

        payload["workflow_steps"] = self._normalize_string_list(payload.get("workflow_steps"))
        payload["recommended_tools"] = self._normalize_tools(payload.get("recommended_tools"))
        payload["key_findings"] = self._normalize_string_list(payload.get("key_findings"))
        payload["reasoning_summary"] = str(payload.get("reasoning_summary") or "").strip()
        payload["next_action"] = str(payload.get("next_action") or "").strip()
        payload["output_mode"] = self._normalize_output_mode(
            payload.get("output_mode"),
            task_type,
        )
        payload["confidence"] = self._normalize_confidence(payload.get("confidence"), task_type)
        payload["status"] = self._normalize_status(payload.get("status"), task_type, payload["confidence"])

        if not payload["workflow_steps"]:
            payload["workflow_steps"] = self._default_workflow(task_type)
        if not payload["recommended_tools"]:
            payload["recommended_tools"] = self._default_tools(task_type)
        if not payload["key_findings"]:
            payload["key_findings"] = self._default_key_findings(task_type)
        if not payload["next_action"]:
            payload["next_action"] = self._default_next_action(task_type)
        if not payload["reasoning_summary"]:
            payload["reasoning_summary"] = self._default_reasoning(task_type)

        output = PlannerOutput.model_validate(payload)
        if output.task_type != hint and hint != "unsupported":
            output.task_type = hint
            output.workflow_steps = self._default_workflow(hint)
            output.recommended_tools = self._default_tools(hint)
            output.output_mode = self._normalize_output_mode(output.output_mode, hint)
            output.key_findings = self._default_key_findings(hint)
            output.next_action = self._default_next_action(hint)
            output.reasoning_summary = (
                "Task classification was normalized with local planner heuristics "
                "to keep supported task routing deterministic."
            )
            output.confidence = max(output.confidence, 0.8)
            output.status = AgentStatus.SUCCESS

        return output

    def _extract_json_object(self, response_text: str) -> dict[str, Any]:
        """Extract a JSON object from a model response."""

        start = response_text.find("{")
        end = response_text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("Planner response did not contain a JSON object.")

        try:
            payload = json.loads(response_text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("Planner response was not valid JSON.") from exc

        if not isinstance(payload, dict):
            raise ValueError("Planner response JSON must be an object.")

        return payload

    def _classify_task_hint(self, user_task: str) -> str:
        """Classify supported task types with local heuristics."""

        task = user_task.lower()
        normalized = f" {re.sub(r'[^a-z0-9]+', ' ', task)} "

        if self._contains_phrase(
            normalized,
            [
                " implement ",
                " implementation plan ",
                " reviewer checked implementation plan ",
            ],
        ):
            return "implementation_plan"
        if " minimal " in normalized and self._contains_phrase(
            normalized,
            [" files ", " change ", " touch ", " modify "],
        ):
            return "minimal_files_to_change"
        if self._contains_phrase(
            normalized,
            [" where ", " find ", " locate ", " implemented ", " location "],
        ):
            return "find_feature_location"
        if self._contains_phrase(
            normalized,
            [" explain ", " how ", " flow ", " code path ", " works "],
        ):
            return "explain_code_path"

        return "unsupported"

    def _contains_phrase(self, normalized_text: str, phrases: list[str]) -> bool:
        """Return whether any normalized phrase appears in normalized text."""

        return any(phrase in normalized_text for phrase in phrases)

    def _normalize_string_list(self, value: Any) -> list[str]:
        """Normalize a model field into a clean list of strings."""

        if not isinstance(value, list):
            return []
        normalized: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized

    def _normalize_tools(self, value: Any) -> list[str]:
        """Normalize and filter recommended tool names."""

        tools = self._normalize_string_list(value)
        filtered: list[str] = []
        for tool_name in tools:
            if tool_name in SUPPORTED_TOOLS and tool_name not in filtered:
                filtered.append(tool_name)
        return filtered

    def _normalize_output_mode(self, value: Any, task_type: str) -> str:
        """Normalize output mode or choose a deterministic default."""

        output_mode = str(value).strip()
        if output_mode in OUTPUT_MODES:
            return output_mode

        if task_type == "explain_code_path":
            return "explanation-oriented"
        if task_type == "find_feature_location":
            return "localization-oriented"
        return "planning-oriented"

    def _normalize_confidence(self, value: Any, task_type: str) -> float:
        """Normalize confidence into a bounded float."""

        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.3 if task_type == "unsupported" else 0.85

        confidence = max(0.0, min(1.0, confidence))
        if task_type == "unsupported":
            return min(confidence, 0.4)
        return max(confidence, 0.75)

    def _normalize_status(self, value: Any, task_type: str, confidence: float) -> AgentStatus:
        """Normalize status into the planner enum."""

        raw_status = str(value).strip().lower()
        if raw_status == AgentStatus.SUCCESS.value:
            status = AgentStatus.SUCCESS
        elif raw_status == AgentStatus.FAILURE.value:
            status = AgentStatus.FAILURE
        else:
            status = AgentStatus.PARTIAL

        if task_type == "unsupported":
            return AgentStatus.PARTIAL if confidence > 0.0 else AgentStatus.FAILURE
        if confidence >= 0.75:
            return AgentStatus.SUCCESS
        return status

    def _default_workflow(self, task_type: str) -> list[str]:
        """Return a deterministic workflow for a supported task type."""

        if task_type == "explain_code_path":
            return [
                "inspect repo structure",
                "search for symbols or entrypoints related to the task",
                "read targeted files and narrow to relevant code paths",
                "summarize the feature flow in plain language",
            ]
        if task_type == "find_feature_location":
            return [
                "inspect repo structure",
                "search for feature-specific keywords and symbols",
                "read the most relevant files to confirm the implementation location",
                "summarize the likely implementation points",
            ]
        if task_type == "minimal_files_to_change":
            return [
                "inspect repo structure",
                "search for the feature entrypoints and related symbols",
                "read the smallest set of relevant files",
                "identify the minimal files to touch",
                "record assumptions for later validation",
            ]
        if task_type == "implementation_plan":
            return [
                "inspect repo structure",
                "search for relevant feature and dependency points",
                "read the most relevant files and interfaces",
                "propose implementation steps",
                "hand off the plan for reviewer scrutiny",
            ]
        return [
            "clarify the user request",
            "narrow the request to one supported planner task type",
        ]

    def _default_tools(self, task_type: str) -> list[str]:
        """Return a deterministic tool recommendation list."""

        if task_type == "explain_code_path":
            return ["list_files", "search_code", "read_file", "read_file_chunk"]
        if task_type == "find_feature_location":
            return ["list_files", "search_code", "read_file"]
        if task_type == "minimal_files_to_change":
            return ["list_files", "search_code", "read_file", "read_file_chunk", "write_note"]
        if task_type == "implementation_plan":
            return ["list_files", "search_code", "read_file", "read_file_chunk", "write_note"]
        return ["write_note"]

    def _default_key_findings(self, task_type: str) -> list[str]:
        """Return concise default findings for the planning stage."""

        if task_type == "unsupported":
            return ["The request does not cleanly match a supported planner task type."]
        return [
            f"Classified task type: {task_type}.",
            "Planner output is a routing plan only and does not include repo analysis.",
        ]

    def _default_next_action(self, task_type: str) -> str:
        """Return the next handoff action for the planner."""

        if task_type == "unsupported":
            return "Request clarification or restate the task in one of the supported planner categories."
        return f"Proceed with the {task_type} workflow using the recommended tools."

    def _default_reasoning(self, task_type: str) -> str:
        """Return a short reasoning summary for the planning stage."""

        if task_type == "unsupported":
            return "The task is ambiguous or out of scope for the current supported planner categories."
        return "The task was classified using the supported planner categories and converted into a minimal workflow."

    def _unsupported_output(
        self,
        user_task: str,
        reason: str,
        confidence: float = 0.2,
        key_findings: list[str] | None = None,
    ) -> PlannerOutput:
        """Return a structured response for unsupported or ambiguous tasks."""

        return PlannerOutput(
            agent_name="planner",
            task_summary=user_task or "Unsupported planner task.",
            reasoning_summary=reason,
            key_findings=key_findings or self._default_key_findings("unsupported"),
            next_action=self._default_next_action("unsupported"),
            confidence=confidence,
            status=AgentStatus.PARTIAL,
            task_type="unsupported",
            workflow_steps=self._default_workflow("unsupported"),
            recommended_tools=self._default_tools("unsupported"),
            output_mode="planning-oriented",
        )

    def _fallback_output(self, user_task: str, hint: str, error_message: str) -> PlannerOutput:
        """Return a heuristic planner result if model output cannot be used."""

        if hint == "unsupported":
            return self._unsupported_output(
                user_task=user_task,
                reason=f"Planner model output could not be validated. {error_message}",
            )

        return PlannerOutput(
            agent_name="planner",
            task_summary=user_task,
            reasoning_summary=(
                "Planner model output could not be validated, so the planner used "
                "local deterministic task classification instead."
            ),
            key_findings=self._default_key_findings(hint),
            next_action=self._default_next_action(hint),
            confidence=0.8,
            status=AgentStatus.SUCCESS,
            task_type=hint,
            workflow_steps=self._default_workflow(hint),
            recommended_tools=self._default_tools(hint),
            output_mode=self._normalize_output_mode("", hint),
        )
