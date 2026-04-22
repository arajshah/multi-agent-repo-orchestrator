"""Reviewer agent for cross-stage quality checks."""

import json
from typing import Any

from config import get_config
from schemas.agent_schemas import (
    AgentStatus,
    AnalystOutput,
    ImplementationPlannerOutput,
    PlannerOutput,
    ReviewIssue,
    ReviewerOutput,
)
from utils.ollama_client import OllamaClient, OllamaClientError, OllamaRequest


class ReviewerAgent:
    """Evaluate prior stage outputs for coherence, grounding, and completeness."""

    def __init__(self, base_url: str | None = None, model_name: str | None = None) -> None:
        """Initialize the reviewer with the configured Ollama client."""

        config = get_config()
        self.model_name = model_name or config.default_model_name
        self.client = OllamaClient(base_url or config.ollama_base_url)

    def run(
        self,
        user_task: str,
        planner_output: PlannerOutput | dict[str, Any],
        analyst_output: AnalystOutput | dict[str, Any],
        implementation_planner_output: ImplementationPlannerOutput | dict[str, Any],
    ) -> ReviewerOutput:
        """Review prior stage outputs and return a structured assessment."""

        planner = self._validate_planner_output(planner_output)
        analyst = self._validate_analyst_output(analyst_output)
        implementation = self._validate_implementation_output(implementation_planner_output)
        review_context = self._build_review_context(user_task, planner, analyst, implementation)
        deterministic = self._deterministic_checks(planner, analyst, implementation, review_context)

        try:
            response_text = self.client.generate(
                OllamaRequest(
                    model_name=self.model_name,
                    prompt=self._build_prompt(user_task, planner, analyst, implementation, review_context),
                )
            )
            return self._parse_and_validate(
                response_text,
                user_task,
                planner,
                analyst,
                implementation,
                review_context,
                deterministic,
            )
        except (OllamaClientError, ValueError):
            try:
                repaired_text = self.client.generate(
                    OllamaRequest(
                        model_name=self.model_name,
                        prompt=self._build_repair_prompt(
                            user_task,
                            planner,
                            analyst,
                            implementation,
                            review_context,
                        ),
                    )
                )
                return self._parse_and_validate(
                    repaired_text,
                    user_task,
                    planner,
                    analyst,
                    implementation,
                    review_context,
                    deterministic,
                )
            except (OllamaClientError, ValueError):
                return self._fallback_output(
                    user_task,
                    planner,
                    analyst,
                    implementation,
                    review_context,
                    deterministic,
                )

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

    def _validate_implementation_output(
        self,
        implementation_output: ImplementationPlannerOutput | dict[str, Any],
    ) -> ImplementationPlannerOutput:
        """Validate an implementation planner output input."""

        if isinstance(implementation_output, ImplementationPlannerOutput):
            return implementation_output
        return ImplementationPlannerOutput.model_validate(implementation_output)

    def _build_review_context(
        self,
        user_task: str,
        planner: PlannerOutput,
        analyst: AnalystOutput,
        implementation: ImplementationPlannerOutput,
    ) -> dict[str, Any]:
        """Assemble a compact cross-stage context for review."""

        strong_evidence = (
            analyst.confidence >= 0.65
            and len(analyst.evidence_snippets) > 0
            and len([file for file in analyst.relevant_files if not file.endswith(".md")]) > 0
        )
        plan_grounds_to_files = set(implementation.minimal_files_to_touch).issubset(
            set(analyst.relevant_files)
        )

        return {
            "user_task": user_task,
            "task_type": planner.task_type,
            "planner_status": planner.status.value,
            "analyst_status": analyst.status.value,
            "implementation_status": implementation.status.value,
            "analyst_relevant_files": analyst.relevant_files,
            "analyst_open_questions": analyst.open_questions,
            "implementation_minimal_files": implementation.minimal_files_to_touch,
            "implementation_proposed_changes": implementation.proposed_changes,
            "implementation_steps": implementation.implementation_steps,
            "implementation_risks": implementation.risks_or_assumptions,
            "strong_evidence": strong_evidence,
            "plan_grounds_to_files": plan_grounds_to_files,
        }

    def _deterministic_checks(
        self,
        planner: PlannerOutput,
        analyst: AnalystOutput,
        implementation: ImplementationPlannerOutput,
        review_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Run deterministic cross-stage quality checks."""

        issues: list[ReviewIssue] = []
        missing_evidence: list[str] = []
        revisions_requested: list[str] = []

        if planner.task_type in {"implementation_plan", "minimal_files_to_change", "find_feature_location"}:
            if not review_context["strong_evidence"]:
                issues.append(
                    ReviewIssue(
                        severity="high",
                        description="The implementation-oriented plan is built on weak or incomplete analyst evidence.",
                    )
                )
                missing_evidence.append(
                    "The analyst did not surface a strong code-grounded implementation point for the requested feature."
                )
                revisions_requested.append(
                    "Gather stronger file-level evidence before finalizing the change plan."
                )

        if implementation.minimal_files_to_touch and not review_context["plan_grounds_to_files"]:
            issues.append(
                ReviewIssue(
                    severity="high",
                    description="The proposed minimal file set is not fully grounded in the analyst's relevant files.",
                )
            )
            revisions_requested.append(
                "Restrict the file set to analyst-confirmed files before approving the plan."
            )

        if not implementation.minimal_files_to_touch and review_context["strong_evidence"]:
            issues.append(
                ReviewIssue(
                    severity="medium",
                    description="Strong analyst evidence exists, but the implementation plan did not narrow to a minimal file set.",
                )
            )
            revisions_requested.append(
                "Identify the smallest plausible subset of grounded files to touch."
            )

        if analyst.open_questions:
            for question in analyst.open_questions[:2]:
                if question not in missing_evidence:
                    missing_evidence.append(question)

        if planner.task_type == "minimal_files_to_change" and len(implementation.minimal_files_to_touch) > 2:
            issues.append(
                ReviewIssue(
                    severity="medium",
                    description="The implementation plan is broader than expected for a minimal-files task.",
                )
            )
            revisions_requested.append(
                "Trim the plan to the narrowest primary and supporting files."
            )

        if planner.task_type == "explain_code_path" and implementation.proposed_changes:
            issues.append(
                ReviewIssue(
                    severity="low",
                    description="The task is explanation-oriented, so any implementation recommendations should stay lightweight.",
                )
            )

        if implementation.confidence > analyst.confidence + 0.2 and not review_context["strong_evidence"]:
            issues.append(
                ReviewIssue(
                    severity="medium",
                    description="The implementation planner is more confident than the available analyst evidence justifies.",
                )
            )
            revisions_requested.append(
                "Lower confidence or gather stronger evidence before treating the plan as execution-ready."
            )

        final_assessment = self._default_final_assessment(issues, missing_evidence, review_context)
        status = self._default_status(issues, review_context)
        confidence = self._default_confidence(issues, review_context)

        return {
            "issues_found": issues,
            "missing_evidence": missing_evidence,
            "revisions_requested": revisions_requested,
            "final_assessment": final_assessment,
            "status": status,
            "confidence": confidence,
        }

    def _build_prompt(
        self,
        user_task: str,
        planner: PlannerOutput,
        analyst: AnalystOutput,
        implementation: ImplementationPlannerOutput,
        review_context: dict[str, Any],
    ) -> str:
        """Build the reviewer prompt."""

        return (
            "You are the Reviewer agent for RepoPilot.\n"
            "Your job is to review earlier stage outputs for coherence, grounding, completeness, and plausibility.\n"
            "Do not perform repo analysis. Do not invent missing evidence.\n"
            "Be fair: approve solid work, but call out unsupported leaps or weak assumptions.\n"
            "Respond with JSON only. No markdown.\n"
            "Return exactly these keys:\n"
            "agent_name, task_summary, reasoning_summary, key_findings, next_action, confidence, "
            "status, issues_found, missing_evidence, revisions_requested, final_assessment\n"
            "Use agent_name='reviewer'.\n"
            "issues_found must be a list of objects with keys severity and description.\n"
            "Status must be one of: success, partial, failure.\n"
            f"User task: {user_task}\n"
            f"Planner output: {planner.model_dump(mode='json')}\n"
            f"Analyst output: {analyst.model_dump(mode='json')}\n"
            f"Implementation planner output: {implementation.model_dump(mode='json')}\n"
            f"Deterministic review context: {json.dumps(review_context)}"
        )

    def _build_repair_prompt(
        self,
        user_task: str,
        planner: PlannerOutput,
        analyst: AnalystOutput,
        implementation: ImplementationPlannerOutput,
        review_context: dict[str, Any],
    ) -> str:
        """Build a stricter repair prompt for malformed review output."""

        return (
            "Return valid JSON only.\n"
            "Required keys: agent_name, task_summary, reasoning_summary, key_findings, next_action, "
            "confidence, status, issues_found, missing_evidence, revisions_requested, final_assessment.\n"
            "Use agent_name='reviewer'.\n"
            "issues_found must be objects with severity and description.\n"
            "Status must be success, partial, or failure.\n"
            f"User task: {user_task}\n"
            f"Planner output: {planner.model_dump(mode='json')}\n"
            f"Analyst output: {analyst.model_dump(mode='json')}\n"
            f"Implementation planner output: {implementation.model_dump(mode='json')}\n"
            f"Deterministic review context: {json.dumps(review_context)}"
        )

    def _parse_and_validate(
        self,
        response_text: str,
        user_task: str,
        planner: PlannerOutput,
        analyst: AnalystOutput,
        implementation: ImplementationPlannerOutput,
        review_context: dict[str, Any],
        deterministic: dict[str, Any],
    ) -> ReviewerOutput:
        """Parse model output, merge deterministic checks, and validate the schema."""

        payload = self._extract_json_object(response_text)
        payload["agent_name"] = "reviewer"
        payload["task_summary"] = str(payload.get("task_summary") or user_task).strip()
        payload["reasoning_summary"] = str(payload.get("reasoning_summary") or "").strip()
        payload["next_action"] = str(payload.get("next_action") or "").strip()
        payload["key_findings"] = self._normalize_string_list(payload.get("key_findings"))
        payload["missing_evidence"] = self._merge_string_lists(
            self._normalize_string_list(payload.get("missing_evidence")),
            deterministic["missing_evidence"],
        )
        payload["revisions_requested"] = self._merge_string_lists(
            self._normalize_string_list(payload.get("revisions_requested")),
            deterministic["revisions_requested"],
        )
        payload["issues_found"] = self._merge_issues(
            payload.get("issues_found"),
            deterministic["issues_found"],
        )
        payload["final_assessment"] = str(payload.get("final_assessment") or "").strip()
        payload["confidence"] = self._normalize_confidence(
            payload.get("confidence"),
            deterministic["confidence"],
            review_context,
        )
        payload["status"] = self._normalize_status(
            payload.get("status"),
            deterministic["status"],
            review_context,
        )

        if not payload["reasoning_summary"]:
            payload["reasoning_summary"] = self._default_reasoning(planner, analyst, implementation, deterministic)
        if not payload["key_findings"]:
            payload["key_findings"] = self._default_key_findings(planner, analyst, implementation, deterministic)
        if not payload["next_action"]:
            payload["next_action"] = self._default_next_action(deterministic)
        if not payload["final_assessment"]:
            payload["final_assessment"] = deterministic["final_assessment"]

        return ReviewerOutput.model_validate(payload)

    def _extract_json_object(self, response_text: str) -> dict[str, Any]:
        """Extract a JSON object from a model response."""

        start = response_text.find("{")
        end = response_text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("Reviewer response did not contain a JSON object.")

        try:
            payload = json.loads(response_text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("Reviewer response was not valid JSON.") from exc

        if not isinstance(payload, dict):
            raise ValueError("Reviewer response JSON must be an object.")

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

    def _merge_string_lists(self, primary: list[str], secondary: list[str]) -> list[str]:
        """Merge two string lists without duplicates."""

        merged = list(primary)
        for item in secondary:
            if item not in merged:
                merged.append(item)
        return merged

    def _merge_issues(self, value: Any, deterministic_issues: list[ReviewIssue]) -> list[ReviewIssue]:
        """Normalize and merge reviewer issues."""

        issues: list[ReviewIssue] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    try:
                        issue = ReviewIssue.model_validate(item)
                    except Exception:
                        continue
                    issues.append(issue)

        descriptions = {issue.description for issue in issues}
        for issue in deterministic_issues:
            if issue.description not in descriptions:
                issues.append(issue)
        return issues

    def _normalize_confidence(
        self,
        value: Any,
        deterministic_confidence: float,
        review_context: dict[str, Any],
    ) -> float:
        """Normalize reviewer confidence."""

        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = deterministic_confidence

        confidence = max(0.0, min(1.0, confidence))
        if review_context["strong_evidence"] and not review_context["analyst_open_questions"]:
            return max(confidence, deterministic_confidence)
        return min(confidence, deterministic_confidence)

    def _normalize_status(
        self,
        value: Any,
        deterministic_status: AgentStatus,
        review_context: dict[str, Any],
    ) -> AgentStatus:
        """Normalize reviewer status."""

        raw_status = str(value).strip().lower()
        if raw_status == AgentStatus.FAILURE.value:
            status = AgentStatus.FAILURE
        elif raw_status == AgentStatus.SUCCESS.value:
            status = AgentStatus.SUCCESS
        else:
            status = AgentStatus.PARTIAL

        if deterministic_status == AgentStatus.FAILURE:
            return AgentStatus.FAILURE
        if deterministic_status == AgentStatus.PARTIAL:
            return AgentStatus.PARTIAL if status != AgentStatus.FAILURE else status
        if not review_context["strong_evidence"] and status == AgentStatus.SUCCESS:
            return AgentStatus.PARTIAL
        return status

    def _default_reasoning(
        self,
        planner: PlannerOutput,
        analyst: AnalystOutput,
        implementation: ImplementationPlannerOutput,
        deterministic: dict[str, Any],
    ) -> str:
        """Return a grounded reasoning summary."""

        if deterministic["issues_found"]:
            return (
                "The review found recoverable gaps between the analyst evidence and the implementation plan, "
                "so the result should be treated as provisional."
            )
        return (
            "The planner, analyst, and implementation outputs are reasonably aligned and the proposed plan "
            "stays within the grounded evidence."
        )

    def _default_key_findings(
        self,
        planner: PlannerOutput,
        analyst: AnalystOutput,
        implementation: ImplementationPlannerOutput,
        deterministic: dict[str, Any],
    ) -> list[str]:
        """Return concise review findings."""

        findings = [f"Reviewed task type: {planner.task_type}."]
        findings.append(
            f"Implementation plan references {len(implementation.minimal_files_to_touch)} grounded files."
        )
        if deterministic["missing_evidence"]:
            findings.append("Some assumptions still need stronger supporting evidence.")
        return findings

    def _default_next_action(self, deterministic: dict[str, Any]) -> str:
        """Return the next action after review."""

        if deterministic["revisions_requested"]:
            return deterministic["revisions_requested"][0]
        return "Proceed with the reviewed plan while preserving the current grounded scope."

    def _default_final_assessment(
        self,
        issues: list[ReviewIssue],
        missing_evidence: list[str],
        review_context: dict[str, Any],
    ) -> str:
        """Return a concise overall assessment."""

        high_issues = [issue for issue in issues if issue.severity == "high"]
        if high_issues and not review_context["strong_evidence"]:
            return "Partial and needs further inspection before implementation."
        if issues or missing_evidence:
            return "Plausible but missing evidence in one or more important areas."
        return "Strong and grounded."

    def _default_status(
        self,
        issues: list[ReviewIssue],
        review_context: dict[str, Any],
    ) -> AgentStatus:
        """Return the reviewer status."""

        high_issues = [issue for issue in issues if issue.severity == "high"]
        if high_issues and not review_context["strong_evidence"]:
            return AgentStatus.PARTIAL
        if issues:
            return AgentStatus.PARTIAL
        return AgentStatus.SUCCESS

    def _default_confidence(
        self,
        issues: list[ReviewIssue],
        review_context: dict[str, Any],
    ) -> float:
        """Return the reviewer confidence."""

        if not review_context["strong_evidence"]:
            return 0.4
        if issues:
            return 0.6
        return 0.8

    def _fallback_output(
        self,
        user_task: str,
        planner: PlannerOutput,
        analyst: AnalystOutput,
        implementation: ImplementationPlannerOutput,
        review_context: dict[str, Any],
        deterministic: dict[str, Any],
    ) -> ReviewerOutput:
        """Return a deterministic review if model output is unusable."""

        return ReviewerOutput(
            agent_name="reviewer",
            task_summary=user_task,
            reasoning_summary=self._default_reasoning(planner, analyst, implementation, deterministic),
            key_findings=self._default_key_findings(planner, analyst, implementation, deterministic),
            next_action=self._default_next_action(deterministic),
            confidence=deterministic["confidence"],
            status=deterministic["status"],
            issues_found=deterministic["issues_found"],
            missing_evidence=deterministic["missing_evidence"],
            revisions_requested=deterministic["revisions_requested"],
            final_assessment=deterministic["final_assessment"],
        )
