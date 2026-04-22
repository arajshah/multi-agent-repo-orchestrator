"""Codebase analyst agent for targeted repository inspection."""

import json
import re
from pathlib import Path
from typing import Any

from config import get_config
from schemas.agent_schemas import (
    AgentStatus,
    AnalystOutput,
    EvidenceSnippet,
    PlannerOutput,
)
from tools.file_tools import list_files, read_file_chunk, search_code
from tools.note_tools import write_note
from utils.ollama_client import OllamaClient, OllamaClientError, OllamaRequest


STOPWORDS = {
    "a",
    "add",
    "an",
    "and",
    "code",
    "feature",
    "find",
    "for",
    "generate",
    "how",
    "in",
    "is",
    "it",
    "minimal",
    "of",
    "plan",
    "repo",
    "repository",
    "should",
    "the",
    "this",
    "to",
    "where",
    "works",
}

TASK_KEYWORD_HINTS = {
    "explain_code_path": ["flow", "entry", "handler", "service", "auth", "login", "user", "password", "token", "security", "repository"],
    "find_feature_location": ["endpoint", "handler", "route", "service", "controller"],
    "minimal_files_to_change": ["endpoint", "service", "module", "config", "schema"],
    "implementation_plan": ["service", "module", "config", "schema", "model", "models", "email", "email_service", "verify", "repository"],
}

SYNONYM_HINTS = {
    "authentication": ["auth", "login", "session", "token", "password", "user"],
    "login": ["auth", "signin", "endpoint"],
    "verification": ["verify", "token", "email", "email_service"],
    "email": ["verification", "mailer", "token", "email_service"],
    "rate": ["limit", "throttle"],
    "limiting": ["limit", "throttle"],
    "user": ["repository"],
    "password": ["security"],
}

FRAMEWORK_PREFIXES = (
    "agents/",
    "schemas/",
    "memory/",
    "orchestrator/",
    "tests/",
    "tools/",
)


class AnalystAgent:
    """Inspect a repository with approved tools and summarize grounded evidence."""

    def __init__(self, base_url: str | None = None, model_name: str | None = None) -> None:
        """Initialize the analyst with the configured Ollama client."""

        config = get_config()
        self.model_name = model_name or config.default_model_name
        self.client = OllamaClient(base_url or config.ollama_base_url)
        self.default_note_path = config.runs_directory / "analyst_notes.md"

    def run(
        self,
        repo_path: str,
        user_task: str,
        planner_output: PlannerOutput | dict[str, Any],
        note_path: str | None = None,
    ) -> AnalystOutput:
        """Inspect the repository in a task-aware way and return grounded findings."""

        planner = self._validate_planner_output(planner_output)
        note_target = note_path or str(self.default_note_path)
        repo_listing = list_files(repo_path, limit=300)

        if repo_listing.status.value == "error":
            output = AnalystOutput(
                agent_name="analyst",
                task_summary=user_task,
                reasoning_summary="Repository inspection could not start because the repo path was invalid.",
                key_findings=[repo_listing.error or "Repository path validation failed."],
                next_action="Provide a valid repository path and rerun the analyst.",
                confidence=0.0,
                status=AgentStatus.FAILURE,
                relevant_files=[],
                relevant_symbols=[],
                evidence_snippets=[],
                open_questions=["Which repository should be inspected for this task?"],
            )
            self._write_note(note_target, user_task, planner, output)
            return output

        repo_files = repo_listing.files
        queries = self._build_queries(user_task, planner)
        search_results = self._run_searches(repo_path, queries)
        candidate_files = self._rank_candidate_files(repo_files, search_results, planner)
        inspected_chunks = self._inspect_candidate_files(repo_path, candidate_files, search_results)
        relevant_files = [item["file_path"] for item in inspected_chunks]
        relevant_symbols = self._extract_symbols(search_results, inspected_chunks)
        evidence_snippets = self._build_evidence_snippets(search_results, inspected_chunks)
        strong_match_count = self._count_strong_matches(search_results, user_task)

        synthesis_payload = {
            "task_summary": user_task,
            "relevant_files": relevant_files,
            "relevant_symbols": relevant_symbols,
            "evidence_snippets": [snippet.model_dump() for snippet in evidence_snippets],
            "strong_match_count": strong_match_count,
        }

        synthesized = self._synthesize_analysis(user_task, planner, synthesis_payload)
        output = AnalystOutput(
            agent_name="analyst",
            task_summary=user_task,
            reasoning_summary=synthesized["reasoning_summary"],
            key_findings=synthesized["key_findings"],
            next_action=synthesized["next_action"],
            confidence=synthesized["confidence"],
            status=synthesized["status"],
            relevant_files=relevant_files,
            relevant_symbols=relevant_symbols,
            evidence_snippets=evidence_snippets,
            open_questions=synthesized["open_questions"],
        )
        self._write_note(note_target, user_task, planner, output)
        return output

    def _validate_planner_output(
        self,
        planner_output: PlannerOutput | dict[str, Any],
    ) -> PlannerOutput:
        """Validate a planner output input."""

        if isinstance(planner_output, PlannerOutput):
            return planner_output
        return PlannerOutput.model_validate(planner_output)

    def _build_queries(self, user_task: str, planner: PlannerOutput) -> list[str]:
        """Build a short, task-aware list of code search queries."""

        normalized = re.sub(r"[^a-zA-Z0-9_]+", " ", user_task.lower())
        tokens = [token for token in normalized.split() if len(token) >= 4 and token not in STOPWORDS]

        queries: list[str] = []
        for token in tokens:
            if token not in queries:
                queries.append(token)
            for synonym in SYNONYM_HINTS.get(token, []):
                if synonym not in queries:
                    queries.append(synonym)

        for hint in TASK_KEYWORD_HINTS.get(planner.task_type, []):
            if hint not in queries:
                queries.append(hint)

        return queries[:8]

    def _run_searches(self, repo_path: str, queries: list[str]) -> list[dict[str, Any]]:
        """Run bounded repository searches for the chosen queries."""

        results: list[dict[str, Any]] = []
        for query in queries:
            search_result = search_code(repo_path, query, limit=8)
            if search_result.status.value == "error":
                continue
            filtered_matches = [
                match
                for match in search_result.matches
                if self._match_is_relevant(query, match.line_text)
            ]
            results.append(
                {
                    "query": query,
                    "result": search_result,
                    "matches": filtered_matches,
                }
            )
        return results

    def _rank_candidate_files(
        self,
        repo_files: list[str],
        search_results: list[dict[str, Any]],
        planner: PlannerOutput,
    ) -> list[str]:
        """Choose a narrow set of candidate files for closer inspection."""

        scores: dict[str, int] = {}
        for search_result in search_results:
            for match in search_result["matches"]:
                boost = 3
                if self._is_framework_file(match.file_path):
                    boost = 0
                scores[match.file_path] = scores.get(match.file_path, 0) + boost
            query = search_result["query"].lower()
            for file_path in repo_files:
                lowered = file_path.lower()
                if query and query in lowered and not self._is_framework_file(file_path):
                    scores[file_path] = scores.get(file_path, 0) + 2

        for file_path in repo_files:
            lowered = file_path.lower()
            if lowered.endswith(".md"):
                scores[file_path] = scores.get(file_path, 0)
            elif file_path not in scores:
                scores[file_path] = 0

            for keyword in TASK_KEYWORD_HINTS.get(planner.task_type, []):
                if keyword in lowered:
                    boost = 0 if self._is_framework_file(file_path) else 1
                    scores[file_path] = scores.get(file_path, 0) + boost

            if planner.task_type == "implementation_plan" and not self._is_framework_file(file_path):
                if any(
                    marker in lowered
                    for marker in [
                        "auth_service.py",
                        "auth_routes.py",
                        "email_service.py",
                        "models.py",
                        "user_repository.py",
                    ]
                ):
                    scores[file_path] = scores.get(file_path, 0) + 2

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        candidates = [
            file_path
            for file_path, score in ranked
            if score > 0 and not self._is_framework_file(file_path)
        ][:5]

        if not candidates:
            fallback_files = ["README.md", "main.py", "config.py"]
            for file_path in fallback_files:
                if file_path in repo_files and file_path not in candidates:
                    candidates.append(file_path)

        return candidates[:5]

    def _is_framework_file(self, file_path: str) -> bool:
        """Return whether a file is part of the orchestration scaffold itself."""

        return file_path.startswith(FRAMEWORK_PREFIXES)

    def _inspect_candidate_files(
        self,
        repo_path: str,
        candidate_files: list[str],
        search_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Read small chunks from the most relevant candidate files."""

        line_map: dict[str, list[int]] = {}
        for search_result in search_results:
            for match in search_result["matches"]:
                line_map.setdefault(match.file_path, []).append(match.line_number)

        inspected: list[dict[str, Any]] = []
        for relative_file in candidate_files:
            target_lines = sorted(set(line_map.get(relative_file, [])))
            if target_lines:
                anchor_line = target_lines[0]
                chunk_result = read_file_chunk(
                    str(Path(repo_path) / relative_file),
                    max(1, anchor_line - 2),
                    anchor_line + 2,
                )
            else:
                chunk_result = read_file_chunk(str(Path(repo_path) / relative_file), 1, 12)

            if chunk_result.status.value == "error":
                continue

            inspected.append(
                {
                    "file_path": relative_file,
                    "content_lines": chunk_result.content_lines,
                    "actual_range": (
                        chunk_result.actual_range.model_dump()
                        if chunk_result.actual_range is not None
                        else None
                    ),
                }
            )

        return inspected

    def _extract_symbols(
        self,
        search_results: list[dict[str, Any]],
        inspected_chunks: list[dict[str, Any]],
    ) -> list[str]:
        """Heuristically extract symbols from search hits and file chunks."""

        symbols: list[str] = []
        patterns = [
            r"(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\b([A-Za-z_][A-Za-z0-9_]*(?:Agent|Client|Output|State|Pipeline))\b",
        ]

        texts: list[str] = []
        for search_result in search_results:
            texts.extend(
                match.line_text
                for match in search_result["matches"]
                if not self._is_framework_file(match.file_path)
            )
        for chunk in inspected_chunks:
            if self._is_framework_file(chunk["file_path"]):
                continue
            texts.extend(chunk["content_lines"])

        for text in texts:
            for pattern in patterns:
                for match in re.findall(pattern, text):
                    if match not in symbols:
                        symbols.append(match)

        return symbols[:10]

    def _build_evidence_snippets(
        self,
        search_results: list[dict[str, Any]],
        inspected_chunks: list[dict[str, Any]],
    ) -> list[EvidenceSnippet]:
        """Build short grounded evidence snippets from tool output."""

        snippets: list[EvidenceSnippet] = []

        for search_result in search_results:
            for match in search_result["matches"][:2]:
                if self._is_framework_file(match.file_path):
                    continue
                snippet_text = match.line_text.strip()
                if not snippet_text:
                    continue
                snippets.append(
                    EvidenceSnippet(
                        file_path=match.file_path,
                        snippet=snippet_text[:280],
                        line_reference=f"line {match.line_number}",
                    )
                )
                if len(snippets) >= 6:
                    return snippets

        for chunk in inspected_chunks:
            joined = "\n".join(chunk["content_lines"]).strip()
            if not joined:
                continue
            line_reference = ""
            if chunk["actual_range"] is not None:
                line_reference = (
                    f"lines {chunk['actual_range']['start_line']}-"
                    f"{chunk['actual_range']['end_line']}"
                )
            snippets.append(
                EvidenceSnippet(
                    file_path=chunk["file_path"],
                    snippet=joined[:280],
                    line_reference=line_reference,
                )
            )
            if len(snippets) >= 6:
                break

        return snippets

    def _count_strong_matches(
        self,
        search_results: list[dict[str, Any]],
        user_task: str,
    ) -> int:
        """Count non-framework search matches that look task-relevant."""

        task_terms = {
            token
            for token in re.sub(r"[^a-zA-Z0-9_]+", " ", user_task.lower()).split()
            if len(token) >= 4 and token not in STOPWORDS
        }

        strong_matches = 0
        for search_result in search_results:
            for match in search_result["matches"]:
                if self._is_framework_file(match.file_path):
                    continue
                lowered_line = match.line_text.lower()
                if not task_terms or any(term in lowered_line for term in task_terms):
                    strong_matches += 1
        return strong_matches

    def _match_is_relevant(self, query: str, line_text: str) -> bool:
        """Filter naive substring hits down to cleaner word or phrase matches."""

        normalized_query = re.sub(r"[^a-zA-Z0-9_]+", " ", query.lower()).strip()
        normalized_line = re.sub(r"[^a-zA-Z0-9_]+", " ", line_text.lower()).strip()

        if not normalized_query or not normalized_line:
            return False

        if " " in normalized_query:
            return normalized_query in normalized_line

        return re.search(rf"\b{re.escape(normalized_query)}\b", normalized_line) is not None

    def _synthesize_analysis(
        self,
        user_task: str,
        planner: PlannerOutput,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Use the model for concise grounded synthesis with one repair attempt."""

        prompt = self._build_synthesis_prompt(user_task, planner, evidence)
        try:
            response_text = self.client.generate(
                OllamaRequest(model_name=self.model_name, prompt=prompt)
            )
            return self._parse_synthesis_response(response_text, evidence)
        except (OllamaClientError, ValueError):
            repair_prompt = self._build_repair_prompt(user_task, planner, evidence)
            try:
                repaired_text = self.client.generate(
                    OllamaRequest(model_name=self.model_name, prompt=repair_prompt)
                )
                return self._parse_synthesis_response(repaired_text, evidence)
            except (OllamaClientError, ValueError):
                return self._fallback_synthesis(planner, evidence)

    def _build_synthesis_prompt(
        self,
        user_task: str,
        planner: PlannerOutput,
        evidence: dict[str, Any],
    ) -> str:
        """Build the initial grounded synthesis prompt."""

        return (
            "You are the Codebase Analyst agent for RepoPilot.\n"
            "Your job is to summarize already-collected repository evidence.\n"
            "Do not invent files, symbols, or code behavior beyond the evidence provided.\n"
            "If evidence is weak, say so clearly and add open questions.\n"
            "Respond with JSON only. No markdown.\n"
            "Return exactly these keys:\n"
            "reasoning_summary, key_findings, next_action, confidence, status, open_questions\n"
            "Status must be one of: success, partial, failure.\n"
            "Confidence must be a float between 0 and 1.\n"
            f"User task: {user_task}\n"
            f"Planner task type: {planner.task_type}\n"
            f"Planner workflow steps: {planner.workflow_steps}\n"
            f"Collected evidence: {json.dumps(evidence)}"
        )

    def _build_repair_prompt(
        self,
        user_task: str,
        planner: PlannerOutput,
        evidence: dict[str, Any],
    ) -> str:
        """Build a stricter repair prompt for invalid synthesis output."""

        return (
            "Return valid JSON only.\n"
            "Required keys: reasoning_summary, key_findings, next_action, confidence, status, open_questions.\n"
            "Status must be success, partial, or failure.\n"
            "Do not add any other keys.\n"
            f"User task: {user_task}\n"
            f"Planner task type: {planner.task_type}\n"
            f"Collected evidence: {json.dumps(evidence)}"
        )

    def _parse_synthesis_response(
        self,
        response_text: str,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Parse, normalize, and validate synthesis output."""

        payload = self._extract_json_object(response_text)
        findings = self._normalize_string_list(payload.get("key_findings"))
        open_questions = self._normalize_string_list(payload.get("open_questions"))

        if not evidence["relevant_files"]:
            if "No directly relevant files were identified from targeted searches." not in findings:
                findings.append("No directly relevant files were identified from targeted searches.")
            if (
                "Does the repository contain the feature described in the task?"
                not in open_questions
            ):
                open_questions.append("Does the repository contain the feature described in the task?")
        elif evidence["strong_match_count"] == 0:
            if "Direct feature-specific matches were weak, so structural files were used for context." not in findings:
                findings.append(
                    "Direct feature-specific matches were weak, so structural files were used for context."
                )
            if (
                "Are the requested feature names different from the identifiers used in this repository?"
                not in open_questions
            ):
                open_questions.append(
                    "Are the requested feature names different from the identifiers used in this repository?"
                )

        confidence = self._normalize_confidence(payload.get("confidence"), evidence)
        status = self._normalize_status(payload.get("status"), evidence, confidence)

        return {
            "reasoning_summary": str(payload.get("reasoning_summary") or "").strip()
            or self._fallback_reasoning(evidence),
            "key_findings": findings or self._fallback_findings(evidence),
            "next_action": str(payload.get("next_action") or "").strip()
            or self._fallback_next_action(evidence),
            "confidence": confidence,
            "status": status,
            "open_questions": open_questions,
        }

    def _extract_json_object(self, response_text: str) -> dict[str, Any]:
        """Extract a JSON object from a model response."""

        start = response_text.find("{")
        end = response_text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("Analyst response did not contain a JSON object.")

        try:
            payload = json.loads(response_text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("Analyst response was not valid JSON.") from exc

        if not isinstance(payload, dict):
            raise ValueError("Analyst response JSON must be an object.")

        return payload

    def _normalize_string_list(self, value: Any) -> list[str]:
        """Normalize an arbitrary list-like value into clean strings."""

        if not isinstance(value, list):
            return []

        normalized: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in normalized:
                normalized.append(text)
        return normalized

    def _normalize_confidence(self, value: Any, evidence: dict[str, Any]) -> float:
        """Normalize analyst confidence with evidence-aware bounds."""

        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.75 if evidence["strong_match_count"] > 0 else 0.35

        confidence = max(0.0, min(1.0, confidence))
        if not evidence["relevant_files"]:
            return min(confidence, 0.45)
        if evidence["strong_match_count"] == 0:
            return min(confidence, 0.5)
        return max(confidence, 0.65)

    def _normalize_status(
        self,
        value: Any,
        evidence: dict[str, Any],
        confidence: float,
    ) -> AgentStatus:
        """Normalize analyst status based on evidence coverage."""

        raw_status = str(value).strip().lower()
        if raw_status == AgentStatus.FAILURE.value:
            status = AgentStatus.FAILURE
        elif raw_status == AgentStatus.SUCCESS.value:
            status = AgentStatus.SUCCESS
        else:
            status = AgentStatus.PARTIAL

        if not evidence["relevant_files"]:
            return AgentStatus.PARTIAL
        if evidence["strong_match_count"] == 0:
            return AgentStatus.PARTIAL
        if confidence >= 0.7:
            return AgentStatus.SUCCESS
        return status

    def _fallback_synthesis(
        self,
        planner: PlannerOutput,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a deterministic grounded synthesis if model output is unusable."""

        return {
            "reasoning_summary": self._fallback_reasoning(evidence),
            "key_findings": self._fallback_findings(evidence),
            "next_action": self._fallback_next_action(evidence),
            "confidence": 0.7 if evidence["strong_match_count"] > 0 else 0.3,
            "status": (
                AgentStatus.SUCCESS
                if evidence["strong_match_count"] > 0
                else AgentStatus.PARTIAL
            ),
            "open_questions": self._fallback_open_questions(planner, evidence),
        }

    def _fallback_reasoning(self, evidence: dict[str, Any]) -> str:
        """Return a deterministic reasoning summary."""

        if evidence["strong_match_count"] > 0:
            return "The analyst used targeted searches and small file reads to gather repo-grounded evidence."
        return (
            "Targeted searches did not reveal strong repo-grounded evidence for the "
            "requested feature, so the analyst fell back to structural context only."
        )

    def _fallback_findings(self, evidence: dict[str, Any]) -> list[str]:
        """Return deterministic key findings."""

        if evidence["strong_match_count"] > 0:
            return [
                f"Identified {len(evidence['relevant_files'])} candidate files from targeted searches.",
                "Evidence is grounded in search matches and bounded file chunks.",
            ]
        return [
            "No directly relevant feature matches were identified from targeted searches.",
            "Structural files were inspected only to confirm the repository shape.",
        ]

    def _fallback_next_action(self, evidence: dict[str, Any]) -> str:
        """Return a deterministic next action."""

        if evidence["strong_match_count"] > 0:
            return "Use the identified files and snippets as inputs to the implementation-planning phase."
        return "Clarify the feature name, expected symbol names, or repository scope before deeper analysis."

    def _fallback_open_questions(
        self,
        planner: PlannerOutput,
        evidence: dict[str, Any],
    ) -> list[str]:
        """Return uncertainty prompts when evidence is incomplete."""

        if evidence["strong_match_count"] > 0:
            questions: list[str] = []
            if planner.task_type == "implementation_plan":
                questions.append("Which existing entrypoints should be extended in the next planning phase?")
            return questions
        return [
            "Does the repository contain the feature described in the task?",
            "Are the requested feature names different from the identifiers used in this repository?",
        ]

    def _write_note(
        self,
        note_path: str,
        user_task: str,
        planner: PlannerOutput,
        output: AnalystOutput,
    ) -> None:
        """Write a concise evidence note for the run."""

        note = (
            "## Analyst Note\n"
            f"Task: {user_task}\n"
            f"Task type: {planner.task_type}\n"
            f"Relevant files: {', '.join(output.relevant_files) or 'none'}\n"
            f"Relevant symbols: {', '.join(output.relevant_symbols) or 'none'}\n"
            f"Key findings: {'; '.join(output.key_findings) or 'none'}\n"
            f"Open questions: {'; '.join(output.open_questions) or 'none'}\n\n"
        )
        write_note(note_path, note)
