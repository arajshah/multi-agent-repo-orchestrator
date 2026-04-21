"""Minimal Ollama HTTP client helpers."""

from typing import Any

from pydantic import BaseModel
import requests


DEFAULT_TIMEOUT_SECONDS = 5
GENERATE_TIMEOUT_SECONDS = 30


class OllamaClientError(Exception):
    """Raised when the local Ollama API cannot satisfy a request."""


class OllamaRequest(BaseModel):
    """Minimal request model for compatibility with the scaffold."""

    model_name: str
    prompt: str


class OllamaClient:
    """Small wrapper around the Ollama helper functions."""

    def __init__(self, base_url: str) -> None:
        """Store the target Ollama base URL."""

        self.base_url = base_url

    def check_server(self) -> dict[str, str]:
        """Verify that the configured Ollama server is reachable."""

        return check_ollama_server(self.base_url)

    def list_models(self) -> list[str]:
        """Return the locally installed models."""

        return list_models(self.base_url)

    def model_exists(self, model_name: str) -> bool:
        """Return whether a model is installed locally."""

        return model_exists(model_name, self.base_url)

    def generate(self, request: OllamaRequest) -> str:
        """Generate text for a prompt using the configured model."""

        result = generate_text(
            prompt=request.prompt,
            model_name=request.model_name,
            base_url=self.base_url,
        )
        return result["response"]


def check_ollama_server(base_url: str) -> dict[str, str]:
    """Verify that the local Ollama server is reachable."""

    endpoint = f"{base_url.rstrip('/')}/api/version"
    try:
        response = requests.get(endpoint, timeout=DEFAULT_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaClientError(
            f"Ollama server not reachable at {base_url}."
        ) from exc

    payload = _parse_json(response, "Unable to read Ollama version response.")
    version = payload.get("version")
    return {
        "base_url": base_url,
        "version": str(version) if version else "unknown",
    }


def list_models(base_url: str) -> list[str]:
    """Return the locally installed Ollama model names."""

    endpoint = f"{base_url.rstrip('/')}/api/tags"
    try:
        response = requests.get(endpoint, timeout=DEFAULT_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaClientError(
            "Unable to list local Ollama models."
        ) from exc

    payload = _parse_json(response, "Unable to read Ollama model list.")
    models = payload.get("models", [])

    if not isinstance(models, list):
        raise OllamaClientError("Unexpected Ollama model list format.")

    model_names: list[str] = []
    for model in models:
        if isinstance(model, dict):
            name = model.get("model") or model.get("name")
            if isinstance(name, str):
                model_names.append(name)

    return model_names


def model_exists(model_name: str, base_url: str) -> bool:
    """Return whether the configured model is installed locally."""

    return model_name in list_models(base_url)


def generate_text(prompt: str, model_name: str, base_url: str) -> dict[str, str]:
    """Run a minimal non-streaming generation request."""

    endpoint = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=GENERATE_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaClientError(
            f"Generation failed for model '{model_name}'."
        ) from exc

    data = _parse_json(response, "Unable to read Ollama generation response.")
    text = data.get("response")

    if not isinstance(text, str):
        raise OllamaClientError("Unexpected Ollama generation response format.")

    return {
        "model": model_name,
        "response": text.strip(),
    }


def _parse_json(response: requests.Response, error_message: str) -> dict[str, Any]:
    """Parse a JSON response and raise a readable client error on failure."""

    try:
        payload = response.json()
    except ValueError as exc:
        raise OllamaClientError(error_message) from exc

    if not isinstance(payload, dict):
        raise OllamaClientError(error_message)

    return payload
