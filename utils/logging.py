"""Minimal logging helpers for future run artifacts."""

from datetime import datetime


def build_log_message(message: str) -> str:
    """Format a timestamped message for future run logging."""

    return f"[{datetime.utcnow().isoformat()}] {message}"
