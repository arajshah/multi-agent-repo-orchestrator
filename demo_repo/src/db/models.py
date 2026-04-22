"""Persistence models used by the demo repository."""

from dataclasses import dataclass


@dataclass
class UserRecord:
    """Stored account data for one user."""

    id: int
    email: str
    password_hash: str
    full_name: str
    role: str
    is_active: bool = True
