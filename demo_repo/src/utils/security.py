"""Password hashing helpers used by the auth service."""

import hashlib


def hash_password(password: str) -> str:
    """Return a deterministic demo password hash."""

    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, stored_hash: str) -> bool:
    """Check a plaintext password against the stored hash."""

    return hash_password(password) == stored_hash
