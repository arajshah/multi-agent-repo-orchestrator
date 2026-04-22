"""User lookup layer backed by in-memory seed data."""

from src.db.models import UserRecord
from src.utils.security import hash_password


_USERS = [
    UserRecord(
        id=1,
        email="alice@example.com",
        password_hash=hash_password("password123"),
        full_name="Alice Admin",
        role="admin",
    ),
    UserRecord(
        id=2,
        email="bruno@example.com",
        password_hash=hash_password("hunter2"),
        full_name="Bruno Builder",
        role="member",
    ),
]


class UserRepository:
    """Read user records from the demo store."""

    def get_by_email(self, email: str) -> UserRecord | None:
        """Return the user with the matching email address."""

        normalized_email = email.strip().lower()
        for user in _USERS:
            if user.email == normalized_email:
                return user
        return None
