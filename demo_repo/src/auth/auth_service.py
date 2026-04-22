"""Core authentication service."""

from src.db.user_repository import UserRepository
from src.utils.security import verify_password


class AuthenticationError(Exception):
    """Raised when login credentials are invalid."""


class AuthService:
    """Authenticate a user and issue a session token."""

    def __init__(self, user_repository: UserRepository, token_service: object) -> None:
        self.user_repository = user_repository
        self.token_service = token_service

    def login(self, email: str, password: str, client_ip: str) -> dict[str, object]:
        """Validate credentials and create a session token."""

        user = self.user_repository.get_by_email(email)
        if user is None:
            raise AuthenticationError("Invalid email or password.")

        if not verify_password(password, user.password_hash):
            raise AuthenticationError("Invalid email or password.")

        if not user.is_active:
            raise AuthenticationError("Account is disabled.")

        # Email verification is intentionally not enforced yet.
        access_token = self.token_service.issue_session(user_id=user.id, client_ip=client_ip)
        return {
            "access_token": access_token,
            "user": {
                "id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role,
            },
        }
