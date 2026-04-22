"""Session token creation utilities."""

from datetime import datetime, timedelta
import secrets

from src.config import SESSION_TOKEN_TTL_SECONDS


class TokenService:
    """Create opaque session tokens for authenticated users."""

    def issue_session(self, user_id: int, client_ip: str) -> str:
        """Return a demo session token string."""

        expires_at = datetime.utcnow() + timedelta(seconds=SESSION_TOKEN_TTL_SECONDS)
        token_body = secrets.token_hex(16)
        return f"session.{user_id}.{client_ip}.{int(expires_at.timestamp())}.{token_body}"
