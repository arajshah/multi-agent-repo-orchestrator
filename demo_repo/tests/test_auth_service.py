"""Lightweight auth service regression notes."""

from src.auth.auth_service import AuthService
from src.auth.token_service import TokenService
from src.db.user_repository import UserRepository


def test_login_returns_a_session_token() -> None:
    """Document the expected happy-path login flow."""

    service = AuthService(UserRepository(), TokenService())
    result = service.login("alice@example.com", "password123", client_ip="127.0.0.1")
    assert result["access_token"].startswith("session.")
