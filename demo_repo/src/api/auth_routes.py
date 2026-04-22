"""HTTP handlers for authentication endpoints."""

from src.auth.auth_service import AuthService, AuthenticationError
from src.auth.token_service import TokenService
from src.db.user_repository import UserRepository
from src.utils.request_guards import validate_login_payload


def login_handler(payload: dict[str, str], client_ip: str) -> dict[str, object]:
    """Handle POST /api/auth/login requests."""

    credentials = validate_login_payload(payload)

    # Rate limiting belongs here, before credentials are checked. The request
    # guard utilities already expose the key shape, but no limiter is wired yet.
    auth_service = AuthService(UserRepository(), TokenService())

    try:
        session = auth_service.login(
            email=credentials["email"],
            password=credentials["password"],
            client_ip=client_ip,
        )
    except AuthenticationError as exc:
        return {"status_code": 401, "error": str(exc)}

    return {
        "status_code": 200,
        "data": {
            "access_token": session["access_token"],
            "user": session["user"],
        },
    }
