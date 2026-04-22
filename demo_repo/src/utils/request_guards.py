"""Request validation helpers for auth endpoints."""


def validate_login_payload(payload: dict[str, str]) -> dict[str, str]:
    """Validate the login request shape."""

    email = payload.get("email", "").strip().lower()
    password = payload.get("password", "")

    if not email or not password:
        raise ValueError("Login requests require email and password.")

    return {"email": email, "password": password}


def build_login_rate_limit_key(email: str, client_ip: str) -> str:
    """Return the key a future login limiter should use."""

    normalized_email = email.strip().lower()
    return f"login:{client_ip}:{normalized_email}"
