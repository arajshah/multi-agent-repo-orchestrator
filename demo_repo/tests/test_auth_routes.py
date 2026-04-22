"""Lightweight route-level checks for the demo API."""

from src.api.auth_routes import login_handler


def test_login_route_returns_200_for_valid_credentials() -> None:
    """Document the current login route contract."""

    response = login_handler(
        {"email": "alice@example.com", "password": "password123"},
        client_ip="127.0.0.1",
    )
    assert response["status_code"] == 200
