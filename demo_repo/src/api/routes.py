"""Route registration for the demo backend."""

from src.api.auth_routes import login_handler


def build_route_table() -> dict[tuple[str, str], object]:
    """Return the static route map used by the demo API."""

    return {
        ("POST", "/api/auth/login"): login_handler,
    }
