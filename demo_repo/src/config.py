"""Configuration constants for the demo backend."""

APP_NAME = "Demo Auth API"
SESSION_TOKEN_TTL_SECONDS = 3600

# These values exist so a future limiter can be wired into the login route.
LOGIN_RATE_LIMIT_WINDOW_SECONDS = 60
LOGIN_RATE_LIMIT_MAX_ATTEMPTS = 5

PASSWORD_RESET_BASE_URL = "https://demo-app.local/reset-password"
