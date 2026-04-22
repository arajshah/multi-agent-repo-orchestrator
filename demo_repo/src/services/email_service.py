"""Email delivery placeholders for the demo backend."""

from src.config import PASSWORD_RESET_BASE_URL


class EmailService:
    """Send transactional emails through a future provider integration."""

    def send_password_reset_email(self, email: str, reset_token: str) -> dict[str, str]:
        """Build a password reset message payload."""

        return {
            "to": email,
            "template": "password_reset",
            "reset_link": f"{PASSWORD_RESET_BASE_URL}?token={reset_token}",
        }

    def send_welcome_email(self, email: str, full_name: str) -> dict[str, str]:
        """Build a welcome email payload."""

        return {
            "to": email,
            "template": "welcome",
            "subject": f"Welcome, {full_name}",
        }
