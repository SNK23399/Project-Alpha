"""
DEGIRO Client - Connection and API access for Core Satellite portfolio.

Handles authentication with automatic in-app approval flow.
Uses singleton pattern to maintain single connection per session.

Credentials are always prompted interactively (secure input, never stored).
"""

import sys
import time
import io
import logging
import getpass

from degiro_connector.trading.api import API as TradingAPI
from degiro_connector.trading.models.credentials import Credentials
from degiro_connector.core.exceptions import DeGiroConnectionError

# Suppress noisy debug output from degiro_connector
logging.getLogger("degiro_connector").setLevel(logging.WARNING)

# Singleton instance
_client_instance = None


class DegiroClient:
    """
    DEGIRO API client with automatic authentication handling.

    Usage:
        from degiro_client import DegiroClient

        client = DegiroClient.get_instance()
        # client.api is the connected TradingAPI
    """

    def __init__(self):
        self.api: TradingAPI = None
        self._connect()

    @classmethod
    def get_instance(cls) -> 'DegiroClient':
        """Get or create the singleton client instance."""
        global _client_instance
        if _client_instance is None:
            _client_instance = cls()
        return _client_instance

    @classmethod
    def reset(cls):
        """Reset the singleton (force new connection on next get_instance)."""
        global _client_instance
        _client_instance = None

    def _connect(self):
        """Connect to DEGIRO with automatic in-app approval handling."""
        # Always prompt for credentials (never stored)
        print("=" * 80)
        print("DEGIRO LOGIN")
        print("=" * 80)
        print("Enter your DEGIRO credentials (input is secure, not stored)")
        print()

        username = input("Username: ").strip()
        password = getpass.getpass("Password: ")

        # Optional TOTP
        totp_secret = None
        totp_input = getpass.getpass("TOTP Secret (press Enter to skip): ")
        if totp_input.strip():
            totp_secret = totp_input.strip()

        print()

        credentials = Credentials(
            username=username,
            password=password,
            totp_secret_key=totp_secret if totp_secret else None,
        )

        print("=" * 80)
        print("DEGIRO: Connecting...")

        self.api = TradingAPI(credentials=credentials)

        # Suppress stderr during connection attempts
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()

        try:
            self.api.connect()
            sys.stderr = old_stderr
            print("Connected successfully!")
            print("=" * 80)

        except DeGiroConnectionError as e:
            sys.stderr = old_stderr

            # Status 12 = in-app confirmation required
            if e.error_details and e.error_details.status == 12:
                self._wait_for_app_approval(e.error_details.in_app_token)
            else:
                self._handle_connection_error(e)
                raise

        except Exception as e:
            sys.stderr = old_stderr
            print(f"Connection failed: {e}")
            raise

    def _wait_for_app_approval(self, in_app_token: str):
        """Wait for user to approve login in DEGIRO app."""
        self.api.credentials.in_app_token = in_app_token

        print("  ACTION REQUIRED:")
        print("  Open the DEGIRO app on your phone")
        print("  Tap 'Yes' to approve this login")
        print("  Waiting for approval", end="", flush=True)

        max_attempts = 24  # 2 minutes

        for attempt in range(max_attempts):
            time.sleep(5)
            print(".", end="", flush=True)

            old_stderr = sys.stderr
            sys.stderr = io.StringIO()

            try:
                self.api.connect()
                sys.stderr = old_stderr
                print("\n")
                print("Connected successfully!")
                print("=" * 80)
                return

            except DeGiroConnectionError as e:
                sys.stderr = old_stderr
                # Status 3 = still waiting
                if e.error_details and e.error_details.status == 3:
                    continue
                raise

            except Exception:
                sys.stderr = old_stderr
                raise

        raise TimeoutError("No approval received within 2 minutes")

    def _handle_connection_error(self, error: DeGiroConnectionError):
        """Handle and report connection errors."""
        print(f"Connection error: {error}")
        if error.error_details:
            if error.error_details.status == 3:
                print("Too many login attempts. Wait before retrying.")
            elif error.error_details.login_failures:
                print(f"Login failures: {error.error_details.login_failures}")


def get_client() -> DegiroClient:
    """Convenience function to get the DEGIRO client instance."""
    return DegiroClient.get_instance()


def get_api() -> TradingAPI:
    """Convenience function to get the connected Trading API."""
    return get_client().api


if __name__ == "__main__":
    print("\nDEGIRO Client Test\n")

    try:
        client = get_client()
        print("\nTest: Fetching account info...")
        info = client.api.get_client_details()
        print(f"Account ID: {info.id if hasattr(info, 'id') else info.get('id', 'N/A')}")
        print("\nClient ready for use!")

    except Exception as e:
        print(f"\nFailed: {e}")
