"""Ensure system trust roots are used for HTTPS (corporate-proxy friendly)."""
from __future__ import annotations


def ensure_system_certificates() -> None:
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        # truststore is best-effort; default certifi roots remain in place.
        pass
