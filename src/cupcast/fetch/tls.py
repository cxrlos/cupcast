from __future__ import annotations

import truststore

_injected = False


def ensure_system_certificates() -> None:
    # Use the OS trust store so TLS works behind corporate/inspecting proxies
    # whose CA is in the system keychain but not in certifi's bundle.
    global _injected
    if not _injected:
        truststore.inject_into_ssl()
        _injected = True
