"""Shared test isolation.

Every test runs with:
- no .env file read (AEGIS_NO_DOTENV=1), so a real key on the machine is never used;
- a fake OpenAI key, so the app's missing-key check does not hide the page;
- no provider selection or Gemini key inherited from the shell;
- outbound TCP connections refused, so no test can reach a model API.
"""

import os
import ipaddress
import socket

import pytest

# Set before any project module is imported by a test module.
os.environ["AEGIS_NO_DOTENV"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

FAKE_OPENAI_KEY = "unit-test-key-not-real"

_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex
_REAL_CREATE_CONNECTION = socket.create_connection


class NetworkBlocked(OSError):
    """Raised when a test tries to open an outbound internet connection."""


def _is_internet(sock):
    return sock.family in (socket.AF_INET, socket.AF_INET6)


def _is_test_loopback(address):
    """Allow Windows asyncio's local socketpair, which uses an ephemeral port."""
    if not isinstance(address, tuple) or len(address) < 2:
        return False
    host, port = address[0], address[1]
    try:
        loopback = ipaddress.ip_address(host).is_loopback
        ephemeral = 49152 <= int(port) <= 65535
    except (TypeError, ValueError):
        return False
    return loopback and ephemeral


def _guard_connect(sock, address, *args, **kwargs):
    if _is_internet(sock) and not _is_test_loopback(address):
        raise NetworkBlocked(f"network access is disabled in tests: {address!r}")
    return _REAL_CONNECT(sock, address, *args, **kwargs)


def _guard_connect_ex(sock, address, *args, **kwargs):
    if _is_internet(sock) and not _is_test_loopback(address):
        raise NetworkBlocked(f"network access is disabled in tests: {address!r}")
    return _REAL_CONNECT_EX(sock, address, *args, **kwargs)


def _guard_create_connection(address, *args, **kwargs):
    raise NetworkBlocked(f"network access is disabled in tests: {address!r}")


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setenv("AEGIS_NO_DOTENV", "1")
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI_KEY)
    for name in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "AEGIS_PROVIDER",
        "GEMINI_CHAT_MODEL",
        "OPENAI_MODEL",
        "OPENAI_TIMEOUT",
        "AEGIS_MAX_RETRIES",
        "MAX_UPLOAD_MB",
        "MAX_PDF_PAGES",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(socket.socket, "connect", _guard_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guard_connect_ex)
    monkeypatch.setattr(socket, "create_connection", _guard_create_connection)
    import retry

    waits = []
    monkeypatch.setattr(retry, "SLEEP", waits.append)
    yield waits
