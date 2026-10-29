"""The suite itself must not depend on a real key or on the network."""

import os
import socket

import pytest

import utils
from conftest import FAKE_OPENAI_KEY, NetworkBlocked


def test_tests_use_a_fake_key_and_skip_dotenv():
    assert os.environ["OPENAI_API_KEY"] == FAKE_OPENAI_KEY
    assert os.environ["AEGIS_NO_DOTENV"] == "1"
    assert "GEMINI_API_KEY" not in os.environ


def test_dotenv_is_not_read_when_disabled(monkeypatch, tmp_path):
    called = []
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: called.append(a))
    utils.load_project_env()
    assert called == []


def test_dotenv_does_not_replace_a_real_key(monkeypatch):
    seen = {}
    import dotenv

    monkeypatch.delenv("AEGIS_NO_DOTENV")
    monkeypatch.setenv("OPENAI_API_KEY", "  ")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda path, override=False: seen.update(override=override))
    utils.load_project_env()
    assert seen == {"override": False}
    # The blank inherited value was removed so the file could supply a key.
    assert "OPENAI_API_KEY" not in os.environ


def test_outbound_connections_are_refused():
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("api.openai.com", 443), timeout=1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkBlocked):
            sock.connect(("127.0.0.1", 9))
    finally:
        sock.close()
