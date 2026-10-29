"""Preflight is fully offline: every SDK client and socket raises if touched."""

import socket
from pathlib import Path

import chromadb
import openai
import pytest

import analyze
import agent
import embed
import preflight

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "docs" / "figures" / "synthetic-contract-v2.pdf"


class Touched(AssertionError):
    pass


def _refuse(*args, **kwargs):
    raise Touched("an external client or socket was used")


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(openai, "OpenAI", _refuse)
    monkeypatch.setattr(openai, "AsyncOpenAI", _refuse)
    monkeypatch.setattr(analyze, "ChatOpenAI", _refuse)
    monkeypatch.setattr(analyze, "OpenAIEmbeddings", _refuse)
    monkeypatch.setattr(agent, "ChatOpenAI", _refuse)
    monkeypatch.setattr(agent, "OpenAIEmbeddings", _refuse)
    monkeypatch.setattr(embed.embedding_functions, "OpenAIEmbeddingFunction", _refuse)
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse)
    monkeypatch.chdir(ROOT)
    store = tmp_path / "chroma_storage"
    monkeypatch.setattr(preflight.Utils, "DB_FOLDER", str(store))
    return store


def _run(argv, capsys):
    with pytest.raises(SystemExit) as caught:
        embed.main(argv)
    return caught.value.code, capsys.readouterr().out


def test_preflight_passes_offline_and_reports_everything(offline, capsys):
    code, out = _run(["playbook", "--preflight"], capsys)
    assert code == 0
    assert "Provider:            openai (default)" in out
    assert "Embedding model:     text-embedding-ada-002" in out
    assert "Chat model:          gpt-4o (default)" in out
    assert "API key present:     yes (OPENAI_API_KEY)" in out
    assert "Playbook:            PASS 30 clauses, 10 categories" in out
    assert "Planned index calls: 1 embeddings request(s) for 30 passages" in out
    assert "playbook_clauses - missing" in out
    assert "Result: PASS" in out
    assert not offline.exists(), "preflight must not create the Chroma folder"


def test_preflight_never_prints_the_key(offline, capsys, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-very-secret-value")
    code, out = _run(["playbook", "--preflight"], capsys)
    assert code == 0
    assert "sk-very-secret" not in out


def test_missing_key_fails(offline, capsys, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    code, out = _run(["playbook", "--preflight"], capsys)
    assert code == 1
    assert "API key present:     no (OPENAI_API_KEY)" in out
    assert "Result: FAIL" in out


def test_pdf_call_plan_is_computed_locally(offline, capsys):
    code, out = _run(["playbook", "--preflight", "--pdf", str(SAMPLE)], capsys)
    assert code == 0
    assert "9 blocks: 8 clauses to review, 1 preamble (not reviewed)" in out
    assert "at most 16 embedding queries and 8 chat calls" in out


def test_rejected_pdf_fails(offline, capsys):
    code, out = _run(["playbook", "--preflight", "--pdf", str(ROOT / "docs/figures/image-only-sample.pdf")], capsys)
    assert code == 1
    assert "OCR is not available" in out


def test_unknown_provider_fails(offline, capsys, monkeypatch):
    monkeypatch.setenv("AEGIS_PROVIDER", "mystery")
    code, out = _run(["playbook", "--preflight"], capsys)
    assert code == 1
    assert "unknown provider" in out


def test_invalid_playbook_fails(offline, capsys, monkeypatch):
    monkeypatch.setattr(embed, "load_playbook", lambda path=None: [{"id": "x"}])
    code, out = _run(["playbook", "--preflight"], capsys)
    assert code == 1
    assert "Playbook:            FAIL" in out


def test_collection_states_are_read_without_network(offline, capsys, tmp_path):
    from test_index_failure import HashEmbedding

    real_client = chromadb.PersistentClient
    store = str(offline)
    client = real_client(path=store)
    client.get_or_create_collection("playbook_clauses", embedding_function=HashEmbedding()).upsert(
        ids=["a"], documents=["x"], embeddings=[[0.1] * 8]
    )
    code, out = _run(["playbook", "--preflight"], capsys)
    assert "playbook_clauses - incomplete (1/30 documents)" in out
    chromadb.api.client.SharedSystemClient.clear_system_cache()


def test_bad_arguments_print_usage(offline, capsys):
    with pytest.raises(SystemExit) as caught:
        embed.main(["playbook", "--preflight", "--oops"])
    assert "Usage" in str(caught.value.code)
