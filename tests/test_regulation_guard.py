"""The EU regulation download and embedding only run on an explicit command."""

import pytest

import embed


def _forbid(*args, **kwargs):
    raise AssertionError("download or embedding started")


def test_no_arguments_prints_usage_and_downloads_nothing(monkeypatch):
    monkeypatch.setattr(embed, "pdf_to_text", _forbid)
    monkeypatch.setattr(embed, "embed_text_in_chromadb", _forbid)
    with pytest.raises(SystemExit) as caught:
        embed.main([])
    assert "Usage" in str(caught.value.code)
    assert "regulation" in str(caught.value.code)


def test_regulation_is_openai_only(monkeypatch):
    monkeypatch.setenv("AEGIS_PROVIDER", "gemini")
    monkeypatch.setattr(embed, "pdf_to_text", _forbid)
    with pytest.raises(SystemExit) as caught:
        embed.main(["regulation"])
    assert "OpenAI only" in str(caught.value.code)


def test_existing_regulation_index_is_not_duplicated(monkeypatch):
    monkeypatch.setattr(embed, "chroma_collection_exists", lambda name, persist_directory=None: True)
    monkeypatch.setattr(embed, "pdf_to_text", _forbid)
    with pytest.raises(SystemExit) as caught:
        embed.main(["regulation"])
    assert "--rebuild" in str(caught.value.code)


def test_regulation_command_runs_the_existing_pipeline(monkeypatch):
    seen = {}
    monkeypatch.setattr(embed, "chroma_collection_exists", lambda name, persist_directory=None: False)
    monkeypatch.setattr(embed, "pdf_to_text", lambda url: "Article 1\\nText of the act.")
    monkeypatch.setattr(embed, "embed_text_in_chromadb", lambda text, name, description: seen.update(text=text))
    embed.main(["regulation"])
    assert seen["text"].startswith("Article 1")
