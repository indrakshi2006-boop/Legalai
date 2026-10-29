"""Optional Q&A tests. The answer functions here are fixtures, not live model reviews."""

from pathlib import Path

import pymupdf as fitz
import pytest
from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

import agent


@pytest.fixture(autouse=True)
def clear_chat_histories():
    agent.NewsChat.store = {}
    yield
    agent.NewsChat.store = {}


def test_session_id_must_be_unique():
    with pytest.raises(ValueError):
        agent.NewsChat("chat1", passages=[])
    with pytest.raises(ValueError):
        agent.NewsChat("  ", passages=[])
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
    assert 'create_chat("chat1")' not in source
    assert "uuid.uuid4().hex" in source


def test_sessions_do_not_share_contract_history():
    passages = [{"id": "clause-001", "text": "The customer pays each invoice within ten days."}]

    def answer(question, sources):
        return "The invoice is due in ten days."

    first = agent.NewsChat("session-a", passages=passages, answerer=answer)
    second = agent.NewsChat("session-b", passages=passages, answerer=answer)
    first.ask("How many days does the customer have to pay the invoice?")
    assert len(first.get_session_history("session-a").messages) == 2
    assert second.get_session_history("session-b").messages == []


def test_contract_answer_includes_the_source_passage():
    seen = {}

    def answer(question, sources):
        seen["sources"] = sources
        return "The customer has ten days to pay."

    chat = agent.NewsChat(
        "session-contract",
        passages=[
            {"id": "clause-001", "text": "The customer pays each invoice within ten days of receipt."},
            {"id": "clause-002", "text": "Only the supplier may terminate immediately without notice."},
        ],
        answerer=answer,
    )
    result = chat.ask("How many days does the customer have to pay an invoice?")
    assert result["supported"] is True
    assert result["sources"][0]["id"] == "clause-001"
    assert "ten days" in result["sources"][0]["text"]
    assert [item["id"] for item in result["sources"]] == ["clause-001"]
    assert seen["sources"][0]["id"] == "clause-001"
    assert chat.rag_chain is None


def test_unsupported_question_does_not_call_the_model():
    called = {"n": 0}

    def answer(question, sources):
        called["n"] += 1
        return "Paris is sunny."

    chat = agent.NewsChat(
        "session-unsupported",
        passages=[{"id": "clause-001", "text": "The customer pays each invoice within ten days."}],
        answerer=answer,
    )
    result = chat.ask("What is the weather in Paris?")
    assert called["n"] == 0
    assert result["supported"] is False
    assert result["sources"] == []
    assert result["answer"] == agent.UNSUPPORTED_ANSWER


def test_model_refusal_is_not_shown_as_a_fact():
    chat = agent.NewsChat(
        "session-refusal",
        passages=[{"id": "clause-001", "text": "The customer pays each invoice within ten days."}],
        answerer=lambda question, sources: "I don't know.",
    )
    result = chat.ask("How many days to pay the invoice?")
    assert result["supported"] is False
    assert result["answer"] == agent.UNSUPPORTED_ANSWER
    assert result["sources"][0]["id"] == "clause-001"


def test_contract_passages_are_not_written_to_a_shared_collection(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("Contract text was sent to a shared collection or embedding client.")

    monkeypatch.setattr(agent, "Chroma", refuse)
    monkeypatch.setattr(agent, "OpenAIEmbeddings", refuse)
    monkeypatch.setattr(agent, "ChatOpenAI", refuse)
    chat = agent.NewsChat(
        "session-private",
        passages=[{"id": "clause-001", "text": "Confidential payment terms of ten days."}],
        answerer=lambda question, sources: "The payment terms are ten days.",
    )
    result = chat.ask("What are the confidential payment terms?")
    assert result["supported"] is True
    assert result["sources"][0]["id"] == "clause-001"
    assert "Confidential payment terms" in chat.passage_index["passages"][0]["text"]


def test_regulation_retriever_returns_source_ids_without_indexing_a_contract():
    class Retriever:
        def invoke(self, question):
            return [
                Document(
                    page_content="Article 5 prohibits that practice.",
                    metadata={"source_id": "act-chunk-5"},
                )
            ]

    chat = agent.NewsChat(
        "session-regulation",
        retriever=Retriever(),
        answerer=lambda question, sources: "Article 5 prohibits that practice.",
    )
    result = chat.ask("What does article 5 prohibit?")
    assert chat.passage_index is None
    assert result["supported"] is True
    assert result["sources"][0]["id"] == "act-chunk-5"
    assert "Article 5" in result["sources"][0]["text"]


def test_empty_regulation_context_is_not_a_fact():
    class Retriever:
        def invoke(self, question):
            return []

    called = {"n": 0}

    def answer(question, sources):
        called["n"] += 1
        return "This practice is prohibited."

    chat = agent.NewsChat("session-empty", retriever=Retriever(), answerer=answer)
    result = chat.ask("What does article 5 prohibit?")
    assert called["n"] == 0
    assert result["supported"] is False
    assert result["answer"] == agent.UNSUPPORTED_ANSWER


def test_chat_model_uses_the_environment_and_timeout(monkeypatch):
    captured = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(agent, "ChatOpenAI", FakeModel)
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_TIMEOUT", "9")
    agent.chat_model()
    assert captured["openai_api_key"] == "unit-test-key"
    assert captured["request_timeout"] == 9.0
    assert captured["temperature"] == 0
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(RuntimeError):
        agent.chat_model()


def _checkbox(page, label):
    matches = [item for item in page.checkbox if item.label == label]
    assert matches, label
    return matches[0]


def _pdf(text):
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(fitz.Rect(72, 72, 540, 700), text)
    data = document.tobytes()
    document.close()
    return data
