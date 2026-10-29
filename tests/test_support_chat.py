"""Local checks for the integrated customer support page and routing."""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

import support_chat


def fake_answerer(system, payload):
    assert "supplied sample support documents" in system
    assert json.loads(payload)["support_documents"]
    return "The sample documents say refunds are accepted within 30 days."


def test_support_workflow_routes_and_keeps_review_gate():
    result = support_chat.respond("I need a refund for my subscription", fake_answerer)
    assert result["category"] == "Billing"
    assert result["approval_required"] is True
    assert "Pending supervisor approval" in result["response"]
    assert "within 30 days" in result["response"]
    assert "Reference documents" in result["response"]


def test_memory_route_uses_previous_issue():
    payloads = []

    def answerer(system, payload):
        payloads.append(json.loads(payload))
        return "Your last issue was about signing in."

    result = support_chat.respond("What was my previous support issue?", answerer, "Cannot log in")
    assert result["category"] == "Memory"
    assert "Cannot log in" in payloads[0]["support_documents"][0]


def test_customer_support_page_opens_with_gemini_configuration(monkeypatch):
    monkeypatch.setenv("AEGIS_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.run()
    app.radio[0].set_value("Customer Support Chat").run()
    assert not app.exception
    assert any(item.value == "Customer Support Chat" for item in app.title)
    assert any("Gemini" in str(item.value) for item in app.caption)


def test_customer_support_shows_missing_gemini_key(monkeypatch):
    monkeypatch.setenv("AEGIS_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.run()
    app.radio[0].set_value("Customer Support Chat").run()
    assert not app.exception
    assert any("Set GEMINI_API_KEY" in str(item.value) for item in app.info)
