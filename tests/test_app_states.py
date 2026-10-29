"""Streamlit AppTest checks for each state of the redesigned page.

Every model call here is a local fixture. A passing test is not a live
model review, and no screenshot is taken from these fixtures.
"""

import json
from pathlib import Path

import httpx
import openai
import pymupdf as fitz
import pytest
from streamlit.testing.v1 import AppTest

import agent
import analyze
import embed
import providers
import report

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
SAMPLE = ROOT / "docs" / "figures" / "synthetic-contract-v2.pdf"
UPLOADED_SAMPLE = ROOT / "docs" / "figures" / "AEGIS_synthetic_contract_test.pdf"
FIXTURE_NOTE = "Fixture comparison used by the interface test. This is not a live model review."
XSS_CONTRACT = (
    "SAMPLE AGREEMENT\n"
    "1. The customer pays <script>alert(1)</script> within ten days of the invoice.\n\n"
    "2. Only the supplier may terminate immediately without notice.\n"
)


def _pdf(text, pages=1):
    document = fitz.open()
    for _ in range(pages):
        page = document.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 540, 760), text)
    data = document.tobytes()
    document.close()
    return data


def _image_pdf():
    document = fitz.open()
    page = document.new_page()
    page.draw_rect(fitz.Rect(72, 72, 200, 200), color=(0, 0, 0), fill=(1, 0, 0))
    data = document.tobytes()
    document.close()
    return data


def _hit(evidence_id, category, text):
    return {"evidence_id": evidence_id, "text": text, "category": category, "title": f"{category} reference",
            "distance": 0.2, "similarity": 0.8}


class Fixture:
    def __init__(self):
        self.compare = 0
        self.retrieve = 0
        self.fail = None

    def retriever(self, text, category=None, k=3):
        self.retrieve += 1
        if "terminat" in text.lower() or category == "Termination":
            return [_hit("termination-01", "Termination", "Either party may terminate on thirty days notice.")]
        return [_hit("payment-01", "Payment", "The customer pays undisputed invoices within thirty days.")]

    def comparer(self, contract_text, hits):
        self.compare += 1
        if self.fail is not None:
            raise self.fail
        return {"evidence_id": hits[0]["evidence_id"], "deviations": ["The fixture recorded a wording difference."],
                "explanation": FIXTURE_NOTE, "severity": "medium"}


@pytest.fixture
def ready(monkeypatch):
    fixture = Fixture()
    monkeypatch.setattr(providers, "index_status", lambda config=None, persist_directory=None, digest=None: {
        "state": "Ready", "detail": "30/30", "count": 30, "expected": 30})
    monkeypatch.setattr(report, "live_pipeline", lambda config=None, policy=None: {
        "retriever": fixture.retriever, "comparer": fixture.comparer, "tuning": analyze.DEFAULT_TUNING,
        "run_info": {"provider": "openai", "embedding_model": "text-embedding-ada-002", "chat_model": "gpt-4o"}})
    return fixture


@pytest.fixture
def missing_index(monkeypatch, tmp_path):
    monkeypatch.setattr(embed.Utils, "DB_FOLDER", str(tmp_path / "no-index"))


def _page():
    page = AppTest.from_file(APP, default_timeout=60)
    page.run()
    assert not page.exception
    return page


def _upload(page, data, name="sample-contract.pdf"):
    page.file_uploader[0].upload(name, data, mime_type="application/pdf").run()
    assert not page.exception
    return page


def _texts(elements):
    return " ".join(str(item.value) for item in elements)


def _all_text(page):
    parts = []
    for kind in ("markdown", "caption", "info", "success", "warning", "error", "text", "code"):
        parts.append(_texts(getattr(page, kind)))
    parts.append(_texts(page.sidebar.markdown))
    return " ".join(parts)


def _button(page, label):
    matches = [button for button in page.button if button.label == label]
    assert matches, label
    return matches[0]


def _checkbox(page, prefix):
    matches = [box for box in page.checkbox if box.label.startswith(prefix)]
    assert matches, prefix
    return matches[0]


def _review(page):
    _checkbox(page, "I agree to send").check().run()
    _button(page, "Review contract").click().run()
    assert not page.exception
    return page


# --- start states --------------------------------------------------------------------


def test_empty_state_header_sidebar_and_privacy(missing_index):
    page = _page()
    assert page.title[0].value == "AEGIS"
    assert "Preliminary review - not legal advice." in _texts(page.warning)
    assert "Choose a contract PDF to start" in _texts(page.info)
    sidebar = _texts(page.sidebar.markdown)
    assert "**Provider:** OpenAI" in sidebar
    assert "text-embedding-ada-002" in sidebar and "gpt-4o" in sidebar
    assert "Text is sent to the configured provider only when you start a review or ask a question" in sidebar
    assert "Playbook index: Missing" in _texts(page.sidebar.error)
    assert "python embed.py playbook" in _texts(page.sidebar.code)
    assert "15 MB and 40 pages" in sidebar


def test_unknown_provider(monkeypatch):
    monkeypatch.setenv("AEGIS_PROVIDER", "mystery")
    page = _page()
    assert "Choose a provider first" in _texts(page.error)
    assert not page.file_uploader


def test_index_ready_shows_in_sidebar(ready):
    page = _page()
    assert "Playbook index: Ready 30/30" in _texts(page.sidebar.success)


# --- upload validation -------------------------------------------------------------------


def test_image_only_pdf_is_rejected_with_next_step(missing_index):
    page = _upload(_page(), _image_pdf())
    errors = _texts(page.error)
    assert "OCR is not available" in errors
    assert "selectable text" in errors


def test_too_large_pdf(monkeypatch, missing_index):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.0005")
    page = _upload(_page(), _pdf(XSS_CONTRACT))
    assert "too large" in _texts(page.error) and "MAX_UPLOAD_MB" in _texts(page.error)


def test_too_many_pages(monkeypatch, missing_index):
    monkeypatch.setenv("MAX_PDF_PAGES", "1")
    page = _upload(_page(), _pdf(XSS_CONTRACT, pages=2))
    assert "2 pages" in _texts(page.error) and "MAX_PDF_PAGES" in _texts(page.error)


def test_not_a_pdf(missing_index):
    page = _upload(_page(), b"hello, not a pdf")
    assert "PDF header" in _texts(page.error)


# --- before the review ----------------------------------------------------------------------


def test_accepted_pdf_shows_clause_count_call_limits_and_consent(ready):
    page = _upload(_page(), SAMPLE.read_bytes(), "synthetic-contract-v2.pdf")
    assert "8 clauses to review and 1 preamble block" in _texts(page.success)
    text = _all_text(page)
    assert "up to 16 text-matching calls and 8 analysis calls to OpenAI" in text
    assert _button(page, "Review contract").disabled is True, "disabled until consent"
    _checkbox(page, "I agree to send this contract's text to OpenAI").check().run()
    assert _button(page, "Review contract").disabled is False
    assert ready.compare == 0 and ready.retrieve == 0


def test_missing_key_blocks_the_review_but_not_the_upload(ready, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    page = _upload(_page(), _pdf(XSS_CONTRACT))
    assert "OPENAI_API_KEY is not set" in _texts(page.error)
    assert _checkbox(page, "I agree").disabled is True
    assert _button(page, "Review contract").disabled is True
    assert "Set OPENAI_API_KEY in .env to ask questions." in _texts(page.info)


def test_missing_index_blocks_the_review_with_the_command(missing_index):
    page = _upload(_page(), _pdf(XSS_CONTRACT))
    assert "playbook index is not ready (Missing" in _texts(page.error)
    assert "python embed.py playbook" in _texts(page.code)
    assert _button(page, "Review contract").disabled is True


def test_gemini_banner_and_cutoff_gate(monkeypatch, missing_index):
    monkeypatch.setenv("AEGIS_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-unit-test-not-real")
    page = _upload(_page(), _pdf(XSS_CONTRACT))
    assert providers.GEMINI_BANNER in _texts(page.error)
    assert "**Provider:** Gemini" in _texts(page.sidebar.markdown)
    assert "gemini-embedding-001" in _texts(page.sidebar.markdown)
    assert "python embed.py playbook" in _texts(page.code)
    assert "with AEGIS_PROVIDER=gemini in .env" in _texts(page.error)
    assert "--inspect-scores" in _texts(page.error)
    assert _button(page, "Review contract").disabled is True
    assert "AIzaSy" not in _all_text(page)


# --- the review and results ----------------------------------------------------------------------


def test_completed_review_dashboard_cards_and_exports(ready):
    page = _review(_upload(_page(), _pdf(XSS_CONTRACT)))
    first = ready.compare
    assert first == 2
    saved = page.session_state["review"]
    assert saved["status"] == "complete"
    assert report.is_preamble(saved["rows"][0])
    assert [m.label for m in page.metric] == ["Prototype health score", "✓ Safe", "⚠ Caution", "✕ Risky", "? Review required"]
    assert [m.value for m in page.metric] == ["40/100", "0", "1", "1", "0"]
    assert "matched a Risky wording rule" in _texts(page.markdown)
    titles = [expander.label.replace("\\", "") for expander in page.expander]
    assert any(title.startswith("✕ Risky: 2. Only the supplier") for title in titles)
    assert any(title.startswith("¶ Preamble - not reviewed") for title in titles)
    labels = [button.label for button in page.get("download_button")]
    assert labels == ["Download CSV", "Download HTML"]
    csv_text = report.to_csv(saved["rows"])
    assert "openai" in csv_text and "gpt-4o" in csv_text
    html_text = report.to_html(saved["rows"], "sample-contract.pdf")
    assert "<script>" not in html_text and "&lt;script&gt;" in html_text

    page.run()
    assert ready.compare == first, "a rerun never repeats calls"
    assert "already holds the review" in _texts(page.info)
    assert not [button for button in page.button if button.label == "Review contract"]


def test_synthetic_contract_drives_summary_heatmap_and_annotated_view(ready):
    page = _review(_upload(_page(), SAMPLE.read_bytes(), "synthetic-contract-v2.pdf"))
    saved = page.session_state["review"]
    assert saved["status"] == "complete"
    assert ready.compare == 8
    assert "Prototype health score" in [metric.label for metric in page.metric]
    assert "Risk heatmap" in [tab.label for tab in page.get("tab")]
    assert page.get("dataframe"), "risk heatmap table renders"
    assert any("Original extracted contract text" in str(element.proto.body) for element in page.get("html"))
    assert report.health_score(saved["rows"]) is not None
    assert len(report.heatmap_rows(saved["rows"])) == len(report.reviewed_rows(saved["rows"]))


@pytest.mark.skipif(not UPLOADED_SAMPLE.exists(), reason="Optional user-uploaded synthetic PDF is not in this checkout")
def test_uploaded_20_clause_pdf_shows_results_only_after_review_and_reuses_rows(ready):
    page = _upload(_page(), UPLOADED_SAMPLE.read_bytes(), UPLOADED_SAMPLE.name)
    assert "20 clauses to review" in _texts(page.success)
    assert "Complete the contract review above" in _texts(page.info)
    assert ready.compare == 0

    page = _review(page)
    saved = page.session_state["review"]
    assert saved["status"] == "complete"
    assert len(report.reviewed_rows(saved["rows"])) == 20
    assert ready.compare == 20
    assert {tab.label for tab in page.get("tab")} >= {
        "Summary",
        "Clause report",
        "Annotated text",
        "Risk heatmap",
    }
    assert page.get("dataframe"), "the heatmap is connected and rendered"

    page.run()
    assert ready.compare == 20, "rendering and rerunning the results must not repeat provider calls"


def test_annotated_view_is_escaped_and_marks_every_block(ready):
    page = _review(_upload(_page(), _pdf(XSS_CONTRACT)))
    markup = " ".join(str(element.proto.body) for element in page.get("html"))
    assert "<script>alert(1)" not in markup
    assert "&lt;script&gt;" in markup
    assert 'class="aegis-source-viewer"' in markup
    assert 'id="source-clause-002"' in markup and 'id="source-clause-003"' in markup
    assert "Original extracted contract text" in markup and "Risky" in markup
    assert "#report-clause-003" in markup


def test_filters_search_and_sort(ready):
    page = _review(_upload(_page(), _pdf(XSS_CONTRACT)))
    page.multiselect(key="filter_labels").set_value(["Risky"]).run()
    assert "Showing 1 of 2 reviewed clauses." in _texts(page.caption)
    page.multiselect(key="filter_labels").set_value([]).run()
    page.text_input(key="search_rows").set_value("ten days").run()
    assert "Showing 1 of 2 reviewed clauses." in _texts(page.caption)
    page.text_input(key="search_rows").set_value("").run()
    page.selectbox(key="sort_rows").set_value("Risk (highest first)").run()
    cards = [e.label.replace("\\", "") for e in page.expander if e.label.startswith(("✕", "⚠"))]
    assert cards[0].startswith("✕ Risky")


def test_no_credit_stops_early_and_keeps_partial_results(ready):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    client = openai.OpenAI(api_key="unit-test-key-not-real", max_retries=0)
    ready.fail = client._make_status_error(
        "m", body={"error": {"code": "credit_balance_exhausted", "type": "insufficient_quota"}},
        response=httpx.Response(429, request=request))
    page = _review(_upload(_page(), _pdf(XSS_CONTRACT)))
    assert ready.compare == 1
    errors_text = _texts(page.error)
    assert "The review stopped early." in errors_text
    assert "Retrying will not help." in errors_text
    assert "rate limit" not in errors_text.lower()
    assert any(e.label.startswith("2 clause(s) could not be compared") for e in page.expander)
    assert [m.value for m in page.metric] == ["Not available", "0", "0", "0", "2"]


def test_network_failure_before_the_review(ready, monkeypatch):
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")

    def broken(config=None, policy=None):
        raise openai.APIConnectionError(request=request)

    monkeypatch.setattr(report, "live_pipeline", broken)
    page = _review(_upload(_page(), _pdf(XSS_CONTRACT)))
    assert "The review could not run." in _texts(page.error)
    assert "Could not reach OpenAI" in _texts(page.error)


# --- contract Q&A ------------------------------------------------------------------------------


def test_contract_qa_supported_and_unsupported(ready, monkeypatch):
    calls = []

    def fake_answer(question, sources, policy=None):
        calls.append(question)
        return "Fixture answer: ten days. This is not a live model answer."

    monkeypatch.setattr(agent, "_answer_with_model", fake_answer)
    page = _upload(_page(), _pdf(XSS_CONTRACT))
    assert "Matching is **lexical**" in _texts(page.markdown)
    box = [item for item in page.text_input if item.label == "Your question"][0]
    box.set_value("How many days to pay the invoice?")
    _button(page, "Ask").click().run()
    assert calls == ["How many days to pay the invoice?"]
    assert "Fixture answer: ten days" in _texts(page.markdown)
    assert "<script>alert(1)</script>" in _texts(page.text), "source shown as plain text"

    box = [item for item in page.text_input if item.label == "Your question"][0]
    box.set_value("Zebra xylophone quantum?")
    _button(page, "Ask").click().run()
    assert len(calls) == 1, "no passage matched, so no model call"
    assert agent.UNSUPPORTED_ANSWER in _texts(page.markdown)
    assert "no model was called" in _texts(page.caption)


def test_contract_qa_refused_under_gemini_gate(monkeypatch, missing_index):
    monkeypatch.setenv("AEGIS_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-unit-test-not-real")
    page = _upload(_page(), _pdf(XSS_CONTRACT))
    assert any("Gemini chat calls are refused" in w for w in [str(x.value) for x in page.warning])
    assert not [b for b in page.button if b.label == "Ask"]


# --- EU AI Act chat -------------------------------------------------------------------------------


def test_regulation_chat_without_index_shows_how_to_build_it(missing_index):
    page = _page()
    assert any(e.label.startswith("EU AI Act questions (optional") for e in page.expander)
    assert "python embed.py regulation" in _texts(page.code)
    assert "can incur a charge" in _texts(page.caption)


def test_regulation_chat_is_openai_only(monkeypatch, missing_index):
    monkeypatch.setenv("AEGIS_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-unit-test-not-real")
    page = _page()
    assert "uses OpenAI only" in _texts(page.info)
