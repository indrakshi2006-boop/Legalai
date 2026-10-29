import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import legal_case
import providers

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
CASES = (
    ("AEGIS_cases_consumer.pdf", "Consumer dispute", "purchase", "The narrative states a refrigerator purchase on 5 March 2026"),
    ("AEGIS_property.pdf", "Property or possession dispute", "claimed_ownership", "Arun says he purchased Plot 27 on 14 June 2018"),
)


def _fixture_report(payload):
    request = json.loads(payload)
    case_type = request["document_type"]
    source = request["source_pages"][0]["text"]
    if case_type == "Consumer dispute":
        quote = "The refrigerator stopped cooling on 18 March 2026"
        detail_text = {
            "purchase": "The narrative states a refrigerator purchase on 5 March 2026; an invoice is listed.",
            "defect": "The narrative reports that the refrigerator stopped cooling on 18 March 2026.",
            "communications": "The narrative describes an email complaint and ticket, followed by a retailer reply.",
            "requested_remedies": "The consumer asks for repair or replacement and reimbursement of stated losses.",
            "available_records": "The narrative lists an invoice, warranty card, emails, service slip, and handwritten estimate.",
            "missing_evidence": "The narrative says receipts or photographs for the food-loss estimate are not attached.",
        }
    else:
        quote = "Arun says he purchased Plot 27 on 14 June 2018"
        detail_text = {
            "claimed_ownership": "The claimant says he purchased Plot 27 under a registered sale deed.",
            "documentary_title_evidence": "A photocopy described as a sale deed is listed; original and updated search are unreviewed.",
            "prior_possession": "The claimant says he last inspected the vacant plot on 2 January 2026.",
            "alleged_occupation": "The claimant says he found a temporary fence and building materials on 15 February 2026.",
            "boundary_evidence": "Certified records and a survey have not yet been confirmed in the narrative.",
            "claimed_permission_or_legal_right_to_occupy": "The respondent reportedly claimed permission from a previous owner; no permission document is listed.",
        }
    assert quote in source
    citation = {"page": 1, "quote": quote}
    cite = lambda field, **extra: {field: extra.pop(field), **extra, "citations": [citation]}
    return {
        "case_summary": {"statement": f"Fictional {case_type.lower()} narrative; claims are not verified.", "citations": [citation]},
        "parties": [cite("name", name="Fictional claimant", role="Claimant", stated_position="Position is attributed in the narrative.")],
        "chronology": [cite("event", date="Date stated in narrative", event="One event in the source narrative.")],
        "documented_facts": [cite("fact", fact="The source describes records currently available.")],
        "allegations": [cite("allegation", allegation="A party's account, not independently verified.", alleged_by="Claimant")],
        "evidence_available": [cite("item", item="A record identified by the narrative.", status="Listed in source", limitation="Contents have not been independently examined.")],
        "missing_information": [{"item": "Underlying records", "why_it_matters": "They may confirm details.", "how_to_verify": "Inspect the original records."}],
        "issues_for_legal_assessment": [cite("issue", issue="Applicable process is unknown.", reason="The supplied pages do not establish jurisdiction.", jurisdiction_note="Check local law.")],
        "questions": [{"question": "What original records are available?", "ask_of": "The claimant"}],
        "next_steps": [{"step": "Collect and review the listed records.", "limitation": "Check local deadlines with a lawyer."}],
        "domain_details": {
            field: {"statement": detail_text[field], "citations": [citation]}
            for field in request["required_domain_fields"]
        },
    }


@pytest.mark.parametrize("filename,case_type,field,expected", CASES)
def test_synthetic_case_is_routed_to_gemini_case_report_without_playbook(monkeypatch, filename, case_type, field, expected):
    path = ROOT / "docs" / "figures" / filename
    if not path.exists():
        pytest.skip(f"Optional synthetic case PDF is not in this checkout: {filename}")
    monkeypatch.setenv("AEGIS_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-key-not-real")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = []

    def fake_gemini(system, user, want_json=True, client=None, config=None, require_inspection=True):
        calls.append((system, user, want_json, require_inspection))
        return json.dumps(_fixture_report(user))

    monkeypatch.setattr(providers, "gemini_chat", fake_gemini)
    page = AppTest.from_file(str(APP), default_timeout=60).run()
    page.radio[0].set_value("Legal Case Analysis").run()
    page.selectbox(key="legal_case_document_type").set_value(case_type).run()
    page.file_uploader(key="legal_case_pdf").upload(filename, path.read_bytes(), mime_type="application/pdf").run()
    assert not page.exception
    assert calls == []

    consent = [item for item in page.checkbox if item.label.startswith("I consent")][0]
    consent.check().run()
    button = [item for item in page.button if item.label == "Analyze legal case"][0]
    button.click().run()
    assert not page.exception
    assert len(calls) == 1
    assert calls[0][2:] == (True, False), "JSON case analysis bypasses only the retrieval-specific similarity gate"
    saved = page.session_state["legal_case_analysis"]["report"]
    assert field in saved["domain_details"]
    assert saved["domain_details"][field]["citations"][0]["page"] == 1
    visible = " ".join(
        str(item.value)
        for kind in ("markdown", "caption", "info", "html", "dataframe")
        for item in page.get(kind)
    )
    assert expected in visible
    assert not page.metric or all("100" not in str(item.value) for item in page.metric)

    page.run()
    assert len(calls) == 1, "rendering a stored report never repeats the Gemini request"


def test_unverifiable_citations_are_removed_and_unknowns_are_preserved():
    raw = {
        "case_summary": {"statement": "A short summary.", "citations": [{"page": 8, "quote": "invented quote"}]},
        "domain_details": {"purchase": {"statement": "Unknown from the supplied document.", "citations": []}},
    }
    pages = ["Only source wording on page one."]
    result = legal_case.validate_report(raw, pages, "Consumer dispute")
    assert result["case_summary"]["citations"] == []
    assert result["domain_details"]["purchase"]["statement"] == legal_case.UNKNOWN
    assert legal_case.completeness(result) == {"cited": 0, "total": 6, "status": "Partial"}


def test_contract_document_type_routes_to_existing_contract_review(monkeypatch):
    monkeypatch.setenv("AEGIS_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-key-not-real")
    page = AppTest.from_file(str(APP), default_timeout=60).run()
    page.radio[0].set_value("Legal Case Analysis").run()
    page.selectbox(key="legal_case_document_type").set_value("Contract").run()
    assert not page.exception
    assert any("playbook comparison" in str(item.value).lower() for item in page.info)
    assert not page.file_uploader
