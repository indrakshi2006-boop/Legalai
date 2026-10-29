"""Exports carry the provider, the models, the date and the preliminary notice."""

import csv
import datetime as DT
import io

import report
import utils


def _rows():
    def retriever(text, category=None, k=3):
        return [
            {
                "evidence_id": "payment-01",
                "text": "The customer pays undisputed invoices within thirty days.",
                "category": "Payment",
                "title": "Invoice timing",
                "distance": 0.2,
                "similarity": 0.8,
            }
        ]

    def comparer(contract_text, hits):
        return {"evidence_id": "payment-01", "deviations": ["ten days rather than thirty"], "explanation": "x", "severity": "medium"}

    return report.run_contract_review(
        "1. The customer pays each invoice within ten days of receipt.\n", retriever=retriever, comparer=comparer
    )


def test_rows_record_the_default_provider_and_models(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    rows = _rows()
    assert rows[0]["provider"] == "openai"
    assert rows[0]["embedding_model"] == utils.EMBEDDING_MODEL == "text-embedding-ada-002"
    assert rows[0]["chat_model"] == "gpt-4o"


def test_csv_has_provider_model_date_and_notice_columns():
    moment = DT.datetime(2026, 10, 29, 4, 30, tzinfo=DT.timezone.utc)
    parsed = list(csv.DictReader(io.StringIO(report.to_csv(_rows(), generated_at=moment))))
    assert parsed[0]["provider"] == "openai"
    assert parsed[0]["embedding_model"] == "text-embedding-ada-002"
    assert parsed[0]["chat_model"] == "gpt-4o"
    assert parsed[0]["generated_utc"] == "2026-10-29 04:30 UTC"
    assert parsed[0]["notice"].startswith("Preliminary review - not legal advice.")


def test_html_has_provider_model_date_and_notice():
    moment = DT.datetime(2026, 10, 29, 4, 30, tzinfo=DT.timezone.utc)
    page = report.to_html(_rows(), "c.pdf", generated_at=moment)
    assert "Preliminary review - not legal advice." in page
    assert "<th>Provider</th><td>openai</td>" in page
    assert "<th>Embedding model</th><td>text-embedding-ada-002</td>" in page
    assert "<th>Chat model</th><td>gpt-4o</td>" in page
    assert "2026-10-29 04:30 UTC" in page


def test_rows_without_run_info_say_not_recorded():
    page = report.to_html([{"clause_id": "c1", "label": "Safe"}], "c.pdf")
    assert "not recorded" in page


def test_model_names_are_escaped_in_html():
    rows = _rows()
    rows[0]["chat_model"] = "<b>model</b>"
    assert "<b>model</b>" not in report.to_html(rows)
