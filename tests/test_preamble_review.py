"""A preamble is shown separately, never sent for comparison, never counted as a risk."""

import csv
import io
from pathlib import Path

import embed
import report

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_sample_banner_is_not_reviewed():
    extracted = embed.read_contract_pdf((ROOT / "docs/figures/synthetic-contract-v2.pdf").read_bytes())
    seen = []

    def retriever(text, category=None, k=3):
        seen.append(text)
        return []

    rows = report.run_contract_review(extracted["text"], retriever=retriever, comparer=lambda *a: None)
    assert len(rows) == 9
    assert report.is_preamble(rows[0])
    assert rows[0]["label"] == ""
    assert not any("FICTIONAL SAMPLE" in text for text in seen)
    counts = report.count_labels(rows)
    assert sum(counts.values()) == 8
    parsed = list(csv.DictReader(io.StringIO(report.to_csv(rows))))
    assert parsed[0]["kind"] == "Preamble - not reviewed"
    assert parsed[0]["label"] == ""
    assert parsed[1]["kind"] == "Clause"
    page = report.to_html(rows, "synthetic-contract-v2.pdf")
    assert "Preamble - not reviewed" in page
    assert "Review required: 8" in page


def test_progress_counts_reviewable_clauses_only():
    text = "SAMPLE AGREEMENT\n1. The customer pays invoices.\n2. Either party may terminate on notice.\n"
    progress = []
    report.run_contract_review(
        text,
        retriever=lambda *a, **k: [],
        comparer=lambda *a: None,
        on_progress=lambda done, total, message: progress.append((done, total, message)),
    )
    assert progress[0] == (0, 2, "Comparing clause 1 of 2")
    assert progress[-1] == (2, 2, "Scoring risk labels")
