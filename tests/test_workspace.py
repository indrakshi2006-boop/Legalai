import pymupdf as fitz

import embed
import report


def _pdf_with_text(pages=1):
    document = fitz.open()
    for index in range(pages):
        page = document.new_page()
        page.insert_text(
            (72, 72),
            f"{index + 1}. The supplier delivers the goods named in this order.\n",
        )
    data = document.tobytes()
    document.close()
    return data


def _image_only_pdf():
    document = fitz.open()
    page = document.new_page()
    page.draw_rect(fitz.Rect(72, 72, 200, 200), color=(0, 0, 0), fill=(1, 0, 0))
    data = document.tobytes()
    document.close()
    return data


def test_pdf_validation():
    assert embed.read_contract_pdf(b"")["error"] == "The uploaded file is empty."
    assert "PDF header" in embed.read_contract_pdf(b"hello")["error"]
    corrupt = embed.read_contract_pdf(b"%PDF-not-really")
    assert corrupt["ok"] is False
    too_big = embed.read_contract_pdf(_pdf_with_text(), max_bytes=20)
    assert too_big["ok"] is False
    assert "too large" in too_big["error"]
    too_long = embed.read_contract_pdf(_pdf_with_text(pages=3), max_pages=2)
    assert too_long["ok"] is False
    assert "3 pages" in too_long["error"]
    image_only = embed.read_contract_pdf(_image_only_pdf())
    assert image_only["ok"] is False
    assert "OCR is not available" in image_only["error"]
    accepted = embed.read_contract_pdf(_pdf_with_text())
    assert accepted["ok"] is True
    assert accepted["page_count"] == 1
    assert "supplier delivers" in accepted["text"]


def test_counts_and_downloads_use_actual_rows():
    rows = [
        {
            "clause_id": "clause-001",
            "heading": "1. Delivery",
            "contract_text": "UNIQUE-CONTRACT-TEXT <script>alert(1)</script>",
            "category": "Payment",
            "label": "Safe",
            "explanation": "No targeted issue.",
            "deviations": [],
            "reference_clause": "REFERENCE-PASSAGE",
            "evidence_id": "payment-01",
            "warnings": ["Semantic similarity ranks retrieved passages only."],
            "deciding_rule": "severity-low",
            "rule_reasons": ["Low severity."],
            "severity": "low",
            "disclaimer": "Not a legal conclusion.",
        },
        {
            "clause_id": "clause-002",
            "heading": "2. Liability",
            "contract_text": "Unlimited liability applies.",
            "category": "Liability",
            "label": "Risky",
            "explanation": "The cap is missing.",
            "deviations": ["Liability is unlimited."],
            "reference_clause": "Liability is capped.",
            "evidence_id": "liability-01",
            "warnings": [],
            "deciding_rule": "liability-unlimited",
            "rule_reasons": ["Unlimited liability was identified."],
            "severity": "high",
            "disclaimer": "Not a legal conclusion.",
        },
    ]
    counts = report.count_labels(rows)
    assert counts == {"Safe": 1, "Caution": 0, "Risky": 1, "Review required": 0}
    csv_text = report.to_csv(rows)
    html_text = report.to_html(rows, "agreement.pdf")
    assert "UNIQUE-CONTRACT-TEXT" in csv_text
    assert "payment-01" in csv_text
    assert "REFERENCE-PASSAGE" in html_text
    assert "<script>" not in html_text
    assert "&lt;script&gt;" in html_text
    assert "Safe: 1" in html_text
    assert "Risky: 1" in html_text
    assert "Caution: 0" in html_text


def test_review_uses_the_pipeline_results_only():
    clauses_seen = {}

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
        clauses_seen["text"] = contract_text
        return {
            "evidence_id": "payment-01",
            "deviations": ["Payment is due in ten days rather than thirty."],
            "explanation": "The period is shorter.",
            "severity": "medium",
        }

    progress = []
    text = (
        "1. The customer pays each invoice within ten days of receipt.\n"
        "2. The supplier delivers the goods to the location named in the order.\n"
    )
    rows = report.run_contract_review(
        text,
        retriever=retriever,
        comparer=comparer,
        on_progress=lambda done, total, message: progress.append((done, total, message)),
    )
    counts = report.count_labels(rows)
    assert sum(counts.values()) == len(rows)
    assert rows[0]["evidence_id"] == "payment-01"
    assert "ten days" in rows[0]["contract_text"]
    assert rows[0]["label"] in {"Safe", "Caution", "Risky"}
    assert progress[-1][2] == "Scoring risk labels"
    assert "sample" not in report.to_csv(rows).lower()
