"""Interface checks for downloads and session reuse.

The comparison function in these tests is a fixture. A passing test is not a
completed live model review.
"""

from pathlib import Path

import pymupdf as fitz
from streamlit.testing.v1 import AppTest

import analyze
import app
import embed
import report


_FIXTURE_NOTE = "Fixture comparison used by the interface test. This is not a live model review."
_CONTRACT = (
    "1. The customer pays <script>alert(1)</script> within ten days of the invoice.\n\n"
    "2. Only the supplier may terminate immediately without notice.\n"
)


def _sample_pdf():
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(fitz.Rect(72, 72, 540, 700), _CONTRACT)
    data = document.tobytes()
    document.close()
    return data


def _hit(evidence_id, category, text):
    return {
        "evidence_id": evidence_id,
        "text": text,
        "category": category,
        "title": evidence_id,
        "distance": 0.2,
        "similarity": 0.8,
    }


def test_fingerprint_includes_configuration():
    data = b"%PDF-1.4 sample"
    first = app.review_fingerprint(data, "gpt-4o|text-embedding-ada-002|60|playbook_clauses|1")
    same = app.review_fingerprint(data, "gpt-4o|text-embedding-ada-002|60|playbook_clauses|1")
    changed_model = app.review_fingerprint(data, "gpt-4o-mini|text-embedding-ada-002|60|playbook_clauses|1")
    changed_file = app.review_fingerprint(data + b"x", "gpt-4o|text-embedding-ada-002|60|playbook_clauses|1")
    assert first == same
    assert first != changed_model
    assert first != changed_file
