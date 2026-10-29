"""Presentation helpers: contrast, icons with text, escaping, filters, sort, profile."""

import ui
import report


def _row(clause_id, label, category="Payment", text="The customer pays.", start=0, **extra):
    row = {"clause_id": clause_id, "label": label, "category": category, "contract_text": text,
           "heading": text[:20], "start": start, "kind": "clause", "deviations": [], "explanation": ""}
    row.update(extra)
    return row


def test_every_label_pair_meets_wcag_aa_and_has_an_icon():
    for label, (icon, ink, paper) in ui.LABEL_STYLE.items():
        assert icon.strip()
        assert ui.contrast_ratio(ink, paper) >= 4.5, label
        assert label in ui.badge_html(label) and icon in ui.badge_html(label)
        assert ui.label_text(label) == f"{icon} {label}"


def test_overall_profile_describes_counts_only():
    rows = [_row("c1", "Risky"), _row("c2", "Caution"), _row("c3", "Safe"), _row("c4", "Review required")]
    profile = ui.overall_profile(rows)
    assert profile.startswith("1 of 4 clauses matched a Risky wording rule")
    assert "1 need caution" in profile and "1 could not be compared" in profile
    assert ui.overall_profile([_row("c1", "Safe")]).startswith("The rules found no targeted wording issue")
    assert "manual review" in ui.overall_profile([_row("c1", "Review required")])
    assert ui.overall_profile([]) == "No clauses were reviewed."


def test_filters_and_sort():
    rows = [
        _row("clause-001", "", kind="preamble", start=0),
        _row("clause-002", "Safe", "Payment", "Invoices are due in thirty days.", 10),
        _row("clause-003", "Risky", "Termination", "Only the supplier may terminate.", 50),
        _row("clause-004", "Caution", "Payment", "A late fee applies.", 90),
    ]
    assert [r["clause_id"] for r in ui.filter_rows(rows)] == ["clause-002", "clause-003", "clause-004"]
    assert [r["clause_id"] for r in ui.filter_rows(rows, labels=["Risky"])] == ["clause-003"]
    assert [r["clause_id"] for r in ui.filter_rows(rows, categories=["Payment"])] == ["clause-002", "clause-004"]
    assert [r["clause_id"] for r in ui.filter_rows(rows, query="LATE fee")] == ["clause-004"]
    ordered = ui.sort_rows(ui.filter_rows(rows), "risk")
    assert [r["label"] for r in ordered] == ["Risky", "Caution", "Safe"]


def test_annotated_html_escapes_text_and_covers_preamble():
    rows = [
        _row("clause-001", "", text="TITLE <b>x</b>", kind="preamble", start=0),
        _row("clause-002", "Risky", text="<script>alert(1)</script>", start=20, explanation="<img src=x>",
             deviations=["<i>d</i>"], evidence_id="termination-01", similarity=0.81234, reference_title="Ref"),
    ]
    markup = ui.annotated_contract_html(rows)
    assert "<script>alert" not in markup and "<img src=x>" not in markup and "<b>x</b>" not in markup
    assert "&lt;script&gt;" in markup
    assert "Preamble - not reviewed" in markup and "similarity 0.812" in markup


def test_markdown_and_html_escaping_helpers():
    assert ui.md_escape("[link](javascript:x) *b*") == "\\[link\\]\\(javascript\\:x\\) \\*b\\*"
    assert ui.plain_html("<b>x</b> **y**") == '<div class="aegis-plain">&lt;b&gt;x&lt;/b&gt; **y**</div>'
    assert "<script>" not in ui.list_html(["<script>"])


def test_pdf_error_help_gives_a_next_step():
    assert "MAX_UPLOAD_MB" in ui.pdf_error_help("This file is too large (99 bytes).")
    assert "MAX_PDF_PAGES" in ui.pdf_error_help("This PDF has 3 pages. The limit is 2 pages.")
    assert "selectable text" in ui.pdf_error_help("It appears to be image-only")


def test_stop_reason_and_failed_rows():
    rows = [
        _row("clause-001", "Review required", error_kind="no_credit",
             warnings=["OpenAI reports no available credit. Retrying will not help. No comparison was made for this clause."]),
        _row("clause-002", "Review required", error_kind="stopped", warnings=["The review stopped after clause-001"]),
        _row("clause-003", "Safe", evidence_id="payment-01"),
    ]
    assert ui.stop_reason(rows) == "OpenAI reports no available credit. Retrying will not help."
    assert [r["clause_id"] for r in ui.failed_rows(rows)] == ["clause-001", "clause-002"]
