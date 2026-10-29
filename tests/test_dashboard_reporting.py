import report
import ui
import segment


def _row(clause_id, label, *, start=0, text="sample clause", kind=segment.CLAUSE, **extra):
    return {
        "clause_id": clause_id,
        "heading": f"Heading {clause_id}",
        "label": label,
        "contract_text": text,
        "start": start,
        "kind": kind,
        "explanation": f"Explanation for {clause_id}",
        "deviations": [f"Difference for {clause_id}"] if label in {"Risky", "Caution"} else [],
        "evidence_id": "PB-1" if label != "Review required" else "",
        **extra,
    }


def test_dashboard_counts_are_exact_and_match_heatmap():
    rows = [
        _row("C1", "Risky", start=30),
        _row("C2", "Caution", start=10),
        _row("C3", "Safe", start=50),
        _row("C4", "Review required", start=70, evidence_id=""),
        _row("P1", "", kind=segment.PREAMBLE),
    ]
    counts = report.count_labels(rows)
    heatmap = report.heatmap_rows(rows)
    assert counts == {"Safe": 1, "Caution": 1, "Risky": 1, "Review required": 1}
    assert [item["Clause"] for item in heatmap] == ["C1", "C2", "C4", "C3"]
    assert {label: sum(item["Risk tier"] == label for item in heatmap) for label in report.LABELS} == counts


def test_health_score_uses_documented_weights_and_excludes_unreviewed():
    rows = [
        _row("C1", "Safe"),
        _row("C2", "Caution"),
        _row("C3", "Risky"),
        _row("C4", "Review required", evidence_id=""),
    ]
    assert report.health_score(rows) == 60
    assert report.health_score([_row("C1", "Review required", evidence_id="")]) is None


def test_heatmap_has_action_and_priority_for_every_reviewed_row():
    rows = [_row("C1", "Risky"), _row("C2", "Safe"), _row("C3", "Review required", evidence_id="")]
    heatmap = report.heatmap_rows(rows)
    assert [item["Priority"] for item in heatmap] == ["High", "Manual review", "Routine"]
    assert all(item["Recommended action"] for item in heatmap)


def test_original_text_highlights_only_exact_clause_offsets_and_links_report():
    source = "Opening. Risky clause. Closing."
    row = _row("C1", "Risky", start=9, text="Risky clause.")
    markup = ui.annotated_source_html(source, [row], "C1")
    assert "Opening. " in markup and "Risky clause." in markup and "Closing." in markup
    assert 'id="source-C1"' in markup
    assert "#report-C1" in markup
    assert ">Risky clause.</mark>" in markup


def test_unmatched_source_offset_is_left_unhighlighted_and_disclosed():
    source = "Original source wording."
    row = _row("C1", "Caution", start=0, text="different clause text")
    markup = ui.annotated_source_html(source, [row])
    assert "Original source wording." in markup
    assert "<mark" not in markup
    assert "could not be highlighted safely" in markup
    assert "C1" in markup
