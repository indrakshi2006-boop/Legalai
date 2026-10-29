"""Presentation helpers for the Streamlit app.

Pure functions: no Streamlit calls, no network. Every piece of contract or
model text is HTML-escaped before it is placed in markup. Labels are never
shown by colour alone: each has an icon and its text.
"""

import html

import report

# Label -> (icon, text colour, background). Contrast of text on background
# is at least 4.5:1 for every pair (checked in tests/test_ui.py).
LABEL_STYLE = {
    "Safe": ("✓", "#1B6B43", "#E6F4EC"),
    "Caution": ("⚠", "#7A4D00", "#FFF2D1"),
    "Risky": ("✕", "#9F1D25", "#FCE7E7"),
    "Review required": ("?", "#2E4372", "#E9EEF8"),
    report.PREAMBLE_STATUS: ("¶", "#4B525E", "#EEF0F2"),
}
RISK_ORDER = {"Risky": 0, "Caution": 1, "Review required": 2, "Safe": 3}
INK = "#1C2433"
SERIF = '"Source Serif 4", "Source Serif Pro", Georgia, "Times New Roman", serif'


def label_of(row):
    return report.PREAMBLE_STATUS if report.is_preamble(row) else (row.get("label") or "Review required")


def label_text(label):
    icon = LABEL_STYLE.get(label, LABEL_STYLE["Review required"])[0]
    return f"{icon} {label}"


def badge_html(label):
    icon, ink, paper = LABEL_STYLE.get(label, LABEL_STYLE["Review required"])
    return (
        f'<span class="aegis-badge" style="color:{ink};background:{paper};border:1px solid {ink};">'
        f'<span aria-hidden="true">{html.escape(icon)}</span> {html.escape(label)}</span>'
    )


def overall_profile(rows):
    """A plain description of the counts. It is not a legal assessment."""
    counts = report.count_labels(rows)
    total = sum(counts.values())
    if total == 0:
        return "No clauses were reviewed."
    if counts["Review required"] == total:
        return f"None of the {total} clauses could be compared, so every clause needs a manual review."
    parts = []
    if counts["Risky"]:
        parts.append(f"{counts['Risky']} of {total} clauses matched a Risky wording rule")
    if counts["Caution"]:
        parts.append(f"{counts['Caution']} need caution")
    if counts["Review required"]:
        parts.append(f"{counts['Review required']} could not be compared and need a manual review")
    if not parts:
        return f"The rules found no targeted wording issue in the {total} clauses compared."
    sentence = "; ".join(parts) + "."
    return sentence[0].upper() + sentence[1:]


def filter_rows(rows, labels=None, categories=None, query=""):
    """Reviewed rows filtered by label, category and a case-insensitive text search."""
    query = (query or "").strip().casefold()
    selected = []
    for row in report.reviewed_rows(rows):
        if labels and (row.get("label") or "Review required") not in labels:
            continue
        category = row.get("category") or "Not classified"
        if categories and category not in categories:
            continue
        if query:
            haystack = " ".join(
                [
                    row.get("clause_id") or "",
                    row.get("heading") or "",
                    row.get("contract_text") or "",
                    row.get("explanation") or "",
                    " ".join(row.get("deviations") or []),
                ]
            ).casefold()
            if query not in haystack:
                continue
        selected.append(row)
    return selected


def sort_rows(rows, by="position"):
    if by == "risk":
        return sorted(rows, key=lambda row: (RISK_ORDER.get(row.get("label"), 2), _position(row)))
    return sorted(rows, key=_position)


def _position(row):
    start = row.get("start")
    if isinstance(start, int):
        return start
    return _sequence(row.get("clause_id"))


def _sequence(clause_id):
    digits = "".join(char for char in str(clause_id or "") if char.isdigit())
    return int(digits) if digits else 0


def failed_rows(rows):
    """Clauses that were not compared because a call failed or the review stopped."""
    return [
        row
        for row in report.reviewed_rows(rows)
        if row.get("error_kind") or (row.get("label") == "Review required" and not row.get("evidence_id"))
    ]


def stop_reason(rows):
    """The first fatal failure in a review, as shown to the user, or ""."""
    import errors as Errors

    for row in report.reviewed_rows(rows):
        if row.get("error_kind") in Errors.FATAL:
            for warning in row.get("warnings") or []:
                if "No comparison was made for this clause." in warning:
                    return warning.replace(" No comparison was made for this clause.", "")
            return (row.get("warnings") or [""])[0]
    return ""


def similarity_text(value):
    if value is None:
        return "not available"
    return f"{float(value):.3f}"


def annotated_contract_html(rows):
    """The whole contract as a marked-up page. Each block expands in place to show its finding.

    Streamlit cannot scroll from one tab to an element in another tab, so each
    clause carries its own summary here (an HTML <details> element) instead of
    linking to the clause card.
    """
    blocks = []
    for row in sorted(rows, key=_position):
        label = label_of(row)
        icon, ink, paper = LABEL_STYLE.get(label, LABEL_STYLE["Review required"])
        clause_id = html.escape(row.get("clause_id") or "")
        text = html.escape(row.get("contract_text") or "")
        if report.is_preamble(row):
            finding = f"<p>{html.escape(report.PREAMBLE_NOTE)}</p>"
        else:
            finding = _finding_html(row)
        blocks.append(
            f'<details class="aegis-clause" id="contract-{clause_id}" style="border-left-color:{ink};">'
            f'<summary><span class="aegis-tag" style="color:{ink};background:{paper};border-color:{ink};">'
            f'<span aria-hidden="true">{html.escape(icon)}</span> {html.escape(label)}</span>'
            f'<span class="aegis-clause-id">{clause_id}</span>'
            f'<span class="aegis-clause-text">{text}</span></summary>'
            f'<div class="aegis-finding">{finding}</div></details>'
        )
    return ANNOTATED_CSS + '<div class="aegis-page" role="document">' + "".join(blocks) + "</div>"


def _finding_html(row):
    parts = []
    category = row.get("category") or "Not classified"
    parts.append(f"<p><strong>Category:</strong> {html.escape(category)}</p>")
    if row.get("evidence_id"):
        parts.append(
            "<p><strong>Closest playbook clause:</strong> "
            f"{html.escape(row.get('evidence_id') or '')} {html.escape(row.get('reference_title') or '')} "
            f"(similarity {html.escape(similarity_text(row.get('similarity')))})</p>"
        )
    else:
        parts.append("<p><strong>No playbook clause matched,</strong> so this clause was not compared.</p>")
    if row.get("explanation"):
        parts.append(f"<p>{html.escape(row['explanation'])}</p>")
    deviations = row.get("deviations") or []
    if deviations:
        items = "".join(f"<li>{html.escape(item)}</li>" for item in deviations)
        parts.append(f"<p><strong>Differences found:</strong></p><ul>{items}</ul>")
    return "".join(parts)


ANNOTATED_CSS = f"""<style>
.aegis-page {{ background:#FFFFFF; border:1px solid #D5DBE3; border-radius:6px; padding:1.5rem 1.75rem;
  font-family:{SERIF}; color:{INK}; max-width:52rem; }}
.aegis-clause {{ border-left:4px solid; padding:0.35rem 0 0.35rem 0.9rem; margin:0 0 0.9rem 0; }}
.aegis-clause > summary {{ cursor:pointer; list-style:none; }}
.aegis-clause > summary::-webkit-details-marker {{ display:none; }}
.aegis-clause > summary:focus-visible {{ outline:2px solid #1D5C7A; outline-offset:3px; }}
.aegis-tag {{ display:inline-block; font-family:"Source Sans 3","Source Sans Pro",system-ui,sans-serif;
  font-size:0.8rem; font-weight:600; border:1px solid; border-radius:3px; padding:0.05rem 0.45rem; margin-right:0.5rem; }}
.aegis-clause-id {{ font-family:"Source Sans 3","Source Sans Pro",system-ui,sans-serif; font-size:0.8rem; color:#4B525E; }}
.aegis-clause-text {{ display:block; white-space:pre-wrap; line-height:1.6; margin-top:0.3rem; font-size:1rem; }}
.aegis-finding {{ font-family:"Source Sans 3","Source Sans Pro",system-ui,sans-serif; font-size:0.92rem;
  background:#F5F7F9; border-radius:4px; padding:0.6rem 0.8rem; margin-top:0.5rem; }}
.aegis-finding p {{ margin:0 0 0.4rem 0; }}
</style>"""

BADGE_CSS = """<style>
.aegis-badge { display:inline-block; font-weight:600; font-size:0.85rem; border-radius:3px; padding:0.05rem 0.5rem; }
.aegis-quote { font-family:%s; white-space:pre-wrap; line-height:1.6; background:#FFFFFF;
  border:1px solid #D5DBE3; border-radius:4px; padding:0.7rem 0.9rem; color:%s; }
.aegis-plain { white-space:pre-wrap; margin:0 0 0.5rem 0; }
.aegis-list { margin:0 0 0.5rem 1.1rem; }
</style>""" % (SERIF, INK)


def quote_html(text):
    return f'<div class="aegis-quote">{html.escape(text or "")}</div>'


def plain_html(text):
    """Untrusted text as an escaped HTML block, so neither HTML nor Markdown in it is interpreted."""
    return f'<div class="aegis-plain">{html.escape(text or "")}</div>'


def list_html(items):
    return "<ul class=\"aegis-list\">" + "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ul>"


_MD_SPECIAL = "\\`*_{}[]()#+-.!<>|~$:"


def md_escape(text):
    """Backslash-escape Markdown so a heading from the contract renders as plain text."""
    return "".join("\\" + char if char in _MD_SPECIAL else char for char in str(text or ""))


PDF_HELP = (
    ("too large", "Split the PDF into smaller parts, or raise MAX_UPLOAD_MB in .env."),
    ("pages. The limit", "Upload a shorter extract, or raise MAX_PDF_PAGES in .env."),
    ("image-only", "Use a PDF with selectable text, for example one exported from Word. Scanned pages need OCR first."),
    ("PDF header", "Choose a .pdf file."),
    ("encrypted", "Remove the password in your PDF reader, save a copy, then upload that copy."),
    ("corrupt", "Open the file in a PDF reader to check it, then export it again."),
    ("empty", "Choose the file again; it arrived with no content."),
)


def pdf_error_help(message):
    for marker, help_text in PDF_HELP:
        if marker in message:
            return help_text
    return "Check the file and upload it again."


def contrast_ratio(foreground, background):
    def channel(value):
        value = value / 255
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    def luminance(colour):
        colour = colour.lstrip("#")
        red, green, blue = (int(colour[index : index + 2], 16) for index in (0, 2, 4))
        return 0.2126 * channel(red) + 0.7152 * channel(green) + 0.0722 * channel(blue)

    lighter, darker = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)
