"""Build the revised AEGIS project report from the implemented prototype."""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
OUT = ROOT / "AEGIS AI-Powered Legal Contract Review Assistant — Final Project Report.docx"

NAVY = (30, 58, 138)
INK = (31, 41, 55)
MUTED = (75, 85, 99)
LINE = (156, 163, 175)
FILL = (248, 250, 252)
BLUE = (239, 246, 255)
AMBER = (255, 251, 235)
RED = (254, 242, 242)
DASH = (107, 114, 128)


FONT_CANDIDATES = {
    False: [r"C:\Windows\Fonts\calibri.ttf", "/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf",
            "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    True: [r"C:\Windows\Fonts\calibrib.ttf", "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf",
           "/Library/Fonts/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
}


def font(size, bold=False):
    """Calibri on Windows; Carlito (metric-compatible), Arial or DejaVu elsewhere."""
    for candidate in FONT_CANDIDATES[bold]:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size)


def wrap(draw, text, width, face):
    words = text.split()
    lines, current = [], ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=face) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def rounded(draw, box, fill, outline, width=2, dashed=False):
    if not dashed:
        draw.rounded_rectangle(box, radius=12, fill=fill, outline=outline, width=width)
        return
    draw.rounded_rectangle(box, radius=12, fill=fill, outline=None)
    x1, y1, x2, y2 = box
    step = 10
    for x in range(x1, x2, step * 2):
        draw.line((x, y1, min(x + step, x2), y1), fill=outline, width=2)
        draw.line((x, y2, min(x + step, x2), y2), fill=outline, width=2)
    for y in range(y1, y2, step * 2):
        draw.line((x1, y, x1, min(y + step, y2)), fill=outline, width=2)
        draw.line((x2, y, x2, min(y + step, y2)), fill=outline, width=2)


def card(draw, box, title, body, fill=FILL, outline=LINE, title_fill=NAVY):
    rounded(draw, box, fill, outline)
    x1, y1, x2, y2 = box
    face_title = font(22, True)
    face_body = font(18)
    draw.text((x1 + 16, y1 + 12), title, font=face_title, fill=title_fill)
    y = y1 + 46
    for paragraph in body:
        for line in wrap(draw, paragraph, (x2 - x1) - 32, face_body):
            draw.text((x1 + 16, y), line, font=face_body, fill=INK)
            y += 24


def arrow(draw, x, y1, y2):
    draw.line((x, y1, x, y2 - 8), fill=NAVY, width=3)
    draw.polygon([(x - 7, y2 - 10), (x + 7, y2 - 10), (x, y2)], fill=NAVY)


def architecture():
    image = Image.new("RGB", (1500, 1160), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 24), "Implemented AEGIS prototype", font=font(32, True), fill=INK)
    draw.text(
        (40, 66),
        "One Streamlit application. Next.js and FastAPI are not part of this build.",
        font=font(20),
        fill=MUTED,
    )
    card(
        draw,
        (40, 110, 1460, 230),
        "Streamlit interface  ·  app.py, ui.py",
        [
            "1 Upload and check, 2 Review (consent, call limits), 3 Results (counts, chart, filters, cards, annotated text, CSV/HTML).",
            "Optional: questions about this contract (lexical matching) and the EU AI Act chat (OpenAI only).",
        ],
        fill=BLUE,
        outline=NAVY,
    )
    arrow(draw, 750, 230, 268)
    boxes = [
        (40, 268, 500, 470, "PDF text  ·  embed.py", ["PyMuPDF reads the file.", "Checks header, size, page count,", "encryption and extractable text.", "Image-only files are rejected. No OCR."]),
        (520, 268, 980, 470, "Clauses  ·  segment.py", ["Deterministic splitting on numbering,", "headings and paragraphs.", "Title or banner blocks are tagged", "preamble and are not reviewed."]),
        (1000, 268, 1460, 470, "Playbook  ·  playbook.json", ["30 illustrative clauses, 10 categories.", "One index per provider, with provider,", "model, dimension and playbook SHA-256", "in its metadata."]),
    ]
    for box in boxes:
        card(draw, box[:4], box[4], box[5])
    arrow(draw, 750, 470, 508)
    card(
        draw,
        (40, 508, 740, 750),
        "Comparison  ·  analyze.py, providers.py",
        [
            "AEGIS_PROVIDER selects OpenAI (default) or Gemini.",
            "No automatic fallback between providers.",
            "Similarity ranks passages. It is not a risk score.",
            "Model returns JSON; an id that was not",
            "retrieved is rejected.",
        ],
        fill=BLUE,
        outline=NAVY,
    )
    card(
        draw,
        (760, 508, 1460, 750),
        "Labels  ·  risk.py",
        [
            "Deterministic rules after the comparison.",
            "Labels: Safe, Caution, Risky, Review required.",
            "A wording difference alone is not Risky.",
            "Missing evidence is never labelled Safe.",
        ],
        fill=AMBER,
        outline=(180, 83, 9),
        title_fill=(180, 83, 9),
    )
    arrow(draw, 750, 750, 788)
    card(
        draw,
        (40, 788, 1460, 960),
        "Calls, results and downloads  ·  retry.py, errors.py, report.py",
        [
            "Every call has a hard limit; only rate limits, 5xx and network errors are retried (at most 3).",
            "No credit or a rejected key stops the remaining clauses; finished clauses are kept.",
            "Rows record provider and models. CSV and escaped HTML are built from them in this session.",
        ],
    )
    rounded(draw, (40, 990, 1460, 1110), (255, 255, 255), DASH, dashed=True)
    draw.text((56, 1006), "Not implemented in this prototype", font=font(20, True), fill=DASH)
    draw.text(
        (56, 1040),
        "Next.js, FastAPI, PostgreSQL, accounts, spaCy, pdfplumber, FAISS, OCR, highlighting on the PDF pages.",
        font=font(18),
        fill=MUTED,
    )
    path = FIG / "architecture.png"
    image.save(path)
    return path


def workflow():
    image = Image.new("RGB", (1500, 1180), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 24), "Review workflow in the current code", font=font(32, True), fill=INK)
    steps = [
        ("1", "Upload a PDF", "Checked on this computer. Nothing is sent."),
        ("2", "Validate the file", "Header, size (15 MB), pages (40), encryption, text."),
        ("3", "Segment", "Clauses to review; preamble shown but not sent."),
        ("4", "Check setup", "Key present, index ready and matching, Gemini gate."),
        ("5", "Consent and limits", "At most 2 embedding + 1 chat call per clause."),
        ("6", "Retrieve, compare, label", "JSON comparison, then the rules."),
        ("7", "Show and export", "Cards, annotated text, CSV and HTML."),
    ]
    y = 90
    for number, title, body in steps:
        draw.ellipse((40, y, 92, y + 52), fill=NAVY)
        draw.text((58, y + 10), number, font=font(22, True), fill="white")
        draw.text((112, y), title, font=font(22, True), fill=INK)
        draw.text((112, y + 30), body, font=font(18), fill=MUTED)
        y += 78
    rounded(draw, (840, 100, 1460, 660), RED, (153, 27, 27))
    draw.text((860, 118), "Stops that do not invent a Safe label", font=font(22, True), fill=(153, 27, 27))
    stops = [
        "Not a PDF, too large, too many pages, encrypted, or image-only: rejected with the next step.",
        "Key missing, index missing, partial or built for the other provider: the Review button stays disabled.",
        "No relevant passage, invalid JSON or a failed call: the clause is Review required.",
        "No credit or a rejected key: the remaining clauses are not sent; finished ones are kept.",
        "Same file and settings: the stored review is shown and nothing is called again.",
    ]
    yy = 170
    face = font(18)
    for item in stops:
        for line in wrap(draw, item, 570, face):
            draw.text((860, yy), line, font=face, fill=INK)
            yy += 24
        yy += 14
    rounded(draw, (40, 690, 1460, 900), BLUE, NAVY)
    draw.text((60, 710), "Separate from the risk review", font=font(22, True), fill=NAVY)
    for offset, text in enumerate(
        [
            "Questions about the contract use lexical matching over this session's clauses and show the passage id.",
            "The model is called only when a passage matches; otherwise the fixed refusal is shown and nothing is sent.",
            "The EU AI Act chat is optional, OpenAI only, and does not read the uploaded contract.",
        ]
    ):
        draw.text((60, 754 + offset * 34), text, font=font(18), fill=INK)
    rounded(draw, (40, 930, 1460, 1120), (255, 255, 255), DASH, dashed=True)
    draw.text((60, 950), "Not demonstrated on a live model", font=font(22, True), fill=DASH)
    draw.text(
        (60, 994),
        "Automated tests use fixture comparisons, a fake Gemini client and a local hash embedding. They call no API.",
        font=font(18),
        fill=INK,
    )
    draw.text(
        (60, 1028),
        "No screenshot in this report shows a completed review, because no authorised live run was performed.",
        font=font(18),
        fill=INK,
    )
    path = FIG / "workflow.png"
    image.save(path)
    return path


def records():
    image = Image.new("RGB", (1500, 860), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 24), "Records the prototype actually produces", font=font(32, True), fill=INK)
    draw.text(
        (40, 68),
        "In-memory review rows and the playbook file. There is no relational database.",
        font=font(20),
        fill=MUTED,
    )
    card(
        draw,
        (40, 130, 720, 660),
        "Playbook clause  ·  playbook.json",
        [
            "id, for example termination-01",
            "category, one of the ten project categories",
            "title",
            "standard_text, original example wording",
            "red_flags, a non-empty list of phrases",
            "Index metadata per collection: provider,",
            "embedding_model, dimension (Gemini 768),",
            "playbook_sha256. A mismatch is refused.",
            "The wording is illustrative, not legal advice.",
        ],
    )
    card(
        draw,
        (780, 130, 1460, 660),
        "One review row per block",
        [
            "clause_id, heading, kind (clause or preamble)",
            "contract_text and its start offset",
            "category, evidence_id, reference title",
            "similarity, reference_clause (retrieved text)",
            "deviations, explanation, severity",
            "label and the deciding rule, warnings",
            "error_kind when a call failed",
            "provider, embedding_model, chat_model",
            "confirmed_violation is always false",
        ],
        fill=BLUE,
        outline=NAVY,
    )
    draw.text(
        (40, 710),
        "CSV and HTML downloads are generated from these rows, with the provider, models, date and a preliminary notice.",
        font=font(20),
        fill=INK,
    )
    draw.text(
        (40, 750),
        "Rows are kept in the app's memory for the browser session. The review does not write the contract text to disk.",
        font=font(20),
        fill=MUTED,
    )
    path = FIG / "records.png"
    image.save(path)
    return path


def shade(cell, color):
    element = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), color)
    shd.set(qn("w:val"), "clear")
    element.append(shd)


def set_run_font(run, name="Times New Roman", size=12, bold=False, color=None, italic=False):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)


def add_page_number(paragraph):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instr)
    run._r.append(end)
    set_run_font(run, size=10, color=MUTED)


def paragraph(document, text, size=12, bold=False, italic=False, center=False, space_after=8, space_before=0, color=None):
    item = document.add_paragraph()
    item.paragraph_format.space_after = Pt(space_after)
    item.paragraph_format.space_before = Pt(space_before)
    item.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    item.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.JUSTIFY
    run = item.add_run(text)
    set_run_font(run, size=size, bold=bold, italic=italic, color=color)
    return item


def heading(document, text, level):
    item = document.add_heading(text, level=level)
    item.paragraph_format.space_before = Pt(16 if level == 1 else 12)
    item.paragraph_format.space_after = Pt(6)
    for run in item.runs:
        set_run_font(run, size=16 if level == 1 else 13, bold=True, color=NAVY)
    return item


def bullet(document, text):
    item = document.add_paragraph(style="List Bullet")
    item.clear()
    item.paragraph_format.space_after = Pt(3)
    run = item.add_run(text)
    set_run_font(run, size=12)
    return item


CONTENTS = [
    "Acknowledgements",
    "Abstract",
    "1. Introduction",
    "1.1 Background",
    "1.2 Problem statement",
    "1.3 Objectives",
    "1.4 Users",
    "1.5 How this revision was checked",
    "2. Requirements and scope",
    "2.1 Functional requirements",
    "2.2 Non-functional behaviour",
    "2.3 Scope",
    "3. System architecture",
    "3.1 What was built",
    "3.2 Module responsibilities",
    "3.3 Model and storage choices",
    "3.4 Why Streamlit was sufficient for the prototype",
    "4. Data and knowledge representation",
    "4.1 The uploaded contract",
    "4.2 Segmentation",
    "4.3 Playbook",
    "4.4 Retrieval",
    "4.5 The review row",
    "5. Privacy and security",
    "6. AI method",
    "6.1 Pipeline",
    "6.2 Comparison",
    "6.3 Labels",
    "6.4 Optional questions",
    "6.5 Limits of the method",
    "7. Workflow",
    "7.1 What a live review would do",
    "7.2 Local pipeline that was executed",
    "8. Tests and evidence",
    "8.1 Command and result",
    "8.2 Tests by module",
    "8.3 Checks that were not run",
    "8.4 Screenshot evidence",
    "8.5 Evaluation that has not been done",
    "9. Findings",
    "9.1 Completed",
    "9.2 Partially completed",
    "9.3 Future work",
    "9.4 Why this is not legal advice",
    "10. Conclusion",
    "References",
    "Appendix A. Environment",
]


def add_toc(document):
    paragraph(
        document,
        "Sections are listed in reading order. Page numbers are printed in the footer. They are not repeated here because this computer has no word processor that can calculate the final page breaks.",
        size=11,
        italic=True,
        space_after=6,
    )
    for title in CONTENTS:
        nested = "." in title[:4]
        item = document.add_paragraph()
        item.paragraph_format.space_after = Pt(2)
        item.paragraph_format.left_indent = Cm(0.75) if nested else Cm(0)
        set_run_font(item.add_run(title), size=12, bold=not nested)


def table(document, caption, headers, rows, widths=None):
    paragraph(document, caption, size=11, bold=True, center=False, space_before=8, space_after=4)
    grid = document.add_table(rows=1, cols=len(headers))
    grid.style = "Table Grid"
    grid.autofit = True
    for index, header in enumerate(headers):
        cell = grid.rows[0].cells[index]
        cell.text = ""
        run = cell.paragraphs[0].add_run(header)
        set_run_font(run, size=10, bold=True, color=(255, 255, 255))
        shade(cell, "1E3A8A")
    for row in rows:
        cells = grid.add_row().cells
        for index, value in enumerate(row):
            cells[index].text = ""
            run = cells[index].paragraphs[0].add_run(value)
            set_run_font(run, size=10)
    if widths:
        for row in grid.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Inches(width)
    document.add_paragraph().paragraph_format.space_after = Pt(6)
    return grid


def figure(document, path, caption):
    document.add_picture(str(path), width=Inches(6.4))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph(document, caption, size=11, italic=True, center=True, space_after=12)


TEST_DESCRIPTIONS = {
    "test_segment.py": "Numbering, nested 1.1, ALL CAPS headings, lettered sub-clauses, tables, preamble, offsets.",
    "test_playbook.py": "Schema, duplicate ids, missing categories, stable reindex.",
    "test_index_failure.py": "Failed or short embedding responses and failed writes leave no partial index.",
    "test_analyze.py": "Category filter, rejected evidence ids, low similarity, timeout and bad JSON.",
    "test_risk.py": "Safe, Caution, Risky, Review required and repeatability.",
    "test_workspace.py": "PDF rejection, counts from real rows, HTML escaping.",
    "test_preamble_review.py": "The banner is shown as preamble, never sent, never counted.",
    "test_export_fields.py": "Provider, models, date and notice in CSV and HTML.",
    "test_errors.py": "No credit separated from rate limits; fatal errors stop the review; codes only.",
    "test_retry.py": "At most 3 retries at 1/2/4 s, Retry-After, never for auth or no credit, budgets.",
    "test_preflight.py": "Offline preflight with SDKs and sockets replaced by functions that raise.",
    "test_providers.py": "Provider selection, Gemini collection, mismatch refusal, no fallback, gate.",
    "test_regulation_guard.py": "The EU Act download runs only on an explicit command.",
    "test_qa.py": "Unique sessions, source ids, unsupported answers, no shared collection.",
    "test_security.py": "Prompt boundary, key not echoed, timeouts, escaped reports.",
    "test_session_ui.py": "Review fingerprint includes the configuration.",
    "test_app_states.py": "Streamlit AppTest for each page state, with fixture pipelines only.",
    "test_ui.py": "Contrast of every label colour, icons, filters, sort, escaping.",
    "test_integration.py": "Imports, temporary hash index, synthetic PDF, CSV and HTML.",
    "test_isolation.py": "Tests use a fake key, skip .env, and cannot open connections.",
}


def run_tests():
    """Run the offline suite now and return what actually happened."""
    import datetime as DT
    import os
    import platform
    import re
    import subprocess
    import sys
    import tempfile
    import xml.etree.ElementTree as ET

    project = ROOT.parent
    with tempfile.TemporaryDirectory() as folder:
        junit = Path(folder) / "junit.xml"
        env = dict(os.environ, AEGIS_NO_DOTENV="1")
        process = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"],
            cwd=project,
            capture_output=True,
            text=True,
            env=env,
        )
        tree = ET.parse(junit)
    suite = tree.getroot()
    if suite.tag == "testsuites":
        suite = suite[0]
    per_file = {}
    for case in suite.iter("testcase"):
        classname = case.get("classname", "")
        parts = classname.split(".")
        name = next((part for part in parts if part.startswith("test_")), classname) + ".py"
        per_file[name] = per_file.get(name, 0) + 1
    summary = process.stdout.strip().splitlines()[-1] if process.stdout.strip() else ""
    warnings = re.search(r"(\d+) warnings?", summary)
    return {
        "date": DT.date.today().strftime("%d %B %Y").lstrip("0"),
        "python": platform.python_version(),
        "system": f"{platform.system()} {platform.machine()}",
        "tests": int(suite.get("tests", 0)),
        "failures": int(suite.get("failures", 0)),
        "errors": int(suite.get("errors", 0)),
        "skipped": int(suite.get("skipped", 0)),
        "seconds": float(suite.get("time", 0.0)),
        "warnings": int(warnings.group(1)) if warnings else 0,
        "summary": summary,
        "per_file": per_file,
        "returncode": process.returncode,
    }


def build():
    FIG.mkdir(parents=True, exist_ok=True)
    tests = run_tests()
    passed = tests["tests"] - tests["failures"] - tests["errors"] - tests["skipped"]
    result_text = (
        f"{passed} passed, {tests['failures']} failed, {tests['errors']} errors, {tests['skipped']} skipped and "
        f"{tests['warnings']} warnings, in {tests['seconds']:.1f} seconds"
    )
    architecture_path = architecture()
    workflow_path = workflow()
    records_path = records()

    document = Document()
    section = document.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run("AEGIS  ·  Course project report  ·  page ")
    set_run_font(run, size=10, color=MUTED)
    add_page_number(footer)

    core = document.core_properties
    core.title = "AEGIS: AI-Powered Legal Contract Review Assistant"
    core.author = "Anjali Agrahari"
    core.subject = "Revised course project report based on the implemented Streamlit prototype"
    core.category = "Academic report"

    paragraph(document, "Vellore Institute of Technology (VIT), Vellore", size=12, center=True, space_after=2)
    paragraph(document, "Course Project Report — Artificial Intelligence", size=12, center=True, space_after=16)
    paragraph(document, "AEGIS: AI-Powered Legal Contract Review Assistant", size=22, bold=True, center=True, space_after=8, color=NAVY)
    paragraph(document, "Revised report describing the implemented prototype", size=13, italic=True, center=True, space_after=16)

    for label, value in [
        ("Student", "Anjali Agrahari"),
        ("Register number", "24BIT0111"),
        ("Team member", "Indrakshi Singh"),
        ("Register number", "24BIT0589"),
        ("Faculty guide", "[To be filled in]"),
        ("Revision date", tests["date"]),
    ]:
        item = document.add_paragraph()
        item.alignment = WD_ALIGN_PARAGRAPH.CENTER
        item.paragraph_format.space_after = Pt(1)
        label_run = item.add_run(f"{label}: ")
        set_run_font(label_run, size=12, bold=True)
        value_run = item.add_run(value)
        set_run_font(value_run, size=12)

    paragraph(
        document,
        f"Each implementation statement below was checked against the project source, and the test figures come from a run made when this document was built ({tests['date']}). Where a feature is only partly built, or was not executed against a live model, the report says so.",
        size=12,
        space_before=14,
    )

    heading(document, "Acknowledgements", 1)
    paragraph(
        document,
        "The authors thank Vellore Institute of Technology for the opportunity to complete this course project. The faculty guide’s name is still to be entered on the title page. No other personal acknowledgement is added here, because none was recorded in the source report.",
    )

    heading(document, "Abstract", 1)
    paragraph(
        document,
        "AEGIS is a prototype that helps a reader inspect an uploaded contract before asking a qualified person to review it. It is a Streamlit application. A user uploads a text-based PDF, PyMuPDF extracts the text, and a deterministic segmenter divides it into clauses, setting aside a title or banner block as a preamble that is not reviewed. Each clause is compared with a 30-clause playbook stored in Chroma. The comparison uses OpenAI by default (text-embedding-ada-002 and a chat model that defaults to gpt-4o); Gemini can be selected explicitly instead, with its own index. A separate rule module assigns Safe, Caution, Risky or Review required. The page shows counts, a chart, filterable clause cards and an annotated view of the contract text, and downloads CSV and HTML that record the provider and models.",
    )
    paragraph(
        document,
        "The prototype does not replace legal advice, and the playbook wording is an original teaching example rather than a legally validated standard. The tool has not been evaluated for legal accuracy: there is no labelled set, so no precision, recall or agreement figure exists. The automated tests check program behaviour with fixtures and call no API. No live review has been run for this report, because the playbook index has not been built; live results remain pending. Next.js and FastAPI appear only as future work.",
    )
    paragraph(
        document,
        "Keywords: contract review, Streamlit, clause playbook, Chroma, retrieval-augmented generation, deterministic risk labels, PyMuPDF.",
        italic=True,
    )

    heading(document, "Contents", 1)
    add_toc(document)

    heading(document, "1. Introduction", 1)
    heading(document, "1.1 Background", 2)
    paragraph(
        document,
        "People who sign vendor agreements, nondisclosure agreements and freelance contracts often need a first reading before they can decide whether to ask a lawyer. Termination, liability, payment and confidentiality clauses are easy to miss in a long document. AEGIS is a course prototype for that first reading. It compares clauses with a project playbook and shows why a label was chosen. It does not decide whether a clause is lawful or enforceable.",
    )
    heading(document, "1.2 Problem statement", 2)
    paragraph(
        document,
        "Build an assistant that accepts a contract PDF, splits it into clauses, compares those clauses with a structured playbook, labels each clause, and explains the comparison from retrieved passages, while stating clearly that the result is not legal advice.",
    )
    heading(document, "1.3 Objectives", 2)
    for text in [
        "Accept a text-based PDF and reject files that cannot be read safely.",
        "Segment the extracted text into clauses without a separate NLP toolkit, and set aside title or banner text.",
        "Keep an illustrative playbook for ten clause families and index it by stable identifiers.",
        "Retrieve playbook passages and ask the configured chat model for a structured comparison.",
        "Apply deterministic rules so that a missing comparison is not labelled Safe.",
        "Show the result in the browser and export CSV and HTML from the same rows.",
        "Keep uploaded contracts out of the shared playbook and regulation collections.",
        "Make every model call visible and bounded: consent before a review, a stated call limit, no hidden retries.",
    ]:
        bullet(document, text)
    heading(document, "1.4 Users", 2)
    table(
        document,
        "Table 1. Intended users of the prototype.",
        ["User", "Intended use"],
        [
            ["Freelancer or student", "See which clauses the prototype flags before signing or before asking for help."],
            ["Small-organisation reader", "Obtain a first-pass list of payment, liability and termination wording."],
            ["Course examiner", "Inspect how a label was derived from a retrieved passage and a rule."],
        ],
    )
    paragraph(
        document,
        "A lawyer is not the primary user. The prototype can at most help someone prepare questions. It does not create a professional relationship and it does not file, negotiate or redline a contract.",
    )
    heading(document, "1.5 How this revision was checked", 2)
    paragraph(
        document,
        "The implemented prototype is the firica/legalai Streamlit application, extended with contract-review modules. The statements in this revision were compared with app.py, ui.py, embed.py, segment.py, analyze.py, providers.py, errors.py, retry.py, preflight.py, risk.py, report.py, agent.py, playbook.json and the test files. The test suite is run by the script that builds this document, and the counts in section 8 are taken from that run. The project copy used for this revision had no Git history attached, so changes were delivered as a series of commits to be applied to the project repository.",
    )

    heading(document, "2. Requirements and scope", 1)
    heading(document, "2.1 Functional requirements", 2)
    paragraph(
        document,
        "Status meanings used in this report: Completed means the behaviour is in the source and was exercised by an automated test or by the running page. Partial means the code exists but a live model run, a built index, or part of the original requirement is missing. Future means the item is not in the source.",
    )
    table(
        document,
        "Table 2. Functional requirements compared with the source.",
        ["ID", "Requirement", "What the code does", "Status"],
        [
            ["FR1", "Upload a contract", "Streamlit file uploader, PDF only.", "Completed"],
            ["FR2", "Validate the file", "Header, size, page count, encryption and empty text are checked, each with a next step.", "Completed"],
            ["FR3", "Extract text", "PyMuPDF. pdfplumber is not used.", "Completed"],
            ["FR4", "Segment clauses", "Numbering, headings and paragraphs; title or banner text is a preamble, not reviewed.", "Completed"],
            ["FR5", "Playbook", "playbook.json has 30 clauses. The index code works on local tests; the index was not built.", "Partial"],
            ["FR6", "Retrieve references", "Searches the provider's own collection after checking its metadata. Tests use fixtures.", "Partial"],
            ["FR7", "Grounded comparison", "The chat model returns JSON. No live call was made for this report.", "Partial"],
            ["FR8", "Risk label", "risk.py assigns four labels with deterministic rules.", "Completed"],
            ["FR9", "Clause report", "Cards show the clause, reference id, title and similarity, differences, explanation and warnings.", "Completed"],
            ["FR10", "Annotated viewer", "The contract text with each clause marked by label; a clause expands in place. The PDF pages are not highlighted.", "Partial"],
            ["FR11", "Dashboard", "Label counts, an overall description, a bar chart, filters by label and category, search and sort.", "Completed"],
            ["FR12", "Disclaimer", "Header, exports and labels say the result is preliminary and not a legal conclusion.", "Completed"],
            ["FR13", "Export", "CSV and HTML from the review rows, with provider, models, date and notice. HTML is escaped.", "Completed"],
            ["FR14", "Optional questions", "Lexical matching over the session's clauses; the model is called only when a passage matches. Not run live.", "Partial"],
            ["FR15", "Provider choice", "OpenAI by default; Gemini when selected, with its own index and gates. Neither was run live.", "Partial"],
            ["FR16", "Preflight", "Offline check of configuration, playbook, index state and call plan, with a PASS or FAIL exit code.", "Completed"],
        ],
    )
    heading(document, "2.2 Non-functional behaviour", 2)
    table(
        document,
        "Table 3. Non-functional behaviour present in the source.",
        ["ID", "Requirement", "Implementation"],
        [
            ["NFR1", "Confidentiality", "Contract passages are not written to any collection. The consent box names the provider before a review."],
            ["NFR2", "Grounding", "A comparison is rejected when its evidence id was not retrieved."],
            ["NFR3", "Transparency", "The card and the export show the evidence id, the retrieved passage and the models used."],
            ["NFR4", "Failures", "No credit, rate limit, bad key, timeout and server errors get distinct messages; failures are Review required, never Safe."],
            ["NFR5", "Bounded calls", "SDK retries are off. At most 3 retries for rate limits, 5xx and network errors; every attempt counts toward a hard limit."],
            ["NFR6", "Configuration", "Keys, provider, models, timeout, retries and upload limits come from the environment (.env)."],
            ["NFR7", "Untrusted uploads", "Contract text is placed in the user message, not in the system prompt, and is escaped on the page and in HTML."],
            ["NFR8", "Accessibility", "Each label has an icon and text, not colour alone; every label colour pair has a contrast ratio of at least 4.5:1."],
        ],
    )
    heading(document, "2.3 Scope", 2)
    paragraph(
        document,
        "The playbook covers termination, liability, indemnity, payment, confidentiality, auto-renewal, intellectual property ownership, non-compete, governing law and dispute resolution. Three illustrative clauses are stored for each category. English text-based PDFs within 15 MB and 40 pages are the supported input. Scanned pages, encrypted files, negotiation, drafting and jurisdiction-specific advice are outside the completed work.",
    )

    heading(document, "3. System architecture", 1)
    heading(document, "3.1 What was built", 2)
    paragraph(
        document,
        "The prototype is one Python application started with streamlit run app.py. There is no separate Next.js frontend and no FastAPI service. The browser talks to Streamlit, which calls the local modules directly. A model provider is called only when the user starts a review after ticking the consent box, or asks a question that matches a passage, and only if the key and a matching playbook index are present.",
    )
    figure(document, architecture_path, "Figure 1. Architecture of the implemented prototype. Dashed items were not built.")
    heading(document, "3.2 Module responsibilities", 2)
    table(
        document,
        "Table 4. Modules in the Streamlit prototype.",
        ["Module", "Responsibility"],
        [
            ["app.py", "Page flow, sidebar, consent, progress, results, downloads and the optional chats."],
            ["ui.py", "Escaped HTML for badges, cards and the annotated contract; filters, sort and the overall description."],
            ["embed.py", "PDF reading, playbook validation, indexing, preflight and inspect-scores commands."],
            ["segment.py", "Deterministic clause segmentation and preamble tagging."],
            ["analyze.py", "Retrieval, comparison and schema checks, with per-provider thresholds."],
            ["providers.py", "Provider selection, the Gemini calls, index metadata checks and the Gemini chat gate."],
            ["errors.py, retry.py", "Classification of provider errors, bounded retries and call limits."],
            ["preflight.py", "Offline checks and call plan."],
            ["risk.py", "Deterministic Safe, Caution, Risky and Review required labels."],
            ["report.py", "Review rows, label counts, CSV and escaped HTML."],
            ["agent.py", "Optional question answering. It does not assign risk labels."],
            ["playbook.json", "Thirty illustrative reference clauses."],
        ],
    )
    heading(document, "3.3 Model and storage choices", 2)
    paragraph(
        document,
        "OpenAI is the default provider and its behaviour is unchanged from the earlier build: text-embedding-ada-002 for indexing and queries, and a chat model that defaults to gpt-4o (OPENAI_MODEL) at temperature 0. The playbook collection is playbook_clauses. Setting AEGIS_PROVIDER=gemini selects Gemini instead: gemini-embedding-001 with the RETRIEVAL_DOCUMENT and RETRIEVAL_QUERY task types, 768 dimensions normalised to unit length as Google's embeddings guide requires below 3,072 dimensions, in a separate cosine collection named playbook_clauses_gemini. The Gemini chat model comes from GEMINI_CHAT_MODEL and defaults to gemini-3.8-flash, which Google's models page listed as stable when checked on 29 October 2026. There is no automatic fallback between providers.",
    )
    paragraph(
        document,
        "Each index stores its provider, embedding model, dimension where fixed, and the SHA-256 of playbook.json in its metadata. A missing, partial, mismatched or other-provider index is refused before any search. Requests time out after OPENAI_TIMEOUT seconds (default 60) for both providers. PostgreSQL, SQLite, FAISS and user accounts were not implemented. .env.example lists the variable names with no values, and .gitignore excludes .env, the virtual environment, chroma_storage and logs.",
    )
    heading(document, "3.4 Why Streamlit was sufficient for the prototype", 2)
    paragraph(
        document,
        "The course prototype needs one user flow: upload, review, inspect and download. Streamlit provides the page, file upload, session state, tabs and charts. Splitting that flow into Next.js and FastAPI would add a second application without changing the review method. One report feature could not be built faithfully in Streamlit: a link from a clause in the annotated view to its card in another tab. The nearest working option, which was built, is that each clause in the annotated view expands in place to show its finding.",
    )

    heading(document, "4. Data and knowledge representation", 1)
    heading(document, "4.1 The uploaded contract", 2)
    paragraph(
        document,
        "The uploaded file stays in the Streamlit session as bytes. PyMuPDF extracts a single text string and a page count. Page coordinates are not stored, so the original PDF pages are not highlighted; the annotated view marks the extracted text instead. The file is not copied into any vector collection.",
    )
    heading(document, "4.2 Segmentation", 2)
    paragraph(
        document,
        "segment_contract recognises numbered clauses such as 1, 1.1 and 1.1.1, headings such as Article, Section and Clause, and short uppercase headings, and falls back to paragraphs. Text before the first clause marker, or a lone title heading at the start, is tagged as a preamble: it is shown as Preamble - not reviewed and is not sent for comparison. A number in a table cell, such as 500 USD or 10 days, is not treated as a clause start, and a short title never absorbs the first clause. Lettered sub-clauses stay with their parent. Fragments shorter than 40 characters without their own clause number are merged, and a clause longer than 1,500 characters is split on blank lines or sentence boundaries. The joined text equals the original. For synthetic-contract-v2.pdf the result is nine blocks: one preamble and eight clauses.",
    )
    heading(document, "4.3 Playbook", 2)
    paragraph(
        document,
        "Each playbook record has an id, category, title, standard_text and a non-empty list of red_flags. Identifiers such as termination-01 are the Chroma ids. Indexing is all or nothing: the embeddings are requested first, and if the request or the write fails, an empty or partial collection is deleted so the app cannot treat it as an index. Validation rejects a duplicate id, an empty field, an unknown category and a missing category. The text was written for this project as an example. It has not been reviewed or validated as a statement of fair or enforceable terms.",
    )
    heading(document, "4.4 Retrieval", 2)
    paragraph(
        document,
        "For each clause, the retriever embeds the clause with the provider's model and searches that provider's collection, first broadly (up to eight passages) and then, when a keyword cue or a clear similarity margin indicates a category, within that category. For OpenAI, passages below a display similarity of 0.35 are dropped. Gemini scores are on a different scale, so its cutoff starts as uninspected: Gemini chat calls are refused until python embed.py --inspect-scores --pdf PATH has been run (embedding requests only) and GEMINI_MIN_SIMILARITY has been set. Any cutoff chosen that way is tuned on the synthetic sample only and is not validated. At most three passages are kept. Similarity ranks passages; it is not the risk label.",
    )
    heading(document, "4.5 The review row", 2)
    paragraph(
        document,
        "After comparison and scoring, report.py builds one dictionary per block. The reference text is the retrieved passage, not wording produced by the model. If no evidence id is accepted, the explanation and deviations are cleared and a warning is stored. Every row records the provider, embedding model and chat model. The session keeps the rows until the uploaded bytes or the provider settings change.",
    )
    figure(document, records_path, "Figure 2. Playbook fields and review-row fields produced by the current modules.")

    heading(document, "5. Privacy and security", 1)
    paragraph(
        document,
        "Reading and checking a PDF happens on the user's computer. A review sends each clause's text to the selected provider for embeddings and comparison, and only after the user ticks a consent box that names the provider. A contract question sends the question and the matching passages, and only if a passage matches. The sidebar states this. Streamlit's usage statistics are off, and the server address is set to localhost, which keeps the app on the user's computer and stops Streamlit's start-up lookup of the public IP address in headless mode. This report does not claim encryption, deletion, anonymisation or isolation between users: the code does not implement them.",
    )
    paragraph(
        document,
        "Gemini data handling. When Gemini is selected, a standing banner reads: Gemini API: use synthetic or non-confidential text only. Google's Gemini API terms (ai.google.dev/gemini-api/terms, effective 23 March 2026, checked 29 October 2026) distinguish unpaid from paid services: for unpaid use, Google says it uses submitted content and responses to improve its products, that human reviewers may read them, and asks users not to submit sensitive, confidential or personal information; for paid use, Google says it does not use prompts or responses to improve its products and logs them for a limited period for policy enforcement. Which tier a given key uses is not determined by this code, and this report does not assert one. OpenAI's retention terms were not audited.",
    )
    table(
        document,
        "Table 5. Security measures that exist, and measures that were not implemented.",
        ["Topic", "What was implemented", "What was not done"],
        [
            ["Secrets", "Keys are read from OPENAI_API_KEY or GEMINI_API_KEY. Output shows only whether a key is set; error text carries short codes, never the key or the provider's message.", "No secrets manager or key rotation."],
            ["Shared index", "Contract text is never written to a collection. Provider collections accept only that provider's precomputed vectors.", "The optional chats still send passages to the provider."],
            ["Hidden calls", "SDK retries are off; retries are bounded and counted; Chroma telemetry is off; the EU Act download needs an explicit command.", "Third-party library behaviour was not audited line by line."],
            ["HTML", "Contract and model text is escaped on the page, in the annotated view and in the HTML export.", "No content security policy is configured."],
            ["Prompt injection", "Uploaded text stays in the user message. Unknown evidence ids are rejected.", "A live adversarial contract was not submitted to a model."],
            ["Accounts", "One local browser session.", "No login, no per-user storage, no deletion workflow."],
        ],
    )

    heading(document, "6. AI method", 1)
    heading(document, "6.1 Pipeline", 2)
    paragraph(
        document,
        "The review pipeline is retrieval-augmented comparison followed by deterministic labelling. Segmentation does not use a learned model. Retrieval uses embeddings. The chat model compares one clause with the passages that were retrieved. The label is then chosen in Python. The model does not choose Safe, Caution, Risky or Review required.",
    )
    heading(document, "6.2 Comparison", 2)
    paragraph(
        document,
        "Both providers receive the same system instruction and payload. The instruction tells the model that the clause is document content, not an instruction, and that it must ignore requests to change the model, the tools or the output schema. The requested JSON keys are evidence_id, deviations, explanation and severity. Severity is low, medium or high and describes a wording difference, not legal risk. If the evidence id is not one of the retrieved ids, the comparison is discarded. OpenAI runs at temperature 0. For Gemini the temperature is left at Google's recommended default of 1.0 for Gemini 3 models, so a Gemini comparison may differ between runs.",
    )
    heading(document, "6.3 Labels", 2)
    table(
        document,
        "Table 6. Risk labels assigned by risk.py.",
        ["Label", "Rule in risk.py"],
        [
            ["Safe", "Low severity and no targeted phrase rule. This means no project rule fired. It does not mean the clause is safe to sign."],
            ["Caution", "Medium or high severity, or a milder targeted phrase, such as a shorter notice period or a ten-day payment term."],
            ["Risky", "A targeted phrase matches, such as immediate one-sided termination, unlimited liability, or a one-sided indemnity. Severity alone does not make a clause Risky."],
            ["Review required", "No evidence, incomplete extraction, a failed call or an invalid comparison. This label is never converted to Safe."],
        ],
    )
    paragraph(
        document,
        "Every scored row sets confirmed_violation to false. The labels are preliminary rule outputs, not legal conclusions. A preamble block receives no label. No retrieval-accuracy, classification-agreement or explanation-quality percentage is reported, because that evaluation was not performed.",
    )
    heading(document, "6.4 Optional questions", 2)
    paragraph(
        document,
        "agent.py is not the labelling engine. Ask about this contract uses lexical matching: it builds a term index of the session's clauses and ranks passages by shared content words. It is not a semantic search. A question with no shared term does not call any model and returns exactly: I can't answer that from the retrieved passages. When passages match, they and the question are sent to the selected provider, and the answer is shown with the passage id and text; an answer the model marks as unsupported is replaced with the same refusal. The EU AI Act chat is optional, uses OpenAI only, and reads a separate regulation collection that has to be built with python embed.py regulation; the page shows that command and its cost warning when the collection is absent.",
    )
    heading(document, "6.5 Limits of the method", 2)
    for text in [
        "Embeddings can rank a passage as near even when a number, a negation or the named party changes the effect.",
        "The model can still describe a retrieved passage incorrectly. The page shows the passage so the reader can check.",
        "Clause-by-clause review misses a definition or an exception that sits in another clause.",
        "The playbook reflects example wording written for the course. Another reader could reasonably disagree with a label.",
        "A scanned contract cannot be read, because OCR was not added.",
        "A Gemini similarity cutoff chosen from the synthetic sample is not validated on other contracts.",
    ]:
        bullet(document, text)

    heading(document, "7. Workflow", 1)
    paragraph(
        document,
        "Figures 3 to 7 were captured on 29 October 2026 from the running application with Chromium (through Playwright), on Linux, with no API key and no playbook index. They show real states of the page; none contains model output. The sample file synthetic-contract-v2.pdf is a fictional one-page document, not a client contract. Figure 8 is the path the code follows after a file is chosen.",
    )
    for name, caption in [
        ("ui-start-no-key-no-index.png", "Figure 3. Start page: header with the preliminary-review note, sidebar with provider, models, missing key, missing index with its command, upload limits and the privacy notice."),
        ("ui-image-only-rejected.png", "Figure 4. An image-only PDF is rejected, with the next step."),
        ("ui-synthetic-v2-checked-review-blocked.png", "Figure 5. synthetic-contract-v2.pdf checked: eight clauses and one preamble block, the call limits (16 embedding and 8 chat calls), and the blockers that keep Review contract disabled."),
        ("ui-eu-chat-index-not-built.png", "Figure 6. The optional EU AI Act chat with no index: the command to build it and the cost warning."),
        ("ui-gemini-mode-banner-and-gates.png", "Figure 7. Gemini selected: the standing banner, Gemini models in the sidebar, and the similarity-cutoff gate."),
    ]:
        figure(document, FIG / name, caption)
    figure(document, workflow_path, "Figure 8. Workflow implemented in app.py, embed.py, analyze.py, risk.py and report.py.")
    heading(document, "7.1 What a live review does", 2)
    paragraph(
        document,
        "After a valid PDF is read, step 2 states the number of clauses and the maximum calls: two embedding queries and one chat call per clause, counting retries. The Review contract button stays disabled until the key is set, the provider's index is ready and matches the current playbook, the Gemini gate (if Gemini is selected) is open, and the consent box is ticked. During the review a progress bar reads clause i of N and each clause is listed as compared or not compared. If the provider reports no credit or rejects the key, the remaining clauses are not sent and the clauses already compared are kept. A rerun with the same file and settings shows the stored review and calls nothing.",
    )
    heading(document, "7.2 Local pipeline that was executed", 2)
    paragraph(
        document,
        "test_local_index_review_and_export built playbook_clauses in a temporary directory with a local hash embedding, read a synthetic PDF, and ran the real segmentation, validation, risk rules and export code with a fixture comparison. The Gemini tests did the same with a fake Gemini client. These show that the modules connect. They do not show that any model produced an explanation.",
    )
    heading(document, "7.3 Live results", 2)
    paragraph(
        document,
        "Pending. No live index build, review or question was run for this revision: in an earlier attempt recorded by the project team, the OpenAI API returned HTTP 429 credit_balance_exhausted, and no further paid call has been authorised. When a live run is authorised, this section will record the provider and models, per-clause similarity scores, validation warnings, labels and counts exactly as produced, the result of one supported and one unsupported question, and screenshots of the completed review.",
    )

    heading(document, "8. Tests and evidence", 1)
    heading(document, "8.1 Command and result", 2)
    paragraph(
        document,
        f"Command: python -m pytest -q, run by docs/build_report.py with Python {tests['python']} on {tests['system']}. Result on {tests['date']}: {result_text}. The suite sets a fake key, does not read .env, and refuses outbound network connections, so no test can call a model API. These tests check program behaviour; they are not a measure of legal-analysis accuracy.",
    )
    paragraph(
        document,
        "A clean installation was verified on 29 October 2026 in a new Python 3.12.3 virtual environment on Linux: pip install -r requirements.txt completed, every module imported with deprecation warnings treated as errors, and the suite passed with the network disabled. Installation on Windows and macOS was not verified. Remaining deprecation notices outside the tests: the optional EU AI Act chat uses LangChain's RunnableWithMessageHistory and InMemoryChatMessageHistory, which LangChain 1.x deprecates in favour of LangGraph persistence.",
    )
    heading(document, "8.2 Tests by module", 2)
    rows = []
    for name in sorted(tests["per_file"]):
        rows.append([name, str(tests["per_file"][name]), TEST_DESCRIPTIONS.get(name, "")])
    table(
        document,
        f"Table 7. Automated tests in the run of {tests['date']} ({tests['tests']} tests).",
        ["File", "Tests", "What is checked"],
        rows,
    )
    heading(document, "8.3 Checks that were not run", 2)
    for text in [
        "python embed.py playbook was not run, so neither playbook index exists and no embedding charge was incurred.",
        "Review contract was not run on any contract with a live model, for either provider.",
        "No contract question was answered by a live model.",
        "python embed.py --inspect-scores was not run, so no Gemini cutoff has been chosen.",
        "No labelled sample was scored by a lawyer, so retrieval accuracy and classification agreement are unknown.",
        "Explanation quality was not rated.",
        "Word pagination, the table of contents and footer page numbers must be checked by opening the file in Word.",
    ]:
        bullet(document, text)
    heading(document, "8.4 Screenshot evidence", 2)
    paragraph(
        document,
        "Figures 3 to 7 are the only screenshots in this report, and each shows a real state of the running page. No screenshot shows clause cards, label counts, the annotated contract, downloads or a contract answer, because those need a live review. No fixture-based or mocked screen was substituted.",
    )
    heading(document, "8.5 Evaluation that has not been done", 2)
    paragraph(
        document,
        f"The prototype has not been evaluated for legal accuracy. There is no labelled set of contracts, no second reviewer and no live sample from which precision, recall or agreement could be calculated. A credible later evaluation would use a small set of contracts labelled by a qualified reviewer, then measure whether the retrieved playbook id is relevant, whether the label agrees with the reviewer, and where the explanation misstates the clause. Those figures are separate from the {tests['tests']} automated tests, which check program behaviour with fixtures.",
    )

    heading(document, "9. Findings", 1)
    heading(document, "9.1 Completed", 2)
    for text in [
        "A three-step Streamlit page with a preliminary-review note, a setup sidebar, consent before any provider call, and stated call limits.",
        "PyMuPDF validation for empty files, bad headers, corrupt files, size, page count, encryption and image-only PDFs, each with a next step.",
        "Deterministic segmentation that preserves the wording, sets aside preamble text and handles tables and lettered sub-clauses.",
        "A 30-clause illustrative playbook with stable identifiers, validation and all-or-nothing indexing.",
        "Comparison code that refuses an evidence id the retriever did not return.",
        "Deterministic labels, including Review required when evidence is missing or a call fails.",
        "Error handling that tells no credit apart from a rate limit, bounded retries and hard call limits.",
        "Results with counts, a chart, filters, search, sort, clause cards and an annotated text view.",
        "CSV and HTML export with provider, models, date and notice; contract text escaped.",
        "OpenAI and Gemini as separate, explicit providers with separate indexes and no fallback.",
        "An offline preflight command and an embeddings-only inspect-scores command.",
    ]:
        bullet(document, text)
    heading(document, "9.2 Partially completed", 2)
    for text in [
        "Retrieval and comparison are implemented for both providers, but no index has been built and no live call was run.",
        "The annotated view marks the extracted text, not the original PDF pages.",
        "Contract question answering uses lexical matching, not a semantic index of the upload.",
        "The EU AI Act chat needs its regulation index, which has not been built for this revision.",
    ]:
        bullet(document, text)
    heading(document, "9.3 Future work", 2)
    for text in [
        "A Next.js interface and a FastAPI service, if the prototype needs a separate client and API.",
        "A database for accounts, saved reviews and deletion.",
        "OCR for scanned PDFs, with its own tests.",
        "A legally reviewed playbook, versioned and limited to a stated jurisdiction.",
        "A labelled evaluation of retrieval relevance, label agreement and explanation quality.",
        "Cross-clause reading of definitions and exceptions.",
        "Migrating the EU AI Act chat away from LangChain's deprecated message-history classes.",
    ]:
        bullet(document, text)
    heading(document, "9.4 Why this is not legal advice", 2)
    paragraph(
        document,
        "The labels come from project rules and example playbook text. They do not incorporate a statute, a case, or a lawyer's review. Safe means only that no targeted project rule fired. Risky means a listed phrase was found. Neither label determines enforceability, fairness or compliance. The interface, the exports and this report state that limit. A person who needs to sign, negotiate or rely on a contract still needs qualified advice.",
    )

    heading(document, "10. Conclusion", 1)
    paragraph(
        document,
        "The implemented AEGIS prototype is a Streamlit application for a first reading of a text-based contract. It extracts text with PyMuPDF, segments clauses deterministically, compares them with an illustrative 30-clause playbook through an explicitly selected provider, and applies auditable rules for Safe, Caution, Risky and Review required. Every provider call is preceded by consent, limited in number, and reported honestly when it fails. The page and the exports show the passage and the models used for each comparison and escape markup found in the contract.",
    )
    paragraph(
        document,
        f"The automated suite ran {tests['tests']} tests without a live model call ({result_text}). That supports the local behaviour of segmentation, validation, error handling, scoring, the interface states and export. It does not support a claim of measured accuracy, legal validation or production readiness. Before a live demonstration, a playbook index must be built, a real PDF reviewed, and the resulting screens captured.",
    )

    heading(document, "References", 1)
    for text in [
        "Chroma. Chroma documentation. https://docs.trychroma.com/",
        "Firica. legalai: a Streamlit application for questions about the EU Artificial Intelligence Act. https://github.com/firica/legalai. The contract-review modules in this project were added to that application.",
        "Google. Gemini API: Embeddings. https://ai.google.dev/gemini-api/docs/embeddings (page last updated 26 October 2026; checked 29 October 2026).",
        "Google. Gemini API: Gemini 3.8 Flash model page. https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash (last updated 22 October 2026; checked 29 October 2026).",
        "Google. Gemini API: API errors. https://ai.google.dev/gemini-api/docs/api-errors (last updated 23 October 2026; checked 29 October 2026).",
        "Google. Gemini API Additional Terms of Service. https://ai.google.dev/gemini-api/terms (effective 23 March 2026; checked 29 October 2026).",
        "Google Cloud. Gemini 3 prompting guide (temperature guidance). https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/gemini-3-prompting-guide (checked 29 October 2026).",
        "LangChain. LangChain documentation. https://python.langchain.com/",
        "OpenAI. Error codes. https://developers.openai.com/api/docs/guides/error-codes (checked 29 October 2026).",
        "OpenAI. API reference. https://platform.openai.com/docs/api-reference",
        "Artifex. PyMuPDF documentation. https://pymupdf.readthedocs.io/",
        "Streamlit. Streamlit documentation. https://docs.streamlit.io/",
    ]:
        bullet(document, text)
    paragraph(
        document,
        "No journal article or statute is cited as evidence that the playbook or the labels are legally correct. No such source was used to validate them.",
        space_before=6,
    )

    heading(document, "Appendix A. Environment", 1)
    table(
        document,
        "Table 8. Environment used for this report.",
        ["Item", "Value used by the prototype"],
        [
            ["Interface", "Streamlit 1.66.1"],
            ["PDF library", "PyMuPDF 2.0"],
            ["Vector store", "ChromaDB 1.5.9; playbook_clauses (OpenAI), playbook_clauses_gemini (Gemini)"],
            ["OpenAI (default)", "openai 3.36.1; text-embedding-ada-002; gpt-4o by default through OPENAI_MODEL"],
            ["Gemini (optional)", "google-genai 2.33.0; gemini-embedding-001 at 768 dimensions; gemini-3.8-flash by default through GEMINI_CHAT_MODEL"],
            ["Timeout and retries", "OPENAI_TIMEOUT, default 60 seconds; AEGIS_MAX_RETRIES, default and maximum 3"],
            ["Upload limits", "MAX_UPLOAD_MB default 15, MAX_PDF_PAGES default 40"],
            ["Tests", f"pytest 9.1.1; {result_text} on {tests['date']}"],
        ],
    )
    paragraph(
        document,
        "Setup. Create a Python 3.12 virtual environment, install requirements.txt, copy .env.example to .env, and set the key for the chosen provider. Run python embed.py playbook --preflight first: it makes no call and prints the plan. Building an index calls the embeddings API and can incur a charge. Starting the page does not, by itself, call a model.",
        space_before=8,
    )

    document.save(OUT)
    return architecture_path, workflow_path, records_path, OUT, tests


if __name__ == "__main__":
    result = build()
    tests = result[-1]
    print(result[3])
    print(f"tests: {tests['summary']}")
