"""Deterministic contract clause segmentation.

Clause boundaries come from numbering and headings. The model is not asked
to invent them. The existing fixed-size chunker remains in embed.py for the
Act retrieval corpus.
"""

import re

MAX_CLAUSE_CHARS = 1500
SHORT_CLAUSE_CHARS = 40

# A clause number followed by the start of a heading or sentence. The character
# after the number must not be lowercase or a digit, so table cells such as
# "10 days" or "1.5 hours" are not clause starts. A bare number with no dot or
# bracket also needs a capitalised word ("1 Services"), so "500 USD" is not one.
_NUMBER_RE = re.compile(
    r"^(?:"
    r"(?:\d{1,3}(?:\.\d{1,3})+\.?|\d{1,3}\.|\(\d{1,3}\)|\d{1,3}\))\s+[^\sa-z0-9]"
    r"|\d{1,3}\s+(?:[A-Z][a-z]|[\"“'‘(])"
    r")"
)
# Words that make a first heading a document title rather than a clause.
_TITLE_RE = re.compile(
    r"(?i)\b(agreement|contract|terms and conditions|order form|statement of work|"
    r"memorandum|addendum|deed|sample|draft)\b"
)
CLAUSE = "clause"
PREAMBLE = "preamble"
_STRUCTURAL_RE = re.compile(
    r"(?i)^(article|section|clause)\s+(\d+(?:\.\d+)*|[IVXLC]{1,8})\b"
)
_SENTENCE_BREAK_RE = re.compile(r"(?<=[A-Za-z][.!?])\s+(?=[A-Z])")


def segment_contract(text):
    """Return clause records in document order.

    Each record has id, heading, text, sequence and kind. Joining the text
    fields reproduces the original document. Empty or unusable input returns [].

    kind is "preamble" for a title, banner or parties block before the first
    clause marker (or a lone title heading at the start), and "clause"
    otherwise. A preamble is shown but not compared with the playbook.
    """
    text = _as_text(text)
    if text is None or not text.strip():
        return []

    spans, mode, anchor = _clause_spans(text)
    spans = _merge_short_spans(text, spans)
    spans = _split_long_spans(text, spans)
    spans = _merge_short_spans(text, spans)
    records = _records(text, spans)
    _mark_preamble(records, mode, anchor)
    return records


def reviewable(records):
    """Records that are compared with the playbook (preamble excluded)."""
    return [record for record in records or [] if record.get("kind", CLAUSE) != PREAMBLE]


def _mark_preamble(records, mode, anchor):
    for record in records:
        record["kind"] = CLAUSE
    if len(records) < 2:
        return
    before_anchor = [record for record in records if anchor is not None and record["start"] < anchor]
    if before_anchor:
        for record in before_anchor:
            record["kind"] = PREAMBLE
    else:
        first = records[0]
        heading = first["heading"]
        if not _NUMBER_RE.match(heading) and not _STRUCTURAL_RE.match(heading):
            title_like = bool(_TITLE_RE.search(heading)) and _is_uppercase_heading(heading)
            lone_title = mode == "paragraph" and _is_uppercase_heading(heading)
            if title_like or lone_title:
                first["kind"] = PREAMBLE
    if all(record["kind"] == PREAMBLE for record in records):
        for record in records:
            record["kind"] = CLAUSE


def _as_text(text):
    if text is None:
        return None
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace")
    if not isinstance(text, str):
        return None
    return text


def _lines(text):
    offset = 0
    for line in text.splitlines(keepends=True):
        yield offset, line
        offset += len(line)


def _clause_spans(text):
    numbered = []
    structural = []
    uppercase = []
    for offset, line in _lines(text):
        stripped = line.strip()
        if not stripped:
            continue
        if _NUMBER_RE.match(stripped):
            numbered.append(offset)
        elif _STRUCTURAL_RE.match(stripped):
            structural.append(offset)
        elif _is_uppercase_heading(stripped):
            uppercase.append(offset)

    if len(numbered) >= 2:
        starts = numbered + structural
        mode = "numbered"
        anchor = min(starts)
    elif len(structural) + len(uppercase) >= 2:
        starts = structural + uppercase
        mode = "headings"
        # "Article 1" / "Section 2" are clause markers; a lone uppercase line may be a title.
        anchor = min(structural) if structural else None
    else:
        starts = _paragraph_starts(text)
        mode = "paragraph"
        anchor = None

    return _spans_from_starts(text, starts), mode, anchor


def _is_uppercase_heading(line):
    if not (4 <= len(line) <= 80):
        return False
    letters = [char for char in line if char.isalpha()]
    if len(letters) < 4 or any(char.islower() for char in letters):
        return False
    words = line.split()
    if not words or len(words) > 10:
        return False
    return not (line.endswith(".") and len(words) > 6)


def _paragraph_starts(text):
    starts = []
    for match in re.finditer(r"\n[ \t]*\n+", text):
        position = match.end()
        if position < len(text) and text[position:].strip():
            starts.append(position)
    if len(starts) >= 1:
        return starts

    for offset, line in _lines(text):
        if line.strip() and offset > 0:
            starts.append(offset)
    return starts


def _spans_from_starts(text, starts):
    points = [0]
    for start in sorted(set(starts)):
        if 0 < start < len(text):
            points.append(start)
    points.append(len(text))
    return [(points[index], points[index + 1]) for index in range(len(points) - 1)]


def _merge_short_spans(text, spans):
    if len(spans) < 2:
        return spans

    merged = []
    index = 0
    while index < len(spans):
        current = spans[index]
        has_next = index + 1 < len(spans)
        if _is_short(text, current) and has_next and _merge_forward(text, current, spans[index + 1]):
            following = spans[index + 1]
            spans[index + 1] = (current[0], following[1])
            index += 1
            continue
        if _is_short(text, current) and merged and not _starts_marker(text, current):
            # A fragment without its own clause marker joins the clause before it.
            # A short numbered clause stays a clause of its own.
            previous_start = merged[-1][0]
            merged[-1] = (previous_start, current[1])
            index += 1
            continue
        merged.append(current)
        index += 1
    return merged


def _merge_forward(text, current, following):
    """Attach a bare heading to its own body, but keep a new clause intact.

    "1. Fees" followed by "1.1 ..." merges, because 1.1 belongs to clause 1.
    A title followed by "Article 1" does not, so a clause is never absorbed
    into the title block before it.
    """
    if not _starts_marker(text, following):
        return True
    if not _starts_marker(text, current):
        return False
    parent = _clause_number(_heading(text[current[0] : current[1]]))
    child = _clause_number(_heading(text[following[0] : following[1]]))
    return bool(parent and child and child.startswith(parent + "."))


_NUMBER_VALUE_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3})*)")


def _clause_number(heading):
    match = _NUMBER_VALUE_RE.match(heading or "")
    return match.group(1).rstrip(".") if match else ""


def _starts_marker(text, span):
    heading = _heading(text[span[0] : span[1]])
    if not heading:
        return False
    return bool(
        _NUMBER_RE.match(heading)
        or _STRUCTURAL_RE.match(heading)
        or _is_uppercase_heading(heading)
    )


def _is_short(text, span):
    return len(text[span[0] : span[1]].strip()) < SHORT_CLAUSE_CHARS


def _split_long_spans(text, spans):
    pieces = []
    for start, end in spans:
        pieces.extend(_split_span(text, start, end))
    return pieces


def _split_span(text, start, end):
    if end - start <= MAX_CLAUSE_CHARS:
        return [(start, end)]

    blank_breaks = []
    for match in re.finditer(r"\n[ \t]*\n+", text[start:end]):
        position = start + match.end()
        if start < position < end and text[position:end].strip():
            blank_breaks.append(position)
    if blank_breaks:
        points = [start, *blank_breaks, end]
        pieces = []
        for left, right in zip(points, points[1:]):
            pieces.extend(_split_span(text, left, right))
        return pieces

    sentence_breaks = []
    for match in _SENTENCE_BREAK_RE.finditer(text[start:end]):
        boundary = start + match.end()
        if start < boundary < end:
            sentence_breaks.append(boundary)
    if not sentence_breaks:
        return [(start, end)]

    pieces = []
    cursor = start
    for boundary in sentence_breaks:
        if boundary - cursor >= MAX_CLAUSE_CHARS and boundary > cursor:
            pieces.append((cursor, boundary))
            cursor = boundary
    if cursor < end:
        pieces.append((cursor, end))
    return pieces or [(start, end)]


def _records(text, spans):
    records = []
    sequence = 1
    carried = ""
    for start, end in spans:
        clause_text = carried + text[start:end]
        carried = ""
        if not clause_text.strip():
            if records:
                records[-1]["text"] += clause_text
            else:
                carried = clause_text
            continue
        records.append(
            {
                "id": f"clause-{sequence:03d}",
                "heading": _heading(clause_text),
                "text": clause_text,
                "sequence": sequence,
                "start": start - len(clause_text) + (end - start),
            }
        )
        sequence += 1
    if carried and records:
        records[-1]["text"] += carried
    return records


def _heading(clause_text):
    for line in clause_text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:120]
    return ""
