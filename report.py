"""Build review rows and downloads from real analysis results."""

import csv
import datetime as DT
import html
import io

import analyze
import errors as Errors
import risk
import segment
import utils as Utils

LABELS = ("Safe", "Caution", "Risky", "Review required")
# Prototype score weights: Safe=100, Caution=60, Risky=20. Review-required
# clauses are excluded because their risk was not assessed, and are reported separately.
HEALTH_POINTS = {"Safe": 100, "Caution": 60, "Risky": 20}
RUN_FIELDS = ("provider", "embedding_model", "chat_model")
PREAMBLE_STATUS = "Preamble - not reviewed"
PREAMBLE_NOTE = (
    "This block is a title, banner or parties section before the first clause. "
    "It was not compared with the playbook and has no risk label."
)
PRELIMINARY_NOTICE = (
    "Preliminary review - not legal advice. Labels are project rule outputs for wording "
    "differences, not legal conclusions, and have not been evaluated for legal accuracy."
)


def live_pipeline(config=None, policy=None):
    """Retriever, comparer and tuning for the selected provider. No fallback.

    The index is checked (count, provider, model, dimension, playbook SHA-256)
    before any search, and Gemini chat is refused until its cutoff gate passes.
    """
    import providers

    config = config or providers.settings()
    allowed, reason = providers.chat_allowed(config)
    if not allowed:
        raise providers.ProviderNotReady(reason)
    providers.require_ready_index(config)
    if config.name == providers.GEMINI:
        retriever = providers.gemini_retriever_factory(policy)
        comparer = providers.gemini_comparer(policy)
    else:
        retriever = analyze.playbook_retriever
        comparer = analyze.compare_with_model
    return {
        "retriever": retriever,
        "comparer": comparer,
        "tuning": analyze.tuning_for(config),
        "run_info": config.run_info(),
    }


def run_contract_review(
    text, retriever=None, comparer=None, on_progress=None, policy=None, run_info=None, tuning=None, on_clause=None
):
    """Segment, compare, and score one contract. Counts come only from these rows.

    A failure that every later call would repeat (rejected key, no credit, call
    limit reached) stops the remaining clauses. They are kept as rows that were
    not compared, so the results already produced are not lost.
    """
    clauses = analyze_ready_clauses(text)
    total = len(segment.reviewable(clauses))
    analyses = []
    stopped = None
    position = 0
    for index, clause in enumerate(clauses, start=1):
        if clause.get("kind") == segment.PREAMBLE:
            analyses.append(None)
            continue
        position += 1
        if stopped is not None:
            analyses.append(_not_compared(clause, index - 1, stopped))
            continue
        if on_progress is not None:
            on_progress(position - 1, total, f"Comparing clause {position} of {total}")
        analysis = analyze.analyze_clause(
            clause,
            index - 1,
            retriever or analyze.playbook_retriever,
            comparer or analyze.compare_with_model,
            policy,
            tuning,
        )
        analyses.append(analysis)
        if on_clause is not None:
            on_clause(position, total, analysis)
        if analysis.get("error_kind") in Errors.FATAL:
            stopped = analysis["clause_id"]
    if on_progress is not None:
        on_progress(total, total, "Scoring risk labels")
    scores = [None if item is None else risk.score_analysis(item) for item in analyses]
    return build_review_rows(clauses, analyses, scores, run_info)


def _not_compared(clause, index, stopped_at):
    clause_id, contract_text = analyze._clause_identity(clause, index)
    warning = (
        f"The review stopped after {stopped_at} because of an error that retrying would repeat, "
        "so no comparison was made for this clause."
    )
    return analyze.validate_analysis(
        analyze._empty_result(clause_id, contract_text, "", [warning], [], error_kind="stopped")
    )


def analyze_ready_clauses(text):
    import embed

    clauses = embed.contract_clauses(text)
    if not clauses:
        raise ValueError("No contract clauses were extracted from this PDF.")
    return clauses


def build_review_rows(clauses, analyses, scores, run_info=None):
    info = dict(Utils.run_info() if run_info is None else run_info)
    rows = []
    for clause, analysis, score in zip(clauses, analyses, scores):
        if analysis is None:
            rows.append(_preamble_row(clause, info))
            continue
        warnings = [str(item) for item in (analysis.get("warnings") or []) if str(item).strip()]
        if score.get("warning"):
            warnings.append(score["warning"])
        rows.append(
            {
                "clause_id": clause.get("id") or analysis.get("clause_id") or "",
                "heading": clause.get("heading") or "",
                "contract_text": analysis.get("contract_text") or clause.get("text") or "",
                "category": analysis.get("category") or "",
                "label": score.get("label") or "",
                "explanation": analysis.get("explanation") or "",
                "deviations": [item for item in (analysis.get("deviations") or []) if isinstance(item, str)],
                "reference_clause": analysis.get("reference_clause") or "",
                "evidence_id": analysis.get("evidence_id") or "",
                "warnings": warnings,
                "deciding_rule": score.get("deciding_rule") or "",
                "rule_reasons": [item.get("reason") or "" for item in (score.get("rules") or [])],
                "severity": analysis.get("severity") or "",
                "disclaimer": score.get("disclaimer") or risk.DISCLAIMER,
                "kind": segment.CLAUSE,
                "start": clause.get("start"),
                "error_kind": analysis.get("error_kind") or "",
                "reference_title": _chosen(analysis).get("title") or "",
                "similarity": _chosen(analysis).get("similarity"),
                "retrieved": [
                    {
                        "evidence_id": hit.get("evidence_id") or "",
                        "title": hit.get("title") or "",
                        "category": hit.get("category") or "",
                        "similarity": hit.get("similarity"),
                    }
                    for hit in (analysis.get("retrieved") or [])
                ],
                **{field: str(info.get(field) or "") for field in RUN_FIELDS},
            }
        )
    return rows


def _chosen(analysis):
    evidence_id = analysis.get("evidence_id") or ""
    for hit in analysis.get("retrieved") or []:
        if hit.get("evidence_id") == evidence_id and evidence_id:
            return hit
    return {}


def _preamble_row(clause, info):
    return {
        "clause_id": clause.get("id") or "",
        "heading": clause.get("heading") or "",
        "contract_text": clause.get("text") or "",
        "category": "",
        "label": "",
        "explanation": "",
        "deviations": [],
        "reference_clause": "",
        "evidence_id": "",
        "warnings": [],
        "deciding_rule": "",
        "rule_reasons": [],
        "severity": "",
        "disclaimer": "",
        "kind": segment.PREAMBLE,
        "start": clause.get("start"),
        "error_kind": "",
        "reference_title": "",
        "similarity": None,
        "retrieved": [],
        **{field: str(info.get(field) or "") for field in RUN_FIELDS},
    }


def is_preamble(row):
    return row.get("kind") == segment.PREAMBLE


def reviewed_rows(rows):
    return [row for row in rows if not is_preamble(row)]


def count_labels(rows):
    """Counts over reviewed clauses only. A preamble has no label and is not counted."""
    counts = {label: 0 for label in LABELS}
    for row in reviewed_rows(rows):
        label = row.get("label")
        if label not in counts:
            label = "Review required"
        counts[label] += 1
    return counts


def health_score(rows):
    """Return a transparent prototype score, or None when nothing was scored.

    Only validated Safe/Caution/Risky results enter the denominator. Clauses
    marked Review required are not silently treated as Safe.
    """
    counts = count_labels(rows)
    scored = sum(counts[label] for label in HEALTH_POINTS)
    if not scored:
        return None
    points = sum(counts[label] * HEALTH_POINTS[label] for label in HEALTH_POINTS)
    return round(points / scored)


def recommended_action(row):
    """Conservative, rule-based next step; does not create new legal analysis."""
    label = row.get("label") or "Review required"
    if label == "Risky":
        return "Prioritize review with qualified counsel; discuss narrowing or balancing the identified wording."
    if label == "Caution":
        return "Clarify the identified difference and consider aligning it with the playbook provision."
    if label == "Safe":
        return "No targeted issue was identified by the current rules; confirm the wording fits the business deal."
    return "Manually compare the full clause with the playbook; no risk conclusion was produced."


def priority_of(row):
    return {
        "Risky": "High",
        "Caution": "Medium",
        "Review required": "Manual review",
        "Safe": "Routine",
    }.get(row.get("label"), "Manual review")


def heatmap_rows(rows):
    """Project the same analyzed result rows into display-ready heatmap records."""
    ordered = sorted(
        reviewed_rows(rows),
        key=lambda row: (ui_risk_order(row.get("label")), _row_position(row)),
    )
    return [
        {
            "Clause": row.get("clause_id") or "",
            "Heading": row.get("heading") or "",
            "Risk tier": row.get("label") or "Review required",
            "Explanation": row.get("explanation") or "No comparison explanation was produced.",
            "Playbook deviation": "; ".join(row.get("deviations") or []) or ("No deviation recorded" if row.get("evidence_id") else "No matching playbook evidence"),
            "Recommended action": recommended_action(row),
            "Priority": priority_of(row),
        }
        for row in ordered
    ]


def ui_risk_order(label):
    return {"Risky": 0, "Caution": 1, "Review required": 2, "Safe": 3}.get(label, 2)


def _row_position(row):
    start = row.get("start")
    if isinstance(start, int):
        return start
    digits = "".join(char for char in str(row.get("clause_id") or "") if char.isdigit())
    return int(digits) if digits else 0


CSV_FIELDS = (
    "clause_id",
    "kind",
    "heading",
    "contract_text",
    "category",
    "label",
    "severity",
    "evidence_id",
    "reference_title",
    "similarity",
    "deviations",
    "explanation",
    "reference_clause",
    "warnings",
    "deciding_rule",
    "provider",
    "embedding_model",
    "chat_model",
    "generated_utc",
    "notice",
)


def _timestamp(generated_at=None):
    moment = generated_at or DT.datetime.now(DT.timezone.utc)
    return moment.astimezone(DT.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def to_csv(rows, generated_at=None):
    stamp = _timestamp(generated_at)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "clause_id": row.get("clause_id") or "",
                "kind": PREAMBLE_STATUS if is_preamble(row) else "Clause",
                "heading": row.get("heading") or "",
                "contract_text": row.get("contract_text") or "",
                "category": row.get("category") or "",
                "label": row.get("label") or "",
                "severity": row.get("severity") or "",
                "evidence_id": row.get("evidence_id") or "",
                "reference_title": row.get("reference_title") or "",
                "similarity": "" if row.get("similarity") is None else f"{float(row['similarity']):.4f}",
                "deviations": " | ".join(row.get("deviations") or []),
                "explanation": row.get("explanation") or "",
                "reference_clause": row.get("reference_clause") or "",
                "warnings": " | ".join(row.get("warnings") or []),
                "deciding_rule": row.get("deciding_rule") or "",
                "provider": row.get("provider") or "",
                "embedding_model": row.get("embedding_model") or "",
                "chat_model": row.get("chat_model") or "",
                "generated_utc": stamp,
                "notice": PRELIMINARY_NOTICE,
            }
        )
    return buffer.getvalue()


def _run_summary(rows):
    values = {}
    for field in RUN_FIELDS:
        found = sorted({str(row.get(field) or "") for row in rows if row.get(field)})
        values[field] = ", ".join(found) or "not recorded"
    return values


def to_html(rows, source_name="contract", generated_at=None):
    counts = count_labels(rows)
    run = _run_summary(rows)
    meta = (
        "<table class=\"meta\">"
        f"<tr><th>Provider</th><td>{html.escape(run['provider'])}</td></tr>"
        f"<tr><th>Embedding model</th><td>{html.escape(run['embedding_model'])}</td></tr>"
        f"<tr><th>Chat model</th><td>{html.escape(run['chat_model'])}</td></tr>"
        f"<tr><th>Generated</th><td>{html.escape(_timestamp(generated_at))}</td></tr>"
        "</table>"
    )
    summary = "".join(
        f"<li>{html.escape(label)}: {counts[label]}</li>" for label in LABELS
    )
    body = []
    for row in rows:
        if is_preamble(row):
            body.append(
                "<article class=\"preamble\">"
                f"<h2>{html.escape(PREAMBLE_STATUS)}: {html.escape(row.get('heading') or row.get('clause_id') or '')}</h2>"
                f"<p>{html.escape(PREAMBLE_NOTE)}</p>"
                f"<pre>{html.escape(row.get('contract_text') or '')}</pre>"
                "</article>"
            )
            continue
        deviations = "".join(f"<li>{html.escape(item)}</li>" for item in row.get("deviations") or [])
        warnings = "".join(f"<li>{html.escape(item)}</li>" for item in row.get("warnings") or [])
        reasons = "".join(f"<li>{html.escape(item)}</li>" for item in row.get("rule_reasons") or [])
        body.append(
            "<article>"
            f"<h2>{html.escape(row.get('heading') or row.get('clause_id') or '')}</h2>"
            f"<p><strong>Risk label:</strong> {html.escape(row.get('label') or '')}</p>"
            f"<p><strong>Category:</strong> {html.escape(row.get('category') or 'Not classified')}</p>"
            f"<p><strong>Evidence ID:</strong> {html.escape(row.get('evidence_id') or 'None')}</p>"
            f"<p><strong>Severity from the comparison:</strong> {html.escape(row.get('severity') or 'None')}</p>"
            f"<h3>Original text</h3><pre>{html.escape(row.get('contract_text') or '')}</pre>"
            f"<h3>Explanation</h3><p>{html.escape(row.get('explanation') or 'No comparison explanation was produced.')}</p>"
            f"<h3>Detected deviations</h3><ul>{deviations or '<li>None</li>'}</ul>"
            f"<h3>Retrieved reference clause</h3><pre>{html.escape(row.get('reference_clause') or 'No reference clause was retrieved.')}</pre>"
            f"<h3>Why this label</h3><ul>{reasons or '<li>None</li>'}</ul>"
            f"<h3>Warnings</h3><ul>{warnings or '<li>None</li>'}</ul>"
            "</article>"
        )
    disclaimer = html.escape(risk.DISCLAIMER)
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        f"<title>Contract review {html.escape(source_name)}</title></head><body>"
        f"<h1>Contract review: {html.escape(source_name)}</h1>"
        f"<p class=\"notice\"><strong>{html.escape(PRELIMINARY_NOTICE)}</strong></p>"
        f"{meta}<p>{disclaimer}</p><ul>{summary}</ul>"
        + "".join(body)
        + "</body></html>"
    )
