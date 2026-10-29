"""AEGIS contract review - Streamlit app.

Flow: 1 Upload and validate (offline), 2 Review (sends text to the selected
provider only after consent), 3 Results (from session memory; reruns never
repeat calls). Optional contract Q&A and EU AI Act chat sit below.
"""

import datetime as DT
import hashlib
import html
import uuid
from pathlib import Path

import altair as ALT
import pandas as PD
import streamlit as ST

import agent as Agent
import embed
import errors as Errors
import providers
import report
import retry as Retry
import segment
import support_chat as SupportChat
import ui
import utils as Utils

PRODUCT = "AEGIS"
PURPOSE = "Compare the clauses of a contract PDF with a reference playbook and flag wording to look at."
NOT_ADVICE = "Preliminary review - not legal advice."


def load_project_env() -> None:
    """Load .env from the project root. See utils.load_project_env."""
    Utils.load_project_env()


load_project_env()


def openai_failure_message(exc: BaseException) -> str | None:
    """User text with the next action for a provider failure, or None."""
    failure = Errors.classify(exc)
    if failure is None:
        return None
    return Errors.user_message(failure)


def failure_message(exc, config=None):
    if isinstance(exc, providers.ProviderNotReady):
        return str(exc)
    failure = Errors.classify(exc, config.label if config else "OpenAI")
    if failure is not None:
        return Errors.user_message(failure)
    return "The request stopped before it finished. Try again; if it repeats, run the preflight check."


def qa_session_id():
    """One id for this browser session. Contract and regulation chats get separate suffixes."""
    current = str(ST.session_state.get("qa_session_id") or "").strip()
    if not current or current == "chat1":
        current = uuid.uuid4().hex
        ST.session_state["qa_session_id"] = current
    return current


def contract_passages(text):
    """Clause records for this upload. They stay in the chat object, not in Chroma."""
    return [
        {"id": clause.get("id") or "", "heading": clause.get("heading") or "", "text": clause.get("text") or ""}
        for clause in embed.contract_clauses(text)
    ]


def review_config_token():
    """Settings that change a review result. The uploaded bytes are combined with this later."""
    playbook = Path(Utils.PLAYBOOK_FILE)
    playbook_stamp = str(playbook.stat().st_mtime_ns) if playbook.exists() else "missing"
    try:
        config = providers.settings()
        parts = [config.name, config.chat_model, config.embedding_model, config.collection, str(config.min_similarity)]
    except ValueError:
        parts = [providers.selected()]
    return "|".join(parts + [str(Utils.comparison_timeout_seconds()), playbook_stamp])


def review_fingerprint(file_bytes, config_token):
    digest = hashlib.sha256(file_bytes).hexdigest()
    return hashlib.sha256(f"{digest}|{config_token}".encode("utf-8")).hexdigest()


def clause_card_html(row):
    """Static HTML of one clause card (used by the security tests). Text is escaped."""
    label = ui.label_of(row)
    deviations = "".join(f"<li>{html.escape(item)}</li>" for item in row.get("deviations") or []) or "<li>None</li>"
    warnings = "".join(f"<li>{html.escape(item)}</li>" for item in row.get("warnings") or []) or "<li>None</li>"
    reasons = "".join(f"<li>{html.escape(item)}</li>" for item in row.get("rule_reasons") or [] if item) or "<li>None</li>"
    return (
        "<section>"
        f"<p>{ui.badge_html(label)}</p>"
        f"<h3>{html.escape(row.get('heading') or row.get('clause_id') or 'Clause')}</h3>"
        f"<p><strong>Category:</strong> {html.escape(row.get('category') or 'Not classified')}</p>"
        f"<p><strong>Evidence ID:</strong> {html.escape(row.get('evidence_id') or 'None')}</p>"
        f"{ui.quote_html(row.get('contract_text') or '')}"
        f"<p>{html.escape(row.get('explanation') or 'No comparison explanation was produced.')}</p>"
        f"<ul>{deviations}</ul>"
        f"{ui.quote_html(row.get('reference_clause') or 'No reference clause was retrieved.')}"
        f"<ul>{reasons}</ul><ul>{warnings}</ul>"
        "</section>"
    )


# --- page sections -----------------------------------------------------------------


def render_header(config):
    ST.markdown(ui.BADGE_CSS, unsafe_allow_html=True)
    ST.title(PRODUCT)
    ST.markdown(PURPOSE)
    ST.warning(
        f"**{NOT_ADVICE}** Labels are rule outputs for wording differences, not legal conclusions, "
        "and the tool has not been evaluated for legal accuracy.",
        icon=":material/gavel:",
    )
    if config is not None and config.name == providers.GEMINI:
        ST.error(f"**{providers.GEMINI_BANNER}**", icon=":material/shield:")


def render_sidebar(config, status):
    with ST.sidebar:
        ST.header("Setup")
        if config is None:
            ST.error(f"Unknown provider `{providers.selected()}`. Set AEGIS_PROVIDER to openai or gemini in .env.")
            return
        ST.markdown(
            f"**Provider:** {config.label}  \n"
            f"**Embedding model:** {config.embedding_model}  \n"
            f"**Chat model:** {config.chat_model}"
        )
        key_text = "set" if config.key_present else "missing"
        ST.markdown(f"**{config.key_name}:** {key_text}")

        state = status["state"]
        if state == "Ready":
            ST.success(f"Playbook index: Ready {status['detail']}", icon=":material/check_circle:")
        elif state == "Wrong provider":
            ST.error(f"Playbook index: Wrong provider. {status['detail']}.", icon=":material/error:")
            ST.code(providers.index_command(config), language="bash", wrap_lines=True)
        else:
            ST.error(f"Playbook index: {state} ({status['detail']})", icon=":material/error:")
            ST.code(providers.index_command(config), language="bash", wrap_lines=True)
        if config.name == providers.GEMINI:
            allowed, _ = providers.chat_allowed(config)
            ST.markdown(f"**Similarity cutoff:** {config.cutoff_state}")
            if not allowed:
                ST.caption("Gemini chat calls are refused until the cutoff step is done (see step 2).")

        ST.subheader("Upload limits")
        ST.markdown(
            f"Text-based PDF, up to {Utils.max_upload_bytes() // (1024 * 1024)} MB and {Utils.max_pdf_pages()} pages. "
            "Scanned (image-only) PDFs are rejected because there is no OCR."
        )

        ST.subheader("Privacy")
        ST.markdown(
            "Uploading and checking a PDF happens on this computer. Contract text leaves it only when you:\n"
            f"- **start a review**: each clause is sent to {config.label} for embeddings "
            f"({config.embedding_model}) and comparison ({config.chat_model});\n"
            f"- **ask a question**: the question and the matching passages are sent to {config.label} "
            f"({config.chat_model}), only if a passage matches.\n\n"
            "The review does not write the contract to disk or into the playbook or regulation indexes. "
            "Results are kept in the app's memory for this browser session."
        )
        if config.name == providers.GEMINI:
            ST.caption(
                f"How Google uses API data depends on the tier; see the Gemini API terms "
                f"({providers.GEMINI_TERMS_URL}, checked {providers.GEMINI_TERMS_CHECKED})."
            )


def render_upload():
    ST.subheader("1  Upload and check the PDF", divider="gray")
    upload = ST.file_uploader("Contract PDF", type=["pdf"], help="Text-based PDF only.")
    if upload is None:
        ST.info("Choose a contract PDF to start. Nothing is sent anywhere at this step.", icon=":material/upload_file:")
        return None
    file_bytes = upload.getvalue()
    extracted = embed.read_contract_pdf(file_bytes, max_bytes=Utils.max_upload_bytes(), max_pages=Utils.max_pdf_pages())
    if not extracted["ok"]:
        ST.error(f"**{extracted['error']}** {ui.pdf_error_help(extracted['error'])}", icon=":material/error:")
        return None
    blocks = embed.contract_clauses(extracted["text"])
    reviewable = segment.reviewable(blocks)
    preamble = len(blocks) - len(reviewable)
    note = f" and {preamble} preamble block (title or parties, not reviewed)" if preamble == 1 else (
        f" and {preamble} preamble blocks (not reviewed)" if preamble else ""
    )
    ST.success(
        f"Read {upload.name}: {extracted['page_count']} page(s), {len(reviewable)} clauses to review{note}.",
        icon=":material/task_alt:",
    )
    return {"name": upload.name, "bytes": file_bytes, "text": extracted["text"], "blocks": blocks, "reviewable": reviewable}


def _start_review(fingerprint):
    ST.session_state["review_request"] = fingerprint


def render_review(config, status, document):
    ST.subheader("2  Review", divider="gray")
    fingerprint = review_fingerprint(document["bytes"], review_config_token())
    saved = ST.session_state.get("review")
    has_result = bool(saved and saved.get("fingerprint") == fingerprint)
    running = ST.session_state.get("review_request") == fingerprint

    count = len(document["reviewable"])
    limits = report_limits(count)
    blockers = review_blockers(config, status)
    ST.markdown(
        f"This review compares **{count} clauses** with the playbook. It makes at most "
        f"**{limits['embed']} embedding calls** and **{limits['chat']} chat calls** to {config.label}. "
        "Retries count toward those limits, so they are never exceeded."
    )
    for message, command in blockers:
        ST.error(message, icon=":material/block:")
        if command:
            ST.code(command, language="bash", wrap_lines=True)

    if has_result:
        ST.info(
            "This session already holds the review of this file with these settings. "
            "It is shown below and will not be run again.",
            icon=":material/history:",
        )
        return fingerprint

    consent = ST.checkbox(
        f"I agree to send this contract's text to {config.label} for this review.",
        key=f"consent-{fingerprint}",
        disabled=bool(blockers) or running,
    )
    ST.button(
        "Review contract",
        type="primary",
        disabled=bool(blockers) or not consent or running,
        on_click=_start_review,
        args=(fingerprint,),
    )
    if running:
        run_review(config, document, fingerprint)
        ST.session_state.pop("review_request", None)
        ST.rerun()
    return fingerprint


def report_limits(count):
    return {"embed": 2 * count, "chat": count}


def review_blockers(config, status):
    blockers = []
    if not config.key_present:
        blockers.append(
            (f"{config.key_name} is not set. Add `{config.key_name}=...` to the .env file in the project folder, then restart the app.", "")
        )
    if status["state"] != "Ready":
        blockers.append(
            (
                f"The {config.label} playbook index is not ready ({status['state']}: {status['detail']}). Build it from the project folder"
                + (", with AEGIS_PROVIDER=gemini in .env:" if config.name == providers.GEMINI else ":"),
                providers.index_command(config),
            )
        )
    allowed, reason = providers.chat_allowed(config)
    if not allowed:
        blockers.append((reason, ""))
    return blockers


def run_review(config, document, fingerprint):
    count = len(document["reviewable"])
    limits = report_limits(count)
    policy = Retry.CallPolicy(provider=config.label, budget=Retry.CallBudget(**limits))
    record = {"fingerprint": fingerprint, "name": document["name"], "status": "error", "message": "", "rows": []}
    with ST.status(f"Reviewing {count} clauses with {config.label}...", expanded=True) as box:
        progress = ST.progress(0.0, text=f"Clause 0 of {count}")
        log = ST.container()

        def on_progress(done, total, message):
            progress.progress(0.0 if not total else min(done / total, 1.0), text=message)

        def on_clause(position, total, analysis):
            if analysis.get("evidence_id"):
                log.markdown(f"✓ {analysis['clause_id']} compared")
            else:
                reason = (analysis.get("warnings") or ["not compared"])[-1]
                log.markdown(f"✕ {analysis['clause_id']} not compared: {reason}")

        try:
            pipeline = report.live_pipeline(config, policy)
            rows = report.run_contract_review(
                document["text"],
                retriever=pipeline["retriever"],
                comparer=pipeline["comparer"],
                policy=policy,
                tuning=pipeline["tuning"],
                run_info=pipeline["run_info"],
                on_progress=on_progress,
                on_clause=on_clause,
            )
        except Exception as exc:
            record["message"] = failure_message(exc, config)
            box.update(label="Review did not start", state="error")
        else:
            record.update(status="complete", rows=rows, finished=DT.datetime.now(DT.timezone.utc))
            stopped = ui.stop_reason(rows)
            record["message"] = stopped
            box.update(label="Review stopped early" if stopped else "Review complete", state="error" if stopped else "complete")
    ST.session_state["review"] = record


def render_results(fingerprint):
    saved = ST.session_state.get("review")
    if not saved or saved.get("fingerprint") != fingerprint:
        return
    ST.subheader("3  Results", divider="gray")
    if saved["status"] == "error":
        ST.error(f"**The review could not run.** {saved['message']}", icon=":material/error:")
        return
    rows = saved["rows"]
    if saved.get("message"):
        ST.error(f"**The review stopped early.** {saved['message']} Results for the clauses before that are kept.", icon=":material/error:")
    failed = ui.failed_rows(rows)
    if failed:
        with ST.expander(f"{len(failed)} clause(s) could not be compared - see why"):
            for row in failed:
                ST.markdown(f"**{row['clause_id']}** {ui.md_escape(row.get('heading') or '')}")
                for warning in row.get("warnings") or []:
                    ST.caption(warning)

    overview, clauses, annotated, downloads = ST.tabs(["Overview", "Clauses", "Annotated contract", "Download"])
    with overview:
        render_overview(rows)
    with clauses:
        render_clause_list(rows)
    with annotated:
        ST.caption(
            "The full contract with each clause marked by its label. Select a clause to see its finding. "
            "Streamlit cannot jump from here to a card in another tab, so each finding opens in place."
        )
        ST.html(ui.annotated_contract_html(rows))
    with downloads:
        render_downloads(saved["name"], rows, saved.get("finished"))


def render_overview(rows):
    counts = report.count_labels(rows)
    columns = ST.columns(4)
    for column, label in zip(columns, report.LABELS):
        column.metric(ui.label_text(label), counts[label])
    ST.markdown(f"**Overall:** {ui.overall_profile(rows)}")
    preamble = [row for row in rows if report.is_preamble(row)]
    if preamble:
        ST.caption(f"{ui.label_text(report.PREAMBLE_STATUS)}: {len(preamble)} block(s), not counted above.")
    frame = PD.DataFrame(
        {
            "Label": [ui.label_text(label) for label in report.LABELS],
            "Clauses": [counts[label] for label in report.LABELS],
            "order": list(range(len(report.LABELS))),
        }
    )
    colours = [ui.LABEL_STYLE[label][1] for label in report.LABELS]
    chart = (
        ALT.Chart(frame)
        .mark_bar()
        .encode(
            x=ALT.X("Clauses:Q", axis=ALT.Axis(tickMinStep=1, format="d")),
            y=ALT.Y("Label:N", sort=ALT.SortField("order"), title=None),
            color=ALT.Color("Label:N", scale=ALT.Scale(domain=list(frame["Label"]), range=colours), legend=None),
            tooltip=["Label", "Clauses"],
        )
        .properties(height=170)
    )
    ST.altair_chart(chart, width="stretch")
    first = rows[0] if rows else {}
    ST.caption(
        f"Provider {first.get('provider') or 'not recorded'}, embedding model {first.get('embedding_model') or 'not recorded'}, "
        f"chat model {first.get('chat_model') or 'not recorded'}."
    )


def render_clause_list(rows):
    reviewed = report.reviewed_rows(rows)
    categories = sorted({row.get("category") or "Not classified" for row in reviewed})
    left, middle, right = ST.columns([2, 2, 1])
    labels = left.multiselect("Label", list(report.LABELS), format_func=ui.label_text, key="filter_labels")
    chosen_categories = middle.multiselect("Category", categories, key="filter_categories")
    order = right.selectbox("Sort by", ["Position in contract", "Risk (highest first)"], key="sort_rows")
    query = ST.text_input("Search clause text", key="search_rows", placeholder="For example: notice, liability, invoice")
    shown = ui.sort_rows(
        ui.filter_rows(rows, labels, chosen_categories, query), "risk" if order.startswith("Risk") else "position"
    )
    ST.caption(f"Showing {len(shown)} of {len(reviewed)} reviewed clauses.")
    for row in shown:
        render_clause_card(row)
    for row in rows:
        if report.is_preamble(row):
            with ST.expander(f"{ui.label_text(report.PREAMBLE_STATUS)}: {ui.md_escape(row.get('heading') or row['clause_id'])}"):
                ST.caption(report.PREAMBLE_NOTE)
                ST.markdown(ui.quote_html(row.get("contract_text")), unsafe_allow_html=True)


def render_clause_card(row):
    label = row.get("label") or "Review required"
    title = f"{ui.label_text(label)}: {ui.md_escape(row.get('heading') or row['clause_id'])}"
    with ST.expander(title):
        ST.markdown(
            f"{ui.badge_html(label)} &nbsp; **Category:** {html.escape(row.get('category') or 'Not classified')} "
            f"&nbsp; **Clause:** {html.escape(row['clause_id'])}",  # badge and ids are escaped HTML
            unsafe_allow_html=True,
        )
        ST.markdown("**Contract text**")
        ST.markdown(ui.quote_html(row.get("contract_text")), unsafe_allow_html=True)
        if row.get("evidence_id"):
            ST.markdown(
                f"**Closest playbook clause:** {ui.md_escape(row['evidence_id'])} "
                f"{ui.md_escape(row.get('reference_title') or '')} (similarity {ui.similarity_text(row.get('similarity'))})"
            )
            ST.markdown(ui.quote_html(row.get("reference_clause")), unsafe_allow_html=True)
            ST.caption("Similarity only ranks playbook passages. It is not a risk score.")
        else:
            ST.markdown("**No playbook clause matched**, so this clause was not compared and needs a manual review.")
        if row.get("explanation"):
            ST.markdown("**In plain English**")
            ST.markdown(ui.plain_html(row["explanation"]), unsafe_allow_html=True)
        if row.get("deviations"):
            ST.markdown("**Differences found**")
            ST.markdown(ui.list_html(row["deviations"]), unsafe_allow_html=True)
        reasons = [item for item in row.get("rule_reasons") or [] if item]
        if reasons:
            ST.markdown("**Why this label**")
            ST.markdown(ui.list_html(reasons), unsafe_allow_html=True)
        if row.get("evidence_id"):
            ST.caption(f"Evidence id: {row['evidence_id']}. Comparison severity: {row.get('severity') or 'none'}.")
        warnings = row.get("warnings") or []
        if warnings:
            ST.markdown("**Warnings**")
            ST.markdown(ui.list_html(warnings), unsafe_allow_html=True)


def render_downloads(source_name, rows, finished):
    stem = Path(source_name).stem or "contract"
    stamp = (finished or DT.datetime.now(DT.timezone.utc)).strftime("%Y%m%d-%H%M")
    ST.markdown(
        "Both files include the provider, the models, the date and the preliminary-review notice. "
        "The contract text in the HTML file is escaped, so markup in a clause shows as text."
    )
    left, right = ST.columns(2)
    left.download_button(
        "Download CSV",
        data=report.to_csv(rows, generated_at=finished).encode("utf-8"),
        file_name=f"aegis-review-{stem}-{stamp}.csv",
        mime="text/csv",
    )
    right.download_button(
        "Download HTML",
        data=report.to_html(rows, source_name, generated_at=finished).encode("utf-8"),
        file_name=f"aegis-review-{stem}-{stamp}.html",
        mime="text/html",
    )


def render_contract_qa(config, document, fingerprint):
    ST.subheader("Ask about this contract")
    ST.markdown(
        "Matching is **lexical**: AEGIS looks for clauses that share words with your question. "
        f"Only when a passage matches are the question and those passages sent to {config.label} "
        f"({config.chat_model}) for an answer. Otherwise you get the refusal below and nothing is sent. "
        "This does not score clauses."
    )
    if not config.key_present:
        ST.info(f"Set {config.key_name} in .env to ask questions.", icon=":material/key:")
        return
    allowed, reason = providers.chat_allowed(config)
    if not allowed:
        ST.warning(reason, icon=":material/block:")
        return
    session_id = f"{qa_session_id()}:contract"
    cache_key = (session_id, fingerprint, config.name)
    if ST.session_state.get("contract_chat_key") != cache_key:
        answerer = providers.gemini_answerer() if config.name == providers.GEMINI else None
        ST.session_state["contract_chat"] = Agent.NewsChat(session_id, passages=contract_passages(document["text"]), answerer=answerer)
        ST.session_state["contract_chat_key"] = cache_key
        ST.session_state["contract_answers"] = []
    with ST.form("contract_question", clear_on_submit=True):
        question = ST.text_input("Your question", placeholder="For example: How many days does the customer have to pay?")
        asked = ST.form_submit_button("Ask")
    if asked and question.strip():
        try:
            result = ST.session_state["contract_chat"].ask(question)
        except Exception as exc:
            ST.error(failure_message(exc, config), icon=":material/error:")
        else:
            ST.session_state["contract_answers"].append({"question": question.strip(), **result})
    for item in reversed(ST.session_state.get("contract_answers") or []):
        render_answer(item)


def render_answer(item):
    with ST.container(border=True):
        ST.markdown("**Question**")
        ST.markdown(ui.plain_html(item["question"]), unsafe_allow_html=True)
        if item.get("supported"):
            ST.markdown("**Answer**")
            ST.markdown(ui.plain_html(item["answer"]), unsafe_allow_html=True)
        else:
            ST.markdown(f"**{Agent.UNSUPPORTED_ANSWER}**")
            ST.caption("No passage supports an answer, so none is stated as fact.")
        for source in item.get("sources") or []:
            ST.caption(f"Source passage {source.get('id') or ''}")
            ST.text(source.get("text") or "")
        if not item.get("sources"):
            ST.caption("No passage matched the words in the question, so no model was called.")


def render_regulation_chat(config):
    with ST.expander("EU AI Act questions (optional, separate from the contract)"):
        ST.markdown(
            "This answers questions about the EU Artificial Intelligence Act from a separate index. "
            "It does not read your contract and is not part of the review."
        )
        if config.name != providers.OPENAI:
            ST.info("The EU AI Act chat uses OpenAI only. Set AEGIS_PROVIDER=openai to use it.", icon=":material/info:")
            return
        if not embed.chroma_collection_exists(Utils.DOCUMENT_COLLECTION):
            ST.info("The EU AI Act index has not been built yet. To build it, run this from the project folder:", icon=":material/info:")
            ST.code("python embed.py regulation", language="bash")
            ST.caption(
                "This downloads the Act and sends all of its text to the OpenAI embeddings API in batches of 100 "
                "chunks. It can incur a charge. The contract review does not need it."
            )
            return
        if not config.key_present:
            ST.info("Set OPENAI_API_KEY in .env to use this chat.", icon=":material/key:")
            return
        session_id = f"{qa_session_id()}:regulation"
        if ST.session_state.get("regulation_chat_id") != session_id:
            ST.session_state["regulation_chat"] = Agent.NewsChat(session_id)
            ST.session_state["regulation_chat_id"] = session_id
            ST.session_state["regulation_answers"] = []
        with ST.form("regulation_question", clear_on_submit=True):
            question = ST.text_input("Question about the EU AI Act")
            asked = ST.form_submit_button("Ask about the Act")
        if asked and question.strip():
            try:
                result = ST.session_state["regulation_chat"].ask(question)
            except Exception as exc:
                ST.error(failure_message(exc, config), icon=":material/error:")
            else:
                ST.session_state["regulation_answers"].append({"question": question.strip(), **result})
        for item in reversed(ST.session_state.get("regulation_answers") or []):
            render_answer(item)


def render_support_chat(config):
    """Customer support workflow as a separate page in the AEGIS app."""
    ST.title("Customer Support Chat")
    if config is None or config.name != providers.GEMINI:
        ST.error("Customer Support Chat uses Gemini. Set AEGIS_PROVIDER=gemini in .env, then restart.")
        return
    ST.caption(
        f"Gemini ({config.chat_model}) answers from the sample support knowledge base. "
        "The free tier may use submitted content to improve Google products; use synthetic or non-confidential questions only."
    )
    if not config.key_present:
        ST.info("Set GEMINI_API_KEY in the project .env file, then restart the app.", icon=":material/key:")
        return
    history = ST.session_state.setdefault("support_chat_messages", [])
    for item in history:
        with ST.chat_message(item["role"]):
            if item.get("error"):
                ST.error(item["content"], icon=":material/error:")
            else:
                ST.markdown(item["content"])
    prompt = ST.chat_input("Ask about plans, billing, accounts, or technical issues")
    if prompt:
        previous_issue = ST.session_state.get("support_last_issue", "No previous issue found.")
        history = ST.session_state["support_chat_messages"]
        history.append({"role": "user", "content": prompt})
        try:
            result = SupportChat.respond(
                prompt,
                answerer=lambda system, user: providers.gemini_support_answer(system, user, config=config),
                previous_issue=previous_issue,
                history=history[:-1],
            )
        except Exception as exc:
            history.append({"role": "assistant", "content": failure_message(exc, config), "error": True})
        else:
            history.append({"role": "assistant", "content": result["response"]})
            if result["category"] != "Memory":
                ST.session_state["support_last_issue"] = prompt
        ST.rerun()


def main():
    ST.set_page_config(page_title="AEGIS LegalAI", page_icon=":material/gavel:", layout="wide")
    try:
        config = providers.settings()
    except ValueError:
        config = None
    with ST.sidebar:
        page = ST.radio("Page", ["Contract Review", "Customer Support Chat"], key="aegis_page")
    if page == "Customer Support Chat":
        render_support_chat(config)
        return
    render_header(config)
    status = providers.index_status(config) if config is not None else None
    render_sidebar(config, status)
    if config is None:
        ST.error("Choose a provider first: set AEGIS_PROVIDER to openai or gemini in .env, then restart.", icon=":material/error:")
        return
    document = render_upload()
    if document is None:
        render_regulation_chat(config)
        return
    fingerprint = render_review(config, status, document)
    render_results(fingerprint)
    ST.divider()
    render_contract_qa(config, document, fingerprint)
    render_regulation_chat(config)


if __name__ == "__main__":
    main()
