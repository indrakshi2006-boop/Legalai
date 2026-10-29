"""Offline preflight for indexing and review.

`python embed.py playbook --preflight [--pdf PATH]` prints what a live run
would use and how many calls it would make, then exits 0 (PASS) or 1 (FAIL).
It makes no network call: it never builds a model client, never sends a
request, and never writes to Chroma. It reads the playbook, the .env key
presence (yes/no only, never the value), the local Chroma folder if it
exists, and the PDF if one is given.
"""

import hashlib
import json as JS
import os as OS
from pathlib import Path

import retry as Retry
import utils as Utils

EMBED_CALLS_PER_CLAUSE = 2  # one broad query, at most one category query
CHAT_CALLS_PER_CLAUSE = 1


def playbook_sha256(path=Utils.PLAYBOOK_FILE):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def planned_review_calls(reviewable_count):
    """Upper bound for one review. Retries count toward the same cap."""
    return {
        "embed": EMBED_CALLS_PER_CLAUSE * reviewable_count,
        "chat": CHAT_CALLS_PER_CLAUSE * reviewable_count,
    }


def provider_settings():
    """Provider, models and key presence. The key value is never returned."""
    import providers

    return providers.describe()


def collection_state(name, expected, persist_directory=None, expected_metadata=None):
    """Read-only look at a local collection. Does not create the Chroma folder."""
    persist_directory = persist_directory or Utils.DB_FOLDER
    if not OS.path.isdir(persist_directory):
        return {"state": "missing", "detail": f"no {Path(persist_directory).name}/ folder", "count": 0}
    import chromadb

    client = chromadb.PersistentClient(path=persist_directory)
    names = [getattr(item, "name", item) for item in client.list_collections()]
    if name not in names:
        return {"state": "missing", "detail": "collection not created", "count": 0}
    collection = client.get_collection(name)
    count = collection.count()
    metadata = dict(collection.metadata or {})
    if count != expected:
        return {"state": "incomplete", "detail": f"{count}/{expected} documents", "count": count, "metadata": metadata}
    mismatched = []
    for key, value in (expected_metadata or {}).items():
        if str(metadata.get(key, "")) != str(value):
            mismatched.append(key)
    if mismatched:
        return {
            "state": "mismatch",
            "detail": f"{count}/{expected} documents, metadata differs: {', '.join(sorted(mismatched))}",
            "count": count,
            "metadata": metadata,
        }
    return {"state": "ready", "detail": f"{count}/{expected} documents", "count": count, "metadata": metadata}


def run(pdf_path=None, out=print):
    """Print the preflight report. Returns True for PASS."""
    import embed
    import segment

    failures = []
    out("AEGIS preflight - offline: no network call, no model client, no Chroma write")

    settings = provider_settings()
    if not settings["supported"]:
        out(f"Provider:            {settings['provider']} - FAIL: unknown provider (use openai or gemini)")
        out("Result: FAIL")
        return False
    out(f"Provider:            {settings['provider']} ({settings['provider_source']})")
    out(f"Embedding model:     {settings['embedding_model']}" + (f", {settings['dimension']} dimensions" if settings.get("dimension") else ""))
    out(f"Chat model:          {settings['chat_model']} ({settings['chat_model_source']})")
    out(f"API key present:     {'yes' if settings['key_present'] else 'no'} ({settings['key_name']})")
    if not settings["key_present"]:
        failures.append(f"{settings['key_name']} is not set in .env")
    out(
        f"Timeout / retries:   {Utils.comparison_timeout_seconds():g} s; up to {Retry.configured_max_retries()} retries "
        "for rate limit, 5xx and network errors only; SDK retries off"
    )
    for note in settings.get("notes", []):
        out(f"Note:                {note}")

    try:
        records = embed.validate_playbook(embed.load_playbook())
        digest = playbook_sha256()
        categories = len({record["category"] for record in records})
        out(f"Playbook:            PASS {len(records)} clauses, {categories} categories, sha256 {digest[:16]}...")
    except Exception as exc:
        records = []
        digest = ""
        out(f"Playbook:            FAIL {type(exc).__name__}: {exc}")
        failures.append("playbook is not valid")

    if records:
        index_calls = settings["index_requests"](len(records))
        characters = sum(len(record["standard_text"]) for record in records)
        out(
            f"Planned index calls: {index_calls} embeddings request(s) for {len(records)} passages "
            f"({characters} characters), 0 chat calls"
        )
        state = collection_state(
            settings["collection"], len(records), expected_metadata=settings["expected_metadata"](digest)
        )
        out(f"Collection:          {settings['collection']} - {state['state']} ({state['detail']})")
        if state["state"] == "mismatch":
            out("                     A review refuses this collection. Rebuild it with: python embed.py playbook")
        if settings.get("other_collections"):
            for other in settings["other_collections"]:
                other_state = collection_state(other, len(records))
                out(f"Other provider:      {other} - {other_state['state']} (not used by this provider)")
        cutoff = settings.get("cutoff_note")
        if cutoff:
            out(f"Similarity cutoff:   {cutoff}")

    if pdf_path:
        data = None
        try:
            data = Path(pdf_path).read_bytes()
        except OSError as exc:
            out(f"Contract PDF:        FAIL cannot read {pdf_path}: {type(exc).__name__}")
            failures.append("PDF could not be read")
        if data is not None:
            extracted = embed.read_contract_pdf(data)
            if not extracted["ok"]:
                out(f"Contract PDF:        FAIL {extracted['error']}")
                failures.append("PDF was rejected")
            else:
                blocks = embed.contract_clauses(extracted["text"])
                reviewable = segment.reviewable(blocks)
                preamble = len(blocks) - len(reviewable)
                calls = planned_review_calls(len(reviewable))
                out(
                    f"Contract PDF:        PASS {Path(pdf_path).name}, {extracted['page_count']} page(s), "
                    f"{len(blocks)} blocks: {len(reviewable)} clauses to review, {preamble} preamble (not reviewed)"
                )
                out(
                    f"Planned review calls: at most {calls['embed']} embedding queries and {calls['chat']} chat calls "
                    "(hard cap; retries count toward it)"
                )

    if failures:
        for reason in failures:
            out(f"FAIL reason:         {reason}")
        out("Result: FAIL")
        return False
    out("Result: PASS")
    return True


def report_json(pdf_path=None):
    """Same checks, returned as data (used by tests)."""
    lines = []
    ok = run(pdf_path, out=lines.append)
    return JS.dumps({"ok": ok, "lines": lines})
