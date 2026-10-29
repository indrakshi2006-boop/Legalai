import errors as Errors
import retry as Retry
import utils as Utils
import os as OS
import json as JS
import sys
from pathlib import Path
from tqdm import tqdm
import requests
import pymupdf as fitz  # PyMuPDF; the `fitz` alias module is deprecated
from chromadb.utils import embedding_functions
import chromadb
import segment

PLAYBOOK_FIELDS = ("id", "category", "title", "standard_text", "red_flags")

def pdf_to_text(url):
    try:
        response = requests.get(url, timeout=Utils.comparison_timeout_seconds())
        pdf_data = response.content
        document = fitz.open(stream=pdf_data, filetype="pdf")
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception:
        print("The regulation PDF could not be downloaded.")
        return ""

def read_contract_pdf(data, max_bytes=None, max_pages=None):
    """Read a contract PDF with PyMuPDF. Image-only files are rejected."""
    if max_bytes is None:
        max_bytes = Utils.max_upload_bytes()
    if max_pages is None:
        max_pages = Utils.max_pdf_pages()
    if data is None:
        return _pdf_error("No file was uploaded.")
    if isinstance(data, str):
        data = data.encode("utf-8")
    if not isinstance(data, (bytes, bytearray)):
        return _pdf_error("The upload could not be read.")
    data = bytes(data)
    if len(data) == 0:
        return _pdf_error("The uploaded file is empty.")
    if len(data) > max_bytes:
        return _pdf_error(
            f"This file is too large ({len(data)} bytes). The limit is {max_bytes} bytes."
        )
    if not data.startswith(b"%PDF"):
        return _pdf_error("Upload a PDF file. This file does not have a PDF header.")

    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception:
        return _pdf_error("This file could not be read as a PDF. It may be corrupt or incomplete.")

    try:
        if getattr(document, "needs_pass", False):
            return _pdf_error("This PDF is encrypted and cannot be read.")
        page_count = int(getattr(document, "page_count", 0) or 0)
        if page_count <= 0:
            return _pdf_error("This PDF has no pages.")
        if page_count > max_pages:
            return _pdf_error(
                f"This PDF has {page_count} pages. The limit is {max_pages} pages."
            )
        parts = []
        for index in range(page_count):
            parts.append(document.load_page(index).get_text() or "")
    except Exception:
        return _pdf_error("This PDF could not be read. It may be corrupt or incomplete.")
    finally:
        document.close()

    text = "\n".join(parts)
    if not text.strip():
        return _pdf_error(
            "This PDF has no extractable text. It appears to be image-only, "
            "and OCR is not available, so it was not accepted."
        )
    return {"ok": True, "text": text, "page_count": page_count, "pages": parts}


def _pdf_error(message):
    return {"ok": False, "error": message}


def chroma_collection_exists(name, persist_directory=Utils.DB_FOLDER):
    """Return whether a usable index is stored.

    playbook_clauses is usable only when its document count equals the
    number of validated playbook records. Any other collection is usable
    when it contains at least one document.
    """
    if not OS.path.isdir(persist_directory):
        return False
    client = chromadb.PersistentClient(path=persist_directory)
    names = []
    for item in client.list_collections():
        names.append(getattr(item, "name", item))
    if name not in names:
        return False
    count = client.get_collection(name).count()
    if name == Utils.PLAYBOOK_COLLECTION:
        expected = len(validate_playbook(load_playbook()))
        return count == expected
    return count > 0


def contract_clauses(text):
    """Segment a contract into clauses.

    The Act retrieval corpus still uses split_text_into_sections. Contract
    review uses these records instead of fixed-size chunks.
    """
    return segment.segment_contract(text)


def split_text_into_sections(text, min_chars_per_section):
    # Fixed-size chunks for the existing Act collection, not contract clauses.
    paragraphs = text.split('\n')
    sections = []
    current_section = ""
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if current_length + paragraph_length + 2 <= min_chars_per_section:  # +2 for the double newline
            current_section += paragraph + '\n\n'
            current_length += paragraph_length + 2  # +2 for the double newline
        else:
            if current_section:
                sections.append(current_section.strip())
            current_section = paragraph + '\n\n'
            current_length = paragraph_length + 2  # +2 for the double newline

    if current_section:  # Add the last section
        sections.append(current_section.strip())

    return sections

def embed_text_in_chromadb(text, document_name, document_description, persist_directory=Utils.DB_FOLDER):
    # Generate embedding for the text using OpenAI embeddings
    openai_ef = openai_embedding_function()    
    documents = split_text_into_sections(text, 1000)
    
    ids = [str(hash(d)) for d in documents]  # Generate unique IDs for each document chunk

    # Metadata for the documents
    metadata = {
        "name": document_name,
        "description": document_description
    }
    metadatas = [metadata] * len(documents)  # Duplicate metadata for each chunk
    
    client = chromadb.PersistentClient(path=persist_directory)
    collection_name = Utils.DOCUMENT_COLLECTION
    collection = client.get_or_create_collection(name=collection_name,embedding_function=openai_ef)

    # create ids from the current count
    count = collection.count()
    print(f"Collection already contains {count} documents")
    ids = [str(i) for i in range(count, count + len(documents))]

    # load the documents in batches of 100
    for i in tqdm(
        range(0, len(documents), 100), desc="Adding documents", unit_scale=100
    ):
        collection.add(
            ids=ids[i : i + 100],
            documents=documents[i : i + 100],
            metadatas=metadatas[i : i + 100],  # type: ignore
        )

    new_count = collection.count()
    print(f"Added {new_count - count} documents")
   
def load_playbook(path=Utils.PLAYBOOK_FILE):
    with open(path, "r", encoding="utf-8") as handle:
        data = JS.load(handle)
    if not isinstance(data, list):
        raise ValueError("Playbook JSON must be a list of clause records.")
    return data


def validate_playbook(records):
    if not isinstance(records, list) or not records:
        raise ValueError("Playbook must contain at least one clause.")

    seen_ids = set()
    found_categories = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Clause at index {index} is not an object.")
        for field in PLAYBOOK_FIELDS:
            if field not in record:
                raise ValueError(f"Clause at index {index} is missing {field}.")

        clause_id = record["id"]
        if not isinstance(clause_id, str) or not clause_id.strip():
            raise ValueError(f"Clause at index {index} has an empty id.")
        if clause_id in seen_ids:
            raise ValueError(f"Duplicate clause id: {clause_id}")
        seen_ids.add(clause_id)

        for field in ("category", "title", "standard_text"):
            value = record[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Clause {clause_id} has an empty {field}.")
        if record["category"] not in Utils.PLAYBOOK_CATEGORIES:
            raise ValueError(f"Clause {clause_id} has an unknown category.")
        found_categories.add(record["category"])

        flags = record["red_flags"]
        if not isinstance(flags, list) or not flags:
            raise ValueError(f"Clause {clause_id} needs at least one red flag.")
        if any(not isinstance(flag, str) or not flag.strip() for flag in flags):
            raise ValueError(f"Clause {clause_id} has an empty red flag.")

    missing = [name for name in Utils.PLAYBOOK_CATEGORIES if name not in found_categories]
    if missing:
        raise ValueError("Playbook is missing categories: " + ", ".join(missing))
    return records


class IndexIncomplete(RuntimeError):
    """The stored playbook collection does not hold exactly the validated records."""


def chroma_client(persist_directory=Utils.DB_FOLDER):
    return chromadb.PersistentClient(path=persist_directory)


def _collection_names(client):
    return [getattr(item, "name", item) for item in client.list_collections()]


def drop_collection_if_incomplete(name, expected_count, persist_directory=Utils.DB_FOLDER, client=None):
    """Delete a playbook collection that is empty or partial. Returns True if deleted."""
    if client is None:
        if not OS.path.isdir(persist_directory):
            return False
        client = chroma_client(persist_directory)
    if name not in _collection_names(client):
        return False
    if client.get_collection(name).count() == expected_count:
        return False
    client.delete_collection(name)
    return True


def build_playbook_index(
    playbook_path=Utils.PLAYBOOK_FILE,
    persist_directory=Utils.DB_FOLDER,
    embedding_function=None,
    policy=None,
):
    """Store the reference clauses in their own Chroma collection, all or nothing.

    The embeddings are requested first, with no Chroma write. Only when every
    vector is back are the records written. If the request or the write fails,
    an empty or partial playbook collection is deleted, so the app never
    treats it as an index; a complete earlier index is left as it was when
    the failure happened before any write. Re-running updates the same ids,
    removes ids no longer in the playbook, and leaves other collections alone.

    Pass embedding_function only for a local check. The default uses
    Utils.EMBEDDING_MODEL and calls the OpenAI embeddings API once, not the chat model.
    """
    if Utils.PLAYBOOK_COLLECTION == Utils.DOCUMENT_COLLECTION:
        raise ValueError("Playbook collection must be separate from the document collection.")

    records = validate_playbook(load_playbook(playbook_path))
    expected = len(records)
    if embedding_function is None:
        embedding_function = openai_embedding_function()
    policy = policy or Retry.default_policy()
    ids = [record["id"] for record in records]
    documents = [record["standard_text"] for record in records]

    try:
        vectors = policy.call("embed", embedding_function, documents)
        if vectors is None or len(vectors) != expected:
            raise IndexIncomplete("The embeddings response did not cover every playbook clause.")
    except Exception:
        drop_collection_if_incomplete(Utils.PLAYBOOK_COLLECTION, expected, persist_directory)
        raise

    client = chroma_client(persist_directory)
    try:
        import providers

        metadata = providers.expected_metadata(providers.settings(providers.OPENAI), _sha256_file(playbook_path))
        metadata["dimension"] = len(vectors[0]) if len(vectors) else 0
        collection = client.get_or_create_collection(
            name=Utils.PLAYBOOK_COLLECTION,
            embedding_function=embedding_function,
            metadata=metadata,
        )
        if dict(collection.metadata or {}) != metadata:
            collection.modify(metadata=metadata)
        stale = [item for item in collection.get(include=[])["ids"] if item not in set(ids)]
        if stale:
            collection.delete(ids=stale)
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=[list(map(float, vector)) for vector in vectors],
            metadatas=[
                {
                    "clause_id": record["id"],
                    "category": record["category"],
                    "title": record["title"],
                    "red_flags": " | ".join(record["red_flags"]),
                }
                for record in records
            ],
        )
        stored = set(collection.get(include=[])["ids"])
        if collection.count() != expected or stored != set(ids):
            raise IndexIncomplete("The playbook collection does not hold exactly the validated clauses.")
    except Exception:
        if Utils.PLAYBOOK_COLLECTION in _collection_names(client):
            client.delete_collection(Utils.PLAYBOOK_COLLECTION)
        raise
    return collection.count()


def _sha256_file(path):
    import hashlib

    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def inspect_scores(pdf_path, out=print):
    """Embeddings only, no chat: print the top playbook matches for each clause.

    One embedding request per reviewable clause (the preamble is skipped). For
    Gemini, the scores are saved (ids and numbers only, no contract text) so
    the chat gate can open once GEMINI_MIN_SIMILARITY is set.
    """
    import datetime as DT
    import statistics

    import providers
    import segment

    config = providers.settings()
    with open(pdf_path, "rb") as handle:
        data = handle.read()
    extracted = read_contract_pdf(data)
    if not extracted["ok"]:
        raise SystemExit(f"Inspect stopped: {extracted['error']}")
    clauses = segment.reviewable(contract_clauses(extracted["text"]))
    providers.require_ready_index(config)
    policy = Retry.CallPolicy(provider=config.label, budget=Retry.CallBudget(embed=len(clauses), chat=0))
    if config.name == providers.GEMINI:
        retriever = providers.gemini_retriever_factory(policy)
    else:
        import analyze

        retriever = analyze.playbook_retriever
    out(
        f"Inspect scores - provider {config.name}, model {config.embedding_model}, "
        f"{len(clauses)} embedding queries, 0 chat calls"
    )
    report = []
    for clause in clauses:
        hits = policy.call("embed", retriever, clause["text"], category=None, k=5)
        out(f"{clause['id']}  {clause['heading'][:60]}")
        for hit in hits:
            out(f"    {hit['similarity']:.4f}  {hit['evidence_id']:<18} {hit['category']}")
        report.append(
            {
                "clause_id": clause["id"],
                "top": [
                    {"evidence_id": hit["evidence_id"], "category": hit["category"], "similarity": hit["similarity"]}
                    for hit in hits
                ],
            }
        )
    tops = [item["top"][0]["similarity"] for item in report if item["top"]]
    if tops:
        out(
            f"Top-1 similarity: min {min(tops):.4f}, median {statistics.median(tops):.4f}, max {max(tops):.4f} "
            f"over {len(tops)} clauses"
        )
    out("These scores come from one synthetic sample. A cutoff chosen from them is not validated.")
    if config.name == providers.GEMINI:
        providers.save_gemini_inspection(
            {
                "embedding_model": providers.GEMINI_EMBEDDING_MODEL,
                "dimension": providers.GEMINI_DIMENSION,
                "playbook_sha256": providers.playbook_sha256(),
                "pdf_sha256": __import__("hashlib").sha256(data).hexdigest(),
                "pdf_name": os_path_basename(pdf_path),
                "created_utc": DT.datetime.now(DT.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "clauses": report,
            }
        )
        out(f"Saved {providers.inspection_path().name}. Set GEMINI_MIN_SIMILARITY in .env to allow Gemini chat calls.")
    return report


def os_path_basename(path):
    return OS.path.basename(str(path))


def load_project_env():
    """Load .env from the project root. See utils.load_project_env."""
    Utils.load_project_env()

def openai_embedding_function():
    """Embedding client for indexing. The key stays in the environment."""
    api_key = OS.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    function = embedding_functions.OpenAIEmbeddingFunction(
        api_key_env_var="OPENAI_API_KEY",  # read from the environment, not stored in the collection
        model_name=Utils.EMBEDDING_MODEL,
    )
    function.client.timeout = Utils.comparison_timeout_seconds()
    # The SDK would otherwise retry every 429 twice, including no-credit errors.
    function.client.max_retries = 0
    return function


def _openai_error_code(exc):
    """Return a short provider error code, never the message or a key."""
    failure = Errors.classify(exc)
    if failure is not None and failure.code:
        return failure.code
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = Errors.safe_code(getattr(current, "code", ""))
        if code:
            return code
        current = current.__cause__ or current.__context__
    return ""


USAGE = (
    "Usage:\n"
    "  python embed.py playbook                      build the playbook index (1 OpenAI request, or 30 Gemini requests)\n"
    "  python embed.py playbook --dry-run            show the indexing plan, no call\n"
    "  python embed.py playbook --preflight [--pdf PATH]   offline checks and call plan\n"
    "  python embed.py --inspect-scores --pdf PATH   embeddings-only similarity scores (no chat)\n"
    "  python embed.py regulation [--rebuild]        download the EU AI Act and embed it with OpenAI (many requests)"
)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    load_project_env()
    try:
        if arguments == ["playbook", "--dry-run"]:
            records = validate_playbook(load_playbook())
            characters = sum(len(record["standard_text"]) for record in records)
            print(
                f"dry-run: 1 embeddings request, model {Utils.EMBEDDING_MODEL}, "
                f"documents {len(records)}, characters {characters}, chat calls 0"
            )
            print("dry-run: no API call and no Chroma write")
            return
        if arguments[:2] == ["playbook", "--preflight"]:
            rest = arguments[2:]
            pdf_path = None
            if rest[:1] == ["--pdf"] and len(rest) == 2:
                pdf_path = rest[1]
            elif rest:
                raise SystemExit(USAGE)
            import preflight

            ok = preflight.run(pdf_path)
            raise SystemExit(0 if ok else 1)
        if arguments[:1] == ["--inspect-scores"] or arguments[:2] == ["playbook", "--inspect-scores"]:
            rest = arguments[1:] if arguments[0] == "--inspect-scores" else arguments[2:]
            if rest[:1] != ["--pdf"] or len(rest) != 2:
                raise SystemExit(USAGE)
            inspect_scores(rest[1])
            return
        if arguments == ["playbook"]:
            import providers

            config = providers.settings()
            if config.name == providers.GEMINI:
                count = providers.build_gemini_index()
            else:
                # Exactly one embeddings request: a retry would go over this budget and stop.
                count = build_playbook_index(policy=Retry.CallPolicy(budget=Retry.CallBudget(embed=1)))
            print(
                f"Playbook index ready in '{config.collection}' "
                f"({count} clauses, provider {config.name}, model {config.embedding_model})."
            )
            return
        if arguments[:1] != ["regulation"] or len(arguments) > 2 or arguments[1:] not in ([], ["--rebuild"]):
            raise SystemExit(USAGE)
        import providers

        if providers.selected() != providers.OPENAI:
            raise SystemExit("The EU regulation index uses OpenAI only. Set AEGIS_PROVIDER=openai to build it.")
        if chroma_collection_exists(Utils.DOCUMENT_COLLECTION) and arguments[1:] != ["--rebuild"]:
            raise SystemExit(
                f"'{Utils.DOCUMENT_COLLECTION}' already has documents. Adding again would duplicate them. "
                "Use `python embed.py regulation --rebuild` to replace it."
            )
        if arguments[1:] == ["--rebuild"] and OS.path.isdir(Utils.DB_FOLDER):
            client = chroma_client()
            if Utils.DOCUMENT_COLLECTION in _collection_names(client):
                client.delete_collection(Utils.DOCUMENT_COLLECTION)

        document_name = "Artificial Intelligence Act"
        document_description = "Artificial Intelligence Act"
        text = pdf_to_text(Utils.EUROPEAN_ACT_URL)
        if not text:
            raise RuntimeError("The regulation PDF could not be downloaded.")
        embed_text_in_chromadb(text, document_name, document_description)
    except SystemExit:
        raise
    except Exception as exc:
        code = _openai_error_code(exc)
        suffix = f": {code}" if code else ""
        message = f"Indexing stopped ({type(exc).__name__}{suffix})."
        failure = Errors.classify(exc)
        if failure is not None:
            message += " " + Errors.user_message(failure)
        raise SystemExit(message) from None


if __name__ == "__main__":
    main()

