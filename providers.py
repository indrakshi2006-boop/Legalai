"""Model provider selection: OpenAI (default) or Gemini, chosen explicitly.

AEGIS_PROVIDER=openai|gemini selects the provider. The default is openai,
and the OpenAI path is unchanged: LangChain/Chroma with
text-embedding-ada-002 and the OPENAI_MODEL chat model at temperature 0.

There is no automatic fallback. If the selected provider fails, the run
fails; it never switches to the other provider, and each provider has its
own Chroma collection that the other never reads or writes.

Gemini facts used here (checked 2026-10-29):
- gemini-embedding-001 accepts task_type RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY
  and output_dimensionality; vectors below 3072 dimensions must be
  normalised by the caller (https://ai.google.dev/gemini-api/docs/embeddings,
  last updated 2026-10-26).
- gemini-3.8-flash is listed as a stable model code
  (https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash, last
  updated 2026-10-22). GEMINI_CHAT_MODEL overrides it.
- Google recommends keeping Gemini 3 temperature at its default of 1.0
  (Vertex AI "Gemini 3 prompting guide"), so this code does not set it.
  Gemini comparisons are therefore not guaranteed to repeat exactly.
"""

from dataclasses import dataclass, field
import hashlib
import json as JS
import math
import os as OS
from pathlib import Path
import time

import utils as Utils

OPENAI = "openai"
GEMINI = "gemini"
DEFAULT_PROVIDER = OPENAI
SUPPORTED = (OPENAI, GEMINI)

GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_DIMENSION = 768
GEMINI_DEFAULT_CHAT_MODEL = "gemini-3.8-flash"
GEMINI_COLLECTION = "playbook_clauses_gemini"
GEMINI_INSPECTION_FILE = "gemini_inspection.json"
GEMINI_INDEX_PAUSE_SECONDS = 0.5
GEMINI_BANNER = "Gemini API: use synthetic or non-confidential text only."
GEMINI_TERMS_URL = "https://ai.google.dev/gemini-api/terms"
GEMINI_TERMS_CHECKED = "2026-10-29"

OPENAI_MIN_SIMILARITY = 0.35
OPENAI_BROAD_K = 8
OPENAI_CATEGORY_SIMILARITY = 0.55
DEFAULT_BROAD_K = 8
TOP_K = 3

# Indirection so tests can avoid real pauses between Gemini index requests.
SLEEP = time.sleep


class ProviderNotReady(RuntimeError):
    """The selected provider cannot make this call yet. The message says why and what to do."""


@dataclass
class Settings:
    name: str
    label: str
    embedding_model: str
    chat_model: str
    chat_model_source: str
    key_name: str
    collection: str
    dimension: int | None
    min_similarity: float | None
    category_similarity: float | None
    broad_k: int
    top_k: int = TOP_K
    cutoff_state: str = "fixed"
    notes: list = field(default_factory=list)

    @property
    def key_present(self):
        return bool(OS.getenv(self.key_name, "").strip())

    def run_info(self):
        return {"provider": self.name, "embedding_model": self.embedding_model, "chat_model": self.chat_model}


def selected():
    """The configured provider name, lower-case. Unknown names are returned as-is."""
    raw = OS.getenv("AEGIS_PROVIDER", "").strip().lower()
    return raw or DEFAULT_PROVIDER


def _float_env(name):
    raw = OS.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if -1.0 <= value <= 1.0 else None


def _int_env(name, default, low, high):
    raw = OS.getenv(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(low, min(value, high))


def gemini_chat_model():
    configured = OS.getenv("GEMINI_CHAT_MODEL", "").strip()
    return configured or GEMINI_DEFAULT_CHAT_MODEL


def settings(name=None):
    """Settings for the selected (or named) provider. Raises ValueError for an unknown name."""
    name = name or selected()
    if name == OPENAI:
        return Settings(
            name=OPENAI,
            label="OpenAI",
            embedding_model=Utils.EMBEDDING_MODEL,
            chat_model=Utils.comparison_model_name(),
            chat_model_source="OPENAI_MODEL" if OS.getenv("OPENAI_MODEL", "").strip() else "default",
            key_name="OPENAI_API_KEY",
            collection=Utils.PLAYBOOK_COLLECTION,
            dimension=None,
            min_similarity=OPENAI_MIN_SIMILARITY,
            category_similarity=OPENAI_CATEGORY_SIMILARITY,
            broad_k=OPENAI_BROAD_K,
        )
    if name == GEMINI:
        cutoff = _float_env("GEMINI_MIN_SIMILARITY")
        inspected = gemini_inspection() is not None
        if cutoff is None:
            state = "uninspected" if not inspected else "inspected, cutoff not set"
        else:
            state = "set" if inspected else "set, but inspect-scores has not been run"
        return Settings(
            name=GEMINI,
            label="Gemini",
            embedding_model=GEMINI_EMBEDDING_MODEL,
            chat_model=gemini_chat_model(),
            chat_model_source="GEMINI_CHAT_MODEL" if OS.getenv("GEMINI_CHAT_MODEL", "").strip() else "default",
            key_name="GEMINI_API_KEY",
            collection=GEMINI_COLLECTION,
            dimension=GEMINI_DIMENSION,
            min_similarity=cutoff,
            category_similarity=cutoff,
            broad_k=_int_env("GEMINI_BROAD_K", DEFAULT_BROAD_K, TOP_K, 30),
            cutoff_state=state,
            notes=[
                GEMINI_BANNER,
                f"Data use depends on the Gemini API tier; see {GEMINI_TERMS_URL} (checked {GEMINI_TERMS_CHECKED}).",
                "The similarity cutoff is tuned on the synthetic sample only and is not validated.",
                "Gemini 3 temperature is left at its default (Google's guidance), so comparisons may vary between runs.",
            ],
        )
    raise ValueError(f"Unknown provider {name!r}. Set AEGIS_PROVIDER to openai or gemini.")


def chat_allowed(config):
    """Gemini chat stays refused until scores were inspected and a cutoff was set."""
    if config.name != GEMINI:
        return True, ""
    if gemini_inspection() is None:
        return False, (
            "Gemini chat calls are refused until the similarity scores are inspected. Run "
            "`python embed.py --inspect-scores --pdf PATH` (embeddings only), then set GEMINI_MIN_SIMILARITY in .env."
        )
    if config.min_similarity is None:
        return False, (
            "Gemini chat calls are refused until a similarity cutoff is set. Read the inspect-scores output, "
            "then set GEMINI_MIN_SIMILARITY in .env."
        )
    return True, ""


def playbook_sha256(path=None):
    return hashlib.sha256(Path(path or Utils.PLAYBOOK_FILE).read_bytes()).hexdigest()


def expected_metadata(config, digest=None):
    """Metadata a usable index must carry. A mismatch is refused before any search."""
    metadata = {
        "provider": config.name,
        "embedding_model": config.embedding_model,
        "playbook_sha256": digest or playbook_sha256(),
    }
    if config.dimension:
        metadata["dimension"] = config.dimension
    return metadata


def describe():
    """Offline description for preflight and the sidebar. Never includes a key value."""
    name = selected()
    source = "AEGIS_PROVIDER" if OS.getenv("AEGIS_PROVIDER", "").strip() else "default"
    try:
        config = settings(name)
    except ValueError:
        return {"supported": False, "provider": name, "provider_source": source}
    other = [GEMINI_COLLECTION] if config.name == OPENAI else [Utils.PLAYBOOK_COLLECTION]
    result = {
        "supported": True,
        "provider": config.name,
        "provider_source": source,
        "embedding_model": config.embedding_model,
        "dimension": config.dimension,
        "chat_model": config.chat_model,
        "chat_model_source": config.chat_model_source,
        "key_name": config.key_name,
        "key_present": config.key_present,
        "collection": config.collection,
        "other_collections": other,
        "index_requests": (lambda count: 1) if config.name == OPENAI else (lambda count: count),
        "expected_metadata": lambda digest: expected_metadata(config, digest),
        "notes": list(config.notes),
    }
    if config.name == GEMINI:
        allowed, reason = chat_allowed(config)
        cutoff = "not set" if config.min_similarity is None else f"{config.min_similarity:g}"
        result["cutoff_note"] = (
            f"{config.cutoff_state} (GEMINI_MIN_SIMILARITY {cutoff}); chat calls "
            + ("allowed" if allowed else "refused")
        )
    return result


# --- Gemini inspection record -------------------------------------------------


def inspection_path():
    return Path(Utils.DB_FOLDER) / GEMINI_INSPECTION_FILE


def gemini_inspection():
    """The saved inspect-scores record, if it matches the current playbook and model."""
    path = inspection_path()
    if not path.is_file():
        return None
    try:
        record = JS.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if record.get("embedding_model") != GEMINI_EMBEDDING_MODEL:
        return None
    if record.get("dimension") != GEMINI_DIMENSION:
        return None
    try:
        if record.get("playbook_sha256") != playbook_sha256():
            return None
    except OSError:
        return None
    return record


def save_gemini_inspection(record):
    path = inspection_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(JS.dumps(record, indent=2), encoding="utf-8")


# --- Gemini calls -------------------------------------------------------------


def _gemini_key():
    key = OS.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise ProviderNotReady("GEMINI_API_KEY is not set. Add it to .env, then restart.")
    return key


def gemini_client():
    """google-genai client. The SDK's own retries are off; retry.py decides."""
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=_gemini_key(),
        http_options=types.HttpOptions(
            timeout=int(Utils.comparison_timeout_seconds() * 1000),  # milliseconds
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def l2_normalise(values):
    vector = [float(value) for value in values]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0 or not math.isfinite(norm):
        raise ValueError("The embedding vector has no length and cannot be normalised.")
    return [value / norm for value in vector]


def gemini_embed(text, task_type, client=None):
    """One embed_content request for one text. Returns a 768-dimension unit vector."""
    from google.genai import types

    client = client or gemini_client()
    response = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=[text],
        config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=GEMINI_DIMENSION),
    )
    embeddings = getattr(response, "embeddings", None) or []
    if len(embeddings) != 1:
        raise ValueError("The Gemini embeddings response did not contain exactly one vector.")
    values = list(getattr(embeddings[0], "values", None) or [])
    if len(values) != GEMINI_DIMENSION:
        raise ValueError(f"The Gemini embedding has {len(values)} dimensions, expected {GEMINI_DIMENSION}.")
    return l2_normalise(values)


def gemini_embed_documents(texts, policy, client=None, pause=None):
    """One request per passage, with a short bounded pause between requests."""
    if pause is None:
        pause = GEMINI_INDEX_PAUSE_SECONDS
    pause = max(0.0, min(float(pause), 5.0))
    client = client or gemini_client()
    vectors = []
    for index, text in enumerate(texts):
        if index and pause:
            SLEEP(pause)
        vectors.append(policy.call("embed", gemini_embed, text, "RETRIEVAL_DOCUMENT", client))
    return vectors


def gemini_chat(system, user, want_json=True, client=None, config=None, require_inspection=True):
    """One generate_content request. Retrieval-grounded chats require inspection."""
    from google.genai import types

    config = config or settings(GEMINI)
    if config.name != GEMINI:
        raise ProviderNotReady("Gemini was not selected. Set AEGIS_PROVIDER=gemini in .env, then restart.")
    if require_inspection:
        allowed, reason = chat_allowed(config)
        if not allowed:
            raise ProviderNotReady(reason)
    client = client or gemini_client()
    options = {"system_instruction": system}
    if want_json:
        options["response_mime_type"] = "application/json"
    response = client.models.generate_content(
        model=config.chat_model,
        contents=user,
        config=types.GenerateContentConfig(**options),
    )
    return str(getattr(response, "text", "") or "")


def gemini_support_answer(system, user, client=None, config=None, policy=None):
    """Generate a support answer with Gemini; this local KB chat needs no vector cutoff."""
    import retry as Retry

    config = config or settings(GEMINI)
    if not config.key_present:
        raise ProviderNotReady("GEMINI_API_KEY is not set. Add it to .env, then restart.")
    policy = policy or Retry.CallPolicy(provider=config.label, max_retries=1)
    return policy.call(
        "chat", gemini_chat, system, user, False, client, config, False
    )


# --- Chroma: provider-specific collections -------------------------------------


class PrecomputedOnly:
    """Placeholder embedding function for provider collections.

    Vectors are always computed by the provider and passed in. If Chroma ever
    tried to embed text itself (for example with its default local model), this
    raises instead, so no other model can write to or query the collection.
    """

    def __init__(self):
        pass

    def __call__(self, input):
        raise RuntimeError("This collection only accepts precomputed provider vectors.")

    def embed_query(self, input):
        raise RuntimeError("This collection only accepts precomputed provider vectors.")

    @staticmethod
    def name():
        return "aegis-precomputed"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return PrecomputedOnly()

    def is_legacy(self):
        return False

    def default_space(self):
        return "cosine"

    def supported_spaces(self):
        return ["cosine"]


def index_status(config=None, persist_directory=None, digest=None):
    """Ready / Missing / Incomplete / Mismatch / Wrong provider, with a short detail.

    Reads local Chroma only. Never creates the Chroma folder.
    """
    import embed

    config = config or settings()
    persist_directory = persist_directory or Utils.DB_FOLDER
    expected = len(embed.validate_playbook(embed.load_playbook()))
    if not OS.path.isdir(persist_directory):
        return {"state": "Missing", "detail": "no index has been built", "count": 0, "expected": expected}
    client = embed.chroma_client(persist_directory)
    names = [getattr(item, "name", item) for item in client.list_collections()]
    if config.collection not in names:
        others = [name for name in (Utils.PLAYBOOK_COLLECTION, GEMINI_COLLECTION) if name in names and name != config.collection]
        if others:
            return {
                "state": "Wrong provider",
                "detail": f"only {', '.join(others)} exists, built for the other provider",
                "count": 0,
                "expected": expected,
            }
        return {"state": "Missing", "detail": "no index for this provider", "count": 0, "expected": expected}
    collection = _open(client, config)
    count = collection.count()
    if count != expected:
        return {"state": "Incomplete", "detail": f"{count}/{expected}", "count": count, "expected": expected}
    wanted = expected_metadata(config, digest)
    stored = dict(collection.metadata or {})
    differs = sorted(key for key, value in wanted.items() if str(stored.get(key, "")) != str(value))
    if differs:
        return {
            "state": "Mismatch",
            "detail": "metadata differs: " + ", ".join(differs),
            "count": count,
            "expected": expected,
        }
    return {"state": "Ready", "detail": f"{count}/{expected}", "count": count, "expected": expected}


def _open(client, config):
    if config.name == GEMINI:
        return client.get_collection(config.collection, embedding_function=PrecomputedOnly())
    return client.get_collection(config.collection)


def require_ready_index(config=None, persist_directory=None):
    """Refuse a missing, partial or mismatched index before any search."""
    config = config or settings()
    status = index_status(config, persist_directory)
    if status["state"] != "Ready":
        raise ProviderNotReady(
            f"The {config.label} playbook index is not ready ({status['state']}: {status['detail']}). "
            f"Run: {index_command(config)}" + (" (with AEGIS_PROVIDER=gemini in .env)" if config.name == GEMINI else "")
        )
    return status


def index_command(config=None):
    """The same command for every provider and shell; .env selects the provider."""
    return "python embed.py playbook"


def gemini_retriever_factory(policy=None, persist_directory=None, client=None):
    """Retriever over playbook_clauses_gemini. Each call is one Gemini embedding request."""
    import embed

    config = settings(GEMINI)
    require_ready_index(config, persist_directory)
    chroma = embed.chroma_client(persist_directory or Utils.DB_FOLDER)
    collection = _open(chroma, config)
    state = {"client": client}

    def retriever(text, category=None, k=TOP_K):
        if state["client"] is None:
            state["client"] = gemini_client()
        vector = gemini_embed(text, "RETRIEVAL_QUERY", state["client"])
        query = {"query_embeddings": [vector], "n_results": k, "include": ["documents", "metadatas", "distances"]}
        if category:
            query["where"] = {"category": category}
        found = collection.query(**query)
        hits = []
        for document, metadata, distance in zip(found["documents"][0], found["metadatas"][0], found["distances"][0]):
            metadata = metadata or {}
            evidence_id = str(metadata.get("clause_id") or "").strip()
            if not evidence_id:
                continue
            hits.append(
                {
                    "evidence_id": evidence_id,
                    "text": document,
                    "category": str(metadata.get("category") or ""),
                    "title": str(metadata.get("title") or ""),
                    "distance": float(distance),
                    # Cosine distance on unit vectors: similarity = 1 - distance = cosine.
                    "similarity": round(1.0 - float(distance), 6),
                }
            )
        return hits

    return retriever


def build_gemini_index(playbook_path=None, persist_directory=None, policy=None, client=None, pause=None):
    """Index the playbook in playbook_clauses_gemini only, all or nothing."""
    import embed
    import retry as Retry

    config = settings(GEMINI)
    persist_directory = persist_directory or Utils.DB_FOLDER
    playbook_path = playbook_path or Utils.PLAYBOOK_FILE
    records = embed.validate_playbook(embed.load_playbook(playbook_path))
    expected = len(records)
    # One request per passage. The default budget allows exactly that many, so a
    # retry that would go over it stops the build instead of adding calls.
    policy = policy or Retry.CallPolicy(provider="Gemini", budget=Retry.CallBudget(embed=expected))
    texts = [record["standard_text"] for record in records]
    try:
        vectors = gemini_embed_documents(texts, policy, client=client or gemini_client(), pause=pause)
    except Exception:
        embed.drop_collection_if_incomplete(GEMINI_COLLECTION, expected, persist_directory)
        raise
    if len(vectors) != expected or any(len(vector) != GEMINI_DIMENSION for vector in vectors):
        embed.drop_collection_if_incomplete(GEMINI_COLLECTION, expected, persist_directory)
        raise embed.IndexIncomplete("The Gemini embeddings did not cover every playbook clause.")

    chroma = embed.chroma_client(persist_directory)
    metadata = expected_metadata(config, playbook_sha256(playbook_path))
    try:
        names = [getattr(item, "name", item) for item in chroma.list_collections()]
        if GEMINI_COLLECTION in names:
            chroma.delete_collection(GEMINI_COLLECTION)
        collection = chroma.create_collection(
            GEMINI_COLLECTION,
            embedding_function=PrecomputedOnly(),
            metadata=metadata,
            configuration={"hnsw": {"space": "cosine"}},
        )
        collection.add(
            ids=[record["id"] for record in records],
            documents=texts,
            embeddings=vectors,
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
        if collection.count() != expected:
            raise embed.IndexIncomplete("The Gemini collection does not hold every playbook clause.")
    except Exception:
        names = [getattr(item, "name", item) for item in chroma.list_collections()]
        if GEMINI_COLLECTION in names:
            chroma.delete_collection(GEMINI_COLLECTION)
        raise
    return collection.count()


def gemini_comparer(policy=None):
    """Comparer with the same prompt, schema and validators as the OpenAI path."""
    import analyze

    config = settings(GEMINI)

    def comparer(contract_text, hits):
        payload = analyze.comparison_payload(contract_text, hits)
        text = gemini_chat(analyze.COMPARISON_SYSTEM_PROMPT, payload, want_json=True, config=config)
        return analyze._parse_model_json(text)

    return comparer


def gemini_answerer():
    """Contract Q&A answerer: same system prompt and passage format as OpenAI."""
    import agent

    config = settings(GEMINI)

    def answer(question, sources):
        messages = agent.qa_prompt_messages(question, sources)
        system = messages[0].content
        user = messages[1].content
        return gemini_chat(system, user, want_json=False, config=config)

    return answer


# --- Uniform interface -----------------------------------------------------------


def embed_documents(texts, config=None, policy=None):
    """Document vectors for indexing. OpenAI: one request for all texts.
    Gemini: one request per text (RETRIEVAL_DOCUMENT), 768-d, L2-normalised."""
    import retry as Retry

    config = config or settings()
    policy = policy or Retry.CallPolicy(provider=config.label)
    if config.name == GEMINI:
        return gemini_embed_documents(texts, policy)
    import embed

    function = embed.openai_embedding_function()
    return policy.call("embed", function, list(texts))


def embed_query(text, config=None):
    """One query vector (one request)."""
    config = config or settings()
    if config.name == GEMINI:
        return gemini_embed(text, "RETRIEVAL_QUERY")
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        openai_api_key=OS.getenv("OPENAI_API_KEY"),
        model=Utils.EMBEDDING_MODEL,
        timeout=Utils.comparison_timeout_seconds(),
        max_retries=0,
    ).embed_query(text)


def chat_json(system, user, config=None):
    """One chat request that returns JSON text."""
    config = config or settings()
    if config.name == GEMINI:
        return gemini_chat(system, user, want_json=True, config=config)
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        openai_api_key=OS.getenv("OPENAI_API_KEY"),
        model=config.chat_model,
        temperature=0,
        request_timeout=Utils.comparison_timeout_seconds(),
        max_retries=0,
    )
    message = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return str(getattr(message, "content", "") or "")
