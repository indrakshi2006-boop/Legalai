"""Provider selection and the Gemini path, with a fake google-genai client.

No test here builds a real client or opens a connection: the OpenAI and
LangChain constructors are replaced with functions that raise, and the
Gemini client is a local fake that records each request.
"""

import csv
import hashlib
import io
import json
import math
import socket
from pathlib import Path
from types import SimpleNamespace

import chromadb
import openai
import pytest
from google.genai import errors as genai_errors

import agent
import analyze
import embed
import errors
import preflight
import providers
import report
import retry
import utils

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "docs" / "figures" / "synthetic-contract-v2.pdf"
GEMINI_SECRET = "AIzaSy-unit-test-secret-not-real"


class Touched(AssertionError):
    pass


def _refuse(*args, **kwargs):
    raise Touched("a client for the other provider was constructed")


class FakeModels:
    def __init__(self, owner):
        self.owner = owner

    def embed_content(self, model, contents, config):
        self.owner.calls.append(("embed", model, config.task_type, config.output_dimensionality, list(contents)))
        if self.owner.fail_embed:
            raise self.owner.fail_embed
        text = contents[0]
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [((digest[i % 32] + i) % 97 + 1) * 3.0 for i in range(config.output_dimensionality)]
        return SimpleNamespace(embeddings=[SimpleNamespace(values=values)])

    def generate_content(self, model, contents, config):
        self.owner.calls.append(("chat", model, config.response_mime_type, contents))
        if self.owner.fail_chat:
            raise self.owner.fail_chat
        return SimpleNamespace(text=self.owner.reply(contents))


class FakeGemini:
    def __init__(self, reply=None):
        self.calls = []
        self.fail_embed = None
        self.fail_chat = None
        self.models = FakeModels(self)
        self.reply = reply or self._first_evidence

    @staticmethod
    def _first_evidence(contents):
        payload = json.loads(contents)
        first = payload["retrieved"][0]["evidence_id"]
        return json.dumps(
            {"evidence_id": first, "deviations": ["Fake difference."], "explanation": "Fake Gemini reply.", "severity": "low"}
        )

    def count(self, kind):
        return sum(1 for call in self.calls if call[0] == kind)


@pytest.fixture
def gemini(monkeypatch, tmp_path):
    """Gemini selected, a fake client, an empty temporary Chroma folder, OpenAI clients forbidden."""
    fake = FakeGemini()
    monkeypatch.setenv("AEGIS_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_SECRET)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(providers, "gemini_client", lambda: fake)
    monkeypatch.setattr(providers, "SLEEP", lambda seconds: None)
    monkeypatch.setattr(utils, "DB_FOLDER", str(tmp_path / "chroma_storage"))
    for module, name in (
        (analyze, "ChatOpenAI"),
        (analyze, "OpenAIEmbeddings"),
        (agent, "ChatOpenAI"),
        (agent, "OpenAIEmbeddings"),
        (openai, "OpenAI"),
    ):
        monkeypatch.setattr(module, name, _refuse)
    monkeypatch.setattr(embed.embedding_functions, "OpenAIEmbeddingFunction", _refuse)
    yield fake
    chromadb.api.client.SharedSystemClient.clear_system_cache()


def _names():
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    client = chromadb.PersistentClient(path=utils.DB_FOLDER)
    return sorted(getattr(item, "name", item) for item in client.list_collections())


def _inspect_and_set_cutoff(monkeypatch, cutoff="0.10"):
    embed.inspect_scores(str(SAMPLE), out=lambda line: None)
    monkeypatch.setenv("GEMINI_MIN_SIMILARITY", cutoff)


# --- selection ------------------------------------------------------------------


def test_default_provider_is_openai_and_unchanged():
    config = providers.settings()
    assert config.name == "openai"
    assert config.embedding_model == "text-embedding-ada-002"
    assert config.chat_model == "gpt-4o"
    assert config.collection == "playbook_clauses"
    assert config.min_similarity == analyze.MIN_SIMILARITY == 0.35
    assert config.broad_k == analyze.BROAD_K == 8
    assert providers.chat_allowed(config) == (True, "")


def test_gemini_selection_and_settings(monkeypatch):
    monkeypatch.setenv("AEGIS_PROVIDER", "Gemini")
    config = providers.settings()
    assert config.name == "gemini"
    assert config.embedding_model == "gemini-embedding-001"
    assert config.dimension == 768
    assert config.collection == "playbook_clauses_gemini"
    assert config.chat_model == "gemini-3.8-flash"
    monkeypatch.setenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash")
    assert providers.settings().chat_model == "gemini-3.6-flash"


def test_support_chat_uses_gemini_without_openai_or_retrieval_cutoff(gemini):
    gemini.reply = lambda _contents: "Gemini support response."
    result = providers.gemini_support_answer(
        "Answer from the sample knowledge base.",
        '{"question":"How much is Pro?"}',
        client=gemini,
        policy=retry.CallPolicy(max_retries=0),
    )
    assert result == "Gemini support response."
    assert gemini.count("chat") == 1
    assert gemini.calls[-1][1:3] == ("gemini-3.8-flash", None)


def test_unknown_provider_is_refused(monkeypatch):
    monkeypatch.setenv("AEGIS_PROVIDER", "other")
    with pytest.raises(ValueError):
        providers.settings()
    assert providers.describe()["supported"] is False


def test_gemini_cutoff_starts_uninspected_and_chat_is_refused(gemini):
    config = providers.settings()
    assert config.cutoff_state == "uninspected"
    assert config.min_similarity is None
    allowed, reason = providers.chat_allowed(config)
    assert allowed is False
    assert "--inspect-scores" in reason
    with pytest.raises(providers.ProviderNotReady):
        providers.gemini_chat("system", "user")
    assert gemini.count("chat") == 0


# --- Gemini calls ----------------------------------------------------------------


def test_gemini_embeddings_are_768_unit_vectors_with_task_types(gemini):
    document = providers.gemini_embed("clause text", "RETRIEVAL_DOCUMENT", gemini)
    query = providers.gemini_embed("question", "RETRIEVAL_QUERY", gemini)
    for vector in (document, query):
        assert len(vector) == 768
        assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0, rel_tol=1e-9)
    assert [call[2] for call in gemini.calls] == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
    assert all(call[1] == "gemini-embedding-001" and call[3] == 768 for call in gemini.calls)


def test_wrong_dimension_is_rejected(gemini):
    gemini.models.embed_content = lambda model, contents, config: SimpleNamespace(
        embeddings=[SimpleNamespace(values=[1.0] * 3072)]
    )
    with pytest.raises(ValueError):
        providers.gemini_embed("x", "RETRIEVAL_QUERY", gemini)


def test_gemini_client_turns_sdk_retries_off(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_SECRET)
    monkeypatch.setenv("OPENAI_TIMEOUT", "12")
    captured = {}
    from google import genai

    monkeypatch.setattr(genai, "Client", lambda **kwargs: captured.update(kwargs) or "client")
    assert providers.gemini_client() == "client"
    options = captured["http_options"]
    assert options.retry_options.attempts == 1
    assert options.timeout == 12000


def test_missing_gemini_key_is_reported(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(providers.ProviderNotReady) as caught:
        providers.gemini_client()
    assert "GEMINI_API_KEY" in str(caught.value)


# --- indexing and collection separation ------------------------------------------


def test_gemini_index_never_exceeds_one_request_per_passage(gemini):
    calls = {"n": 0}
    real = gemini.models.embed_content

    def flaky(model, contents, config):
        calls["n"] += 1
        if calls["n"] in (3, 4):
            raise genai_errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE"}})
        return real(model, contents, config)

    gemini.models.embed_content = flaky
    with pytest.raises(errors.BudgetExceeded):
        providers.build_gemini_index()
    assert calls["n"] == 30, "two retries used budget, so the last passages were not requested"
    assert "playbook_clauses_gemini" not in _names()


def test_gemini_index_one_request_per_passage_in_its_own_collection(gemini):
    count = providers.build_gemini_index()
    assert count == 30
    assert gemini.count("embed") == 30
    assert all(call[2] == "RETRIEVAL_DOCUMENT" for call in gemini.calls)
    assert _names() == ["playbook_clauses_gemini"]
    status = providers.index_status()
    assert status["state"] == "Ready"
    client = chromadb.PersistentClient(path=utils.DB_FOLDER)
    collection = client.get_collection("playbook_clauses_gemini", embedding_function=providers.PrecomputedOnly())
    metadata = collection.metadata
    assert metadata["provider"] == "gemini"
    assert metadata["embedding_model"] == "gemini-embedding-001"
    assert metadata["dimension"] == 768
    assert metadata["playbook_sha256"] == providers.playbook_sha256()
    assert collection.configuration["hnsw"]["space"] == "cosine"
    assert sorted(collection.get(include=[])["ids"]) == sorted(r["id"] for r in embed.load_playbook())


def test_gemini_index_pauses_between_requests(gemini, monkeypatch):
    waits = []
    monkeypatch.setattr(providers, "SLEEP", waits.append)
    providers.build_gemini_index(pause=99)
    assert len(waits) == 29
    assert set(waits) == {5.0}, "the pause is bounded to 5 s"


def test_gemini_index_failure_leaves_no_collection(gemini):
    request_error = genai_errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE"}})
    gemini.fail_embed = request_error
    with pytest.raises(genai_errors.ServerError):
        providers.build_gemini_index()
    assert "playbook_clauses_gemini" not in _names()
    # A 503 is retried (3 times) inside the default budget of 30 requests in total.
    assert gemini.count("embed") == 4


def test_gemini_never_writes_to_the_openai_collection(gemini, monkeypatch):
    from test_index_failure import HashEmbedding

    monkeypatch.setenv("AEGIS_PROVIDER", "openai")
    embed.build_playbook_index(persist_directory=utils.DB_FOLDER, embedding_function=HashEmbedding())
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    client = chromadb.PersistentClient(path=utils.DB_FOLDER)
    before = client.get_collection("playbook_clauses").get(include=["embeddings"])
    monkeypatch.setenv("AEGIS_PROVIDER", "gemini")
    providers.build_gemini_index()
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    client = chromadb.PersistentClient(path=utils.DB_FOLDER)
    after = client.get_collection("playbook_clauses").get(include=["embeddings"])
    assert before["ids"] == after["ids"]
    assert [list(v) for v in before["embeddings"]] == [list(v) for v in after["embeddings"]]
    assert len(after["embeddings"][0]) == 8, "the OpenAI collection still holds its own vectors"


def test_provider_collections_refuse_text_embedding(gemini):
    providers.build_gemini_index()
    client = chromadb.PersistentClient(path=utils.DB_FOLDER)
    collection = client.get_collection("playbook_clauses_gemini", embedding_function=providers.PrecomputedOnly())
    with pytest.raises(Exception):
        collection.add(ids=["x"], documents=["embed me with something else"])
    with pytest.raises(Exception):
        collection.query(query_texts=["embed me"], n_results=1)


# --- mismatch refusal -------------------------------------------------------------


@pytest.mark.parametrize(
    "key,value",
    [("playbook_sha256", "0" * 64), ("embedding_model", "other-model"), ("dimension", 3072), ("provider", "openai")],
)
def test_metadata_mismatch_is_refused_before_any_search(gemini, key, value):
    providers.build_gemini_index()
    client = chromadb.PersistentClient(path=utils.DB_FOLDER)
    collection = client.get_collection("playbook_clauses_gemini", embedding_function=providers.PrecomputedOnly())
    metadata = dict(collection.metadata)
    metadata[key] = value
    collection.modify(metadata=metadata)
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    before = gemini.count("embed")
    assert providers.index_status()["state"] == "Mismatch"
    with pytest.raises(providers.ProviderNotReady) as caught:
        providers.gemini_retriever_factory()
    assert "Mismatch" in str(caught.value)
    assert gemini.count("embed") == before, "no query was embedded"


def test_wrong_provider_index_is_reported(gemini, monkeypatch):
    from test_index_failure import HashEmbedding

    embed.build_playbook_index(persist_directory=utils.DB_FOLDER, embedding_function=HashEmbedding())
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    assert providers.index_status()["state"] == "Wrong provider"
    monkeypatch.setenv("AEGIS_PROVIDER", "openai")
    assert providers.index_status()["state"] == "Ready"


def test_openai_index_records_provider_metadata(tmp_path, monkeypatch):
    from test_index_failure import HashEmbedding

    monkeypatch.setattr(utils, "DB_FOLDER", str(tmp_path))
    embed.build_playbook_index(persist_directory=str(tmp_path), embedding_function=HashEmbedding())
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    collection = chromadb.PersistentClient(path=str(tmp_path)).get_collection("playbook_clauses")
    assert collection.metadata["provider"] == "openai"
    assert collection.metadata["embedding_model"] == "text-embedding-ada-002"
    assert collection.metadata["playbook_sha256"] == providers.playbook_sha256()
    assert providers.index_status()["state"] == "Ready"
    chromadb.api.client.SharedSystemClient.clear_system_cache()


# --- the review through the Gemini pipeline ---------------------------------------


def test_gemini_review_after_inspection_uses_only_gemini(gemini, monkeypatch):
    providers.build_gemini_index()
    calls_after_index = len(gemini.calls)
    _inspect_and_set_cutoff(monkeypatch, cutoff="-1")
    inspected = len(gemini.calls) - calls_after_index
    assert inspected == 8, "inspect-scores: one embedding query per reviewable clause"
    assert gemini.count("chat") == 0, "inspect-scores makes no chat call"
    record = providers.gemini_inspection()
    assert record and len(record["clauses"]) == 8
    assert "contract_text" not in json.dumps(record) and "FICTIONAL" not in json.dumps(record)

    text = embed.read_contract_pdf(SAMPLE.read_bytes())["text"]
    policy = retry.CallPolicy(provider="Gemini", budget=retry.CallBudget(embed=16, chat=8))
    pipeline = report.live_pipeline(policy=policy)
    rows = report.run_contract_review(
        text,
        retriever=pipeline["retriever"],
        comparer=pipeline["comparer"],
        policy=policy,
        tuning=pipeline["tuning"],
        run_info=pipeline["run_info"],
    )
    assert len(rows) == 9 and report.is_preamble(rows[0])
    assert all(row["provider"] == "gemini" for row in rows)
    assert all(row["chat_model"] == "gemini-3.8-flash" for row in rows)
    assert all(row["embedding_model"] == "gemini-embedding-001" for row in rows)
    assert policy.attempts["chat"] <= 8 and policy.attempts["embed"] <= 16
    assert all(call[2] == "application/json" for call in gemini.calls if call[0] == "chat")
    parsed = list(csv.DictReader(io.StringIO(report.to_csv(rows))))
    assert {row["provider"] for row in parsed} == {"gemini"}
    page = report.to_html(rows, "synthetic-contract-v2.pdf")
    assert "<th>Provider</th><td>gemini</td>" in page
    assert "<th>Chat model</th><td>gemini-3.8-flash</td>" in page


def test_cutoff_alone_does_not_open_the_chat_gate(gemini, monkeypatch):
    monkeypatch.setenv("GEMINI_MIN_SIMILARITY", "0.5")
    allowed, reason = providers.chat_allowed(providers.settings())
    assert not allowed and "inspect" in reason


def test_cutoff_blocks_low_similarity_before_gemini_comparison(gemini, monkeypatch):
    monkeypatch.setenv("GEMINI_MIN_SIMILARITY", "0.73")
    called = {"value": False}

    def retriever(_text, category=None, k=8):
        return [{
            "evidence_id": "payment-01", "text": "Customer pays invoices within thirty days.",
            "category": "Payment", "title": "Invoice timing", "distance": 0.271,
            "similarity": 0.729,
        }]

    def comparer(*_args):
        called["value"] = True
        raise AssertionError("low-confidence contract text must not reach Gemini")

    result = analyze.analyze_clause(
        {"id": "sample-1", "text": "The customer pays all invoices at once."},
        0,
        retriever,
        comparer,
        tuning=analyze.tuning_for(providers.settings()),
    )
    assert not called["value"]
    assert not result["evidence_id"]
    assert any("No relevant playbook clause" in warning for warning in result["warnings"])


def test_inspection_alone_does_not_open_the_chat_gate(gemini):
    providers.build_gemini_index()
    embed.inspect_scores(str(SAMPLE), out=lambda line: None)
    allowed, reason = providers.chat_allowed(providers.settings())
    assert not allowed and "GEMINI_MIN_SIMILARITY" in reason
    with pytest.raises(providers.ProviderNotReady):
        report.live_pipeline()


def test_inspection_is_invalidated_by_a_playbook_change(gemini, monkeypatch, tmp_path):
    providers.build_gemini_index()
    embed.inspect_scores(str(SAMPLE), out=lambda line: None)
    assert providers.gemini_inspection() is not None
    changed = tmp_path / "playbook.json"
    changed.write_text(Path(utils.PLAYBOOK_FILE).read_text(encoding="utf-8") + "\n", encoding="utf-8")
    monkeypatch.setattr(utils, "PLAYBOOK_FILE", str(changed))
    assert providers.gemini_inspection() is None


# --- no fallback --------------------------------------------------------------------


def test_gemini_failure_does_not_fall_back_to_openai(gemini, monkeypatch):
    providers.build_gemini_index()
    _inspect_and_set_cutoff(monkeypatch, cutoff="-1")
    gemini.fail_chat = genai_errors.ClientError(
        402, {"error": {"code": 402, "message": "secret detail", "status": "PAYMENT_REQUIRED"}}
    )
    text = "1. The customer pays each invoice within ten days of receipt.\n2. Either party may terminate on notice.\n"
    pipeline = report.live_pipeline()
    rows = report.run_contract_review(
        text, retriever=pipeline["retriever"], comparer=pipeline["comparer"], tuning=pipeline["tuning"], run_info=pipeline["run_info"]
    )
    assert all(row["label"] == "Review required" for row in rows)
    assert "Retrying will not help." in " ".join(rows[0]["warnings"])
    assert "secret detail" not in json.dumps(rows)
    assert gemini.count("chat") == 1, "no retry for no-credit, and the review stopped"
    # OpenAI constructors raise Touched: reaching this line means none was built.


def test_openai_failure_does_not_use_gemini(monkeypatch):
    monkeypatch.setattr(providers, "gemini_client", _refuse)

    def failing_retriever(text, category=None, k=3):
        request = __import__("httpx").Request("POST", "https://api.openai.com/v1/embeddings")
        raise openai.APIConnectionError(request=request)

    rows = report.run_contract_review(
        "1. The customer pays each invoice within ten days of receipt.\n", retriever=failing_retriever, comparer=_refuse
    )
    assert rows[0]["label"] == "Review required"
    assert rows[0]["provider"] == "openai"


def test_regulation_chat_is_openai_only(gemini, monkeypatch):
    with pytest.raises(Exception):
        agent.NewsChat("session-x")  # would need OpenAI; the app never offers it under Gemini


# --- preflight and key handling ------------------------------------------------------


def test_gemini_preflight_offline(gemini, capsys, monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    with pytest.raises(SystemExit) as caught:
        embed.main(["playbook", "--preflight", "--pdf", str(SAMPLE)])
    out = capsys.readouterr().out
    assert caught.value.code == 0
    assert "Provider:            gemini (AEGIS_PROVIDER)" in out
    assert "Embedding model:     gemini-embedding-001, 768 dimensions" in out
    assert "Chat model:          gemini-3.8-flash (default)" in out
    assert "API key present:     yes (GEMINI_API_KEY)" in out
    assert "Planned index calls: 30 embeddings request(s) for 30 passages" in out
    assert "playbook_clauses_gemini - missing" in out
    assert "Similarity cutoff:   uninspected" in out and "chat calls refused" in out
    assert providers.GEMINI_BANNER in out
    assert "https://ai.google.dev/gemini-api/terms" in out
    assert GEMINI_SECRET not in out
    assert gemini.calls == []


def test_gemini_preflight_without_key_fails(gemini, capsys, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(SystemExit) as caught:
        embed.main(["playbook", "--preflight"])
    assert caught.value.code == 1
    assert "API key present:     no (GEMINI_API_KEY)" in capsys.readouterr().out


def test_key_never_appears_in_descriptions_or_errors(gemini):
    assert GEMINI_SECRET not in json.dumps(providers.describe(), default=str)
    failure = errors.classify(genai_errors.ClientError(400, {"error": {"status": "INVALID_ARGUMENT", "message": GEMINI_SECRET}}))
    assert GEMINI_SECRET not in errors.user_message(failure)
    assert errors.safe_code(GEMINI_SECRET) == ""


# --- Gemini error classification -------------------------------------------------------


def _genai(code, status=None, reason=None, extra=None):
    body = {"error": {"code": code, "message": "m", "status": status}}
    if reason or extra:
        body["error"]["details"] = [dict(reason and {"reason": reason} or {}, **(extra or {}))]
    cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return cls(code, body)


@pytest.mark.parametrize(
    "exc,kind",
    [
        (_genai(402, "PAYMENT_REQUIRED"), errors.NO_CREDIT),
        (_genai(400, "FAILED_PRECONDITION"), errors.NO_CREDIT),
        (_genai(400, "INVALID_ARGUMENT", reason="API_KEY_INVALID"), errors.AUTH),
        (_genai(403, "PERMISSION_DENIED"), errors.PERMISSION),
        (_genai(429, "RESOURCE_EXHAUSTED"), errors.RATE_LIMIT),
        (_genai(429, "RESOURCE_EXHAUSTED", extra={"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}), errors.DAILY_QUOTA),
        (_genai(503, "UNAVAILABLE"), errors.SERVER),
        (_genai(504, "DEADLINE_EXCEEDED"), errors.TIMEOUT),
        (_genai(400, "INVALID_ARGUMENT"), errors.BAD_REQUEST),
    ],
)
def test_gemini_errors_are_classified(exc, kind):
    failure = errors.classify(exc, "Gemini")
    assert failure.kind == kind
    assert failure.provider == "Gemini"
    message = errors.user_message(failure)
    if kind in (errors.NO_CREDIT, errors.DAILY_QUOTA):
        assert "will not help" in message


def test_gemini_retry_delay_is_read_from_retry_info():
    exc = _genai(429, "RESOURCE_EXHAUSTED", extra={"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"})
    assert errors.classify(exc).retry_after == 7.0


def test_gemini_rate_limit_is_retried_but_payment_is_not(isolated_environment):
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _genai(429, "RESOURCE_EXHAUSTED")
        return "ok"

    assert retry.CallPolicy(provider="Gemini").call("embed", flaky) == "ok"
    assert attempts["n"] == 2

    def broke():
        attempts["n"] += 1
        raise _genai(402, "PAYMENT_REQUIRED")

    attempts["n"] = 0
    with pytest.raises(genai_errors.ClientError):
        retry.CallPolicy(provider="Gemini").call("embed", broke)
    assert attempts["n"] == 1
