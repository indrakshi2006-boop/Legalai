"""Local pipeline check. This does not call OpenAI and is not a live review."""

import hashlib
import shutil

import chromadb
from chromadb import EmbeddingFunction
import pymupdf as fitz

import agent
import analyze
import app
import embed
import report
import risk
import segment
import utils


class HashEmbedding(EmbeddingFunction):
    """Local vectors so this check does not call the embeddings API."""

    def __init__(self):
        pass

    def __call__(self, input):
        vectors = []
        for text in input:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([byte / 255.0 for byte in digest[:8]])
        return vectors

    @staticmethod
    def name():
        return "hash-embedding"

    def get_config(self):
        return {"source": "test"}

    @staticmethod
    def build_from_config(config):
        return HashEmbedding()


def test_modules_import():
    assert app.qa_session_id
    assert agent.NewsChat
    assert analyze.analyze_clause
    assert embed.build_playbook_index
    assert report.to_html
    assert risk.score_analysis
    assert segment.segment_contract
    assert utils.PLAYBOOK_COLLECTION != utils.DOCUMENT_COLLECTION


def test_local_index_review_and_export(monkeypatch, tmp_path):
    def forbid_live_client(*args, **kwargs):
        raise AssertionError("A live model client was constructed.")

    monkeypatch.setattr(analyze, "ChatOpenAI", forbid_live_client)
    monkeypatch.setattr(analyze, "OpenAIEmbeddings", forbid_live_client)
    monkeypatch.setattr(agent, "ChatOpenAI", forbid_live_client)
    monkeypatch.setattr(agent, "OpenAIEmbeddings", forbid_live_client)

    directory = str(tmp_path)
    count = embed.build_playbook_index(
        playbook_path=utils.PLAYBOOK_FILE,
        persist_directory=directory,
        embedding_function=HashEmbedding(),
    )
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    client = chromadb.PersistentClient(path=directory)
    collection = client.get_collection(utils.PLAYBOOK_COLLECTION, embedding_function=HashEmbedding())
    stored = collection.get(include=["documents", "metadatas"])
    records = embed.load_playbook(utils.PLAYBOOK_FILE)
    assert count == len(records)
    assert set(stored["ids"]) == {record["id"] for record in records}
    by_id = {
        item_id: (document, metadata)
        for item_id, document, metadata in zip(stored["ids"], stored["documents"], stored["metadatas"])
    }
    assert all(metadata["clause_id"] == item_id for item_id, metadata in zip(stored["ids"], stored["metadatas"]))

    extracted = embed.read_contract_pdf(
        _pdf(
            "1. The customer pays <script>alert(1)</script> within ten days of the invoice.\n\n"
            "2. Only the supplier may terminate immediately without notice.\n"
        )
    )
    assert extracted["ok"] is True

    def retriever(text, category=None, k=3):
        if "terminat" in text.lower():
            chosen = "Termination"
        elif "pay" in text.lower() or "invoice" in text.lower():
            chosen = "Payment"
        else:
            chosen = category
        matches = [record for record in records if chosen is None or record["category"] == chosen]
        hits = []
        for record in matches[:k]:
            document, metadata = by_id[record["id"]]
            hits.append(
                {
                    "evidence_id": metadata["clause_id"],
                    "text": document,
                    "category": metadata["category"],
                    "title": metadata["title"],
                    "distance": 0.2,
                    "similarity": 0.8,
                }
            )
        return hits

    def comparer(contract_text, hits):
        hit = hits[0]
        return {
            "evidence_id": hit["evidence_id"],
            "deviations": ["Local fixture difference. This is not a live model review."],
            "explanation": "Local fixture comparison. This is not a live model review.",
            "severity": "medium",
        }

    rows = report.run_contract_review(extracted["text"], retriever=retriever, comparer=comparer)
    assert len(rows) >= 2
    assert {row["clause_id"] for row in rows}
    assert all(row["label"] in report.LABELS for row in rows)
    assert all(row["evidence_id"] in by_id for row in rows)
    assert all("not a live model review" in row["explanation"] for row in rows)
    csv_text = report.to_csv(rows)
    html_text = report.to_html(rows, "sample-contract.pdf")
    assert "clause-001" in csv_text
    assert "<script>" not in html_text
    assert "&lt;script&gt;" in html_text
    assert "Local fixture comparison" in html_text
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    shutil.rmtree(directory, ignore_errors=True)


def _pdf(text):
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(fitz.Rect(72, 72, 540, 700), text)
    data = document.tobytes()
    document.close()
    return data
