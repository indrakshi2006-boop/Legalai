"""A failed index build never leaves an empty or partial playbook collection."""

import hashlib

import chromadb
import pytest
from chromadb import EmbeddingFunction

import embed
import utils


class HashEmbedding(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input):
        return [[byte / 255.0 for byte in hashlib.sha256(text.encode()).digest()[:8]] for text in input]

    @staticmethod
    def name():
        return "hash-embedding"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return HashEmbedding()


class FailingEmbedding(HashEmbedding):
    def __call__(self, input):
        raise RuntimeError("embeddings request failed")


class ShortEmbedding(HashEmbedding):
    def __call__(self, input):
        return super().__call__(input)[:-1]


@pytest.fixture
def store(tmp_path):
    yield str(tmp_path)
    chromadb.api.client.SharedSystemClient.clear_system_cache()


def _names(path):
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    client = chromadb.PersistentClient(path=path)
    return [getattr(item, "name", item) for item in client.list_collections()]


def test_failed_embedding_request_creates_nothing(store):
    with pytest.raises(RuntimeError):
        embed.build_playbook_index(persist_directory=store, embedding_function=FailingEmbedding())
    assert utils.PLAYBOOK_COLLECTION not in _names(store)
    assert embed.chroma_collection_exists(utils.PLAYBOOK_COLLECTION, store) is False


def test_short_embedding_response_creates_nothing(store):
    with pytest.raises(embed.IndexIncomplete):
        embed.build_playbook_index(persist_directory=store, embedding_function=ShortEmbedding())
    assert utils.PLAYBOOK_COLLECTION not in _names(store)


def test_existing_empty_collection_is_removed_after_a_failure(store):
    client = chromadb.PersistentClient(path=store)
    client.get_or_create_collection(utils.PLAYBOOK_COLLECTION, embedding_function=HashEmbedding())
    with pytest.raises(RuntimeError):
        embed.build_playbook_index(persist_directory=store, embedding_function=FailingEmbedding())
    assert utils.PLAYBOOK_COLLECTION not in _names(store)


def test_failed_write_deletes_the_partial_collection(store, monkeypatch):
    real_upsert = chromadb.api.models.Collection.Collection.upsert

    def partial_upsert(self, ids, documents=None, embeddings=None, metadatas=None, **kwargs):
        real_upsert(self, ids=ids[:5], documents=documents[:5], embeddings=embeddings[:5], metadatas=metadatas[:5])
        raise RuntimeError("disk full")

    monkeypatch.setattr(chromadb.api.models.Collection.Collection, "upsert", partial_upsert)
    with pytest.raises(RuntimeError):
        embed.build_playbook_index(persist_directory=store, embedding_function=HashEmbedding())
    assert utils.PLAYBOOK_COLLECTION not in _names(store)


def test_complete_index_survives_a_failed_request_before_any_write(store):
    assert embed.build_playbook_index(persist_directory=store, embedding_function=HashEmbedding()) == 30
    with pytest.raises(RuntimeError):
        embed.build_playbook_index(persist_directory=store, embedding_function=FailingEmbedding())
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    assert embed.chroma_collection_exists(utils.PLAYBOOK_COLLECTION, store) is True


def test_ids_removed_from_the_playbook_are_removed_from_the_index(store, tmp_path):
    import json

    embed.build_playbook_index(persist_directory=store, embedding_function=HashEmbedding())
    records = embed.load_playbook(utils.PLAYBOOK_FILE)
    extra = dict(records[0], id="termination-99")
    path = tmp_path / "pb.json"
    path.write_text(json.dumps(records + [extra]), encoding="utf-8")
    assert embed.build_playbook_index(str(path), persist_directory=store, embedding_function=HashEmbedding()) == 31
    assert embed.build_playbook_index(persist_directory=store, embedding_function=HashEmbedding()) == 30
