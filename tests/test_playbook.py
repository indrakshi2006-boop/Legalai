import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import chromadb
from chromadb import EmbeddingFunction

import embed
import utils


class HashEmbedding(EmbeddingFunction):
    """Deterministic local vectors so index tests do not call OpenAI."""

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


class PlaybookValidationTest(unittest.TestCase):
    def test_playbook_file_is_valid(self):
        records = embed.load_playbook(utils.PLAYBOOK_FILE)
        embed.validate_playbook(records)
        self.assertGreaterEqual(len(records), 30)
        self.assertEqual(len(records), len({record["id"] for record in records}))
        self.assertEqual(
            set(utils.PLAYBOOK_CATEGORIES),
            {record["category"] for record in records},
        )

    def test_duplicate_id_is_rejected(self):
        records = embed.load_playbook(utils.PLAYBOOK_FILE)
        records.append(dict(records[0]))
        with self.assertRaises(ValueError):
            embed.validate_playbook(records)

    def test_missing_category_is_rejected(self):
        records = [
            record
            for record in embed.load_playbook(utils.PLAYBOOK_FILE)
            if record["category"] != "Payment"
        ]
        with self.assertRaises(ValueError):
            embed.validate_playbook(records)

    def test_empty_required_field_is_rejected(self):
        records = embed.load_playbook(utils.PLAYBOOK_FILE)
        records[0]["title"] = "  "
        with self.assertRaises(ValueError):
            embed.validate_playbook(records)


class PlaybookIndexTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_reindex_updates_the_same_ids(self):
        client = chromadb.PersistentClient(path=self.temp_dir)
        unrelated = client.get_or_create_collection(
            name=utils.DOCUMENT_COLLECTION,
            embedding_function=HashEmbedding(),
        )
        unrelated.upsert(ids=["keep-me"], documents=["unrelated document"])

        first_count = embed.build_playbook_index(
            playbook_path=utils.PLAYBOOK_FILE,
            persist_directory=self.temp_dir,
            embedding_function=HashEmbedding(),
        )
        second_count = embed.build_playbook_index(
            playbook_path=utils.PLAYBOOK_FILE,
            persist_directory=self.temp_dir,
            embedding_function=HashEmbedding(),
        )
        self.assertEqual(first_count, second_count)
        self.assertGreaterEqual(first_count, 30)

        playbook_path = Path(self.temp_dir) / "playbook.json"
        records = embed.load_playbook(utils.PLAYBOOK_FILE)
        records[0]["standard_text"] += " Updated marker."
        playbook_path.write_text(json.dumps(records), encoding="utf-8")
        third_count = embed.build_playbook_index(
            playbook_path=str(playbook_path),
            persist_directory=self.temp_dir,
            embedding_function=HashEmbedding(),
        )

        chromadb.api.client.SharedSystemClient.clear_system_cache()
        checked = chromadb.PersistentClient(path=self.temp_dir)
        playbook = checked.get_collection(
            utils.PLAYBOOK_COLLECTION,
            embedding_function=HashEmbedding(),
        )
        stored = playbook.get(ids=[records[0]["id"]], include=["documents", "metadatas"])
        self.assertEqual(third_count, first_count)
        self.assertEqual(playbook.count(), first_count)
        self.assertIn("Updated marker.", stored["documents"][0])
        self.assertEqual(stored["metadatas"][0]["clause_id"], records[0]["id"])
        self.assertEqual(stored["metadatas"][0]["category"], records[0]["category"])

        document_collection = checked.get_collection(
            utils.DOCUMENT_COLLECTION,
            embedding_function=HashEmbedding(),
        )
        self.assertEqual(document_collection.count(), 1)
        self.assertEqual(document_collection.get(ids=["keep-me"])["ids"], ["keep-me"])
        self.assertNotEqual(utils.PLAYBOOK_COLLECTION, utils.DOCUMENT_COLLECTION)
        self.assertEqual(utils.EMBEDDING_MODEL, "text-embedding-ada-002")

    def test_playbook_index_requires_every_validated_record(self):
        records = embed.validate_playbook(embed.load_playbook(utils.PLAYBOOK_FILE))
        self.assertEqual(len(records), 30)
        client = chromadb.PersistentClient(path=self.temp_dir)
        client.get_or_create_collection(
            name=utils.PLAYBOOK_COLLECTION,
            embedding_function=HashEmbedding(),
        )
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        self.assertFalse(
            embed.chroma_collection_exists(utils.PLAYBOOK_COLLECTION, self.temp_dir)
        )

        chromadb.api.client.SharedSystemClient.clear_system_cache()
        partial = chromadb.PersistentClient(path=self.temp_dir)
        partial.get_collection(
            utils.PLAYBOOK_COLLECTION,
            embedding_function=HashEmbedding(),
        ).upsert(
            ids=[records[0]["id"]],
            documents=[records[0]["standard_text"]],
            metadatas=[
                {
                    "clause_id": records[0]["id"],
                    "category": records[0]["category"],
                    "title": records[0]["title"],
                    "red_flags": " | ".join(records[0]["red_flags"]),
                }
            ],
        )
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        self.assertFalse(
            embed.chroma_collection_exists(utils.PLAYBOOK_COLLECTION, self.temp_dir)
        )

        embed.build_playbook_index(
            playbook_path=utils.PLAYBOOK_FILE,
            persist_directory=self.temp_dir,
            embedding_function=HashEmbedding(),
        )
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        self.assertTrue(
            embed.chroma_collection_exists(utils.PLAYBOOK_COLLECTION, self.temp_dir)
        )


if __name__ == "__main__":
    unittest.main()
